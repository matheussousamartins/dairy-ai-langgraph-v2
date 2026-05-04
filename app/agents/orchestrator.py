"""
agents/orchestrator.py â€" Orquestrador multi-agente com execuÃ§Ã£o paralela

Fluxo do grafo:
  classify â†’ route â†’ execute (paralelo) â†’ consolidate â†’ END
                â†˜ respond_direct â†’ consolidate â†’ END

Agente 3 (Regulatórios) é incluído em toda pergunta de laticínios.
Agente 0 (Base Geral) é incluído apenas para glossário e terminologia —
sua base não cobre queries técnicas ou regulatórias genéricas.

Execução paralela:
  Todos os agentes rodam ao mesmo tempo via asyncio.gather + ainvoke.
  Latência total = tempo do agente mais lento (não a soma).
"""

import asyncio
import os
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Annotated, Tuple
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from app.config import (
    LLM_MODEL,
    CLASSIFIER_TEMPERATURE,
    CONSOLIDATION_TEMPERATURE,
    CONSOLIDATION_TIMEOUT_SEC,
    DIRECT_TEMPERATURE,
    ORCHESTRATOR_FASTPATH,
    CLASSIFICATION_CACHE_SIZE,
    ENABLE_GENERAL_INDEX_FALLBACK,
    GENERAL_INDEX_FALLBACK_SEARCH_TYPE,
    GENERAL_INDEX_FALLBACK_PER_TABLE_K,
    GENERAL_INDEX_FALLBACK_FINAL_K,
    GENERAL_INDEX_FALLBACK_MIN_RESULTS,
    GENERAL_INDEX_FALLBACK_MAX_TABLES,
    GENERAL_INDEX_FALLBACK_ONLY_ON_WEAK,
    GENERAL_INDEX_FALLBACK_REQUIRE_DAIRY_SIGNAL,
    ENABLE_WEB_FALLBACK,
    WEB_FALLBACK_PROVIDER,
    WEB_FALLBACK_TIMEOUT_SEC,
    WEB_FALLBACK_MAX_RESULTS,
    WEB_FALLBACK_MAX_SOURCES,
    WEB_FALLBACK_ONLY_ON_WEAK,
    WEB_FALLBACK_REQUIRE_DAIRY_SIGNAL,
    WEB_FALLBACK_REQUIRE_GENERAL_FALLBACK_FIRST,
    WEB_FALLBACK_FETCH_FULLTEXT,
    WEB_FALLBACK_MAX_PAGE_CHARS,
    WEB_FALLBACK_MAX_SNIPPET_CHARS,
    WEB_FALLBACK_ALLOWED_DOMAINS,
    MATCH_THRESHOLD,
)
from app.agents.prompts import get_orchestrator_prompt
from app.agents.agent_config import AGENTS, get_agent_by_id
from app.agents.base_agent import get_agent_graph
from app.rag.search import embed_query, search_general_knowledge_base
from app.tools.web_fallback import (
    search_web_duckduckgo,
    enrich_results_with_page_content,
    build_web_fallback_evidence,
)

# Tempo máximo de espera por agente (segundos)
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "12"))
_SPECIALISTS_DESC = "".join(
    f"  {agent['agent_id']} = {agent['name']}\n"
    for agent in AGENTS
    if agent["agent_id"] not in (0, 3, 5, 6)
)

_CLASSIFICATION_CACHE: "OrderedDict[str, List[int]]" = OrderedDict()
_MAX_CLASSIFICATION_CACHE = max(0, CLASSIFICATION_CACHE_SIZE)
_ROUTING_RULES_PATH = Path("docs/orchestrator/routing_rules.yaml")


def _load_routing_rules() -> Dict[str, Any]:
    """Carrega regras de sinalização de docs/orchestrator/routing_rules.yaml.

    Se o arquivo não existir ou falhar no parse, loga um warning e retorna
    sets vazios — o sistema ainda funciona via LLM classifier com menor
    precisão no fast-path. Nenhuma exceção é propagada.
    """
    import logging
    _log = logging.getLogger(__name__)

    _empty: Dict[str, Any] = {
        "greetings": frozenset(),
        "dairy_signal_terms": frozenset(),
        "quality_lab_terms": frozenset(),
        "regulatory_strong_terms": frozenset(),
        "legal_requirement_phrases": frozenset(),
        "quality_strong_terms": frozenset(),
        "general_knowledge_terms": frozenset(),
        "fermented_strong_terms": frozenset(),
        "cheese_strong_terms": frozenset(),
        "intent_patterns_by_agent": {},
        "low_precision_keywords": frozenset(),
        "hint_noise_terms": frozenset(),
        "hint_noise_tokens": frozenset(),
        "specialist_strong_hints_default": {},
    }

    if not _ROUTING_RULES_PATH.exists():
        _log.warning(
            "routing_rules.yaml não encontrado em %s — fast-path com sets vazios. "
            "Roteamento delegado ao LLM classifier.",
            _ROUTING_RULES_PATH,
        )
        return _empty

    try:
        import yaml  # type: ignore
        raw = yaml.safe_load(_ROUTING_RULES_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        _log.warning(
            "Erro ao carregar routing_rules.yaml: %s — fast-path com sets vazios.", exc
        )
        return _empty

    def _fs(data: Any) -> frozenset:
        if isinstance(data, list):
            return frozenset(str(x).strip() for x in data if str(x).strip())
        return frozenset()

    domain = raw.get("domain_signals", {}) or {}
    regulatory = domain.get("regulatory", {}) or {}
    quality = domain.get("quality_leite", {}) or {}
    fermented = domain.get("fermented", {}) or {}
    cheese = domain.get("cheese", {}) or {}
    general = domain.get("general_knowledge", {}) or {}
    noise = raw.get("noise_control", {}) or {}

    # Intent patterns: dict[int, tuple[str, ...]]
    intent_patterns: Dict[int, tuple] = {}
    for k, v in (raw.get("intent_patterns_by_agent", {}) or {}).items():
        try:
            aid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, list):
            intent_patterns[aid] = tuple(str(p) for p in v if str(p).strip())

    # Specialist strong hints: dict[int, set]
    specialist_hints: Dict[int, set] = {}
    for k, v in (raw.get("specialist_strong_hints", {}) or {}).items():
        try:
            aid = int(k)
        except (TypeError, ValueError):
            continue
        specialist_hints[aid] = (
            {str(h).strip() for h in v if str(h).strip()} if isinstance(v, list) else set()
        )

    return {
        "greetings": _fs(raw.get("greetings", [])),
        "dairy_signal_terms": _fs(raw.get("dairy_signal_terms", [])),
        "quality_lab_terms": _fs(raw.get("quality_lab_terms", [])),
        "regulatory_strong_terms": _fs(regulatory.get("strong_terms", [])),
        "legal_requirement_phrases": _fs(regulatory.get("legal_requirement_phrases", [])),
        "quality_strong_terms": _fs(quality.get("strong_terms", [])),
        "general_knowledge_terms": _fs(general.get("terms", [])),
        "fermented_strong_terms": _fs(fermented.get("strong_terms", [])),
        "cheese_strong_terms": _fs(cheese.get("strong_terms", [])),
        "intent_patterns_by_agent": intent_patterns,
        "low_precision_keywords": _fs(raw.get("low_precision_keywords", [])),
        "hint_noise_terms": _fs(noise.get("hint_noise_terms", [])),
        "hint_noise_tokens": _fs(noise.get("hint_noise_tokens", [])),
        "specialist_strong_hints_default": specialist_hints,
    }


_ROUTING_RULES = _load_routing_rules()

# Sets de sinalização carregados do YAML — nomes preservados para compatibilidade
# com todas as funções de detecção (_is_strong_regulatory_signal, etc.)
_GREETINGS: frozenset                    = _ROUTING_RULES["greetings"]
_DAIRY_TERMS: frozenset                  = _ROUTING_RULES["dairy_signal_terms"]
_QUALITY_LAB_TERMS: frozenset            = _ROUTING_RULES["quality_lab_terms"]
_REGULATORY_STRONG_TERMS: frozenset      = _ROUTING_RULES["regulatory_strong_terms"]
_LEGAL_REQUIREMENT_DIRECT_PHRASES: frozenset = _ROUTING_RULES["legal_requirement_phrases"]
_QUALITY_STRONG_TERMS: frozenset         = _ROUTING_RULES["quality_strong_terms"]
_GENERAL_KNOWLEDGE_TERMS: frozenset      = _ROUTING_RULES["general_knowledge_terms"]
_FERMENTED_STRONG_TERMS: frozenset       = _ROUTING_RULES["fermented_strong_terms"]
_CHEESE_STRONG_TERMS: frozenset          = _ROUTING_RULES["cheese_strong_terms"]
_INTENT_PATTERNS_BY_AGENT: Dict[int, tuple] = _ROUTING_RULES["intent_patterns_by_agent"]

_CLASSIFIER_FEW_SHOTS = """
FEW-SHOTS (padrao esperado):
- Pergunta: "Qual teste qualitativo detecta formaldeído no leite e qual indicação de positivo?"
  agent_ids: [3,4]
  confidence: 0.96
  reason: "Teste qualitativo/adulterante em leite -> Qualidade do Leite. Sem sinal de glossário."
  alternatives: [5]

- Pergunta: "Quais açúcares são responsáveis pelo escurecimento do queijo na pizza?"
  agent_ids: [3,1]
  confidence: 0.94
  reason: "Tecnologia de queijo (browning/derretimento) -> Queijos. Sem sinal de glossário."
  alternatives: [5]

- Pergunta: "Quanto da lactose pode ser transformada em ácido lático pelas bactérias do iogurte?"
  agent_ids: [3,2]
  confidence: 0.95
  reason: "Fermentação em iogurte -> Fermentados. Sem sinal de glossário."
  alternatives: [4]

- Pergunta: "Posso rotular como light se reduzir só 10% de sódio?"
  agent_ids: [3]
  confidence: 0.97
  reason: "Critério de rotulagem regulatória -> Regulatórios. Sem sinal de glossário."
  alternatives: [6]

- Pergunta: "Explique diferença entre ESD e EST no leite e impacto tecnológico."
  agent_ids: [3,4]
  confidence: 0.93
  reason: "Composição físico-química e impacto tecnológico -> Qualidade do Leite. Sem glossário."
  alternatives: [1]

- Pergunta: "Se eu usar aroma de fumaça, a rotulagem vira defumado ou sabor defumado?"
  agent_ids: [3]
  confidence: 0.97
  reason: "Denominação de venda/rotulagem regulatória. Sem sinal de glossário."
  alternatives: [1]

- Pergunta: "Como calcular acidez titulável em leite fluido pelo método Dornic?"
  agent_ids: [3,4]
  confidence: 0.96
  reason: "Método analítico de qualidade do leite (Dornic). Sem sinal de glossário."
  alternatives: []

- Pergunta: "Qual é a capital da França?"
  agent_ids: []
  confidence: 0.99
  reason: "Fora do escopo de laticínios."
  alternatives: []

- Pergunta: "No relatório final, posso escrever Starter ou preciso usar o termo padronizado do glossário?"
  agent_ids: [0,3]
  confidence: 0.92
  reason: "Sinal explícito de glossário/terminologia institucional -> agente 0 obrigatório."
  alternatives: [2]

- Pergunta: "Como rotular o provolone fresco quando for usado aroma de fumaça?"
  agent_ids: [3]
  confidence: 0.95
  reason: "Rotulagem e denominação de venda são temas regulatórios. Sem glossário."
  alternatives: [1]

- Pergunta: "Qual rotação e tempo da centrifugação no ácido siálico?"
  agent_ids: [3,4]
  confidence: 0.95
  reason: "Método analítico laboratorial da IN 68 -> Qualidade do Leite. Sem glossário."
  alternatives: []

- Pergunta: "Em qual faixa de pH deve ser feito o corte da coalhada no queijo?"
  agent_ids: [3,1]
  confidence: 0.93
  reason: "Parâmetro de processo de fabricação de queijo -> Queijos. Sem glossário."
  alternatives: [2]

- Pergunta: "Como rotular o provolone maturado quando for usado aroma de fumaça?"
  agent_ids: [3]
  confidence: 0.95
  reason: "Rotulagem/denominação normativa -> Regulatórios. Sem glossário."
  alternatives: [1,6]

- Pergunta: "No relatório final, posso manter Rennet em inglês ou devo padronizar o termo?"
  agent_ids: [0,3]
  confidence: 0.92
  reason: "Padronização terminológica/glossário — 'Rennet' é sinal explícito de glossário. Agente 0 obrigatório."
  alternatives: [1,2]

- Pergunta: "Como CCS elevada influencia o risco de amargor no queijo?"
  agent_ids: [3,1,4]
  confidence: 0.88
  reason: "CCS é qualidade do leite (4), impacto no queijo (amargor) é tecnologia (1). Sem glossário."
  alternatives: []

- Pergunta: "Quais fatores de processo aumentam o risco de CLC no queijo?"
  agent_ids: [3,1]
  confidence: 0.92
  reason: "CLC é defeito técnico de queijo -> Queijos. Sem sinal de glossário."
  alternatives: []

- Pergunta: "Na prevenção de estufamento tardio, quais abordagens o documento compara?"
  agent_ids: [3,1]
  confidence: 0.93
  reason: "Estufamento tardio é defeito técnico de queijo (Clostridium) -> Queijos."
  alternatives: []

- Pergunta: "Na prática de captação de leite, qual tensão o texto aponta ao falar de qualidade e prevenção de amargor?"
  agent_ids: [3,1]
  confidence: 0.90
  reason: "Amargor é defeito técnico de queijo -> Queijos. 'Qualidade' aqui é contexto, não domínio."
  alternatives: [4]

- Pergunta: "Como NSLAB influenciam a formação de CLC?"
  agent_ids: [3,1]
  confidence: 0.92
  reason: "CLC é defeito de queijo, NSLAB são bactérias adjuntas de queijo -> Queijos."
  alternatives: []

- Pergunta: "Para formar textura grana correta, o que precisa estar alinhado no processo?"
  agent_ids: [3,1]
  confidence: 0.91
  reason: "Textura grana é característica de queijos duros -> Queijos. Sem glossário."
  alternatives: []

- Pergunta: "Para dizer que um produto não contém gordura total, basta zerar gordura total?"
  agent_ids: [3]
  confidence: 0.96
  reason: "Alegação de ausência de nutriente é rotulagem regulatória (RDC 54). Sem glossário."
  alternatives: []

- Pergunta: "Sou obrigado a ter local para produto suspeito, reinspeção e aproveitamento condicional?"
  agent_ids: [3]
  confidence: 0.96
  reason: "Requisito de instalações para fiscalização — RIISPOA. Sem glossário."
  alternatives: []

- Pergunta: "Como deve ser a denominação quando há ingredientes adicionais?"
  agent_ids: [3]
  confidence: 0.95
  reason: "Denominação de venda é rotulagem regulatória. Sem sinal de glossário."
  alternatives: [1]

- Pergunta: "Qual resultado positivo no método B para análise do leite?"
  agent_ids: [3,4]
  confidence: 0.94
  reason: "Método B é método analítico da IN 68 -> Qualidade do Leite. Sem glossário."
  alternatives: []

- Pergunta: "As proteínas do soro conseguem formar gel com a mesma eficiência da caseína?"
  agent_ids: [3,2]
  confidence: 0.88
  reason: "Geleificação de proteínas do soro é relevante para fermentados -> Fermentados."
  alternatives: [1]

- Pergunta: "Qual categoria de iogurte é descrita como indulgente e aveludada?"
  agent_ids: [3,2]
  confidence: 0.91
  reason: "Categorias sensoriais de iogurte -> Fermentados. Sem sinal de glossário."
  alternatives: []

- Pergunta: "Quais características sensoriais o minas padrão deve apresentar segundo a normativa?"
  agent_ids: [3]
  confidence: 0.96
  reason: "Padrão de identidade sensorial definido em IN -> Regulatórios. Sem glossário."
  alternatives: [1]

- Pergunta: "Como deve ser rotulado o provolone defumado quando usado aroma artificial?"
  agent_ids: [3]
  confidence: 0.96
  reason: "Denominação de venda com aroma artificial -> Regulatórios. Sem glossário."
  alternatives: [1]

- Pergunta: "O minas frescal pode conter substâncias estranhas de acordo com a IN vigente?"
  agent_ids: [3]
  confidence: 0.97
  reason: "Padrão de identidade/substâncias estranhas -> Regulatórios. Sem glossário."
  alternatives: []

- Pergunta: "Quais as formas de apresentação permitidas para a ricota segundo a normativa?"
  agent_ids: [3]
  confidence: 0.96
  reason: "Formas de apresentação definidas em IN -> Regulatórios. Sem glossário."
  alternatives: [1]

- Pergunta: "Quais cuidados de segurança devo ter ao trabalhar com inflamáveis no laboratório de leite?"
  agent_ids: [3,4]
  confidence: 0.93
  reason: "Segurança com inflamáveis está nas Recomendações Gerais da IN 68 -> Qualidade do Leite."
  alternatives: []

- Pergunta: "Em que unidades se expressa a concentração m/m e m/v nos métodos da IN 68?"
  agent_ids: [3,4]
  confidence: 0.94
  reason: "Notação m/m e m/v nos métodos analíticos da IN 68 -> Qualidade do Leite."
  alternatives: []

- Pergunta: "Como preparar a solução de azul de metileno para os testes da IN 68?"
  agent_ids: [3,4]
  confidence: 0.94
  reason: "Azul de metileno é indicador nos métodos da IN 68 -> Qualidade do Leite."
  alternatives: []

- Pergunta: "Qual o comprimento de onda utilizado na leitura de absorbância para o método de ácido sórbico?"
  agent_ids: [3,4]
  confidence: 0.93
  reason: "Espectrofotometria nos métodos quantitativos da IN 68 -> Qualidade do Leite."
  alternatives: []

- Pergunta: "Não inalar vapores é recomendação de qual documento da IN 68?"
  agent_ids: [3,4]
  confidence: 0.95
  reason: "Recomendação de segurança nas Recomendações Gerais da IN 68 -> Qualidade do Leite."
  alternatives: []

- Pergunta: "O que caracteriza o leite cru em termos de obtenção e condições do animal?"
  agent_ids: [3,4]
  confidence: 0.94
  reason: "Caracterização do leite cru cobre legislação (RIISPOA) e qualidade do leite (IN 62) -> Agentes 3 e 4."
  alternatives: []

- Pergunta: "Como executar o teste de gelatina no leite?"
  agent_ids: [3,4]
  confidence: 0.95
  reason: "Teste qualitativo da IN 68 -> Qualidade do Leite (4). Regulatório como contexto (3). Sem glossário."
  alternatives: []
"""

_LOW_PRECISION_KEYWORDS: frozenset       = _ROUTING_RULES["low_precision_keywords"]
_HINT_NOISE_TERMS: frozenset             = _ROUTING_RULES["hint_noise_terms"]
_HINT_NOISE_TOKENS: frozenset            = _ROUTING_RULES["hint_noise_tokens"]
_SPECIALIST_STRONG_HINTS_DEFAULT: Dict[int, set] = _ROUTING_RULES["specialist_strong_hints_default"]

_ROUTING_BASELINE_IDS = [0, 3]
_ROUTING_CONFIDENCE_THRESHOLDS = {
    "high": 0.86,
    "medium": 0.55,
}
_SPECIALISTS_PER_BUCKET = {
    "high": 1,
    "medium": 2,
    "low": 3,
}
_FALLBACK_MAX_ATTEMPTS = 1
_FALLBACK_EXTRA_SPECIALISTS = {
    "high": 1,
    "medium": 2,
    "low": 2,
}
# Mapa de vizinhanca entre especialistas (extraido da taxonomia day1).
_NEAREST_SPECIALIST_MAP: Dict[int, List[int]] = {
    1: [],
    2: [1],
    3: [1, 4],
    4: [2],
    5: [1, 4],
    6: [1, 2],
}


def _load_taxonomy_nearest_map() -> Dict[int, List[int]]:
    """Carrega vizinhanca de dominios a partir da taxonomia (se disponivel)."""
    taxonomy_path = Path("docs/orchestrator/day1/AGENT_ROUTING_TAXONOMY.yaml")
    if not taxonomy_path.exists():
        return dict(_NEAREST_SPECIALIST_MAP)
    try:
        import yaml  # type: ignore
    except Exception:
        return dict(_NEAREST_SPECIALIST_MAP)

    try:
        raw = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8")) or {}
        agents = raw.get("agents", {}) or {}
        loaded: Dict[int, List[int]] = {}
        for k, info in agents.items():
            try:
                aid = int(k)
            except (TypeError, ValueError):
                continue
            if aid in _ROUTING_BASELINE_IDS:
                continue
            confusion = info.get("confusion_with", []) if isinstance(info, dict) else []
            near: List[int] = []
            for raw_id in confusion or []:
                try:
                    nid = int(raw_id)
                except (TypeError, ValueError):
                    continue
                _no_kb = {5, 6}
                if nid not in _ROUTING_BASELINE_IDS and nid not in _no_kb and 0 <= nid <= 6 and nid not in near:
                    near.append(nid)
            if near:
                loaded[aid] = near
        if loaded:
            return loaded
    except Exception:
        pass
    return dict(_NEAREST_SPECIALIST_MAP)


_NEAREST_SPECIALIST_MAP = _load_taxonomy_nearest_map()
_AGENT_TABLE_BY_ID: Dict[int, str] = {
    int(agent["agent_id"]): str(agent.get("table_name", ""))
    for agent in AGENTS
    if str(agent.get("table_name", "")).strip()
}
_ALL_AGENT_TABLES: List[str] = []
for _agent in AGENTS:
    _table = str(_agent.get("table_name", "")).strip()
    if _table and _table not in _ALL_AGENT_TABLES:
        _ALL_AGENT_TABLES.append(_table)


def _load_specialist_strong_hints() -> Dict[int, set]:
    """Carrega hints fortes de especialistas derivados do rag_queries.

    Fonte esperada:
      docs/orchestrator/day1/ROUTING_SPECIALIST_HINTS.yaml

    Se indisponível, retorna apenas hints default.
    """
    def _norm(value: str) -> str:
        v = (value or "").lower().strip()
        v = unicodedata.normalize("NFKD", v)
        v = "".join(ch for ch in v if not unicodedata.combining(ch))
        v = re.sub(r"\s+", " ", v)
        return v

    merged: Dict[int, set] = {}
    agent_keyword_tokens: Dict[int, set] = {}
    for agent in AGENTS:
        try:
            aid = int(agent.get("agent_id", -1))
        except (TypeError, ValueError):
            continue
        if aid in (0, 3):
            continue
        toks: set = set()
        for raw_kw in (agent.get("keywords", []) or []):
            kw_norm = _norm(str(raw_kw))
            for tk in kw_norm.split():
                if len(tk) >= 4 and tk not in _LOW_PRECISION_KEYWORDS:
                    toks.add(tk)
        agent_keyword_tokens[aid] = toks

    for aid, hints in _SPECIALIST_STRONG_HINTS_DEFAULT.items():
        merged[aid] = {_norm(str(h)) for h in hints if str(h).strip()}

    hints_path = Path("docs/orchestrator/day1/ROUTING_SPECIALIST_HINTS.yaml")
    if not hints_path.exists():
        return merged

    try:
        import yaml  # type: ignore
        raw = yaml.safe_load(hints_path.read_text(encoding="utf-8")) or {}
        specialists = raw.get("specialists", {}) or {}
        for raw_aid, info in specialists.items():
            try:
                aid = int(raw_aid)
            except (TypeError, ValueError):
                continue
            if aid in (0, 3):
                continue
            hints = info.get("strong_hints_normalized", []) if isinstance(info, dict) else []
            if not isinstance(hints, list):
                continue
            bucket = merged.setdefault(aid, set())
            default_bucket = merged.get(aid, set())
            for hint in hints:
                normalized = _norm(str(hint))
                if normalized:
                    # Segurança de produção: aceita do arquivo apenas hints
                    # multi-palavra (mais específicos) ou já presentes no default.
                    if normalized in _HINT_NOISE_TERMS:
                        continue
                    if any(ch.isdigit() for ch in normalized):
                        continue
                    if any(tok in _HINT_NOISE_TOKENS for tok in normalized.split()):
                        continue
                    if normalized not in default_bucket and len(normalized.split()) > 2:
                        continue
                    if normalized not in default_bucket and " " not in normalized:
                        continue
                    if normalized in _LOW_PRECISION_KEYWORDS:
                        continue
                    if normalized not in default_bucket:
                        tokens = [t for t in normalized.split() if len(t) >= 4]
                        kw_tokens = agent_keyword_tokens.get(aid, set())
                        if tokens and kw_tokens and not any(t in kw_tokens for t in tokens):
                            continue
                    bucket.add(normalized)
    except Exception:
        return merged

    return merged


_SPECIALIST_STRONG_HINTS = _load_specialist_strong_hints()


def _choose_primary_agent_id(agent_ids: List[int], route_text: str = "") -> int:
    """Escolhe o agente principal para exibicao ao cliente.

    Regra:
    1) Preferir especialistas de dominio (1..6, exceto 3 se houver 1..2/4..6).
    2) Se nao houver especialista de dominio, preferir 3 (Regulatorios).
    3) Por fim, usar 0 (Base Geral).
    """
    if not agent_ids:
        return 0

    domain_primary = _infer_domain_primary_from_text(route_text, agent_ids)
    if domain_primary is not None:
        return domain_primary

    # Especialistas de dominio (na ordem de relevancia original)
    for aid in agent_ids:
        if aid not in (0, 3):
            return aid

    if 3 in agent_ids:
        return 3
    if 0 in agent_ids:
        return 0
    return agent_ids[0]


def _sanitize_math_for_ui(text: str) -> str:
    """Converte trechos matematicos em LaTeX para texto simples amigavel ao front."""
    if not text:
        return text

    out = str(text)
    # Delimitadores comuns de math mode
    out = re.sub(r"\\\[(.*?)\\\]", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\\\((.*?)\\\)", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\$\$(.*?)\$\$", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\$(.*?)\$", r"\1", out, flags=re.DOTALL)

    # Comandos latex usuais em respostas de calculo
    out = out.replace(r"\times", "x")
    out = out.replace(r"\cdot", "x")
    out = out.replace(r"\,", " ")
    out = out.replace("\\n", "\n")
    out = out.replace("\t", " ")
    out = re.sub(r"\\text\{([^}]*)\}", r"\1", out)

    # Limpeza de comandos LaTeX residuais
    out = re.sub(r"\\[a-zA-Z]+", "", out)
    out = out.replace("{", "").replace("}", "")

    # Remove wrappers visuais do tipo [ ... ] quando estiverem sozinhos na linha
    cleaned_lines = []
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]") and len(s) >= 2:
            s = s[1:-1].strip()
        cleaned_lines.append(s if s else line.strip())
    out = "\n".join(cleaned_lines)

    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def _dedupe_paragraphs(text: str) -> str:
    """Remove paragrafos duplicados mantendo a primeira ocorrencia."""
    if not text:
        return text
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    seen = set()
    kept: List[str] = []
    for p in parts:
        key = _normalize_text(p)
        if key in seen:
            continue
        seen.add(key)
        kept.append(p)
    return "\n\n".join(kept).strip()


def _normalize_mul_symbols(text: str) -> str:
    return (
        text.replace("×", "x")
        .replace("*", "x")
        .replace("X", "x")
    )


def _enforce_dornic_canonical_formula(user_text: str, text: str) -> str:
    """Garante forma canonica da formula Dornic quando a pergunta for desse tema.

    Formula canonica (IN 68): Acidez (Dornic) = V x f x 0,9 x 10
    """
    if not text:
        return text
    q = _normalize_text(user_text)
    if "dornic" not in q and not ("acidez" in q and "titul" in q):
        return text

    out = text
    # Corrige forma incompleta observada em consolidacoes conflitantes.
    patterns = [
        r"Acidez\s*\(?°?\s*D(?:ornic)?\)?\s*=\s*V\s*[x\*]\s*f\s*[x\*]\s*10",
        r"Acidez\s*\('?\s*Dornic\s*'?\)\s*=\s*V\s*[x\*]\s*f\s*[x\*]\s*10",
    ]
    for pat in patterns:
        out = re.sub(
            pat,
            "Acidez (Dornic) = V x f x 0,9 x 10",
            out,
            flags=re.IGNORECASE,
        )

    # Se houver duas secoes de formula Dornic, mantem so a primeira ocorrencia.
    lines = out.splitlines()
    new_lines: List[str] = []
    seen_formula_line = False
    for line in lines:
        ln_norm = _normalize_mul_symbols(line)
        is_dornic_formula = (
            "acidez" in ln_norm.lower()
            and "dornic" in ln_norm.lower()
            and "=" in ln_norm
            and "v" in ln_norm.lower()
            and "f" in ln_norm.lower()
            and "10" in ln_norm
        )
        if is_dornic_formula:
            if seen_formula_line:
                continue
            seen_formula_line = True
        new_lines.append(line)
    out = "\n".join(new_lines).strip()
    return out


def _postprocess_consolidated_answer(user_text: str, text: str) -> str:
    out = _sanitize_math_for_ui(text or "")
    out = _enforce_dornic_canonical_formula(user_text, out)
    out = _dedupe_paragraphs(out)
    out = _strip_leading_uncertainty_prefix(out)
    # Remove cauda de ressalva genérica quando a resposta já é factualmente completa.
    # Só aplica se o texto tem substância suficiente antes da ressalva (>80 chars).
    stripped = _strip_uncertainty_tail(out)
    if stripped and len(stripped) > 80:
        out = stripped
    out = _sanitize_math_for_ui(out)
    return out


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _strip_profile_suffix(text: str) -> str:
    if "\n[Perfil" in text:
        return text.split("\n[Perfil", 1)[0]
    return text


def _extract_current_user_segment(text: str) -> str:
    marker = "\n[Pergunta atual]\n"
    if marker in text:
        return text.rsplit(marker, 1)[1].strip()
    if text.strip().startswith("[Pergunta atual]"):
        return text.split("[Pergunta atual]", 1)[1].strip()
    return text


def _extract_recent_context_block(text: str) -> str:
    if "[Contexto recente da conversa]" not in text:
        return ""
    context_block = text.split("[Contexto recente da conversa]", 1)[1]
    if "[Pergunta atual]" in context_block:
        context_block = context_block.split("[Pergunta atual]", 1)[0]
    return context_block.strip()


def _has_recent_context_block(text: str) -> bool:
    return bool(_extract_recent_context_block(text))


def _is_conversation_recap_request(text_norm: str) -> bool:
    if not text_norm:
        return False

    strong_phrases = (
        "o que conversamos",
        "sobre o que conversamos",
        "conversamos recentemente",
        "o que falamos antes",
        "o que falamos",
        "falamos recentemente",
        "me explique sobre o que conversamos",
        "me lembre do que conversamos",
        "resuma o que conversamos",
        "resuma o que falamos",
        "retome o que falamos",
        "continue de onde paramos",
        "retome a conversa",
    )
    if any(phrase in text_norm for phrase in strong_phrases):
        return True

    return bool(
        re.search(
            r"\b(resuma|retome|relembre|recapitule|continue|explique)\b.*\b(conversa|conversamos|falamos)\b",
            text_norm,
        )
    )


def _build_contextual_search_query(text: str) -> str:
    current = _strip_profile_suffix(_extract_current_user_segment(text)).strip()
    if not current:
        return ""

    if "[Contexto recente da conversa]" not in text or "[Pergunta atual]" not in text:
        return current

    context_block = text.split("[Contexto recente da conversa]", 1)[1]
    context_block = context_block.split("[Pergunta atual]", 1)[0]

    user_snippets: List[str] = []
    for raw_line in context_block.splitlines():
        line = raw_line.strip()
        line_norm = unicodedata.normalize("NFKD", line)
        line_norm = "".join(ch for ch in line_norm if not unicodedata.combining(ch))
        line_norm = line_norm.lower()
        if not line_norm.startswith("usuario:"):
            continue
        snippet = line.split(":", 1)[1].strip()
        snippet = _strip_profile_suffix(snippet)
        if snippet:
            user_snippets.append(snippet)

    if not user_snippets:
        return current

    combined = " | ".join(user_snippets[-2:] + [current]).strip(" |")
    combined = re.sub(r"\s+", " ", combined).strip()
    if len(combined) <= 320:
        return combined
    return combined[:317].rstrip() + "..."


def _is_objective_question(text: str) -> bool:
    q = _normalize_text(text)
    if not q:
        return False
    patterns = (
        r"^(quem e|qual e|quais sao|quanto e|onde fica|onde e|quando|como se chama)\b",
        r"^(quem|qual|quais|quanto|onde|quando)\b",
    )
    return any(re.search(p, q) for p in patterns)


def _looks_uncertain(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return True
    uncertainty_markers = (
        "nao encontrei informacao suficiente",
        "nao encontrei informacoes suficientes",
        "nao encontrei informacoes especificas",
        "nao tenho informacao suficiente",
        "nao tenho informacoes suficientes",
        "nao ha informacao suficiente",
        "nao ha evidencia suficiente",
        "nao ha evidencias suficientes",
        "nao ha dados suficientes",
        "sem informacao suficiente",
        "informacao insuficiente",
        "faltam evidencias",
        "pode ser",
        "talvez",
        "recomenda-se verificar",
        "aconselhavel verificar",
        "consultar fontes adicionais",
        "com o meu conhecimento atual",
        "com o seu conhecimento atual",
        "nao foi possivel identificar",
        "nao foi possivel encontrar",
        "evidencia insuficiente",
        "nao disponho de informacao",
    )
    return any(marker in t for marker in uncertainty_markers)


def _strip_uncertainty_tail(text: str) -> str:
    """Remove cauda de ressalva genérica quando houver fato já respondido.

    Ex.: "X é Y. No entanto, ..." -> "X é Y."
    """
    if not text:
        return ""
    out = str(text).strip()

    # Corta no início de conectores de ressalva.
    m = re.search(r"\b(No entanto|Por[ée]m|Contudo)\b", out, flags=re.IGNORECASE)
    if m:
        out = out[: m.start()].strip()

    # Corta no início de frases de falta de evidência genérica.
    m2 = re.search(
        r"\b(a base atual n[ãa]o trouxe|nao trouxe informacao suficiente|"
        r"faltam evidencias|com o meu conhecimento atual|"
        r"recomenda-se verificar|aconselh[aá]vel verificar)\b",
        out,
        flags=re.IGNORECASE,
    )
    if m2:
        out = out[: m2.start()].strip()

    # Limpeza de pontuação residual
    out = re.sub(r"[;,:\-–—]+$", "", out).strip()
    return out


def _strip_leading_uncertainty_prefix(text: str) -> str:
    if not text:
        return ""
    out = str(text).strip()
    out = re.sub(
        r"^\s*(?:Com base no meu conhecimento atual|Com o meu conhecimento atual|Com base nas informacoes disponiveis),\s*",
        "",
        out,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    return out


def _extract_factual_candidate(text: str) -> Optional[str]:
    """Extrai parte factual útil de uma resposta mista (fato + ressalva)."""
    cleaned = _sanitize_math_for_ui(text or "")
    cleaned = _strip_leading_uncertainty_prefix(cleaned)
    if not cleaned:
        return None
    head = _strip_uncertainty_tail(cleaned)
    if not head:
        return None
    if len(head.split()) < 3:
        return None
    if _looks_uncertain(head):
        return None
    return head


def _prefer_direct_fact_response(
    user_text: str,
    responses: List[Dict[str, Any]],
) -> Optional[str]:
    """When question is objective, prefer a direct factual specialist answer."""
    if not _is_objective_question(user_text):
        return None

    direct = []
    for r in responses:
        if not r.get("success") or not r.get("response"):
            continue
        factual = _extract_factual_candidate(str(r.get("response", "")))
        if factual:
            item = dict(r)
            item["factual_response"] = factual
            direct.append(item)
    if len(direct) != 1:
        return None

    # Prefer domain specialist over transversal agents for objective facts.
    chosen = direct[0]
    aid = int(chosen.get("agent_id", -1))
    if aid in (0, 3):
        specialists = [r for r in direct if int(r.get("agent_id", -1)) not in (0, 3)]
        if len(specialists) == 1:
            chosen = specialists[0]
    return _sanitize_math_for_ui(str(chosen.get("factual_response", "")))


def _prefer_regulatory_requirement_response(
    user_text: str,
    responses: List[Dict[str, Any]],
) -> Optional[str]:
    """Para requisito legal explícito, prioriza a resposta factual do Agente 3."""
    if not _is_legal_requirement_regulatory_signal(_normalize_text(user_text)):
        return None

    for r in responses:
        if int(r.get("agent_id", -1)) != 3:
            continue
        if not r.get("success") or not r.get("response"):
            continue
        factual = _extract_factual_candidate(str(r.get("response", "")))
        if factual:
            return _sanitize_math_for_ui(factual)
    return None


def _build_keyword_sets() -> Dict[int, set]:
    keyword_sets: Dict[int, set] = {}
    for agent in AGENTS:
        aid = agent["agent_id"]
        if aid in (0, 3):
            continue
        raw_keywords = agent.get("keywords", []) or []
        words = {
            _normalize_text(str(k))
            for k in raw_keywords
            if isinstance(k, str) and len(_normalize_text(k)) >= 4
        }
        keyword_sets[aid] = words
    return keyword_sets


_SPECIALIST_KEYWORDS = _build_keyword_sets()


def _contains_keyword(text_norm: str, keyword_norm: str) -> bool:
    if not keyword_norm:
        return False
    # Match por fronteira para evitar falso-positivo por substring.
    pattern = rf"(?<!\w){re.escape(keyword_norm)}(?!\w)"
    return re.search(pattern, text_norm) is not None


def _keyword_weight(aid: int, keyword_norm: str) -> int:
    strong = _SPECIALIST_STRONG_HINTS.get(aid, set())
    if keyword_norm in strong:
        return 3
    if keyword_norm in _LOW_PRECISION_KEYWORDS:
        return 0
    if " " in keyword_norm:
        return 2
    if len(keyword_norm) >= 9:
        return 2
    return 1


def _cache_get(cache_key: str) -> Optional[List[int]]:
    if _MAX_CLASSIFICATION_CACHE <= 0:
        return None
    cached = _CLASSIFICATION_CACHE.get(cache_key)
    if cached is None:
        return None
    _CLASSIFICATION_CACHE.move_to_end(cache_key)
    return list(cached)


def _cache_set(cache_key: str, agent_ids: List[int]) -> None:
    if _MAX_CLASSIFICATION_CACHE <= 0:
        return
    _CLASSIFICATION_CACHE[cache_key] = list(agent_ids)
    _CLASSIFICATION_CACHE.move_to_end(cache_key)
    while len(_CLASSIFICATION_CACHE) > _MAX_CLASSIFICATION_CACHE:
        _CLASSIFICATION_CACHE.popitem(last=False)


def _looks_like_greeting_only(text_norm: str) -> bool:
    if not text_norm:
        return False
    if text_norm in _GREETINGS:
        return True
    if len(text_norm.split()) <= 4 and any(text_norm.startswith(g) for g in _GREETINGS):
        return True
    return False


def _contains_dairy_signal(text_norm: str) -> bool:
    if any(term in text_norm for term in _DAIRY_TERMS):
        return True
    if re.search(r"\b(in|rdc|rtiq)\s*\d{1,4}\b", text_norm):
        return True
    return False


def _contains_any_phrase(text_norm: str, phrases: set) -> bool:
    for phrase in phrases:
        p = _normalize_text(str(phrase))
        if not p:
            continue
        if " " in p:
            if p in text_norm:
                return True
        else:
            if re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text_norm):
                return True
    return False


def _is_strong_regulatory_signal(text_norm: str) -> bool:
    if not text_norm:
        return False
    if _is_legal_requirement_regulatory_signal(text_norm):
        return True
    if _contains_any_phrase(text_norm, _REGULATORY_STRONG_TERMS):
        return True
    if re.search(r"\b(in|rdc)\s*\d{1,4}\b", text_norm):
        return True
    return False


def _is_strong_quality_signal(text_norm: str) -> bool:
    if not text_norm:
        return False
    return _contains_any_phrase(text_norm, _QUALITY_STRONG_TERMS)


def _is_strong_fermented_signal(text_norm: str) -> bool:
    if not text_norm:
        return False
    return _contains_any_phrase(text_norm, _FERMENTED_STRONG_TERMS)


def _is_strong_cheese_signal(text_norm: str) -> bool:
    if not text_norm:
        return False
    return _contains_any_phrase(text_norm, _CHEESE_STRONG_TERMS)


def _is_glossary_or_normalization_signal(text_norm: str) -> bool:
    if not text_norm:
        return False
    explicit_markers = (
        "termo correto",
        "significado esperado",
        "como deve ser padronizado",
        "preciso usar o termo padronizado",
        "em ingles",
        "em inglês",
        "padronizar",
        "padronizado",
        "glossario",
        "glossário",
    )
    if any(m in text_norm for m in explicit_markers):
        return True
    # Linguagem corporativa/produto da base geral com "termo" ou "significado".
    brand_terms = ("coagusens", "coagutrack", "rennet", "starter", "endogenous", "phage", "fagos")
    if any(t in text_norm for t in brand_terms) and ("termo" in text_norm or "significado" in text_norm):
        return True
    # Instrumentos e índices de coagulação — sempre Base Geral (fix confusão 0→1)
    # Perguntas sobre Coagutrack/CoaguSens/quimosina/índice C/P são conceitos do
    # glossário base, mesmo sem a palavra "termo" na pergunta.
    coag_instruments = ("coagutrack", "coagusens", "coagulometro", "coagulômetro")
    if any(t in text_norm for t in coag_instruments):
        return True
    if "fermento repicado" in text_norm:
        return True
    if ("indice" in text_norm or "índice") and "c/p" in text_norm:
        return True
    if "enzima coagulante" in text_norm and "queij" not in text_norm:
        return True
    return False


def _is_labeling_regulatory_signal(text_norm: str) -> bool:
    if not text_norm:
        return False
    labeling_terms = (
        "rotular",
        "rotulado",   # "Como deve ser rotulado o provolone?" — "rotular" não match "rotulado"
        "rotulagem",
        "denominacao",
        "denominação",
        "denominacao de venda",
        "denominação de venda",
        "embalagem",
    )
    if any(t in text_norm for t in labeling_terms):
        # Denominação de venda é sempre regulatória, com ou sem produto explícito
        # (fix confusão 3→1: "Como deve ser a denominação quando há ingredientes adicionais?")
        if "denominacao" in text_norm or "denominação" in text_norm:
            return True
        dairy_product_terms = (
            "provolone", "ricota", "minas", "queijo", "iogurte", "bebida lactea",
            "bebida láctea", "cream cheese", "mussarela", "muçarela",
            "sobremesa lactea", "sobremesa láctea", "requeijao", "requeijão",
        )
        if any(p in text_norm for p in dairy_product_terms):
            return True
    # Alegações de ausência/composição sem o termo "rotulagem" (fix confusão 3→1)
    # Ex.: "Para dizer que um produto não contém gordura total..."
    absence_claims = (
        "nao contem", "não contém", "isento de", "sem adicao de",
        "valor nulo", "zerar gordura", "zerar acucar",
    )
    nutritional_terms = (
        "gordura", "acucar", "açúcar", "sodio", "sódio",
        "calorias", "energetico", "energético",
    )
    if any(ac in text_norm for ac in absence_claims) and any(nt in text_norm for nt in nutritional_terms):
        return True
    return False


def _is_legal_requirement_regulatory_signal(text_norm: str) -> bool:
    if not text_norm:
        return False
    if _contains_any_phrase(text_norm, _LEGAL_REQUIREMENT_DIRECT_PHRASES):
        return True

    has_requirement_marker = (
        "exigid" in text_norm
        or "obrigat" in text_norm
        or "minimo legal" in text_norm
        or "mínimo legal" in text_norm
    )
    if not has_requirement_marker:
        return False

    requirement_subjects = (
        "periodo minimo", "período mínimo",
        "prazo minimo", "prazo mínimo",
        "tempo minimo", "tempo mínimo",
        "maturacao", "maturação",
        "limite minimo", "limite mínimo",
        "limite maximo", "limite máximo",
        "teor minimo", "teor mínimo",
        "deve sofrer maturacao", "deve sofrer maturação",
    )
    return any(subject in text_norm for subject in requirement_subjects)


def _is_normative_regulatory_signal(text_norm: str) -> bool:
    if not text_norm:
        return False
    if _is_legal_requirement_regulatory_signal(text_norm):
        return True
    markers = (
        "norma", "normas", "regulamento", "requisito legal", "requisitos legais",
        "instrução normativa", "instrucao normativa", "rdc", "riispoa", "decreto", "art.",
    )
    if any(m in text_norm for m in markers):
        return True
    return bool(re.search(r"\b(in|rdc)\s*\d{1,4}\b", text_norm))


def _is_troubleshooting_defect_intent(text_norm: str) -> bool:
    if not text_norm:
        return False
    markers = (
        "defeito", "defeitos", "diagnostico", "diagnóstico", "causa", "causas",
        "corrigir", "correcao", "correção", "estufamento", "rançoso", "rancoso",
        "trinca", "olhadura", "contaminacao", "contaminação", "problema", "problemas",
    )
    return any(m in text_norm for m in markers)


def _is_ambiguous_cheese_fermented_signal(text_norm: str) -> bool:
    return _is_strong_cheese_signal(text_norm) and _is_strong_fermented_signal(text_norm)


def _is_general_knowledge_signal(text_norm: str) -> bool:
    if not text_norm:
        return False
    if (
        _is_strong_regulatory_signal(text_norm)
        or _is_strong_quality_signal(text_norm)
        or _is_strong_fermented_signal(text_norm)
        or _is_strong_cheese_signal(text_norm)
    ):
        return False
    if _is_glossary_or_normalization_signal(text_norm):
        return True
    if _contains_any_phrase(text_norm, _GENERAL_KNOWLEDGE_TERMS):
        return True
    if re.match(r"^quem\b", text_norm) and (
        "distribui" in text_norm or "fabricante" in text_norm or "empresa" in text_norm
    ):
        return True
    return False


def _infer_domain_primary_from_text(text: str, candidate_ids: List[int]) -> Optional[int]:
    text_norm = _normalize_text(text)
    ids = _sanitize_agent_ids(candidate_ids)
    if not ids:
        return None

    # Glossário/padronização antes do sinal normativo: algumas queries de glossário
    # contêm "norma" ou "regulamento" e não devem ser capturadas pelo Agent 3.
    if _is_glossary_or_normalization_signal(text_norm) and 0 in ids:
        return 0

    if _is_legal_requirement_regulatory_signal(text_norm) and 3 in ids:
        return 3

    if _is_normative_regulatory_signal(text_norm) and 3 in ids:
        # Regra especialista-em-contexto: se a query cita uma norma MAS o tema central
        # é de um domínio especialista (fermentados, queijos, qualidade do leite),
        # o especialista é o primário — Agent 3 permanece no plano como co-piloto.
        if _is_strong_fermented_signal(text_norm) and 2 in ids:
            return 2
        if _is_strong_cheese_signal(text_norm) and 1 in ids:
            return 1
        if _is_strong_quality_signal(text_norm) and 4 in ids:
            return 4
        return 3

    # Rotulagem/denominação/embalagem em produto lácteo é tema regulatório.
    if _is_labeling_regulatory_signal(text_norm) and 3 in ids:
        return 3

    # Precedencia forte: regulatorio e qualidade (metodos analiticos) devem dominar.
    if _is_strong_regulatory_signal(text_norm) and 3 in ids:
        return 3
    if _is_strong_quality_signal(text_norm) and 4 in ids:
        # CCS/mastite + impacto em queijo → Agente 1 é o primário (fix confusão 1→4)
        # Ex.: "Como CCS elevada influencia o amargor no queijo?"
        _cheese_quality_impact = (
            "amargor", "proteolise", "proteólise", "rendimento",
            "maturacao", "maturação", "coagulacao", "coagulação",
            "plasmina", "lipolise", "lipólise", "filagem", "prensagem",
            "rendimento de queijo", "recuperacao de gordura",
        )
        if _is_strong_cheese_signal(text_norm) and 1 in ids:
            if any(t in text_norm for t in _cheese_quality_impact):
                return 1
        return 4
    if _is_strong_fermented_signal(text_norm) and 2 in ids:
        # "clc" é defeito de queijo, não fermentado
        if 1 in ids and ("queijo" in text_norm or "coalhada" in text_norm or "clc" in text_norm):
            return 1
        return 2
    if _is_strong_cheese_signal(text_norm) and 1 in ids:
        # Defeitos técnicos em queijo → Agente 1 (fix confusão 1→5)
        # Agente 5 é para diagnóstico visual; termos de defeito agora pertencem a Agent 1
        return 1

    # Regras por especialista com alta precisao.
    specialist_order = [4, 2, 1, 5, 6]
    for aid in specialist_order:
        if aid not in ids:
            continue
        hints = _SPECIALIST_STRONG_HINTS.get(aid, set())
        if any(_contains_keyword(text_norm, h) for h in hints):
            return aid

    # Perguntas gerais/institucionais: manter Base Geral como primaria.
    if _is_general_knowledge_signal(text_norm) and 0 in ids:
        return 0

    return None


def _should_include_agent_0(text_norm: str) -> bool:
    """Agent 0 só entra no plano quando há sinal explícito de glossário ou conhecimento geral transversal."""
    return (
        _is_glossary_or_normalization_signal(text_norm)
        or _is_general_knowledge_signal(text_norm)
    )


def _rule_based_route(user_text: str) -> Optional[List[int]]:
    result = _rule_based_route_impl(user_text)
    if result is None:
        return None
    # [] explícito = saudação/off-topic — preserva o sinal em vez de converter para None
    if not result:
        return result
    text_norm = _normalize_text(_strip_profile_suffix(user_text))
    if not _should_include_agent_0(text_norm):
        result = [x for x in result if x != 0]
    return result if result else None


def _rule_based_route_impl(user_text: str) -> Optional[List[int]]:
    text = _normalize_text(_strip_profile_suffix(user_text))
    if not text:
        return []

    if _looks_like_greeting_only(text):
        return []

    # Intenção de glossário/padronização institucional (Base Geral).
    if _is_glossary_or_normalization_signal(text):
        return [0, 3]

    # IN 68 é especificamente um documento de métodos analíticos de Qualidade do Leite.
    # Se a pergunta menciona IN 68 + termos de laboratório/análise, inclui Agente 4.
    # Deve ser verificado ANTES de _is_normative_regulatory_signal (que captura "in 68" via regex).
    if re.search(r"\bin\s*68\b", text) and (
        any(t in text for t in _QUALITY_LAB_TERMS) or _is_strong_quality_signal(text)
    ):
        return [0, 3, 4]

    # Perguntas sobre requisito mínimo/obrigatório/exigido devem ser respondidas
    # pelo regulatório para evitar transformar contexto técnico em exigência legal.
    if _is_legal_requirement_regulatory_signal(text):
        return [0, 3]

    # Contexto normativo explícito com domínio especialista identificado:
    # inclui o especialista no plano — Agent 3 co-piloto, especialista é primário.
    if _is_normative_regulatory_signal(text):
        if _is_strong_fermented_signal(text):
            return [0, 3, 2]
        if _is_strong_cheese_signal(text):
            return [0, 3, 1]
        if _is_strong_quality_signal(text):
            return [0, 3, 4]
        return [0, 3]

    # Precedencia forte de domínio:
    # 1) Regulatório domina quando houver sinal normativo claro.
    # 2) Métodos/qualidade domina para perguntas analíticas de laboratório.
    if _is_labeling_regulatory_signal(text):
        if _is_strong_fermented_signal(text):
            return [0, 3, 2]
        if _is_strong_cheese_signal(text):
            return [0, 3, 1]
        return [0, 3]
    if _is_strong_regulatory_signal(text):
        if _is_strong_fermented_signal(text):
            return [0, 3, 2]
        if _is_strong_cheese_signal(text):
            return [0, 3, 1]
        return [0, 3]
    if _is_strong_quality_signal(text):
        # CCS/mastite + impacto em queijo → incluir Agente 1 (fix confusão 1→4)
        if _is_strong_cheese_signal(text):
            return [0, 3, 4, 1]
        return [0, 3, 4]
    if _is_ambiguous_cheese_fermented_signal(text):
        return [0, 3, 1, 2]
    if _is_strong_fermented_signal(text):
        # "clc" (cristais de lactato/cálcio) é defeito de queijo, não fermentado
        if "queijo" in text or "coalhada" in text or "clc" in text:
            return [0, 3, 1]
        return [0, 3, 2]
    if _is_strong_cheese_signal(text):
        # Defeitos/troubleshooting em queijo → Agente 1 (Ha-La Biotec cobre)
        # Agente 5 = diagnóstico visual por imagem (KB ainda não ingerida)
        # (fix confusão 1→5: removido roteamento para Agente 5 em contexto de queijo)
        return [0, 3, 1]

    # Perguntas de laboratorio/controle de qualidade (inclui seguranca em lab)
    # devem consultar Qualidade do Leite mesmo sem termos "dairy" explicitos.
    if any(term in text for term in _QUALITY_LAB_TERMS):
        return [0, 3, 4]

    # Regras determinísticas para intents críticas (alta precisão).
    intent_scores: Dict[int, int] = {}
    for aid, patterns in _INTENT_PATTERNS_BY_AGENT.items():
        score = 0
        for pat in patterns:
            if re.search(pat, text):
                score += 1
        if score > 0:
            intent_scores[aid] = score

    if intent_scores:
        # Se houver empates entre especialistas não-regulatórios, delega ao LLM
        # para evitar escolha arbitrária em perguntas híbridas.
        sorted_scores = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        best_aid, best_score = sorted_scores[0]
        tied_best = [aid for aid, sc in sorted_scores if sc == best_score]
        non_reg_tied = [aid for aid in tied_best if aid != 3]
        if len(non_reg_tied) > 1:
            return None

        # Regras regulatórias ficam no baseline [0,3].
        if best_aid == 3:
            return [0, 3]
        return _sanitize_agent_ids([0, 3, best_aid])

    specialist_scores: List[tuple[int, int, int]] = []  # (aid, weighted, hits)
    for aid, keywords in _SPECIALIST_KEYWORDS.items():
        weighted = 0
        hits = 0
        for kw in keywords:
            if not kw:
                continue
            if _contains_keyword(text, kw):
                w = _keyword_weight(aid, kw)
                if w <= 0:
                    continue
                weighted += w
                hits += 1
        if hits > 0:
            specialist_scores.append((aid, weighted, hits))

    specialist_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # Alta confiança: >=2 hits úteis, ou 1 hit forte (peso >=3).
    high_conf = [aid for aid, weighted, hits in specialist_scores if hits >= 2 or weighted >= 3]
    if high_conf:
        return _sanitize_agent_ids([0, 3] + high_conf[:3])

    # Confiança média: 1 hit específico relevante.
    medium_conf = [aid for aid, weighted, hits in specialist_scores if weighted >= 2 and hits >= 1]
    if medium_conf:
        return _sanitize_agent_ids([0, 3] + medium_conf[:2])

    # Se parece pergunta técnica de laticínios, evita cair em [0,3] cego;
    # delega ao classificador LLM para escolher especialista.
    if _contains_dairy_signal(text) and _is_objective_question(text):
        return None

    # Domínio dairy genérico (não objetivo) segue baseline.
    if _contains_dairy_signal(text):
        return [0, 3]

    # Baixa confiança: deixar o classificador LLM decidir.
    return None


# Agentes sem KB ingerida — removidos de qualquer rota até a base estar disponível.
# LEMBRETE: remover 5 e 6 deste set quando as KBs forem ingeridas:
#   - Agente 5 (defeitos visuais): aguardando imagens do cliente
#   - Agente 6 (formulação): aguardando documentos do cliente
_AGENTS_WITHOUT_KB: set[int] = {5, 6}


def _sanitize_agent_ids(raw_ids: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for aid in raw_ids:
        if 0 <= aid <= 6 and aid not in seen and aid not in _AGENTS_WITHOUT_KB:
            seen.add(aid)
            out.append(aid)
    return out


def _clamp_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        conf = 0.50
    return max(0.0, min(1.0, conf))


def _confidence_to_bucket(confidence: float) -> str:
    if confidence >= _ROUTING_CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if confidence >= _ROUTING_CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _estimate_fastpath_confidence(route_text: str, agent_ids: List[int]) -> float:
    text_norm = _normalize_text(route_text)
    if not agent_ids:
        return 0.98
    if _is_strong_regulatory_signal(text_norm) or _is_strong_quality_signal(text_norm):
        return 0.86
    if any(aid not in _ROUTING_BASELINE_IDS for aid in agent_ids):
        return 0.84
    if _contains_dairy_signal(text_norm):
        return 0.72
    return 0.60


def _recalibrate_confidence(route_text: str, agent_ids: List[int], raw_confidence: float) -> float:
    """Recalibra confiança para evitar excesso de 'high' em casos ambíguos."""
    conf = _clamp_confidence(raw_confidence)
    ids = _sanitize_agent_ids(agent_ids)
    text_norm = _normalize_text(route_text)
    specialists = [aid for aid in ids if aid not in _ROUTING_BASELINE_IDS]

    # Sem agente: fora de escopo costuma ter alta confiança.
    if not ids:
        return conf

    # Baseline puro para pergunta objetiva dairy deve cair para medium.
    if set(ids) <= set(_ROUTING_BASELINE_IDS) and _contains_dairy_signal(text_norm) and _is_objective_question(text_norm):
        conf = min(conf, 0.69)

    # Regulatório/laboratorial são domínios de alta especificidade.
    if _is_strong_regulatory_signal(text_norm):
        if 3 in ids:
            conf = max(conf, 0.84)
        # Evita superconfiança quando há especialista concorrendo com regulatório.
        if any(aid not in (0, 3) for aid in ids):
            conf = min(conf, 0.78)

    if _is_strong_quality_signal(text_norm):
        if 4 in ids:
            conf = max(conf, 0.84)
        if any(aid not in (0, 3, 4) for aid in ids):
            conf = min(conf, 0.78)

    # Mais de um especialista tende a aumentar ambiguidade.
    if len(specialists) > 1:
        conf = min(conf, 0.74)

    # Se parece pergunta geral institucional, evita high artificial.
    if _is_general_knowledge_signal(text_norm) and 0 in ids and len(specialists) == 0:
        conf = min(conf, 0.70)

    return _clamp_confidence(conf)


def _apply_dairy_hard_constraints(route_text: str, agent_ids: List[int]) -> List[int]:
    if not agent_ids:
        return []
    text_norm = _normalize_text(route_text)
    if not _contains_dairy_signal(text_norm):
        return agent_ids
    out = list(agent_ids)
    # 0 e 3 sao obrigatorios para perguntas com sinal de lacteos.
    if 0 not in out:
        out.insert(0, 0)
    if 3 not in out:
        insert_at = 1 if 0 in out else 0
        out.insert(insert_at, 3)
    # Mantem ordenacao com baseline no topo.
    baseline = [aid for aid in _ROUTING_BASELINE_IDS if aid in out]
    tail = [aid for aid in out if aid not in baseline]
    return _sanitize_agent_ids(baseline + tail)


def _apply_domain_guardrails(route_text: str, agent_ids: List[int], alternatives: List[int]) -> Tuple[List[int], List[int]]:
    """Aplica guardrails de domínio para reduzir confusão entre especialistas."""
    text_norm = _normalize_text(route_text)
    ids = _sanitize_agent_ids(agent_ids)
    alts = _sanitize_agent_ids(alternatives)

    if not ids and not _contains_dairy_signal(text_norm):
        return ids, alts

    if _is_legal_requirement_regulatory_signal(text_norm):
        ids = _sanitize_agent_ids([0, 3] if 0 in ids or _should_include_agent_0(text_norm) else [3])
        alts = [aid for aid in alts if aid in (3,)]
        return ids, alts

    if _is_strong_regulatory_signal(text_norm):
        ids = _sanitize_agent_ids([0, 3] + [aid for aid in ids if aid not in _ROUTING_BASELINE_IDS and aid == 3])
        alts = [aid for aid in alts if aid not in (1, 2, 4, 5, 6)]
        return ids, alts

    if _is_strong_quality_signal(text_norm):
        if 4 not in ids:
            ids = _sanitize_agent_ids(ids + [4])
        # Evita competição desnecessária com outros especialistas em perguntas de método.
        alts = [aid for aid in alts if aid in (4,)]
        return ids, alts

    return ids, alts


def _build_execution_plan(
    route_text: str,
    chosen_ids: List[int],
    alternatives: Optional[List[int]],
    bucket: str,
) -> List[int]:
    chosen = _sanitize_agent_ids(chosen_ids)
    alts = _sanitize_agent_ids(alternatives or [])

    if not chosen:
        return []

    text_norm = _normalize_text(route_text)
    has_dairy_signal = _contains_dairy_signal(text_norm)
    has_specialist = any(aid not in _ROUTING_BASELINE_IDS for aid in chosen)
    has_baseline_pair = all(aid in chosen for aid in _ROUTING_BASELINE_IDS)
    is_dairy_route = has_dairy_signal or has_specialist or has_baseline_pair
    should_include_agent_0 = _should_include_agent_0(text_norm)
    strong_reg = _is_strong_regulatory_signal(text_norm)
    strong_quality = _is_strong_quality_signal(text_norm)
    ambiguous_12 = _is_ambiguous_cheese_fermented_signal(text_norm)

    if is_dairy_route:
        # Agent 3 segue como copiloto baseline do domínio lácteo.
        # Agent 0 só entra quando houver sinal explícito de glossário/base geral.
        if should_include_agent_0 and 0 not in chosen:
            chosen.insert(0, 0)
        if 3 not in chosen:
            insert_at = 1 if 0 in chosen else 0
            chosen.insert(insert_at, 3)
        chosen = _sanitize_agent_ids(chosen)

        base = [aid for aid in _ROUTING_BASELINE_IDS if aid in chosen]
        specialists = [aid for aid in chosen if aid not in _ROUTING_BASELINE_IDS]

        # Completa especialistas com alternativas relevantes.
        for aid in alts:
            if aid not in _ROUTING_BASELINE_IDS and aid not in specialists:
                specialists.append(aid)

        # Guardrail de precedencia: evita ruido de especialista quando a pergunta
        # e claramente normativa ou de metodo analitico de qualidade.
        if strong_reg:
            specialists = []
        elif strong_quality:
            specialists = [aid for aid in specialists if aid == 4]

        max_specialists = _SPECIALISTS_PER_BUCKET.get(bucket, 3)

        # Ambiguidade real entre queijo e fermentado:
        # permite 2 especialistas para elevar recall@3 sem abrir demais.
        if ambiguous_12 and 1 in specialists and 2 in specialists:
            max_specialists = max(max_specialists, 2)

        # Se houver especialista na classificação/alternativas, garante ao menos 1.
        if specialists and max_specialists < 1:
            max_specialists = 1

        selected_specialists = specialists[:max_specialists]
        if strong_reg or not selected_specialists:
            plan = base + selected_specialists
        else:
            regulatory_tail = [aid for aid in base if aid == 3]
            general_tail = [aid for aid in base if aid == 0]
            plan = selected_specialists + regulatory_tail + general_tail
    else:
        max_agents = _SPECIALISTS_PER_BUCKET.get(bucket, 3)
        merged = chosen + [aid for aid in alts if aid not in chosen]
        plan = merged[:max_agents]

    return _sanitize_agent_ids(plan)[:5]


def _merge_agent_responses(
    previous: List[Dict[str, Any]],
    current: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Mescla respostas por agent_id, preservando melhor evidência."""
    merged: Dict[int, Dict[str, Any]] = {}
    for item in previous:
        aid = int(item.get("agent_id", -1))
        if aid >= 0:
            merged[aid] = dict(item)
    for item in current:
        aid = int(item.get("agent_id", -1))
        if aid < 0:
            continue
        old = merged.get(aid)
        if old is None:
            merged[aid] = dict(item)
            continue
        old_ok = bool(old.get("success")) and bool(old.get("response"))
        new_ok = bool(item.get("success")) and bool(item.get("response"))
        # Prioriza resposta nova se ela melhora sucesso/evidencia.
        if (new_ok and not old_ok) or (new_ok and old_ok):
            merged[aid] = dict(item)
    # Mantem ordem: respostas novas primeiro, depois remanescentes.
    current_order = [int(i.get("agent_id", -1)) for i in current]
    previous_order = [int(i.get("agent_id", -1)) for i in previous]
    ordered_ids: List[int] = []
    for aid in current_order + previous_order:
        if aid >= 0 and aid in merged and aid not in ordered_ids:
            ordered_ids.append(aid)
    return [merged[aid] for aid in ordered_ids]


def _collect_fallback_candidates(state: OrchestratorState) -> List[int]:
    chosen = _sanitize_agent_ids(state.get("chosen_agent_ids", []))
    plan = _sanitize_agent_ids(state.get("execution_plan", chosen))
    alternatives = _sanitize_agent_ids(state.get("routing_alternatives", []))

    seed_specialists = [aid for aid in plan if aid not in _ROUTING_BASELINE_IDS]
    if not seed_specialists:
        seed_specialists = [aid for aid in chosen if aid not in _ROUTING_BASELINE_IDS]

    candidates: List[int] = []
    for aid in seed_specialists:
        for near in _NEAREST_SPECIALIST_MAP.get(aid, []):
            if near not in _ROUTING_BASELINE_IDS and near not in candidates:
                candidates.append(near)

    for aid in alternatives:
        if aid not in _ROUTING_BASELINE_IDS and aid not in candidates:
            candidates.append(aid)

    already_planned = set(plan)
    raw = [aid for aid in candidates if aid not in already_planned]
    return _sanitize_agent_ids(raw)


def _has_weak_or_conflicting_evidence(responses: List[Dict[str, Any]]) -> bool:
    successful = [
        r for r in responses
        if r.get("success") and str(r.get("response", "")).strip()
    ]
    if not successful:
        return True

    factual_count = 0
    uncertain_count = 0
    for item in successful:
        txt = str(item.get("response", ""))
        if _extract_factual_candidate(txt):
            factual_count += 1
        if _looks_uncertain(txt):
            uncertain_count += 1

    if factual_count == 0:
        return True
    if uncertain_count == len(successful):
        return True
    return False


def _has_specialist_factual_evidence(responses: List[Dict[str, Any]]) -> bool:
    for item in responses:
        aid = int(item.get("agent_id", -1))
        if aid in _ROUTING_BASELINE_IDS:
            continue
        if not item.get("success"):
            continue
        txt = str(item.get("response", "")).strip()
        if not txt:
            continue
        if _extract_factual_candidate(txt):
            return True
    return False


def _requires_specialist_primary_evidence(user_text: str) -> bool:
    text_norm = _normalize_text(_strip_profile_suffix(user_text))
    if not text_norm:
        return False
    if _is_legal_requirement_regulatory_signal(text_norm):
        return False
    if _is_glossary_or_normalization_signal(text_norm):
        return False
    if _is_general_knowledge_signal(text_norm):
        return False
    return _contains_dairy_signal(text_norm)


def _get_specialist_primary_with_regulatory_context(
    user_text: str,
    responses: List[Dict[str, Any]],
    preferred_agent_id: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if _is_legal_requirement_regulatory_signal(_normalize_text(user_text)):
        return None, None

    specialist_candidates: List[Dict[str, Any]] = []
    regulatory_response: Optional[Dict[str, Any]] = None

    for item in responses:
        if not item.get("success") or not str(item.get("response", "")).strip():
            continue
        aid = int(item.get("agent_id", -1))
        factual = _extract_factual_candidate(str(item.get("response", "")))
        if aid == 3:
            if factual:
                regulatory_response = dict(item)
                regulatory_response["response"] = factual
            continue
        if aid in _ROUTING_BASELINE_IDS:
            continue
        if factual:
            row = dict(item)
            row["response"] = factual
            specialist_candidates.append(row)

    if not specialist_candidates:
        return None, regulatory_response

    primary = next(
        (item for item in specialist_candidates if int(item.get("agent_id", -1)) == preferred_agent_id),
        specialist_candidates[0],
    )
    return primary, regulatory_response


def _should_trigger_fallback(state: OrchestratorState) -> Tuple[bool, str]:
    attempts = int(state.get("fallback_attempts", 0) or 0)
    if attempts >= _FALLBACK_MAX_ATTEMPTS:
        return False, "max_attempts_reached"

    responses = state.get("agent_responses", []) or []
    candidates = _collect_fallback_candidates(state)
    if not candidates:
        return False, "no_candidates"

    bucket = str(state.get("routing_bucket", "medium"))
    if bucket == "low":
        return True, "low_confidence_bucket"

    user_text = _get_last_user_text(state.get("messages", []))

    if bucket == "medium":
        # Conservador por padrão, mas dispara se especialista não trouxe evidência factual.
        if _requires_specialist_primary_evidence(user_text) and not _has_specialist_factual_evidence(responses):
            return True, "medium_no_specialist_factual_evidence"
        return False, "medium_conservative"

    if _requires_specialist_primary_evidence(user_text) and not _has_specialist_factual_evidence(responses):
        return True, "no_specialist_factual_evidence"

    if _has_weak_or_conflicting_evidence(responses):
        return True, "weak_or_conflicting_evidence"

    return False, "sufficient_evidence"


def _collect_general_fallback_tables(state: "OrchestratorState") -> List[str]:
    """Seleciona tabelas para fallback geral priorizando contexto atual."""
    ordered_agent_ids: List[int] = []

    for aid in _sanitize_agent_ids(state.get("execution_plan", [])):
        if aid not in ordered_agent_ids:
            ordered_agent_ids.append(aid)
    for aid in _sanitize_agent_ids(state.get("chosen_agent_ids", [])):
        if aid not in ordered_agent_ids:
            ordered_agent_ids.append(aid)
    for aid in _sanitize_agent_ids(state.get("routing_alternatives", [])):
        if aid not in ordered_agent_ids:
            ordered_agent_ids.append(aid)

    # Completa com todos os agentes para realmente virar "base geral".
    for aid in sorted(_AGENT_TABLE_BY_ID):
        if aid not in ordered_agent_ids:
            ordered_agent_ids.append(aid)

    tables: List[str] = []
    for aid in ordered_agent_ids:
        table = _AGENT_TABLE_BY_ID.get(aid, "").strip()
        if table and table not in tables:
            tables.append(table)

    if not tables:
        tables = list(_ALL_AGENT_TABLES)

    cap = max(1, int(GENERAL_INDEX_FALLBACK_MAX_TABLES))
    return tables[:cap]


def _should_use_general_index_fallback(
    state: "OrchestratorState",
    successful_responses: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    if not ENABLE_GENERAL_INDEX_FALLBACK:
        return False, "general_index_disabled"

    # Evita uso em saudação/off-topic.
    if not state.get("chosen_agent_ids"):
        return False, "no_dairy_route"

    user_text = _strip_profile_suffix(_get_last_user_text(state.get("messages", [])))
    text_norm = _normalize_text(user_text)
    if not text_norm:
        return False, "empty_query"

    if GENERAL_INDEX_FALLBACK_REQUIRE_DAIRY_SIGNAL and not _contains_dairy_signal(text_norm):
        return False, "no_dairy_signal"

    if _requires_specialist_primary_evidence(user_text) and not _has_specialist_factual_evidence(successful_responses):
        return True, "no_specialist_factual_evidence"

    if not successful_responses:
        return True, "no_successful_specialist_response"

    if not GENERAL_INDEX_FALLBACK_ONLY_ON_WEAK:
        return True, "enabled_always"

    weak = _has_weak_or_conflicting_evidence(successful_responses)
    if weak:
        return True, "weak_or_conflicting_specialist_evidence"
    return False, "specialist_evidence_sufficient"


def _render_general_fallback_evidence(results: List[Dict[str, Any]], top_n: int = 4) -> str:
    snippets: List[str] = []
    for item in (results or [])[: max(1, top_n)]:
        metadata = item.get("metadata") or {}
        source_table = str(metadata.get("source_table", "")).strip() or "base_geral_unificada"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        content = re.sub(r"\s+", " ", content)
        if len(content) > 520:
            content = content[:520].rstrip() + "..."
        snippets.append(f"[{source_table}] {content}")
    return "\n".join(snippets).strip()


async def _fetch_general_index_fallback_evidence(
    state: "OrchestratorState",
) -> Tuple[str, str]:
    """Busca evidências em índice geral unificado.

    Retorna (evidence_text, reason). evidence_text vazio indica sem evidência útil.
    """
    user_text = _build_contextual_search_query(_get_last_user_text(state.get("messages", [])))
    tables = _collect_general_fallback_tables(state)
    if not user_text or not tables:
        return "", "general_index_no_query_or_tables"

    try:
        rows = await asyncio.to_thread(
            search_general_knowledge_base,
            user_text,
            tables,
            GENERAL_INDEX_FALLBACK_SEARCH_TYPE,
            max(1, int(GENERAL_INDEX_FALLBACK_PER_TABLE_K)),
            max(1, int(GENERAL_INDEX_FALLBACK_FINAL_K)),
            MATCH_THRESHOLD,
        )
    except Exception:
        return "", "general_index_search_error"

    min_results = max(1, int(GENERAL_INDEX_FALLBACK_MIN_RESULTS))
    if len(rows) < min_results:
        return "", "general_index_insufficient_results"

    evidence = _render_general_fallback_evidence(rows, top_n=min(6, len(rows)))
    if not evidence:
        return "", "general_index_empty_evidence"
    return evidence, "general_index_evidence_collected"


def _append_reason(reason: str, marker: str) -> str:
    base = (reason or "").strip()
    mark = (marker or "").strip()
    if not mark:
        return base
    return f"{base} | {mark}" if base else mark


def _should_use_web_fallback(
    state: "OrchestratorState",
    successful_responses: List[Dict[str, Any]],
    general_attempted: bool,
) -> Tuple[bool, str]:
    if not ENABLE_WEB_FALLBACK:
        return False, "web_fallback_disabled"

    if WEB_FALLBACK_PROVIDER != "duckduckgo":
        return False, "web_provider_not_supported"

    user_text = _strip_profile_suffix(_get_last_user_text(state.get("messages", [])))
    text_norm = _normalize_text(user_text)
    if not text_norm:
        return False, "web_empty_query"

    if WEB_FALLBACK_REQUIRE_DAIRY_SIGNAL and not _contains_dairy_signal(text_norm):
        return False, "web_no_dairy_signal"

    # Se o índice geral está desabilitado, o "require general first" não faz sentido
    # pois general_used nunca será True — nesse caso, ignora a restrição.
    if WEB_FALLBACK_REQUIRE_GENERAL_FALLBACK_FIRST and ENABLE_GENERAL_INDEX_FALLBACK and not general_attempted:
        return False, "web_requires_general_fallback_first"

    if _requires_specialist_primary_evidence(user_text) and not _has_specialist_factual_evidence(successful_responses):
        return True, "web_no_specialist_factual_evidence"

    if not successful_responses:
        return True, "web_no_specialist_evidence"

    if WEB_FALLBACK_ONLY_ON_WEAK and not _has_weak_or_conflicting_evidence(successful_responses):
        return False, "web_specialist_evidence_sufficient"

    return True, "web_weak_or_conflicting_evidence"


async def _fetch_web_fallback_evidence(
    state: "OrchestratorState",
) -> Tuple[str, List[Dict[str, str]], str]:
    user_text = _build_contextual_search_query(_get_last_user_text(state.get("messages", [])))
    if not user_text:
        return "", [], "web_no_query"

    try:
        rows = await asyncio.to_thread(
            search_web_duckduckgo,
            user_text,
            WEB_FALLBACK_ALLOWED_DOMAINS,
            max(1, int(WEB_FALLBACK_MAX_RESULTS)),
            float(WEB_FALLBACK_TIMEOUT_SEC),
            max(120, int(WEB_FALLBACK_MAX_SNIPPET_CHARS)),
        )
    except Exception:
        return "", [], "web_search_error"

    if not rows:
        return "", [], "web_no_whitelisted_results"

    if WEB_FALLBACK_FETCH_FULLTEXT:
        try:
            rows = await asyncio.to_thread(
                enrich_results_with_page_content,
                rows,
                float(WEB_FALLBACK_TIMEOUT_SEC),
                max(600, int(WEB_FALLBACK_MAX_PAGE_CHARS)),
            )
        except Exception:
            # Mantém snippets mesmo se enriquecimento falhar.
            pass

    evidence_text, sources = build_web_fallback_evidence(
        rows,
        max_sources=max(1, int(WEB_FALLBACK_MAX_SOURCES)),
    )
    if not evidence_text or not sources:
        return "", [], "web_insufficient_evidence"
    return evidence_text, sources, "web_evidence_collected"


def _render_web_sources_block(sources: List[Dict[str, str]]) -> str:
    domains: List[str] = []
    for item in sources or []:
        domain = str(item.get("domain", "")).strip()
        if domain and domain not in domains:
            domains.append(domain)
    if not domains:
        return ""
    return "_Fonte: " + ", ".join(domains) + "_"


def _looks_like_unusable_consolidation_answer(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    markers = (
        "nao foi possivel consolidar",
        "nao foi possivel obter uma resposta",
        "nao foi possivel responder",
        "resposta confiavel no momento",
        "tente reformular sua pergunta",
    )
    return any(marker in normalized for marker in markers)


def _build_evidence_grounded_fallback_answer(
    user_text: str,
    specialist_responses: List[Dict[str, Any]],
    general_evidence_text: str,
    web_evidence_text: str,
    web_sources: List[Dict[str, str]],
) -> str:
    factual_blocks: List[str] = []
    for item in specialist_responses:
        factual = _extract_factual_candidate(str(item.get("response", "")))
        if factual:
            factual_blocks.append(factual)

    if factual_blocks:
        joined = "\n\n".join(factual_blocks[:2]).strip()
        return _postprocess_consolidated_answer(user_text, joined)

    bullets: List[str] = []
    for raw in (general_evidence_text or "").splitlines():
        line = re.sub(r"^\[[^\]]+\]\s*", "", raw.strip())
        if len(line) < 40:
            continue
        bullets.append(line)
        if len(bullets) >= 3:
            break

    if not bullets:
        for raw in (web_evidence_text or "").splitlines():
            line = re.sub(r"^\[Fonte \d+\]\s*", "", raw.strip())
            if len(line) < 40:
                continue
            bullets.append(line)
            if len(bullets) >= 3:
                break

    if bullets:
        intro = (
            "Encontrei evidências relacionadas ao tema, mas a consolidação automática falhou. "
            "O que a base trouxe de mais útil foi:"
        )
        answer = intro + "\n\n- " + "\n- ".join(bullets)
        sources_block = _render_web_sources_block(web_sources) if web_sources else ""
        if sources_block:
            answer = (answer.strip() + "\n\n" + sources_block).strip()
        return _postprocess_consolidated_answer(user_text, answer)

    return _postprocess_consolidated_answer(
        user_text,
        "Não foi possível consolidar a resposta automaticamente, mas houve falha técnica durante a etapa final. "
        "Tente reformular a pergunta ou consultar o agente especialista direto.",
    )


async def _ainvoke_consolidation_with_timeout(
    state: "OrchestratorState",
    prompt: str,
) -> str:
    response = await asyncio.wait_for(
        _get_consolidation_model(_resolve_state_model(state)).ainvoke(
            [HumanMessage(content=prompt)]
        ),
        timeout=float(CONSOLIDATION_TIMEOUT_SEC),
    )
    return str(response.content or "")


# ============================================================
# Estado do orquestrador
# ============================================================

class OrchestratorState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    llm_model: str
    chosen_agent_ids: List[int]
    chosen_agent_names: List[str]
    execution_plan: List[int]
    agent_responses: List[Dict[str, Any]]
    final_response: str
    primary_agent_id: int
    primary_agent_name: str
    user_profile: Optional[Dict[str, Any]]
    routing_confidence: float
    routing_bucket: str
    routing_reason: str
    routing_alternatives: List[int]
    fallback_used: bool
    fallback_attempts: int
    fallback_trigger: str
    previous_agent_responses: List[Dict[str, Any]]
    general_index_fallback_used: bool
    web_fallback_used: bool
    web_fallback_sources: List[Dict[str, str]]
    needs_clarification: bool


# ============================================================
# Schema de classificação
# ============================================================

class ClassificationResult(BaseModel):
    """
    agent_ids: Lista de IDs relevantes, ordenada por relevância.
               Deve SEMPRE incluir 0 e 3 para perguntas de laticínios.
               [] apenas para saudações ou tópicos fora do setor.
    confidence: Grau de confiança do roteamento (0.0 a 1.0).
    reason/reasoning: Justificativa breve (para debug).
    alternatives: IDs alternativos relevantes para fallback/planner.
    """
    agent_ids: List[int]
    confidence: float = 0.50
    reason: str = ""
    alternatives: List[int] = []
    reasoning: str = ""


# ============================================================
# Lazy init dos modelos
# ============================================================

_classifier_models: Dict[str, Any] = {}
_consolidation_models: Dict[str, Any] = {}
_direct_models: Dict[str, Any] = {}


def _resolve_state_model(state: OrchestratorState) -> str:
    return str(state.get("llm_model") or LLM_MODEL)


def _get_classifier(model_name: str):
    if model_name not in _classifier_models:
        _classifier_models[model_name] = ChatOpenAI(model=model_name, temperature=CLASSIFIER_TEMPERATURE).with_structured_output(
            ClassificationResult,
            method="function_calling",
        )
    return _classifier_models[model_name]


def _get_consolidation_model(model_name: str):
    if model_name not in _consolidation_models:
        _consolidation_models[model_name] = ChatOpenAI(model=model_name, temperature=CONSOLIDATION_TEMPERATURE)
    return _consolidation_models[model_name]


def _get_direct_model(model_name: str):
    if model_name not in _direct_models:
        _direct_models[model_name] = ChatOpenAI(model=model_name, temperature=DIRECT_TEMPERATURE)
    return _direct_models[model_name]


# ============================================================
# Nó CLASSIFY
# ============================================================

def _get_last_user_text(messages: List[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _build_classification_state(
    route_text: str,
    agent_ids: List[int],
    confidence: float = 0.50,
    reason: str = "",
    alternatives: Optional[List[int]] = None,
) -> OrchestratorState:
    confidence = _recalibrate_confidence(route_text, agent_ids, confidence)
    bucket = _confidence_to_bucket(confidence)
    alternatives_ids = _sanitize_agent_ids(alternatives or [])

    if not agent_ids:
        return {
            "chosen_agent_ids": [],
            "chosen_agent_names": [],
            "execution_plan": [],
            "primary_agent_id": 0,
            "primary_agent_name": "Assistente Geral",
            "agent_responses": [],
            "final_response": "",
            "routing_confidence": confidence,
            "routing_bucket": bucket,
            "routing_reason": reason or "sem_agentes",
            "routing_alternatives": alternatives_ids,
            "fallback_used": False,
            "fallback_attempts": 0,
            "fallback_trigger": "",
            "previous_agent_responses": [],
            "general_index_fallback_used": False,
            "web_fallback_used": False,
            "web_fallback_sources": [],
        }

    sanitized_ids = _sanitize_agent_ids(agent_ids)
    execution_plan = _build_execution_plan(
        route_text=route_text,
        chosen_ids=sanitized_ids,
        alternatives=alternatives_ids,
        bucket=bucket,
    )

    agent_names = []
    for aid in sanitized_ids:
        cfg = get_agent_by_id(aid)
        agent_names.append(cfg["name"] if cfg else f"Agente {aid}")

    primary_agent_id = _choose_primary_agent_id(execution_plan or sanitized_ids, route_text=route_text)
    primary_cfg = get_agent_by_id(primary_agent_id)
    primary_agent_name = (
        primary_cfg["name"]
        if primary_cfg
        else next(
            (name for aid, name in zip(sanitized_ids, agent_names) if aid == primary_agent_id),
            "Assistente Geral",
        )
    )

    return {
        "chosen_agent_ids": sanitized_ids,
        "chosen_agent_names": agent_names,
        "execution_plan": execution_plan,
        "primary_agent_id": primary_agent_id,
        "primary_agent_name": primary_agent_name,
        "agent_responses": [],
        "final_response": "",
        "routing_confidence": confidence,
        "routing_bucket": bucket,
        "routing_reason": reason or "classificacao_llm",
        "routing_alternatives": alternatives_ids,
        "fallback_used": False,
        "fallback_attempts": 0,
        "fallback_trigger": "",
        "previous_agent_responses": [],
        "general_index_fallback_used": False,
        "web_fallback_used": False,
        "web_fallback_sources": [],
    }


async def classify(state: OrchestratorState) -> OrchestratorState:
    """Identifica quais agentes devem ser consultados.

    Agente 3 é incluído em toda pergunta de laticínios.
    Agente 0 entra apenas com sinal explícito de glossário/terminologia —
    o filtro _should_include_agent_0 é aplicado após a classificação LLM.
    """
    messages = state.get("messages", [])
    user_text = _get_last_user_text(messages)

    if not user_text:
        return _build_classification_state(route_text="", agent_ids=[], confidence=1.0, reason="mensagem_vazia")

    route_text = _strip_profile_suffix(user_text)
    current_question_norm = _normalize_text(_extract_current_user_segment(route_text))
    if _is_conversation_recap_request(current_question_norm):
        return _build_classification_state(
            route_text=route_text,
            agent_ids=[],
            confidence=0.98 if _has_recent_context_block(route_text) else 0.80,
            reason="conversation_recap",
        )

    cache_key = _normalize_text(route_text)

    cached_ids = _cache_get(cache_key)
    if cached_ids is not None:
        return _build_classification_state(
            route_text=route_text,
            agent_ids=cached_ids,
            confidence=0.95,
            reason="cache_hit",
        )

    if ORCHESTRATOR_FASTPATH:
        fast_ids = _rule_based_route(route_text)
        if fast_ids is not None:
            if fast_ids:
                _cache_set(cache_key, fast_ids)
            return _build_classification_state(
                route_text=route_text,
                agent_ids=fast_ids,
                confidence=_estimate_fastpath_confidence(route_text, fast_ids),
                reason="fastpath_rule_based",
            )

    system_prompt = get_orchestrator_prompt()

    classification_instruction = f"""

Com base na pergunta do usuário, identifique quais agentes devem ser consultados.

REGRAS DE INCLUSÃO:
- Agente 3 (Regulatórios): incluir em TODA pergunta de laticínios.
- Agente 0 (Base Geral): incluir SOMENTE se a pergunta envolver glossário,
  padronização de termos ("qual termo usar", "como chamar"), marcas/fabricantes/
  distribuidores/equipamentos específicos, ou saudação/off-topic. NÃO incluir
  agente 0 em perguntas puramente técnicas, analíticas, de processo ou regulatórias
  — a base do agente 0 não cobre esses temas e só adiciona ruído.
- Especialistas 1-4: adicionar apenas se a pergunta for claramente desse domínio.

ESPECIALISTAS DISPONÍVEIS:
{_SPECIALISTS_DESC}
FORMATO DA RESPOSTA:
- Saudação / off-topic → []
- Pergunta de glossário/terminologia → [0, 3]
- Pergunta regulatória/técnica → [3] ou [3, X]
- Pergunta de glossário + especialidade → [0, 3, X]
- Máx 5 IDs. Ordene por relevância: agente mais relevante primeiro.

ALÉM DOS IDs, informe:
- confidence: número entre 0.0 e 1.0
- reason: justificativa curta
- alternatives: IDs alternativos relevantes (sem repetir os principais)

REGRAS DE DESEMPATE (OBRIGATÓRIAS):
- Se a pergunta for de glossário, padronização de termos, "qual termo usar" ou "significado esperado",
  priorize [0,3] e NÃO escolha especialista como primário.
- Se a pergunta envolver rotulagem/denominação/embalagem de produto lácteo,
  priorize [0,3] (regulatório), mesmo que cite nome de queijo.
- Se a pergunta mencionar "norma", "regulamento", "IN", "RDC", "decreto" ou "artigo",
  priorize [0,3] e evite priorizar formulação (6) como agente principal.
- Se a pergunta pedir requisito mínimo/obrigatório/exigido ("período mínimo exigido",
  "mínimo legal", "deve sofrer maturação"), trate como regulatória e priorize [0,3].
- Se a pergunta for de método analítico/laboratorial (Dornic, titulação, HCl, NaOH, IN 68,
  absorbância, comprimento de onda, centrifugação, m/m, m/v etc.),
  inclua 4 e priorize 4 como especialista — IN 68 é documento de métodos do Agente 4, não regulatório.
- Se a pergunta citar fermentação em queijo/coalhada de processo (corte de coalhada, pH de corte),
  priorize 1 (queijos) e não 2.
- Se a pergunta for de padronização de termo/glossário ("qual termo usar", "significado esperado"),
  priorize [0,3] e evite especialistas como primários.
- Evite super-especializar perguntas institucionais ou terminológicas.

 {_CLASSIFIER_FEW_SHOTS}
 """

    classifier = _get_classifier(_resolve_state_model(state))
    result = await classifier.ainvoke([
        SystemMessage(content=system_prompt + classification_instruction),
        HumanMessage(content=user_text),
    ])

    # Valida IDs (0-6), preserva ordem, remove duplicatas
    agent_ids = _sanitize_agent_ids(result.agent_ids)
    alternatives = _sanitize_agent_ids(getattr(result, "alternatives", []) or [])
    confidence = _clamp_confidence(getattr(result, "confidence", 0.50))
    reason = (getattr(result, "reason", "") or getattr(result, "reasoning", "") or "").strip()

    # Hard constraints para sinais claros de dominio lacteo.
    agent_ids = _apply_dairy_hard_constraints(route_text, agent_ids)
    alternatives = [aid for aid in alternatives if aid not in agent_ids]
    agent_ids, alternatives = _apply_domain_guardrails(route_text, agent_ids, alternatives)

    # Filtro de agente 0 — aplica o mesmo critério do fast-path:
    # só entra se há sinal explícito de glossário, terminologia ou conhecimento geral.
    # A base do agente 0 (verdades_absolutas + glossário) não cobre queries técnicas
    # ou regulatórias genéricas — incluí-lo nessas queries só adiciona latência e ruído.
    text_norm_for_a0 = _normalize_text(route_text)
    if not _should_include_agent_0(text_norm_for_a0):
        agent_ids = [aid for aid in agent_ids if aid != 0]

    if not agent_ids:
        return _build_classification_state(
            route_text=route_text,
            agent_ids=[],
            confidence=confidence,
            reason=reason or "sem_dominio_relevante",
            alternatives=alternatives,
        )

    _cache_set(cache_key, agent_ids)
    classification = _build_classification_state(
        route_text=route_text,
        agent_ids=agent_ids,
        confidence=confidence,
        reason=reason or "classificacao_llm",
        alternatives=alternatives,
    )

    if _should_ask_clarification(
        chosen_ids=classification.get("chosen_agent_ids", []),
        bucket=classification.get("routing_bucket", "medium"),
        confidence=classification.get("routing_confidence", 0.5),
        reason=classification.get("routing_reason", ""),
        messages=messages,
    ):
        classification["needs_clarification"] = True

    return classification

# ============================================================
# Roteamento condicional
# ============================================================

def route(state: OrchestratorState) -> str:
    if state.get("needs_clarification"):
        return "ask_clarification"
    planned = state.get("execution_plan")
    if planned is not None:
        return "respond_direct" if not planned else "execute"
    return "respond_direct" if not state.get("chosen_agent_ids") else "execute"


def route_after_execute(state: OrchestratorState) -> str:
    should_fallback, _ = _should_trigger_fallback(state)
    return "fallback_reclassify" if should_fallback else "consolidate"


def route_after_fallback(state: OrchestratorState) -> str:
    trigger = str(state.get("fallback_trigger", ""))
    if trigger == "fallback_no_plan_change":
        return "consolidate"
    return "execute"


# ============================================================
# NÃ³ EXECUTE â€" execuÃ§Ã£o paralela
# ============================================================

async def execute(state: OrchestratorState) -> OrchestratorState:
    """Invoca todos os agentes em PARALELO via asyncio.gather.

    LatÃªncia total â‰ˆ tempo do agente mais lento (nÃ£o a soma).
    Cada agente tem timeout individual de AGENT_TIMEOUT segundos.
    """
    agent_ids = state.get("execution_plan") or state.get("chosen_agent_ids", [])
    agent_names = [
        (get_agent_by_id(aid) or {}).get("name", f"Agente {aid}")
        for aid in agent_ids
    ]

    user_text = _get_last_user_text(state.get("messages", []))
    search_query = _build_contextual_search_query(user_text)

    if not user_text or not search_query:
        return {"agent_responses": []}

    # Computa o embedding da query UMA vez para todos os agentes paralelos.
    # Sem isso, cada agente chamaria embed_query() independentemente para a mesma string
    # — (N-1) chamadas duplicadas à OpenAI (~150ms cada desperdiçadas).
    shared_embedding: Optional[List[float]] = None
    try:
        shared_embedding = await asyncio.to_thread(
            embed_query, search_query
        )
    except Exception:
        pass  # Fallback: cada agente computa seu próprio embedding normalmente.

    async def call_one(agent_id: int, agent_name: str) -> Dict[str, Any]:
        try:
            graph = get_agent_graph(agent_id, _resolve_state_model(state))
            result = await asyncio.wait_for(
                graph.ainvoke({
                    "messages": [HumanMessage(content=user_text)],
                    "llm_model": _resolve_state_model(state),
                    "precomputed_embedding": shared_embedding,
                }),
                timeout=AGENT_TIMEOUT,
            )
            agent_msgs = result.get("messages", [])
            agent_text = ""
            for msg in reversed(agent_msgs):
                if isinstance(msg, AIMessage):
                    content = msg.content
                    if isinstance(content, list):
                        agent_text = "\n".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        )
                    elif isinstance(content, str):
                        agent_text = content
                    if agent_text:
                        break
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "response": agent_text,
                "success": bool(agent_text),
            }
        except asyncio.TimeoutError:
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "response": f"{agent_name}: timeout ao consultar base de conhecimento.",
                "success": False,
            }
        except Exception as e:
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "response": f"Erro ao consultar {agent_name}: {e}",
                "success": False,
            }

    # Dispara todos os agentes ao mesmo tempo
    current_responses = await asyncio.gather(
        *[call_one(aid, name) for aid, name in zip(agent_ids, agent_names)]
    )

    previous_responses = state.get("previous_agent_responses", []) or []
    responses = _merge_agent_responses(previous_responses, list(current_responses))

    successful_ids = [r["agent_id"] for r in responses if r.get("success")]
    candidate_ids = successful_ids or agent_ids
    primary_agent_id = _choose_primary_agent_id(candidate_ids, route_text=user_text)
    primary_cfg = get_agent_by_id(primary_agent_id)
    primary_agent_name = (
        primary_cfg["name"] if primary_cfg else "Assistente Geral"
    )

    return {
        "agent_responses": list(responses),
        "primary_agent_id": primary_agent_id,
        "primary_agent_name": primary_agent_name,
        "previous_agent_responses": [],
    }


# ============================================================
# NÃ³ FALLBACK_RECLASSIFY â€" segunda passada inteligente
# ============================================================

async def fallback_reclassify(state: OrchestratorState) -> OrchestratorState:
    route_text = _strip_profile_suffix(_get_last_user_text(state.get("messages", [])))
    current_chosen = _sanitize_agent_ids(state.get("chosen_agent_ids", []))
    current_plan = _sanitize_agent_ids(state.get("execution_plan", current_chosen))
    current_bucket = str(state.get("routing_bucket", "medium"))
    current_conf = _clamp_confidence(state.get("routing_confidence", 0.50))
    current_alts = _sanitize_agent_ids(state.get("routing_alternatives", []))
    current_reason = str(state.get("routing_reason", ""))
    attempts = int(state.get("fallback_attempts", 0) or 0)

    should_fallback, trigger = _should_trigger_fallback(state)
    if not should_fallback:
        return {"fallback_trigger": trigger}

    fallback_candidates = _collect_fallback_candidates(state)
    if not fallback_candidates:
        return {"fallback_trigger": "no_candidates"}

    extra_cap = _FALLBACK_EXTRA_SPECIALISTS.get(current_bucket, 2)
    selected_extra = fallback_candidates[:extra_cap]

    # Expande escolhidos e recalcula plano mantendo bucket original.
    new_chosen = _sanitize_agent_ids(current_chosen + selected_extra)
    new_alts = _sanitize_agent_ids(current_alts + fallback_candidates[extra_cap:])
    new_conf = max(current_conf, 0.65)
    new_reason = (
        f"{current_reason} | fallback_second_pass:{trigger}"
        if current_reason
        else f"fallback_second_pass:{trigger}"
    )

    rebuilt = _build_classification_state(
        route_text=route_text,
        agent_ids=new_chosen,
        confidence=new_conf,
        reason=new_reason,
        alternatives=new_alts,
    )
    # Mantem respostas da primeira passada para mescla posterior.
    rebuilt["previous_agent_responses"] = list(state.get("agent_responses", []) or [])
    rebuilt["fallback_used"] = True
    rebuilt["fallback_attempts"] = attempts + 1
    rebuilt["fallback_trigger"] = trigger
    # Garante que o novo plano realmente evoluiu.
    if rebuilt.get("execution_plan") == current_plan:
        rebuilt["fallback_trigger"] = "fallback_no_plan_change"
        return {"fallback_trigger": "fallback_no_plan_change"}
    return rebuilt


# ============================================================
# Clarificação — detecção e nó
# ============================================================

_AGENT_DOMAIN_LABELS: Dict[int, str] = {
    1: "fabricação e tecnologia de queijos",
    2: "fermentados (iogurtes, kefir, skyr)",
    4: "qualidade do leite e métodos analíticos",
    5: "diagnóstico visual de defeitos",
    6: "formulação de produtos lácteos",
}


def _should_ask_clarification(
    chosen_ids: List[int],
    bucket: str,
    confidence: float,
    reason: str,
    messages: List[AnyMessage],
) -> bool:
    """Retorna True quando a pergunta é genuinamente ambígua e merece clarificação.

    Critérios conservadores — só dispara quando bucket=low E a query
    é curta/vaga OU há múltiplos especialistas com confiança muito baixa.
    Fast-path e cache hits são sempre decisivos e nunca chegam aqui.
    """
    if bucket != "low":
        return False

    specialists = [aid for aid in chosen_ids if aid not in _ROUTING_BASELINE_IDS]
    if not specialists:
        return False

    # Fast-path e cache já resolveram — não perguntar
    if "fastpath" in reason or "cache_hit" in reason:
        return False

    # Anti-loop: se o assistente já fez uma pergunta recentemente, não repetir
    recent_ai = [m for m in messages[-6:] if isinstance(m, AIMessage)]
    for msg in recent_ai:
        txt = (msg.content or "") if isinstance(msg.content, str) else ""
        if "?" in txt:
            return False

    user_text = _get_last_user_text(messages)
    current = _extract_current_user_segment(_strip_profile_suffix(user_text))
    current_norm = _normalize_text(current)
    words = current_norm.split()

    # Sinal de laticínio obrigatório — não perguntar em off-topic
    if not _contains_dairy_signal(current_norm):
        return False

    # Query muito curta e vaga
    if len(words) < 5:
        return True

    # Múltiplos especialistas com confiança muito baixa
    if len(specialists) >= 2 and confidence < 0.50:
        return True

    return False


async def ask_clarification(state: OrchestratorState) -> OrchestratorState:
    """Gera uma pergunta de clarificação direcionada ao usuário.

    Chamado quando o orquestrador tem baixa confiança e a pergunta é ambígua
    entre especialistas. A resposta do usuário na próxima mensagem irá
    resolver a ambiguidade naturalmente pelo contexto da conversa.
    """
    user_text = _get_last_user_text(state.get("messages", []))
    current = _extract_current_user_segment(_strip_profile_suffix(user_text))
    chosen = state.get("chosen_agent_ids", [])
    specialists = [aid for aid in chosen if aid not in _ROUTING_BASELINE_IDS]

    domain_options = [
        _AGENT_DOMAIN_LABELS[aid]
        for aid in specialists
        if aid in _AGENT_DOMAIN_LABELS
    ]

    if domain_options:
        options_str = " ou ".join(domain_options)
        system = (
            "Você é o assistente do DairyApp AI, especializado em tecnologia de laticínios. "
            "O usuário fez uma pergunta um pouco ampla ou ambígua. "
            "Faça UMA pergunta curta e direta para entender melhor o que ele precisa, "
            f"considerando que pode ser sobre: {options_str}. "
            "Seja cordial, não explique os agentes internamente, apenas pergunte o que "
            "o usuário quer saber. Responda em português brasileiro. Máximo 2 frases."
        )
    else:
        system = (
            "Você é o assistente do DairyApp AI, especializado em tecnologia de laticínios. "
            "O usuário fez uma pergunta um pouco ampla. Peça para ele detalhar melhor "
            "o que precisa saber, com uma pergunta curta e cordial. "
            "Responda em português brasileiro. Máximo 2 frases."
        )

    response = await _get_direct_model(_resolve_state_model(state)).ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=current or user_text),
    ])

    question = _sanitize_math_for_ui(response.content or "")
    if not question:
        question = "Poderia detalhar um pouco mais sua dúvida? Assim consigo te direcionar melhor."

    return {
        "final_response": question,
        "messages": [AIMessage(content=question)],
        "agent_responses": [],
        "needs_clarification": True,
    }


# ============================================================
# NÃ³ RESPOND_DIRECT â€" saudaÃ§Ãµes e off-topic
# ============================================================

async def respond_direct(state: OrchestratorState) -> OrchestratorState:
    """Resposta direta para saudações e mensagens off-topic (sem RAG)."""
    user_text = _get_last_user_text(state.get("messages", []))
    current_text = _extract_current_user_segment(user_text)
    current_norm = _normalize_text(current_text)

    if _is_conversation_recap_request(current_norm):
        if _has_recent_context_block(user_text):
            system = (
                "Voce e o assistente geral do Dairy AI (DairyApp). "
                "O usuario pediu um resumo do que acabou de ser discutido e voce recebeu "
                "um bloco [Contexto recente da conversa]. Resuma APENAS o que esta nesse "
                "contexto. Priorize sintese executiva em 3 a 5 bullets curtos ou um "
                "paragrafo objetivo. Destaque fatos tecnicos, conclusoes e pendencias. "
                "Nao consulte RAG, nao invente fatos e nao reabra a classificacao por dominio. "
                "Se houver contradicao no proprio contexto, aponte isso explicitamente."
            )
        else:
            system = (
                "Voce e o assistente geral do Dairy AI (DairyApp). "
                "O usuario pediu um resumo da conversa, mas nenhum contexto recente foi fornecido. "
                "Explique isso de forma curta e educada e convide o usuario a retomar o tema "
                "com uma nova pergunta objetiva."
            )
    else:
        system = (
            "Voce e o assistente geral do Dairy AI (DairyApp), especializado em tecnologia "
            "de laticinios. Em saudacoes e primeira interacao, apresente-se de forma curta "
            "como Dairy AI e diga em uma frase como pode ajudar. Depois disso, evite repetir "
            "apresentacoes e va direto ao ponto. Quando pertinente, sugira perguntas tecnicas "
            "sobre queijos, fermentados, regulatorios, qualidade do leite, diagnostico de "
            "defeitos ou formulacao. Responda em portugues brasileiro."
        )

    response = await _get_direct_model(_resolve_state_model(state)).ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=user_text),
    ])

    final_text = _sanitize_math_for_ui(response.content or "")
    return {
        "agent_responses": [],
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
    }


# ============================================================
# NÃ³ CONSOLIDATE â€" fusÃ£o das respostas
# ============================================================

async def consolidate(state: OrchestratorState) -> OrchestratorState:
    """Funde as respostas dos agentes em uma resposta coerente.

    1 agente bem-sucedido â†’ repassa direto (sem chamada LLM extra).
    2+ agentes â†’ LLM funde preservando todos os dados tÃ©cnicos.
    """
    # Veio de respond_direct: já tem final_response
    if not state.get("chosen_agent_ids") and state.get("final_response"):
        final_text = state.get("final_response") or ""
        user_text = _get_last_user_text(state.get("messages", []))
        final_text = _postprocess_consolidated_answer(user_text, final_text)
        if final_text:
            msgs = state.get("messages", [])
            if msgs and isinstance(msgs[-1], AIMessage) and (msgs[-1].content or "") == final_text:
                return {}
            return {"messages": [AIMessage(content=final_text)]}
        return {}

    successful = [
        r for r in state.get("agent_responses", [])
        if r.get("success") and r.get("response")
    ]
    user_text = _get_last_user_text(state.get("messages", []))

    # Fallback final em base geral unificada (controlado por flag).
    # Só entra quando a evidência dos especialistas está ausente/fraca.
    general_should_use, general_trigger = _should_use_general_index_fallback(state, successful)
    general_evidence_text = ""
    general_used = False
    if general_should_use:
        general_evidence_text, evidence_status = await _fetch_general_index_fallback_evidence(state)
        if general_evidence_text:
            general_used = True
            general_trigger = _append_reason(general_trigger, evidence_status)

    # Última camada: web fallback com whitelist de domínios confiáveis.
    web_should_use, web_trigger = _should_use_web_fallback(state, successful, general_should_use)
    web_evidence_text = ""
    web_sources: List[Dict[str, str]] = []
    web_used = False
    if web_should_use:
        web_evidence_text, web_sources, web_status = await _fetch_web_fallback_evidence(state)
        if web_evidence_text and web_sources:
            web_used = True
            web_trigger = _append_reason(web_trigger, web_status)

    if not successful:
        if general_used or web_used:
            evidence_blocks = []
            if general_used and general_evidence_text:
                evidence_blocks.append(f"BASE GERAL UNIFICADA:\n{general_evidence_text}")
            if web_used and web_evidence_text:
                evidence_blocks.append(f"WEB (DOMÍNIOS CONFIÁVEIS):\n{web_evidence_text}")
            evidence_blob = "\n\n".join(evidence_blocks).strip()
            prompt = (
                "Você é o assistente geral do DairyApp AI. "
                "Responda à pergunta do usuário com base SOMENTE nas evidências abaixo. "
                "Se a evidência não for suficiente para afirmar algo com segurança, diga isso explicitamente.\n\n"
                f"PERGUNTA: {user_text}\n\n"
                f"EVIDÊNCIAS:\n{evidence_blob}\n\n"
                "Resposta:"
            )
            try:
                response_text = await _ainvoke_consolidation_with_timeout(state, prompt)
                final_text = _postprocess_consolidated_answer(user_text, response_text)
                if _looks_like_unusable_consolidation_answer(final_text):
                    final_text = _build_evidence_grounded_fallback_answer(
                        user_text,
                        successful,
                        general_evidence_text,
                        web_evidence_text,
                        web_sources,
                    )
            except Exception:
                final_text = _build_evidence_grounded_fallback_answer(
                    user_text,
                    successful,
                    general_evidence_text,
                    web_evidence_text,
                    web_sources,
                )
            sources_block = _render_web_sources_block(web_sources) if web_used else ""
            if sources_block:
                final_text = (final_text.strip() + "\n\n" + sources_block).strip()
            fallback_marker = []
            if general_used:
                fallback_marker.append(f"general_index_fallback:{general_trigger}")
            if web_used:
                fallback_marker.append(f"web_fallback:{web_trigger}")
            return {
                "final_response": final_text,
                "messages": [AIMessage(content=final_text)],
                "general_index_fallback_used": general_used,
                "web_fallback_used": web_used,
                "web_fallback_sources": web_sources if web_used else [],
                "fallback_used": True,
                "fallback_trigger": " | ".join(fallback_marker),
                "routing_reason": _append_reason(
                    str(state.get("routing_reason", "")),
                    "web_fallback_used" if web_used else "general_index_fallback_used",
                ),
            }

        final_text = (
            "Não foi possível obter uma resposta no momento. "
            "Por favor, tente reformular sua pergunta."
        )
        final_text = _postprocess_consolidated_answer(user_text, final_text)
        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
        }

    # 1 agente: repassa direto (econômico), exceto se fallback geral ou web foi acionado.
    # Só faz fast-path se a resposta tem conteúdo factual — caso contrário cai no _all_uncertain
    # abaixo, evitando propagar "não tenho informação" diretamente ao usuário.
    if len(successful) == 1 and not general_used and not web_used:
        single_fact = _extract_factual_candidate(str(successful[0]["response"]))
        if single_fact:
            final_text = _postprocess_consolidated_answer(user_text, single_fact)
            return {
                "final_response": final_text,
                "messages": [AIMessage(content=final_text)],
            }

    # Todos os agentes retornaram apenas incerteza (sem conteúdo factual).
    # Neste caso, não misturamos as respostas "sem info" com a evidência de fallback
    # pois isso produziria respostas duplicadas e confusas.
    _all_uncertain = successful and not any(
        _extract_factual_candidate(str(r.get("response", ""))) for r in successful
    )
    if _all_uncertain:
        if general_used or web_used:
            # Consolida usando SOMENTE a evidência de fallback — ignora respostas "sem info".
            evidence_blocks = []
            if general_used and general_evidence_text:
                evidence_blocks.append(f"BASE GERAL UNIFICADA:\n{general_evidence_text}")
            if web_used and web_evidence_text:
                evidence_blocks.append(f"WEB (DOMÍNIOS CONFIÁVEIS):\n{web_evidence_text}")
            evidence_blob = "\n\n".join(evidence_blocks).strip()
            fallback_only_prompt = (
                "Você é o assistente geral do DairyApp AI. "
                "Responda à pergunta do usuário com base SOMENTE nas evidências abaixo. "
                "Se a evidência não cobre o dado exato, informe de forma objetiva o que foi encontrado e o que não estava disponível. "
                "Não mencione agentes internos, bases de conhecimento ou ferramentas internas.\n\n"
                f"PERGUNTA: {user_text}\n\n"
                f"EVIDÊNCIAS:\n{evidence_blob}\n\n"
                "Resposta:"
            )
            try:
                response_text = await _ainvoke_consolidation_with_timeout(state, fallback_only_prompt)
                final_text = _postprocess_consolidated_answer(user_text, response_text)
                if _looks_like_unusable_consolidation_answer(final_text):
                    final_text = _build_evidence_grounded_fallback_answer(
                        user_text, [], general_evidence_text, web_evidence_text, web_sources
                    )
            except Exception:
                final_text = _build_evidence_grounded_fallback_answer(
                    user_text, [], general_evidence_text, web_evidence_text, web_sources
                )
            sources_block = _render_web_sources_block(web_sources) if web_used else ""
            if sources_block:
                final_text = (final_text.strip() + "\n\n" + sources_block).strip()
            fallback_marker = []
            if general_used:
                fallback_marker.append(f"general_index_fallback:{general_trigger}")
            if web_used:
                fallback_marker.append(f"web_fallback:{web_trigger}")
            return {
                "final_response": final_text,
                "messages": [AIMessage(content=final_text)],
                "general_index_fallback_used": general_used,
                "web_fallback_used": web_used,
                "web_fallback_sources": web_sources if web_used else [],
                "fallback_used": True,
                "fallback_trigger": " | ".join(fallback_marker),
                "routing_reason": _append_reason(
                    str(state.get("routing_reason", "")),
                    "web_fallback_used" if web_used else "general_index_fallback_used",
                ),
            }
        else:
            # Nenhum fallback trouxe evidência — não exibe resposta incerta do agente
            # ao usuário; retorna mensagem neutra sem expor "não encontrei informação".
            final_text = (
                "No momento não foi possível localizar dados suficientes sobre este tema "
                "nas fontes disponíveis. Tente reformular a pergunta com mais detalhes "
                "ou consulte diretamente a documentação técnica."
            )
            return {
                "final_response": final_text,
                "messages": [AIMessage(content=final_text)],
            }

    # 2+ agentes (ou 1 + fallback geral): consolida com LLM.

    regulatory_preferred = (
        _prefer_regulatory_requirement_response(user_text, successful)
        if not general_used and not web_used
        else None
    )
    if regulatory_preferred:
        regulatory_preferred = _postprocess_consolidated_answer(user_text, regulatory_preferred)
        return {
            "final_response": regulatory_preferred,
            "messages": [AIMessage(content=regulatory_preferred)],
        }

    # Em perguntas objetivas, quando houver um único especialista com
    # resposta factual direta, devolve essa resposta sem adicionar ressalvas.
    preferred = _prefer_direct_fact_response(user_text, successful) if not general_used and not web_used else None
    if preferred:
        preferred = _postprocess_consolidated_answer(user_text, preferred)
        return {
            "final_response": preferred,
            "messages": [AIMessage(content=preferred)],
        }

    # Se existe pelo menos uma resposta factual, remove respostas que são
    # apenas ressalva/ausência para não "contaminar" a consolidação.
    factual_responses = []
    for r in successful:
        factual = _extract_factual_candidate(str(r.get("response", "")))
        if factual:
            item = dict(r)
            item["response"] = factual
            factual_responses.append(item)
    if factual_responses:
        successful = factual_responses

    # Otimização: única fonte de evidência factual → repassa direto, sem LLM de consolidação.
    # Consolidar uma única resposta não agrega valor e custa 800ms–2s desnecessários.
    if len(successful) == 1 and not general_used and not web_used:
        final_text = _postprocess_consolidated_answer(user_text, successful[0]["response"])
        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
        }

    # Separa respostas por papel: especialistas de domínio (1,2,4,5,6) vs baseline (0,3).
    # Especialistas = conteúdo técnico principal; Agent 3 = complemento regulatório;
    # Agent 0 = terminologia/glossário (contexto de suporte).
    _specialist_resps = [
        r for r in successful if int(r.get("agent_id", -1)) not in _ROUTING_BASELINE_IDS
    ]
    _regulatory_resp = next(
        (r for r in successful if int(r.get("agent_id", -1)) == 3), None
    )
    _general_resp = next(
        (r for r in successful if int(r.get("agent_id", -1)) == 0), None
    )

    _specialist_block = "".join(
        f"\n--- {r['agent_name']} ---\n{r['response']}\n"
        for r in _specialist_resps
    )
    _regulatory_block = (
        f"\n--- {_regulatory_resp['agent_name']} ---\n{_regulatory_resp['response']}\n"
        if _regulatory_resp else ""
    )
    _general_block = (
        f"\n--- {_general_resp['agent_name']} ---\n{_general_resp['response']}\n"
        if _general_resp else ""
    )
    _fallback_block = ""
    if general_used and general_evidence_text:
        _fallback_block += f"\n--- Base Geral Unificada (fallback) ---\n{general_evidence_text}\n"
    if web_used and web_evidence_text:
        _fallback_block += f"\n--- Web (domínios confiáveis) ---\n{web_evidence_text}\n"

    if _specialist_resps:
        # Caminho hierárquico: especialistas como base, regulatório como complemento.
        _prompt_body = f"PERGUNTA: {user_text}\n\nCONTEÚDO TÉCNICO PRINCIPAL:{_specialist_block}"
        if _regulatory_block:
            _prompt_body += f"\nCONTEXTO REGULATÓRIO COMPLEMENTAR:{_regulatory_block}"
        if _general_block:
            _prompt_body += f"\nTERMINOLOGIA / BASE GERAL:{_general_block}"
        if _fallback_block:
            _prompt_body += f"\nEVIDÊNCIA ADICIONAL:{_fallback_block}"

        consolidation_prompt = (
            "Você é o assistente geral do DairyApp AI.\n"
            "Regras de composição da resposta:\n"
            "1. Use o CONTEÚDO TÉCNICO PRINCIPAL como base da resposta\n"
            "2. Acrescente do CONTEXTO REGULATÓRIO COMPLEMENTAR apenas o que for diretamente "
            "relevante para a pergunta — não repita o que o técnico já cobriu\n"
            "3. Se técnico e norma divergirem em um ponto específico, a norma prevalece naquele ponto\n"
            "4. Se a pergunta for sobre requisito mínimo/obrigatório/exigido, trate o contexto "
            "regulatório como critério definitivo; não transforme prática técnica em exigência legal\n"
            "5. Preserve todos os dados técnicos (temperaturas, pHs, prazos, concentrações)\n"
            "6. NÃO invente fatos além das evidências fornecidas\n"
            "7. NÃO adicione ressalvas genéricas se a pergunta principal já foi respondida\n"
            "8. NÃO mencione agentes internos, bases de conhecimento ou ferramentas\n"
            "9. Tom técnico e profissional em português brasileiro\n\n"
            + _prompt_body
            + "\n\nResposta final:"
        )
    else:
        # Sem especialistas de domínio — apenas baseline (regulatório + geral + fallback).
        _all_block = _regulatory_block + _general_block + _fallback_block
        consolidation_prompt = (
            "Você é o assistente geral do DairyApp AI. "
            "Responda com base SOMENTE nas evidências abaixo. "
            "Preserve todos os dados. NÃO invente fatos. "
            "NÃO mencione agentes ou ferramentas internas. "
            "Tom técnico em português brasileiro.\n\n"
            f"PERGUNTA: {user_text}\n\n"
            f"EVIDÊNCIAS:{_all_block}\n"
            "Resposta:"
        )

    try:
        response_text = await _ainvoke_consolidation_with_timeout(state, consolidation_prompt)
        final_text = _postprocess_consolidated_answer(user_text, response_text)
        if _looks_like_unusable_consolidation_answer(final_text):
            final_text = _build_evidence_grounded_fallback_answer(
                user_text,
                successful,
                general_evidence_text,
                web_evidence_text,
                web_sources,
            )
    except Exception:
        final_text = _build_evidence_grounded_fallback_answer(
            user_text,
            successful,
            general_evidence_text,
            web_evidence_text,
            web_sources,
        )

    if web_used:
        sources_block = _render_web_sources_block(web_sources)
        if sources_block:
            final_text = (final_text.strip() + "\n\n" + sources_block).strip()

    payload: OrchestratorState = {
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
    }
    if general_used or web_used:
        payload["general_index_fallback_used"] = general_used
        payload["web_fallback_used"] = web_used
        payload["web_fallback_sources"] = web_sources if web_used else []
        marker = []
        if general_used:
            marker.append(f"general_index_fallback:{general_trigger}")
        if web_used:
            marker.append(f"web_fallback:{web_trigger}")
        payload.update(
            {
                "fallback_used": True,
                "fallback_trigger": " | ".join(marker),
                "routing_reason": _append_reason(
                    str(state.get("routing_reason", "")),
                    "web_fallback_used" if web_used else "general_index_fallback_used",
                ),
            }
        )
    return payload


# ============================================================
# Montagem e compilação do grafo
# ============================================================

def build_orchestrator_graph() -> Any:
    graph = StateGraph(OrchestratorState)

    graph.add_node("classify", classify)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("execute", execute)
    graph.add_node("fallback_reclassify", fallback_reclassify)
    graph.add_node("respond_direct", respond_direct)
    graph.add_node("consolidate", consolidate)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route,
        {"ask_clarification": "ask_clarification", "execute": "execute", "respond_direct": "respond_direct"},
    )

    graph.add_edge("ask_clarification", END)

    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {"fallback_reclassify": "fallback_reclassify", "consolidate": "consolidate"},
    )
    graph.add_conditional_edges(
        "fallback_reclassify",
        route_after_fallback,
        {"execute": "execute", "consolidate": "consolidate"},
    )
    graph.add_edge("respond_direct", "consolidate")
    graph.add_edge("consolidate", END)

    return graph.compile()


# ============================================================
# Instância global (lazy cache)
# ============================================================

_orchestrator_graph = None


def get_orchestrator_graph() -> Any:
    global _orchestrator_graph
    if _orchestrator_graph is None:
        _orchestrator_graph = build_orchestrator_graph()
    return _orchestrator_graph
