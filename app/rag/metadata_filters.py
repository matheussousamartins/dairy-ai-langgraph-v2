# -*- coding: utf-8 -*-
"""
rag/metadata_filters.py - Classificacao de intencao para filtros de busca (V2 Single-Agent)

Dois niveis:
  1. Fast-path keyword (<1ms): saudacoes e sinais inequivocos (IN 76, dornic, kefir...).
     Reutiliza orch_signals.py + routing_rules.yaml — sem LLM.
  2. LLM classifier (gpt-4o-mini, ~300ms): todos os outros casos.
     Retorna lista de dominios em JSON estruturado. Cache LRU de 512 entradas.

Por que dois niveis?
  - Fast-path evita LLM para ~30% das queries (saudacoes, queries com termos
    inequivocos como "IN 76", "dornic", "kefir").
  - LLM resolve ambiguidade cruzada: "CCS para Parmesao" vai para queijos, nao
    para qualidade — porque o contexto ("Parmesao") e levado em conta.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

from app.agents.orch_signals import (
    _ROUTING_RULES,
    _contains_dairy_signal,
    _is_labeling_regulatory_signal,
    _is_legal_requirement_regulatory_signal,
    _is_normative_regulatory_signal,
    _is_strong_cheese_signal,
    _is_strong_regulatory_signal,
    _looks_like_greeting_only,
)
from app.agents.orch_text import _normalize_text
from app.agents.agent_config import AGENTS
from app.config import (
    SINGLE_AGENT_CLASSIFIER_MODEL,
    SINGLE_AGENT_CLASSIFIER_TIMEOUT_SEC,
    SINGLE_AGENT_CLASSIFIER_CACHE_SIZE,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

def _table(agent_id: int) -> str:
    for agent in AGENTS:
        if agent["agent_id"] == agent_id:
            return agent["table_name"]
    raise ValueError(f"agent_id {agent_id} nao encontrado em AGENTS")


_TABLE_QUEIJOS      = _table(1)
_TABLE_FERMENTADOS  = _table(2)
_TABLE_REGULATORIOS = _table(3)
_TABLE_QUALIDADE    = _table(4)
_TABLE_DEFEITOS     = _table(5)
_TABLE_FORMULACAO   = _table(6)

_DOMAIN_TO_TABLE = {
    "queijos":      _TABLE_QUEIJOS,
    "fermentados":  _TABLE_FERMENTADOS,
    "regulatorios": _TABLE_REGULATORIOS,
    "qualidade":    _TABLE_QUALIDADE,
    "defeitos":     _TABLE_DEFEITOS,
    "formulacao":   _TABLE_FORMULACAO,
}

# Sinais de dominio lidos do YAML (mesmos que o orquestrador V1 usa)
_FERMENTED_TERMS: frozenset = _ROUTING_RULES.get("domain_signals_fermented", frozenset())
_QUALITY_TERMS: frozenset   = _ROUTING_RULES.get("domain_signals_quality", frozenset())

_DEFECT_TERMS = frozenset({
    "defeito", "estufamento", "contaminacao", "amargo", "rancoso",
    "mofo", "fungo", "coliforme", "trinca", "olhadura", "casca",
    "clostridium", "diagnostico", "causa raiz", "acao corretiva",
    "troubleshooting", "rancidez", "amargor", "butirico", "heterolactic",
})

_FORMULATION_TERMS = frozenset({
    "formulacao", "ingrediente", "estabilizante", "espessante",
    "shelf-life", "shelf life", "validade", "composicao",
    "doce de leite", "requeijao", "cream cheese",
    "balanco de massa", "desenvolvimento", "ficha tecnica",
    "aromatizante",
})

_RAW_MILK_PROCESS_TERMS = frozenset({
    "leite cru", "leite refrigerado", "armazenado", "armazenamento",
    "estocado", "estocagem", "dois dias", "2 dias", "48 horas",
    "pasteurizacao", "pasteurização", "pasteurizacao convencional",
    "pasteurização convencional", "tratamento termico", "tratamento térmico",
    "termizacao", "termização",
})


# ---------------------------------------------------------------------------
# QueryIntent
# ---------------------------------------------------------------------------

@dataclass
class QueryIntent:
    """Resultado da analise de intencao de uma query."""
    search_tables: List[str]
    domain: str
    needs_regulatory: bool = False
    is_greeting: bool = False
    question_type: str = "general"
    classified_by: str = "keyword"
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fast-path keyword (sinais inequivocos)
# ---------------------------------------------------------------------------

def _has_fermented_signal(norm: str) -> bool:
    if _FERMENTED_TERMS:
        return any(t in norm for t in _FERMENTED_TERMS)
    return bool(re.search(r"\b(iogurte|kefir|fermentado|fermentacao|sinerese|skyr|nslab)\b", norm))


def _has_quality_signal(norm: str) -> bool:
    if _QUALITY_TERMS:
        return any(t in norm for t in _QUALITY_TERMS)
    return bool(re.search(r"\b(ccs|cbt|dornic|crioscopia|alizarol|mastite|celulas somaticas)\b", norm))


def _has_defect_signal(norm: str) -> bool:
    return any(t in norm for t in _DEFECT_TERMS)


def _has_formulation_signal(norm: str) -> bool:
    return any(t in norm for t in _FORMULATION_TERMS)


def _has_raw_milk_process_signal(norm: str) -> bool:
    return (
        any(t in norm for t in _RAW_MILK_PROCESS_TERMS)
        and any(t in norm for t in ("leite cru", "leite refrigerado", "pasteurizacao", "pasteurização"))
        and any(t in norm for t in ("tratamento", "armazen", "estoc", "termiz", "pasteur"))
    )


def _keyword_classify(norm: str) -> Optional[QueryIntent]:
    """Tenta classificar por keyword. Retorna None se ambiguo (vai para LLM)."""

    if _looks_like_greeting_only(norm):
        return QueryIntent(search_tables=[], domain="greeting", is_greeting=True)

    # Processo/qualidade do leite cru: perguntas operacionais sobre tratamento
    # antes de pasteurização pertencem ao especialista técnico. Regulação entra
    # só como complemento, nunca como fonte primária única.
    if _has_raw_milk_process_signal(norm):
        return QueryIntent(
            search_tables=[_TABLE_QUALIDADE, _TABLE_REGULATORIOS],
            domain="raw_milk_process_quality",
            needs_regulatory=True,
            question_type="general",
        )

    if _is_legal_requirement_regulatory_signal(norm):
        return QueryIntent(
            search_tables=[_TABLE_REGULATORIOS],
            domain="regulatory_legal_requirement",
            needs_regulatory=True,
            question_type="regulatory",
        )

    regulatory = _is_strong_regulatory_signal(norm) or _is_normative_regulatory_signal(norm)
    labeling   = _is_labeling_regulatory_signal(norm)
    cheese     = _is_strong_cheese_signal(norm)

    if regulatory or labeling:
        if cheese:
            return QueryIntent(
                search_tables=[_TABLE_QUEIJOS, _TABLE_REGULATORIOS],
                domain="cheese_regulatory",
                needs_regulatory=True,
                question_type="regulatory",
            )
        return QueryIntent(
            search_tables=[_TABLE_REGULATORIOS],
            domain="regulatory",
            needs_regulatory=True,
            question_type="regulatory",
        )

    # Termos inequivocos de fermentados (sem cruzamento de dominio)
    if _has_fermented_signal(norm) and not cheese:
        tables = [_TABLE_FERMENTADOS]
        if _is_strong_regulatory_signal(norm):
            tables.append(_TABLE_REGULATORIOS)
        return QueryIntent(search_tables=tables, domain="fermented")

    # Formulacao pura
    if _has_formulation_signal(norm) and not cheese and not _has_quality_signal(norm):
        return QueryIntent(search_tables=[_TABLE_FORMULACAO], domain="formulation")

    # Ambiguo — deixa para o LLM
    return None


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = """Voce e o classificador de dominio do DairyApp AI.

DOMINIOS (retorne exatamente esses nomes):
- queijos: tecnologia de fabricacao e maturacao de queijos; processo (coagulacao, filagem, prensagem, salga, dessoragem); ingredientes de processo (coalho, GDL, CaCl2, soro-fermento, culturas adjuntas, sal); qualidade do leite COMO FATOR que impacta o queijo (CCS, psicrotroficas, Clostridium, antibioticos, mastite em contexto de fabricacao); defeitos tecnicos de queijo (estufamento, amargor, olhadura, trinca, rancidez); maturacao e bioquimica (proteolise, lipolise, grana, browning, derretimento)
- fermentados: iogurte, kefir, leite fermentado, bebida lactea fermentada, coalhada, skyr; culturas lacticas e probioticos; sinerese, viscosidade, gel de iogurte; pos-acidificacao; EPS
- regulatorios: instrucoes normativas (IN 76, IN 77, IN 30, IN 46, IN 22, IN 65-74), RDCs, RIISPOA, Codex, FDA, EU; rotulagem e denominacao de venda; padroes de identidade e qualidade (RTIQ); alegacoes nutricionais; prazos de adequacao
- qualidade: metodos analiticos do leite (IN 68, crioscopia, Gerber, Dornic, alizarol como METODO); deteccao de fraudes (aguagem, neutralizantes, conservantes); CCS/CBT como MEDICAO (nao como fator de processo); instrumentacao laboratorial
- defeitos: diagnostico e troubleshooting de defeitos quando a pergunta nao menciona queijo especifico; causa raiz generica; acoes corretivas sem produto definido
- formulacao: formulacao de PRODUTOS FINAIS (doce de leite, requeijao cremoso, cream cheese, sobremesa lactea, bebida lactea nao fermentada); estabilizantes, espessantes, emulsificantes em produtos prontos; shelf-life de produto acabado; balanco de massa

RETORNE APENAS JSON VALIDO. Formato: {"dominios": ["queijos"], "confidence": 0.95}
- dominios: lista de 1 ou 2 strings do conjunto acima
- Use 2 dominios apenas quando a pergunta combina claramente aspecto tecnico de produto + norma/regulacao
- confidence: float 0.0-1.0"""

# Few-shot examples: (query_normalizada, dominios_corretos)
# Cobrem os casos de fronteira mais frequentes — usados como historico na chamada LLM
_CLASSIFIER_FEW_SHOTS = [
    # --- QUEIJOS: processo e ingredientes ---
    ("por que a gdl pode ajudar na coagulacao mas nao substitui o soro-fermento",
     '{"dominios": ["queijos"], "confidence": 0.97}'),
    ("qual a funcao do cacl2 na coagulacao do leite para queijo prato",
     '{"dominios": ["queijos"], "confidence": 0.97}'),
    ("como deve ser o soro-fermento tipico dos queijos grana em acidez ph e populacao microbiana",
     '{"dominios": ["queijos"], "confidence": 0.98}'),
    ("por que a massa precisa ter caseina pouco degradada antes da filagem",
     '{"dominios": ["queijos"], "confidence": 0.98}'),
    ("por que a temperatura da salmoura e critica nos grana italianos",
     '{"dominios": ["queijos"], "confidence": 0.97}'),
    ("qual o risco de usar cultura tipica de iogurte como base dominante para parmesao",
     '{"dominios": ["queijos"], "confidence": 0.96}'),
    # --- QUEIJOS: qualidade do leite como fator de processo ---
    ("qual contagem de celulas somaticas e recomendada para leite destinado a parmesao",
     '{"dominios": ["queijos"], "confidence": 0.95}'),
    ("por que alta contagem de celulas somaticas prejudica mais um parmesao do que um queijo de maturacao curta",
     '{"dominios": ["queijos"], "confidence": 0.96}'),
    ("qual nivel de psicrotroficos e incompativel com producao regular de parmesao de alta qualidade",
     '{"dominios": ["queijos"], "confidence": 0.95}'),
    ("por que leite com antibiotico e incompativel com parmesao",
     '{"dominios": ["queijos"], "confidence": 0.96}'),
    ("por que a pasteurizacao nao resolve sozinha o problema de leite refrigerado com muita pseudomonas",
     '{"dominios": ["queijos"], "confidence": 0.94}'),
    # --- QUEIJOS: defeitos tecnicos ---
    ("se o queijo estufa meses depois com crateras trincas odor butirico e sabor rancoso qual contaminacao rastrear",
     '{"dominios": ["queijos"], "confidence": 0.97}'),
    ("por que bacterias propionicas sao problema em parmesao mesmo sendo uteis em queijos com olhaduras",
     '{"dominios": ["queijos"], "confidence": 0.96}'),
    ("o que caracteriza sabor amargo em queijos duros e por que grana bem elaborado tende a controlar esse defeito",
     '{"dominios": ["queijos"], "confidence": 0.97}'),
    ("em que condicao pode ocorrer estufamento tardio por clostridium tyrobutyricum na mussarela",
     '{"dominios": ["queijos"], "confidence": 0.97}'),
    # --- QUEIJOS: maturacao e bioquimica ---
    ("por que fratura granulosa e sabor picante nao podem ser exigidos de queijos duros com cura de 2 ou 3 meses",
     '{"dominios": ["queijos"], "confidence": 0.97}'),
    ("quais fatores de processo e maturacao aumentam o risco de amargor em queijos semiduros",
     '{"dominios": ["queijos"], "confidence": 0.96}'),
    ("como a proteolise secundaria contribui para o desenvolvimento da grana no parmesao",
     '{"dominios": ["queijos"], "confidence": 0.98}'),
    # --- QUEIJOS + REGULATORIOS: aspecto tecnico + norma ---
    ("quais caracteristicas regulatorias e sensoriais sao esperadas para parmesao e outros queijos duros no brasil",
     '{"dominios": ["queijos", "regulatorios"], "confidence": 0.93}'),
    ("quais limites microbiologicos sao usados para coliformes e estafilococos coagulase positiva na mussarela",
     '{"dominios": ["queijos", "regulatorios"], "confidence": 0.92}'),
    ("ate onde a legislacao permite subir a umidade da mussarela e por que isso nao resolve o problema tecnologico",
     '{"dominios": ["queijos", "regulatorios"], "confidence": 0.93}'),
    ("qual padronizacao de leite e indicada para parmesao brasileiro conforme a norma vigente",
     '{"dominios": ["queijos", "regulatorios"], "confidence": 0.91}'),
    # --- REGULATORIOS: norma pura ---
    ("quais instrucoes normativas regulamentam a producao de leite cru refrigerado no brasil",
     '{"dominios": ["regulatorios"], "confidence": 0.98}'),
    ("o que a in 76 estabelece como limite de ccs para leite tipo a",
     '{"dominios": ["regulatorios"], "confidence": 0.97}'),
    ("quais aditivos alimentares sao permitidos em iogurte conforme a legislacao brasileira",
     '{"dominios": ["regulatorios"], "confidence": 0.96}'),
    ("como deve ser feita a rotulagem nutricional de queijo com alegacao light",
     '{"dominios": ["regulatorios"], "confidence": 0.96}'),
    ("o riispoa exige registro no sif para producao de queijo destinado ao comercio interestadual",
     '{"dominios": ["regulatorios"], "confidence": 0.98}'),
    # --- QUALIDADE: metodo analitico ---
    ("como funciona o metodo de medicao de ccs pelo contador eletronico descrito na in 68",
     '{"dominios": ["qualidade"], "confidence": 0.97}'),
    ("qual o procedimento para determinacao de acidez titulavel pelo metodo dornic",
     '{"dominios": ["qualidade"], "confidence": 0.98}'),
    ("como detectar aguagem no leite pelo metodo crioscopia",
     '{"dominios": ["qualidade"], "confidence": 0.97}'),
    ("quais sao os metodos analiticos oficiais para deteccao de neutralizantes no leite in 68",
     '{"dominios": ["qualidade"], "confidence": 0.97}'),
    ("como interpretar resultado do alizarol para estabilidade termica do leite",
     '{"dominios": ["qualidade"], "confidence": 0.96}'),
    # --- FERMENTADOS ---
    ("por que a gdl pode ser usada como acidificante em iogurte de corte",
     '{"dominios": ["fermentados"], "confidence": 0.93}'),
    ("como controlar a sinerese em iogurte grego sem uso de amido",
     '{"dominios": ["fermentados"], "confidence": 0.97}'),
    ("qual a diferenca entre fermentacao termofila e mesofila no kefir",
     '{"dominios": ["fermentados"], "confidence": 0.97}'),
    ("por que a pos-acidificacao compromete a vida util do iogurte batido",
     '{"dominios": ["fermentados"], "confidence": 0.97}'),
    ("como o exopolissacarideo produzido pelo lactobacillus bulgaricus afeta a textura do iogurte",
     '{"dominios": ["fermentados"], "confidence": 0.97}'),
    # --- FORMULACAO: produto final ---
    ("qual estabilizante usar para evitar sinérese em requeijao cremoso UHT",
     '{"dominios": ["formulacao"], "confidence": 0.96}'),
    ("como calcular o balanco de massa para doce de leite pastoso com 70 brix",
     '{"dominios": ["formulacao"], "confidence": 0.97}'),
    ("qual o shelf-life esperado para cream cheese com atmosfera modificada",
     '{"dominios": ["formulacao"], "confidence": 0.96}'),
    ("quais espessantes sao permitidos em sobremesa lactea sabor baunilha",
     '{"dominios": ["formulacao"], "confidence": 0.95}'),
]


@lru_cache(maxsize=SINGLE_AGENT_CLASSIFIER_CACHE_SIZE)
def _llm_classify_cached(query_norm: str) -> Optional[List[str]]:
    """Classifica com few-shot LLM. Cache LRU por query normalizada."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        llm = ChatOpenAI(
            model=SINGLE_AGENT_CLASSIFIER_MODEL,
            temperature=0,
            max_tokens=60,
        )

        messages = [SystemMessage(content=_CLASSIFIER_SYSTEM)]
        for q_example, answer in _CLASSIFIER_FEW_SHOTS:
            messages.append(HumanMessage(content=q_example))
            messages.append(AIMessage(content=answer))
        messages.append(HumanMessage(content=query_norm))

        response = llm.invoke(messages)
        raw = (response.content or "").strip()
        parsed = json.loads(raw)
        dominios = parsed.get("dominios", [])
        if isinstance(dominios, list) and dominios:
            return tuple(dominios)
    except Exception as exc:
        _log.warning("LLM classifier falhou: %s", exc)
    return None


async def _llm_classify_async(query_norm: str) -> Optional[List[str]]:
    """Wrapper async com timeout para o classificador LLM."""
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _llm_classify_cached, query_norm),
            timeout=SINGLE_AGENT_CLASSIFIER_TIMEOUT_SEC,
        )
        return list(result) if result else None
    except asyncio.TimeoutError:
        _log.warning("LLM classifier timeout apos %ss", SINGLE_AGENT_CLASSIFIER_TIMEOUT_SEC)
        return None


def _dominios_to_intent(dominios: List[str], norm: str) -> QueryIntent:
    """Converte lista de dominios LLM para QueryIntent com tabelas."""
    tables = []
    for d in dominios:
        t = _DOMAIN_TO_TABLE.get(d)
        if t and t not in tables:
            tables.append(t)

    if not tables:
        tables = [_TABLE_QUEIJOS, _TABLE_REGULATORIOS]

    domain = "_".join(dominios) if dominios else "unknown"
    needs_regulatory = _TABLE_REGULATORIOS in tables

    question_type = "regulatory" if needs_regulatory else "general"

    return QueryIntent(
        search_tables=tables,
        domain=domain,
        needs_regulatory=needs_regulatory,
        question_type=question_type,
        classified_by="llm",
    )


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def classify_query_intent(query: str) -> QueryIntent:
    """Classificacao sincrona (keyword only). Usar em contextos sync."""
    norm = _normalize_text(query)
    result = _keyword_classify(norm)
    if result:
        return result

    # Fallback keyword sem LLM — nao tem como chamar async aqui
    if _is_strong_cheese_signal(norm) or _contains_dairy_signal(norm):
        return QueryIntent(
            search_tables=[_TABLE_QUEIJOS, _TABLE_REGULATORIOS],
            domain="dairy_generic",
            needs_regulatory=True,
            classified_by="keyword_fallback",
        )

    return QueryIntent(
        search_tables=[_TABLE_QUEIJOS, _TABLE_REGULATORIOS],
        domain="unknown",
        classified_by="keyword_fallback",
    )


async def classify_query_intent_async(query: str) -> QueryIntent:
    """Classificacao async com LLM para casos ambiguos.

    Fast-path keyword para sinais inequivocos (saudacao, normas, fermentados puros).
    LLM apenas quando keyword nao resolve.
    """
    norm = _normalize_text(query)

    # Tenta fast-path primeiro
    result = _keyword_classify(norm)
    if result:
        _log.debug("classify fast-path: %s -> %s", norm[:50], result.domain)
        return result

    # LLM para casos ambiguos
    dominios = await _llm_classify_async(norm)
    if dominios:
        intent = _dominios_to_intent(dominios, norm)
        _log.debug("classify LLM: %s -> %s (tables=%s)", norm[:50], intent.domain, intent.search_tables)
        return intent

    # Fallback final se LLM falhou
    _log.warning("classify fallback para dairy_generic: %s", norm[:60])
    if _is_strong_cheese_signal(norm) or _contains_dairy_signal(norm):
        return QueryIntent(
            search_tables=[_TABLE_QUEIJOS, _TABLE_REGULATORIOS],
            domain="dairy_generic",
            needs_regulatory=True,
            classified_by="keyword_fallback",
        )

    return QueryIntent(
        search_tables=[_TABLE_QUEIJOS, _TABLE_REGULATORIOS],
        domain="unknown",
        classified_by="keyword_fallback",
    )
