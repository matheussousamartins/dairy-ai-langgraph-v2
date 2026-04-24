"""
agents/orchestrator.py â€" Orquestrador multi-agente com execuÃ§Ã£o paralela

Fluxo do grafo:
  classify â†’ route â†’ execute (paralelo) â†’ consolidate â†’ END
                â†˜ respond_direct â†’ consolidate â†’ END

Agentes 0 (Base Geral) e 3 (Regulatórios) são SEMPRE incluídos
para qualquer pergunta sobre laticÃ­nios â€" o classificador Ã© instruÃ­do
a retorná-los obrigatoriamente.

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
from app.rag.search import search_general_knowledge_base
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
_GREETINGS = {
    "oi", "ola", "olá", "bom dia", "boa tarde", "boa noite",
    "e ai", "e aí", "tudo bem", "blz", "beleza",
}
_DAIRY_TERMS = {
    "leite", "lacteo", "laticinio", "laticinios", "queijo",
    "iogurte", "fermentado", "ricota", "requeijao", "mussarela",
    "coalhada", "soro", "pasteurizacao", "ccs", "cbt", "rtiq", "rdc",
}
_QUALITY_LAB_TERMS = {
    "laboratorio", "analise", "analitico", "amostra", "coleta",
    "controle de qualidade", "qualidade", "bpl", "boas praticas",
    "incendio", "emergencia", "evacuacao", "extintor", "epi", "brigada",
    "reagente", "reagentes", "capela", "residuo", "residuos",
    "sistema fechado", "titulacao", "titulação", "pipeta", "balanca", "balança",
    # Segurança laboratorial IN 68 (fix confusão 4→0)
    # "Evite armazenar INFLAMÁVEIS próximos de chama" e "Não inalar vapores"
    # são seções das Recomendações Gerais da IN 68 (Agente 4)
    "inflamavel", "inflamaveis", "inflamável", "inflamáveis",
    "inalar", "nao inalar", "não inalar",
    # Notação de concentração usada em IN 68 (m/m, m/v)
    "m/m", "m/v", "massa em massa", "massa em volume",
    # Indicadores/soluções de referência da IN 68
    "azul de metileno",
}

_REGULATORY_STRONG_TERMS = {
    "rdc", "riispoa", "sif", "dipoa", "anvisa", "mapa",
    "instrução normativa", "instrucao normativa", "decreto",
    "resolucao", "resolução", "rotulagem", "rotulo", "rótulo",
    "art.", "artigo", "prazo de adequacao", "prazo de adequação",
    "comercio interestadual", "comércio interestadual",
    "informacao nutricional complementar", "informação nutricional complementar",
    "alegacao nutricional", "alegação nutricional", "light", "zero",
    "aditivo alimentar", "coadjuvante de tecnologia",
    "ingredientes obrigatorios", "ingredientes obrigatórios",
    "rotular", "rotulagem", "denominacao de venda", "denominação de venda",
    "norma", "normas", "regulamento", "requisito legal", "requisitos legais",
    # Linguagem operacional RIISPOA/MAPA (fix confusão 3→1)
    "produto suspeito", "reinspeção", "reinspcao", "aproveitamento condicional",
    "condenacao", "condenação", "estabelecimento registrado",
    "registro no sif", "registro no mapa", "sou obrigado a ter",
    # Alegações nutricionais de ausência/redução (RDC 54) (fix confusão 3→1)
    "nao contem gordura", "não contém gordura", "isento de gordura",
    "sem adicao de gordura", "nao contem acucar", "não contém açúcar",
    "valor energetico reduzido", "teor reduzido",
    # Padrões de composição mínima de produto (fix confusão 3→1)
    "teor minimo", "teor máximo", "teor maximo", "composicao minima",
    "minimos para sobremesa", "minimos para bebida", "minimos para iogurte",
    # Linguagem de padrão de identidade — INs 65/66/71/72/73/74 (fix confusão 3→1)
    # "Quais características sensoriais o minas padrão deve ter?" → regulatório
    "padrao de identidade", "padrão de identidade",
    "identidade e qualidade", "regulamento tecnico de identidade",
    "formas de apresentacao", "formas de apresentação",
    "substancias estranhas", "substâncias estranhas",
    "particulas estranhas", "partículas estranhas",
    "caracteristicas sensoriais", "características sensoriais",
    "sabor caracteristico", "sabor característico",
    "aroma caracteristico", "aroma característico",
    "impurezas de qualquer natureza",
}

_QUALITY_STRONG_TERMS = {
    "in 68", "dornic", "crioscopia", "acidez titulavel", "acidez titulável",
    "formaldeido", "formaldeído", "cbt", "ccs", "alizarol", "titulacao",
    "titulação", "hcl", "naoh", "espectrofotometro", "espectrofotômetro",
    "amostra", "triplicata", "pericia", "perícia",
    "transmitancia", "transmitância", "centrifugacao", "centrifugação",
    "acido sialico", "ácido siálico", "kmno4", "kh2po4", "na2hpo4",
    "tampao", "tampão", "espectrofotometrico", "espectrofotométrico",
    "gelatina",
    # Métodos analíticos IN 68 (fix confusão 4→3)
    "metodo a", "metodo b", "metodo c", "metodo d",
    "leite fluido", "leite beneficiado", "beneficiamento do leite",
    "mastite", "celulas somaticas", "células somáticas",
    # Termos espectrofotométricos / IN 68 (fix confusão 4→3)
    # "absorbância" e "comprimento de onda" são de métodos analíticos do leite, não regulatórios
    "absorbancia", "absorbância", "comprimento de onda",
    # Ácido sórbico: conservante regulado por IN 68 em métodos qualitativos/quantitativos
    "acido sorbico", "ácido sórbico",
    # Leite cru: contexto de caracterização físico-química (Qualidade do Leite)
    "leite cru",
}

_GENERAL_KNOWLEDGE_TERMS = {
    "empresa", "fabricante", "distribui", "distribuidor",
    "referencia", "referência", "documento", "conceito", "definicao",
    "definição", "glossario", "glossário",
    "padronizado", "padronizar", "padronizacao", "padronização",
    "termo correto", "significado esperado", "em ingles", "em inglês",
    "starter", "rennet", "endogenous", "phage", "fagos",
    "coagusens", "coagutrack", "quimosina",
    # Instrumentos e índices de coagulação (fix confusão 0→1)
    "coagulometro", "coagulômetro", "indice c/p", "índice c/p",
    "fermento repicado", "enzima coagulante", "forca de gel", "força de gel",
    "indice de coagulação", "índice de coagulação",
}

_FERMENTED_STRONG_TERMS = {
    "iogurte", "kefir", "leite fermentado", "fermentacao", "fermentação",
    "acido latico", "ácido lático", "sinérese", "sinerese",
    "pos-acidificacao", "pós-acidificação", "cultura lactica", "cultura láctica",
    "starter", "nslab",
    # Descritores sensoriais de fermentados (fix confusão 2→1)
    "indulgente", "aveludado", "aveludada",
    "ph de estabilizacao", "ph de estabilização",
    # Proteínas do soro em contexto de gel/fermentados (fix confusão 2→1)
    "proteinas do soro", "proteínas do soro", "soroproteina", "soroprotéina",
    "beta-lactoglobulina", "beta lactoglobulina",
    "skyr",
    # Termos exclusivos de fermentados extraídos dos docs (fix confusão 2→3)
    "exopolissacarideo", "exopolissacarídeo", "eps",
    "incubacao de iogurte", "incubação de iogurte",
    "bebida lactea fermentada", "bebida láctea fermentada",
    "bebida lactea", "bebida láctea",
    "fermentado base vegetal", "fermentado em base vegetal",
    "bifidobacterium", "lactobacillus bulgaricus", "l. bulgaricus",
    "acidifix", "cultura termofila", "cultura termófila",
    "hotfil", "hot fill", "dressing de creme",
    "isomero latico", "isômero lático",
    "viscosidade do gel", "consistencia do gel", "consistência do gel",
    "coalhada de iogurte",
}

_CHEESE_STRONG_TERMS = {
    "queijo", "mussarela", "muçarela", "mozarela", "mucarela",
    "provolone", "ricota", "coalho", "filagem", "prensagem",
    "maturacao", "maturação", "cheddar", "cottage", "minas meia cura",
    "coalhada", "corte da coalhada", "agua de lavagem", "água de lavagem",
    "soro", "coagulação", "coagulacao", "coalho",
    # Proteólise é processo central de maturação de queijo (fix confusão 1→4)
    "proteolise", "proteólise", "proteolisis",
    # Termos exclusivos de tecnologia de queijos extraídos dos docs (fix 1→4/1→3)
    "browning", "maillard no queijo", "derretimento do queijo",
    "rendimento de fabricacao", "rendimento de fabricação",
    "recuperacao de gordura", "recuperação de gordura",
    "plasmina", "lipolise no queijo", "lipólise no queijo",
    "desmineralizacao", "desmineralização",
    "sinerese da coalhada", "sinérese da coalhada",
    "olhadura", "trinca no queijo",
    "cristal de lactato", "cristais de lactato",
    "grana padano", "parmigiano", "parmesao", "parmesão",
    "cultivo adjunto", "cultivos adjuntos",
    "indice c/p", "índice c/p",
    "sal na umidade", "porcentagem de sal",
    "rendimento queijo",
}

# Regras determinísticas de alta precisão para intents críticas.
# Texto já chega normalizado (sem acento e em minúsculas).
_INTENT_PATTERNS_BY_AGENT: Dict[int, tuple[str, ...]] = {
    # Agente 1: Tecnologia de Queijos
    1: (
        r"\bpizza\b",
        r"\bmu?ss?arela\b|\bmozarela\b|\bmucarela\b",
        r"\bderretimento\b",
        r"\bbrowning\b",
        r"\bfilagem\b",
        r"\bprensagem\b",
        r"\bprovolone\b",
        # Defeitos técnicos de queijo (fix confusão 1→5)
        # Esses defeitos estão documentados nos artigos Ha-La Biotec do Agente 1
        r"\bestufamento\b",
        r"\bclc\b",
        r"\bbutiric[ao]\b",
        r"\bamargor\b",
        r"\bolfatura\b|\bolhadura\b|\btrinca\b",
    ),
    # Agente 2: Fermentados
    2: (
        r"\biogurte\b",
        r"\bfermentacao\b",
        r"\bsinerese\b|\bsinerese\b",
        r"\bacido\s+latico\b",
        r"\bkefir\b",
        r"\bcoalhada\b",
    ),
    # Agente 3: Regulatórios
    3: (
        r"\brotulag",
        r"\brdc\b",
        r"\bin\s*\d{1,4}\b",
        r"\briispoa\b",
        r"\bart\.?\s*\d+\b",
        r"\binstrucao\s+normativa\b",
        r"\bmedida\s+provisoria\b",
    ),
    # Agente 4: Qualidade do Leite
    4: (
        r"\bformaldeido\b",
        r"\bdornic\b",
        r"\bccs\b",
        r"\bcbt\b",
        r"\bacidez\s+titulavel\b",
        r"\bcrioscopia\b",
        r"\balizarol\b",
    ),
}

_CLASSIFIER_FEW_SHOTS = """
FEW-SHOTS (padrao esperado):
- Pergunta: "Qual teste qualitativo detecta formaldeído no leite e qual indicação de positivo?"
  agent_ids: [0,3,4]
  confidence: 0.96
  reason: "Teste qualitativo e fraude/adulterante em leite -> Qualidade do Leite."
  alternatives: [5]

- Pergunta: "Quais açúcares são responsáveis pelo escurecimento do queijo na pizza?"
  agent_ids: [0,3,1]
  confidence: 0.94
  reason: "Tema tecnológico de queijo/pizza (browning/derretimento) -> Tecnologia de Queijos."
  alternatives: [5]

- Pergunta: "Quanto da lactose pode ser transformada em ácido lático pelas bactérias do iogurte?"
  agent_ids: [0,3,2]
  confidence: 0.95
  reason: "Fermentação em iogurte e metabolismo de lactose -> Fermentados."
  alternatives: [4]

- Pergunta: "Posso rotular como light se reduzir só 10% de sódio?"
  agent_ids: [0,3]
  confidence: 0.97
  reason: "Critério de rotulagem regulatória -> Regulatórios."
  alternatives: [6]

- Pergunta: "Explique diferença entre ESD e EST no leite e impacto tecnológico."
  agent_ids: [0,3,4]
  confidence: 0.93
  reason: "Composição físico-química do leite e impacto tecnológico -> Qualidade do Leite."
  alternatives: [1]

- Pergunta: "Se eu usar aroma de fumaça, a rotulagem vira defumado ou sabor defumado?"
  agent_ids: [0,3]
  confidence: 0.97
  reason: "Denominação de venda/rotulagem regulatória."
  alternatives: [1]

- Pergunta: "Como calcular acidez titulável em leite fluido pelo método Dornic?"
  agent_ids: [0,3,4]
  confidence: 0.96
  reason: "Método analítico de qualidade do leite (Dornic)."
  alternatives: []

- Pergunta: "Qual é a capital da França?"
  agent_ids: []
  confidence: 0.99
  reason: "Fora do escopo de laticínios."
  alternatives: []

- Pergunta: "No relatório final, posso escrever Starter ou preciso usar o termo padronizado do glossário?"
  agent_ids: [0,3]
  confidence: 0.92
  reason: "Pergunta de padronização terminológica/glossário institucional."
  alternatives: [2]

- Pergunta: "Como rotular o provolone fresco quando for usado aroma de fumaça?"
  agent_ids: [0,3]
  confidence: 0.95
  reason: "Rotulagem e denominação de venda são temas regulatórios."
  alternatives: [1]

- Pergunta: "Qual rotação e tempo da centrifugação no ácido siálico?"
  agent_ids: [0,3,4]
  confidence: 0.95
  reason: "Método analítico laboratorial de qualidade do leite."
  alternatives: []

- Pergunta: "Em qual faixa de pH deve ser feito o corte da coalhada no queijo?"
  agent_ids: [0,3,1]
  confidence: 0.93
  reason: "Parâmetro de processo de fabricação de queijo."
  alternatives: [2]

- Pergunta: "Como rotular o provolone maturado quando for usado aroma de fumaça?"
  agent_ids: [0,3]
  confidence: 0.95
  reason: "Pergunta normativa de rotulagem/denominação, mesmo citando queijo."
  alternatives: [1,6]

- Pergunta: "No relatório final, posso manter Rennet em inglês ou devo padronizar o termo?"
  agent_ids: [0,3]
  confidence: 0.92
  reason: "Padronização terminológica de glossário institucional."
  alternatives: [1,2]

- Pergunta: "Como CCS elevada influencia o risco de amargor no queijo?"
  agent_ids: [0,3,1,4]
  confidence: 0.88
  reason: "CCS é métrica de qualidade do leite (agente 4), mas o impacto no queijo (amargor) é tecnologia de queijos (agente 1). Agente 1 como primário."
  alternatives: [4]

- Pergunta: "Quais fatores de processo aumentam o risco de CLC no queijo?"
  agent_ids: [0,3,1]
  confidence: 0.92
  reason: "CLC é defeito técnico de queijo coberto nas bases Ha-La Biotec. Agente 1 — não Agente 5 (que é diagnóstico visual)."
  alternatives: []

- Pergunta: "Na prevenção de estufamento tardio, quais abordagens o documento compara?"
  agent_ids: [0,3,1]
  confidence: 0.93
  reason: "Estufamento tardio é defeito técnico de queijo (Clostridium) — Agente 1 tem artigos sobre isso."
  alternatives: []

- Pergunta: "Para dizer que um produto não contém gordura total, basta zerar gordura total?"
  agent_ids: [0,3]
  confidence: 0.96
  reason: "Alegação de ausência de nutriente é critério de rotulagem regulatória (RDC 54). Tema regulatório, não de produto."
  alternatives: []

- Pergunta: "Sou obrigado a ter local para produto suspeito, reinspeção e aproveitamento condicional?"
  agent_ids: [0,3]
  confidence: 0.96
  reason: "Requisito de instalações físicas para fiscalização — linguagem operacional RIISPOA."
  alternatives: []

- Pergunta: "Como deve ser a denominação quando há ingredientes adicionais?"
  agent_ids: [0,3]
  confidence: 0.95
  reason: "Denominação de venda é tema de rotulagem regulatória, mesmo sem citar produto específico."
  alternatives: [1]

- Pergunta: "Qual resultado positivo no método B para análise do leite?"
  agent_ids: [0,3,4]
  confidence: 0.94
  reason: "Método B é método analítico da IN 68 — Qualidade do Leite."
  alternatives: []

- Pergunta: "As proteínas do soro conseguem formar gel com a mesma eficiência da caseína?"
  agent_ids: [0,3,2]
  confidence: 0.88
  reason: "Propriedades de geleificação de proteínas do soro são relevantes para fermentados (iogurte). Agente 2."
  alternatives: [1]

- Pergunta: "Qual categoria de iogurte é descrita como indulgente e aveludada?"
  agent_ids: [0,3,2]
  confidence: 0.91
  reason: "Categorias sensoriais de iogurte — Fermentados (Agente 2)."
  alternatives: []

- Pergunta: "Quais características sensoriais o minas padrão deve apresentar segundo a normativa?"
  agent_ids: [0,3]
  confidence: 0.96
  reason: "Padrão de identidade sensorial de produto lácteo é definido em Instrução Normativa — Regulatórios (Agente 3), não Queijos."
  alternatives: [1]

- Pergunta: "Como deve ser rotulado o provolone defumado quando usado aroma artificial?"
  agent_ids: [0,3]
  confidence: 0.96
  reason: "Denominação de venda e rotulagem com aroma artificial é critério regulatório — Agente 3."
  alternatives: [1]

- Pergunta: "O minas frescal pode conter substâncias estranhas de acordo com a IN vigente?"
  agent_ids: [0,3]
  confidence: 0.97
  reason: "Substâncias estranhas no contexto de padrão de identidade são requisito regulatório. Tema de Agente 3, não de Queijos."
  alternatives: []

- Pergunta: "Quais as formas de apresentação permitidas para a ricota segundo a normativa?"
  agent_ids: [0,3]
  confidence: 0.96
  reason: "Formas de apresentação são definidas nos regulamentos técnicos de identidade (INs). Agente 3."
  alternatives: [1]

- Pergunta: "Quais cuidados de segurança devo ter ao trabalhar com inflamáveis no laboratório de leite?"
  agent_ids: [0,3,4]
  confidence: 0.93
  reason: "Segurança com inflamáveis em laboratório está nas Recomendações Gerais da IN 68 — Qualidade do Leite (Agente 4)."
  alternatives: []

- Pergunta: "Em que unidades se expressa a concentração m/m e m/v nos métodos da IN 68?"
  agent_ids: [0,3,4]
  confidence: 0.94
  reason: "Notação m/m e m/v são unidades de concentração nos métodos analíticos da IN 68 — Qualidade do Leite."
  alternatives: []

- Pergunta: "Como preparar a solução de azul de metileno para os testes da IN 68?"
  agent_ids: [0,3,4]
  confidence: 0.94
  reason: "Azul de metileno é indicador usado nos métodos analíticos da IN 68 — Qualidade do Leite."
  alternatives: []

- Pergunta: "Qual o comprimento de onda utilizado na leitura de absorbância para o método de ácido sórbico?"
  agent_ids: [0,3,4]
  confidence: 0.93
  reason: "Absorbância e comprimento de onda são parâmetros de espectrofotometria nos métodos quantitativos da IN 68 — Qualidade do Leite."
  alternatives: []

- Pergunta: "Não inalar vapores é recomendação de qual documento da IN 68?"
  agent_ids: [0,3,4]
  confidence: 0.95
  reason: "Recomendação de segurança 'Não inalar vapores' está nas Recomendações Gerais da IN 68 — Qualidade do Leite."
  alternatives: []
"""

# Palavras muito amplas que não devem disparar especialista sozinhas.
_LOW_PRECISION_KEYWORDS = {
    "leite", "qualidade", "queijo", "acidez", "pH", "ph", "cultura",
    "fermentado", "defeito", "problema", "sabor", "textura", "ingrediente",
    "receita", "validade", "analise", "norma",
}
_HINT_NOISE_TERMS = {
    "edicao", "edição", "dados", "serie", "série", "grafico", "gráfico",
    "imagem", "linha", "eixo", "bibliografia", "nacional", "oficiais",
    "metodos", "métodos", "comum",
}
_HINT_NOISE_TOKENS = {
    "edicao", "edição", "parte", "trimestral", "jan", "fev", "mar",
    "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez",
    "todo", "mundo", "popular",
}

# Hints de alta precisão default para fast-path (fallback local).
_SPECIALIST_STRONG_HINTS_DEFAULT: Dict[int, set] = {
    1: {
        "mussarela", "muçarela", "mozarela", "filagem", "prensagem", "maturacao",
        "maturação", "provolone", "pizza", "browning", "derretimento",
        # Defeitos técnicos de queijo — cobertos pelos artigos Ha-La Biotec (fix confusão 1→5)
        "estufamento", "clostridium", "rancoso", "rançoso", "olhadura", "trinca",
        "amargor", "defeito butirico", "defeito butírico", "clc", "heterolatic",
        "butirico", "butírico",
    },
    2: {
        "iogurte", "kefir", "coalhada", "sinérese", "sinerese",
        "acido latico", "ácido lático", "pos-acidificacao", "pós-acidificação",
        "skyr", "nslab",
    },
    4: {
        "dornic", "ccs", "cbt", "crioscopia", "alizarol", "formaldeido",
        "formaldeído", "acidez titulavel", "acidez titulável",
        "mastite", "celulas somaticas",
    },
    5: {
        # Agent 5 terá dados de diagnóstico visual de defeitos (imagens).
        # Termos técnicos de defeitos movidos para Agent 1 (Ha-La Biotec cobre isso).
        # Deixar vazio por ora — será preenchido quando a KB visual for ingerida.
    },
    6: {
        "formulacao", "formulação", "shelf-life", "shelf life", "estabilizante",
        "espessante", "aromatizante",
    },
}

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
        key = re.sub(r"\s+", " ", p).strip().lower()
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
        "nao ha evidencia suficiente",
        "faltam evidencias",
        "pode ser",
        "talvez",
        "recomenda-se verificar",
        "aconselhavel verificar",
        "consultar fontes adicionais",
        "com o meu conhecimento atual",
        "com o seu conhecimento atual",
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


def _extract_factual_candidate(text: str) -> Optional[str]:
    """Extrai parte factual útil de uma resposta mista (fato + ressalva)."""
    cleaned = _sanitize_math_for_ui(text or "")
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


def _is_normative_regulatory_signal(text_norm: str) -> bool:
    if not text_norm:
        return False
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
        if 1 in ids and ("queijo" in text_norm or "coalhada" in text_norm):
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


def _rule_based_route(user_text: str) -> Optional[List[int]]:
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
        if "queijo" in text or "coalhada" in text:
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
    strong_reg = _is_strong_regulatory_signal(text_norm)
    strong_quality = _is_strong_quality_signal(text_norm)
    ambiguous_12 = _is_ambiguous_cheese_fermented_signal(text_norm)

    if is_dairy_route:
        # Se a classificação já apontou domínio lácteo (baseline e/ou especialista),
        # mantém o par obrigatório 0+3 no plano de execução.
        if 0 not in chosen:
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

        plan = base + specialists[:max_specialists]
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

    if bucket == "medium":
        return False, "medium_conservative"

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
    user_text = _strip_profile_suffix(_get_last_user_text(state.get("messages", [])))
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
    general_used: bool,
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

    if WEB_FALLBACK_REQUIRE_GENERAL_FALLBACK_FIRST and not general_used:
        return False, "web_requires_general_fallback_first"

    if not successful_responses:
        return True, "web_no_specialist_evidence"

    if WEB_FALLBACK_ONLY_ON_WEAK and not _has_weak_or_conflicting_evidence(successful_responses):
        return False, "web_specialist_evidence_sufficient"

    return True, "web_weak_or_conflicting_evidence"


async def _fetch_web_fallback_evidence(
    state: "OrchestratorState",
) -> Tuple[str, List[Dict[str, str]], str]:
    user_text = _strip_profile_suffix(_get_last_user_text(state.get("messages", [])))
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
    rows: List[str] = []
    for item in sources or []:
        title = str(item.get("title", "")).strip() or "Fonte"
        domain = str(item.get("domain", "")).strip()
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        label = f"{title} ({domain})" if domain else title
        rows.append(f"- {label}: {url}")
    if not rows:
        return ""
    return "Fontes externas consultadas:\n" + "\n".join(rows)


# ============================================================
# Estado do orquestrador
# ============================================================

class OrchestratorState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
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

_classifier_model = None
_consolidation_model = None
_direct_model = None


def _get_classifier():
    global _classifier_model
    if _classifier_model is None:
        _classifier_model = ChatOpenAI(model=LLM_MODEL, temperature=CLASSIFIER_TEMPERATURE).with_structured_output(
            ClassificationResult,
            method="function_calling",
        )
    return _classifier_model


def _get_consolidation_model():
    global _consolidation_model
    if _consolidation_model is None:
        _consolidation_model = ChatOpenAI(model=LLM_MODEL, temperature=CONSOLIDATION_TEMPERATURE)
    return _consolidation_model


def _get_direct_model():
    global _direct_model
    if _direct_model is None:
        _direct_model = ChatOpenAI(model=LLM_MODEL, temperature=DIRECT_TEMPERATURE)
    return _direct_model


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

    Agentes 0 e 3 são SEMPRE obrigatórios para qualquer pergunta
    de laticÃ­nios â€" o prompt instrui o LLM explicitamente.
    """
    messages = state.get("messages", [])
    user_text = _get_last_user_text(messages)

    if not user_text:
        return _build_classification_state(route_text="", agent_ids=[], confidence=1.0, reason="mensagem_vazia")

    route_text = _strip_profile_suffix(user_text)
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

REGRA OBRIGATÓRIA:
- Para QUALQUER pergunta relacionada a laticínios (produtos, processos,
  ingredientes, fabricantes, distribuidores, equipamentos, normas, qualidade,
  defeitos, formulação, legislação), SEMPRE inclua os agentes 0 e 3 na lista.
- Agente 0 (Base Geral Dairy): glossário, produtos, fabricantes, ingredientes,
  distribuidores, equipamentos — base de conhecimento transversal.
- Agente 3 (Regulatórios por País): normas, legislação, requisitos legais.

ESPECIALISTAS (adicione apenas se a pergunta for claramente desse domínio):
{_SPECIALISTS_DESC}
FORMATO DA RESPOSTA:
- Saudação / off-topic (sem relação com laticínios) → []
- Pergunta de laticínios sem especialidade clara → [0, 3]
- Pergunta com especialidade clara → [0, 3, X]
- Pergunta com múltiplas especialidades → [0, 3, X, Y] (máx 5 IDs)
- Ordene por relevância: o agente mais relevante primeiro.

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

    classifier = _get_classifier()
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

    if not agent_ids:
        return _build_classification_state(
            route_text=route_text,
            agent_ids=[],
            confidence=confidence,
            reason=reason or "sem_dominio_relevante",
            alternatives=alternatives,
        )

    _cache_set(cache_key, agent_ids)
    return _build_classification_state(
        route_text=route_text,
        agent_ids=agent_ids,
        confidence=confidence,
        reason=reason or "classificacao_llm",
        alternatives=alternatives,
    )

# ============================================================
# Roteamento condicional
# ============================================================

def route(state: OrchestratorState) -> str:
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

    if not user_text:
        return {"agent_responses": []}

    async def call_one(agent_id: int, agent_name: str) -> Dict[str, Any]:
        try:
            graph = get_agent_graph(agent_id)
            result = await asyncio.wait_for(
                graph.ainvoke({"messages": [HumanMessage(content=user_text)]}),
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
# NÃ³ RESPOND_DIRECT â€" saudaÃ§Ãµes e off-topic
# ============================================================

async def respond_direct(state: OrchestratorState) -> OrchestratorState:
    """Resposta direta para saudações e mensagens off-topic (sem RAG)."""
    user_text = _get_last_user_text(state.get("messages", []))

    system = (
        "Voce e o assistente geral do Dairy AI (DairyApp), especializado em tecnologia "
        "de laticinios. Em saudacoes e primeira interacao, apresente-se de forma curta "
        "como Dairy AI e diga em uma frase como pode ajudar. Depois disso, evite repetir "
        "apresentacoes e va direto ao ponto. Quando pertinente, sugira perguntas tecnicas "
        "sobre queijos, fermentados, regulatorios, qualidade do leite, diagnostico de "
        "defeitos ou formulacao. Responda em portugues brasileiro."
    )

    response = await _get_direct_model().ainvoke([
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
    web_should_use, web_trigger = _should_use_web_fallback(state, successful, general_used)
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
                response = await _get_consolidation_model().ainvoke([HumanMessage(content=prompt)])
                final_text = _postprocess_consolidated_answer(user_text, response.content or "")
            except Exception:
                final_text = _postprocess_consolidated_answer(
                    user_text,
                    "A base geral unificada trouxe evidências, mas não foi possível consolidar uma resposta confiável no momento.",
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

    # 1 agente: repassa direto (econômico), exceto se fallback geral foi acionado.
    if len(successful) == 1 and not general_used:
        single_fact = _extract_factual_candidate(str(successful[0]["response"]))
        final_text = _postprocess_consolidated_answer(
            user_text,
            single_fact or successful[0]["response"],
        )
        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
        }

    # 2+ agentes (ou 1 + fallback geral): consolida com LLM.

    # Em perguntas objetivas, quando houver um único especialista com
    # resposta factual direta, devolve essa resposta sem adicionar ressalvas.
    preferred = _prefer_direct_fact_response(user_text, successful) if not general_used else None
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

    responses_text = "".join(
        f"\n--- {r['agent_name']} ---\n{r['response']}\n"
        for r in successful
    )
    if general_used and general_evidence_text:
        responses_text += (
            f"\n--- Base Geral Unificada (fallback) ---\n"
            f"{general_evidence_text}\n"
        )
    if web_used and web_evidence_text:
        responses_text += (
            f"\n--- Web (fallback em domínios confiáveis) ---\n"
            f"{web_evidence_text}\n"
        )

    consolidation_prompt = (
        "Você é o assistente geral do DairyApp AI. Recebeu respostas de múltiplos "
        "especialistas para a pergunta do usuário. Sua tarefa:\n"
        "- Fundir em UMA resposta coerente e completa\n"
        "- Preservar TODOS os dados técnicos (temperaturas, pHs, normas, prazos)\n"
        "- Não perder informação de nenhum especialista\n"
        "- NÃO adicionar fatos novos que não estejam nas respostas dos especialistas\n"
        "- Se houver lacuna de evidência, diga explicitamente que a base atual não trouxe informação suficiente\n"
        "- Evite misturar produtos/rotinas diferentes sem indicar a diferença\n"
        "- Não mencionar que consultou múltiplos agentes internos\n"
        "- Se houver evidência web, trate como fonte externa e preserve rastreabilidade\n"
        "- Tom técnico e profissional em português brasileiro\n\n"
        f"PERGUNTA: {user_text}\n\n"
        f"RESPOSTAS DOS ESPECIALISTAS:{responses_text}\n"
        "Resposta unificada:"
    )

    try:
        response = await _get_consolidation_model().ainvoke(
            [HumanMessage(content=consolidation_prompt)]
        )
        final_text = _postprocess_consolidated_answer(user_text, response.content or "")
    except Exception:
        final_text = _postprocess_consolidated_answer(
            user_text,
            "\n\n".join(r["response"] for r in successful),
        )

    payload: OrchestratorState = {
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
    }
    if general_used or web_used:
        sources_block = _render_web_sources_block(web_sources) if web_used else ""
        if sources_block:
            final_text = (final_text.strip() + "\n\n" + sources_block).strip()
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
    graph.add_node("execute", execute)
    graph.add_node("fallback_reclassify", fallback_reclassify)
    graph.add_node("respond_direct", respond_direct)
    graph.add_node("consolidate", consolidate)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route,
        {"execute": "execute", "respond_direct": "respond_direct"},
    )

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

