"""
agent_config.py — Configuração dos 6 agentes especialistas

Este arquivo define os metadados de cada agente: nome, ID, tabela de embeddings,
descrição para o orquestrador, e parâmetros de busca customizáveis.

No projeto original (ai-agent-sales), essa informação não existia como arquivo
separado — havia apenas um agente principal com intents de CRM. Aqui, como temos
6 agentes independentes, cada um com sua base de conhecimento, precisamos de um
registro centralizado que diga:
  - "Agente 1 busca na tabela embeddings_agente_1_queijos"
  - "Agente 3 usa hybrid_rrf porque legislação tem termos exatos"
  - "O orquestrador sabe que perguntas sobre pH do leite vão para o Agente 4"

Outros módulos importam daqui:
  - base_agent.py → usa table_name e search_config para criar a tool de busca
  - orchestrator.py → usa description para classificar a pergunta
  - webapp.py → usa agent_id para rotear o endpoint para o agente correto
  - prompts.py → usa agent_id para selecionar o system prompt adequado

Formato: lista de dicionários. Cada dicionário é um agente.
"""

import os
from typing import Dict, List, Optional, Any


# ============================================================
# Definição dos 6 agentes especialistas
# ============================================================
# Cada agente tem os seguintes campos:
#
# agent_id (int):
#   Identificador numérico único do agente (1 a 6).
#   Usado nos endpoints (/webhook/agente-{agent_id}) e nos logs.
#
# name (str):
#   Nome curto do agente. Aparece no campo agent_name da response
#   e nos logs de interação.
#
# table_name (str):
#   Nome da tabela de embeddings no Supabase onde os documentos
#   deste agente estão armazenados. Cada agente tem sua tabela
#   própria para evitar contaminação entre domínios.
#   O formato segue o padrão: embeddings_agente_{id}_{dominio}
#
# description (str):
#   Texto que o orquestrador usa para decidir quando rotear
#   uma pergunta para este agente. Deve ser específico o suficiente
#   para distinguir de outros agentes.
#   Exemplo: "queijo mussarela" → Agente 1, não Agente 6.
#
# keywords (list[str]):
#   Palavras-chave que ajudam o orquestrador na classificação.
#   São usadas como dica adicional quando a descrição é ambígua.
#
# search_config (dict):
#   Parâmetros de busca específicos deste agente.
#   Se não definidos, usa os valores padrão de config.py.
#   Permite customizar search_type, k, reranker por agente.
#   Exemplo: Agente 3 (regulatórios) usa hybrid_rrf porque
#   legislação tem termos exatos como "IN 76", "RDC 331".

AGENTS: List[Dict[str, Any]] = [
    {
        "agent_id": 0,
        "name": "Base Geral Dairy",
        "table_name": "embeddings_agente_0_base_geral",
        "description": (
            "Base transversal de conhecimento geral da empresa (glossário, "
            "verdades absolutas, definições canônicas e convenções de linguagem). "
            "Deve apoiar todas as respostas para manter coerência terminológica "
            "e diretrizes institucionais."
        ),
        "keywords": [
            "glossário", "termo", "definição", "conceito", "verdades absolutas",
            "diretriz", "padrão", "institucional", "base geral",
        ],
        "search_config": {
            "search_type": "hybrid_rrf",
            "k": 4,
        },
    },
    {
        "agent_id": 1,
        "name": "Tecnologia de Queijos",
        "table_name": "embeddings_agente_1_queijos",
        "description": (
            "Especialista em tecnologia de fabricação de queijos. "
            "Cobre processos de fabricação (coagulação, corte da massa, filagem, "
            "prensagem, salga, maturação), rendimento, parâmetros de processo "
            "(temperatura, pH, tempo), equipamentos e boas práticas de fabricação. "
            "Tipos: mussarela, minas frescal, prato, coalho, provolone, "
            "parmesão, gorgonzola, brie, camembert."
        ),
        "keywords": [
            "queijo", "mussarela", "coalho", "coagulação", "filagem",
            "maturação", "salga", "prensagem", "rendimento", "massa",
            "fermento", "coalho", "soro", "cura", "defumação",
        ],
        "search_config": {
            # Busca semântica é suficiente para perguntas técnicas
            # sobre processos de fabricação (conceituais, não terminológicas)
        },
    },
    {
        "agent_id": 2,
        "name": "Fermentados",
        "table_name": "embeddings_agente_2_fermentados",
        "description": (
            "Especialista em produtos lácteos fermentados. "
            "Cobre fabricação de iogurte, kefir, leite fermentado, coalhada, "
            "bebida láctea, culturas láticas, probióticos, parâmetros de "
            "fermentação (curvas de pH, temperatura, tempo), bacteriófagos, "
            "reologia e textura de fermentados."
        ),
        "keywords": [
            "iogurte", "kefir", "fermentado", "fermentação", "cultura",
            "probiótico", "lactobacillus", "streptococcus", "acidez",
            "coalhada", "bebida láctea", "bacteriófago", "pH",
        ],
        "search_config": {
            # Fermentados mistura conteúdo técnico + termos específicos de mercado.
            # Híbrido melhora recuperação semântica e por palavra-chave.
            "search_type": "hybrid_rrf",
            "k": 7,
        },
    },
    {
        "agent_id": 3,
        "name": "Regulatórios por País",
        "table_name": "embeddings_agente_3_regulatorios",
        "description": (
            "Especialista em legislação e regulamentação de laticínios. "
            "Cobre normas brasileiras (MAPA, ANVISA), internacionais "
            "(FDA, EU, Codex Alimentarius, Mercosul), instruções normativas "
            "(IN 76, IN 77, IN 30, IN 46, IN 22), RDCs (RDC 331, RDC 259), "
            "RIISPOA (Decreto 9.013/2017), padrões de identidade e qualidade "
            "(INs 65, 66, 71, 72, 73, 74 — características sensoriais, "
            "substâncias estranhas, formas de apresentação), "
            "rotulagem, RTIQ de produtos lácteos. "
            "NÃO cobre métodos analíticos — esse é domínio do Agente 4."
        ),
        "keywords": [
            "legislação", "norma", "IN", "RDC", "RIISPOA", "MAPA",
            "ANVISA", "FDA", "regulamento", "instrução normativa",
            "padrão", "microbiológico", "rotulagem", "Codex",
        ],
        # Regulatórios se beneficia de busca híbrida porque o usuário
        # frequentemente busca por termos exatos: "IN 76", "RDC 331",
        # "Art. 15", que a busca textual (FTS) captura melhor que a
        # busca semântica sozinha.
        "search_config": {
            "search_type": "hybrid_rrf",
            "k": 6,  # Mais chunks porque legislação precisa de contexto amplo
        },
    },
    {
        "agent_id": 4,
        "name": "Qualidade do Leite",
        "table_name": "embeddings_agente_4_qualidade_leite",
        "description": (
            "Especialista em qualidade da matéria-prima leite. "
            "Cobre análises físico-químicas (acidez Dornic, crioscopia, densidade, "
            "gordura, proteína, lactose, ESD, EST), contagem de células somáticas (CCS), "
            "contagem bacteriana total (CBT), detecção de fraudes e adulterações "
            "(aguagem, neutralizantes, conservantes, reconstituintes), "
            "métodos analíticos oficiais IN 68 (qualitativos, quantitativos, "
            "espectrofotometria, absorbância, comprimento de onda, m/m, m/v, "
            "azul de metileno, ácido sórbico, recomendações de segurança laboratorial), "
            "fatores que afetam qualidade (raça, alimentação, estação, manejo), "
            "pagamento por qualidade."
        ),
        "keywords": [
            "leite", "qualidade", "CCS", "CBT", "acidez", "crioscopia",
            "gordura", "proteína", "fraude", "aguagem", "neutralizante",
            "conservante", "análise", "Dornic", "alizarol", "densidade",
            "laboratório", "amostra", "controle de qualidade", "BPL",
            "segurança", "incêndio", "emergência", "evacuação", "EPI",
        ],
        "search_config": {
            # Base 4 costuma ter conteúdo heterogêneo (métodos + boas práticas).
            # Híbrido tende a recuperar melhor termos técnicos e operacionais.
            "search_type": "hybrid_rrf",
            "k": 7,
        },
    },
    {
        "agent_id": 5,
        "name": "Diagnóstico de Defeitos",
        "table_name": "embeddings_agente_5_defeitos",
        "description": (
            "Especialista em diagnóstico de problemas e defeitos em "
            "produtos lácteos. Cobre defeitos sensoriais (sabor, textura, "
            "aparência, casca), estufamento precoce e tardio, contaminação "
            "microbiológica, defeitos de processo, análise de causa raiz, "
            "ações corretivas, troubleshooting de problemas em fábrica."
        ),
        "keywords": [
            "defeito", "problema", "estufamento", "contaminação",
            "sabor", "textura", "aparência", "amargo", "rançoso",
            "mofo", "fungo", "Clostridium", "coliforme", "casca",
            "olhadura", "trinca", "descoloração",
        ],
        # Defeitos também se beneficia de busca híbrida porque termos
        # técnicos específicos (ex: "Clostridium tyrobutyricum",
        # "estufamento tardio") são melhor capturados por FTS.
        "search_config": {
            "search_type": "hybrid_rrf",
        },
    },
    {
        "agent_id": 6,
        "name": "Formulação e Desenvolvimento",
        "table_name": "embeddings_agente_6_formulacao",
        "description": (
            "Especialista em formulação e desenvolvimento de novos "
            "produtos lácteos. Cobre formulações base (iogurte, bebida "
            "láctea, doce de leite, requeijão, cream cheese), fichas "
            "técnicas de ingredientes (estabilizantes, espessantes, "
            "conservantes, aromatizantes), balanço de massa, tabelas "
            "de composição, substituição de ingredientes, shelf-life, "
            "estabilidade e desenvolvimento de embalagens."
        ),
        "keywords": [
            "formulação", "receita", "ingrediente", "estabilizante",
            "espessante", "conservante", "shelf-life", "validade",
            "composição", "balanço", "doce de leite", "requeijão",
            "cream cheese", "bebida láctea", "desenvolvimento",
        ],
        "search_config": {},
    },
]


# ============================================================
# Funções auxiliares para acesso rápido
# ============================================================

# Mapa em memÃ³ria para acesso O(1) por ID.
_AGENT_MAP: Dict[int, Dict[str, Any]] = {agent["agent_id"]: agent for agent in AGENTS}

def get_agent_by_id(agent_id: int) -> Optional[Dict[str, Any]]:
    """Retorna a configuração de um agente pelo ID.
    
    Uso: config = get_agent_by_id(1)  →  { agent_id: 1, name: "Tecnologia de Queijos", ... }
    
    Retorna None se o agent_id não existir (ex: agent_id=7).
    Isso permite que o webapp.py retorne 404 para IDs inválidos.
    """
    return _AGENT_MAP.get(agent_id)


def get_all_agents() -> List[Dict[str, Any]]:
    """Retorna a lista completa de agentes.
    
    Usado pelo orquestrador para montar o prompt de classificação
    (lista todos os agentes disponíveis e suas descrições).
    """
    return AGENTS


def get_agent_descriptions_for_orchestrator() -> str:
    """Gera um texto formatado com a descrição de cada agente.
    
    Esse texto é injetado no system prompt do orquestrador para que
    o LLM saiba quais agentes existem e quando usar cada um.
    
    Formato de saída:
    1. Tecnologia de Queijos → Especialista em fabricação de queijos...
    2. Fermentados → Especialista em produtos lácteos fermentados...
    ...
    
    Usado em: prompts.py → ORCHESTRATOR_SYSTEM_PROMPT
    """
    lines = []
    for agent in AGENTS:
        lines.append(
            f"{agent['agent_id']}. {agent['name']} → {agent['description']}"
        )
    return "\n".join(lines)


def get_search_config(agent_id: int) -> Dict[str, Any]:
    """Retorna os parâmetros de busca para um agente específico.
    
    Se o agente tem search_config customizado, retorna ele.
    Se não, retorna um dicionário vazio (o módulo de busca vai
    usar os defaults de config.py).
    
    Exemplo:
      get_search_config(3)  →  {"search_type": "hybrid_rrf", "k": 6}
      get_search_config(1)  →  {}  (usa defaults)
    
    Usado em: base_agent.py → ao criar a tool de busca do agente.
    """
    agent = get_agent_by_id(agent_id)
    if not agent:
        return {}

    cfg = dict(agent.get("search_config", {}))

    # Overrides via env para testes rápidos sem editar código.
    # Exemplo:
    #   AGENT_4_SEARCH_TYPE=hybrid_rrf
    #   AGENT_4_K=6
    env_prefix = f"AGENT_{agent_id}_"
    env_search_type = os.getenv(f"{env_prefix}SEARCH_TYPE", "").strip()
    env_k = os.getenv(f"{env_prefix}K", "").strip()
    env_use_hyde = os.getenv(f"{env_prefix}USE_HYDE", "").strip().lower()

    if env_search_type:
        cfg["search_type"] = env_search_type
    if env_k.isdigit():
        cfg["k"] = int(env_k)
    if env_use_hyde in ("1", "true", "yes"):
        cfg["use_hyde"] = True
    elif env_use_hyde in ("0", "false", "no"):
        cfg["use_hyde"] = False

    return cfg
