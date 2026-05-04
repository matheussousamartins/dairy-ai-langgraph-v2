"""
config.py â€” ConfiguraÃ§Ãµes centralizadas do DairyApp AI

Este arquivo Ã© importado por TODOS os outros mÃ³dulos do projeto.
Ele lÃª variÃ¡veis de ambiente do arquivo .env e expÃµe como constantes Python.

Por que centralizar?
- Trocar de modelo (gpt-4o-mini â†’ gpt-4o) = mudar 1 linha no .env
- Trocar de banco (Supabase â†’ Hetzner) = mudar 1 URL no .env
- Ativar reranking ou HyDE = mudar 1 flag no .env
- Nenhum arquivo de cÃ³digo precisa ser editado para mudar comportamento

Hierarquia de prioridade:
1. VariÃ¡vel de ambiente do sistema (export VAR=valor)
2. VariÃ¡vel no arquivo .env
3. Valor padrÃ£o definido aqui (fallback para desenvolvimento)
"""

import os
from dotenv import load_dotenv

# ============================================================
# Carrega o arquivo .env para o ambiente do processo
# ============================================================
# load_dotenv() procura um arquivo chamado .env no diretÃ³rio atual
# e carrega cada linha KEY=VALUE como variÃ¡vel de ambiente.
# Se a variÃ¡vel jÃ¡ existir no sistema, ela NÃƒO Ã© sobrescrita
# (variÃ¡veis do sistema tÃªm prioridade).
load_dotenv()


def _parse_optional_non_negative_int_env(var_name: str, default: str = "") -> int | None:
    """Lê um inteiro não negativo de env, aceitando None explícito.

    Valores aceitos para None: "", "none", "null", "false", "off", "disabled".
    """
    raw = os.getenv(var_name, default)
    normalized = (raw or "").strip().lower()
    if normalized in {"", "none", "null", "false", "off", "disabled"}:
        return None
    try:
        value = int(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Variável {var_name} inválida: '{raw}'. Use inteiro >= 0 ou 'none'."
        ) from exc
    if value < 0:
        raise ValueError(
            f"Variável {var_name} inválida: {value}. Use inteiro >= 0 ou 'none'."
        )
    return value


def _parse_csv_env(var_name: str, default: str = "") -> list[str]:
    raw = os.getenv(var_name, default)
    items = [item.strip() for item in (raw or "").split(",") if item.strip()]
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


# ============================================================
# OPENAI â€” Chaves e modelos
# ============================================================

# Chave da API da OpenAI. NecessÃ¡ria para:
# - Gerar embeddings (text-embedding-3-small)
# - Chamar o LLM para chat (gpt-4o-mini ou outro)
# - HyDE (query expansion, se ativado)
# Sem essa chave, nada funciona.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Modelo usado para CHAT (respostas dos agentes e orquestrador).
# gpt-4o-mini Ã© o melhor custo-benefÃ­cio para produÃ§Ã£o.
# Para testes com qualidade mÃ¡xima, use "gpt-4o".
# Para economia mÃ¡xima durante desenvolvimento, use "gpt-3.5-turbo".
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
ALLOWED_CHAT_MODELS = _parse_csv_env("ALLOWED_CHAT_MODELS", LLM_MODEL)

# Temperaturas por papel â€” controla criatividade vs. determinismo.
# 0 = determinÃ­stico (ideal para classificaÃ§Ã£o), 1 = mais criativo.
AGENT_TEMPERATURE         = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
CONSOLIDATION_TEMPERATURE = float(os.getenv("CONSOLIDATION_TEMPERATURE", "0.3"))
CONSOLIDATION_TIMEOUT_SEC = float(os.getenv("CONSOLIDATION_TIMEOUT_SEC", "10"))
DIRECT_TEMPERATURE        = float(os.getenv("DIRECT_TEMPERATURE", "0.5"))
CLASSIFIER_TEMPERATURE    = float(os.getenv("CLASSIFIER_TEMPERATURE", "0"))
AGENT_PROMPT_MODE         = os.getenv("AGENT_PROMPT_MODE", "compact").strip().lower()
ORCHESTRATOR_FASTPATH     = os.getenv("ORCHESTRATOR_FASTPATH", "true").strip().lower() == "true"
CLASSIFICATION_CACHE_SIZE = int(os.getenv("CLASSIFICATION_CACHE_SIZE", "256"))
ORCHESTRATOR_CONTEXT_MEMORY_ENABLED = (
    os.getenv("ORCHESTRATOR_CONTEXT_MEMORY_ENABLED", "true").strip().lower() == "true"
)
ORCHESTRATOR_CONTEXT_MAX_MESSAGES = int(
    os.getenv("ORCHESTRATOR_CONTEXT_MAX_MESSAGES", "6")
)
ORCHESTRATOR_CONTEXT_MAX_CHARS = int(
    os.getenv("ORCHESTRATOR_CONTEXT_MAX_CHARS", "1200")
)
ORCHESTRATOR_CONTEXT_TRIGGER_MAX_CHARS = int(
    os.getenv("ORCHESTRATOR_CONTEXT_TRIGGER_MAX_CHARS", "220")
)

# Modelo usado para gerar EMBEDDINGS (vetores dos documentos e queries).
# text-embedding-3-small gera vetores de 1536 dimensÃµes.
# Ã‰ o modelo recomendado pela OpenAI para RAG â€” barato e eficiente.
# IMPORTANTE: se trocar o modelo, os vetores existentes ficam incompatÃ­veis.
# Seria necessÃ¡rio re-ingerir todos os documentos.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# DimensÃ£o dos vetores gerados pelo modelo de embeddings.
# text-embedding-3-small = 1536 dimensÃµes.
# Esse valor Ã© usado nas tabelas SQL (vector(1536)) e nas funÃ§Ãµes de busca.
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))


# ============================================================
# BANCO DE DADOS â€” Duas conexÃµes separadas
# ============================================================

# Connection string do SUPABASE (vector store).
# Usado para: tabelas de embeddings (6 tabelas, uma por agente),
# funÃ§Ãµes de busca (kb_vector_search, kb_hybrid_search).
# Formato: postgresql://user:password@host:port/database
# Exemplo: postgresql://postgres.abc123:senha@aws-0-sa-east-1.pooler.supabase.com:5432/postgres
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")

# Connection string do POSTGRES HETZNER (dados operacionais).
# Usado para: chat_memories (histÃ³rico de conversa),
# interaction_logs (analytics), users, user_sessions,
# ingested_documents (rastreabilidade).
# Formato: postgresql://user:password@host:port/database
# Exemplo: postgresql://postgres:senha@5.161.236.220:5432/postgres
HETZNER_DB_URL = os.getenv("HETZNER_DB_URL", "")

# Pool de conexões (ajustável por ambiente).
# Defaults conservadores para evitar saturar Session Pooler do Supabase.
SUPABASE_DB_POOL_MIN_SIZE = int(os.getenv("SUPABASE_DB_POOL_MIN_SIZE", "1"))
SUPABASE_DB_POOL_MAX_SIZE = int(os.getenv("SUPABASE_DB_POOL_MAX_SIZE", "4"))
SUPABASE_DB_POOL_TIMEOUT_SEC = float(os.getenv("SUPABASE_DB_POOL_TIMEOUT_SEC", "12"))
SUPABASE_DB_POOL_RECONNECT_TIMEOUT_SEC = float(
    os.getenv("SUPABASE_DB_POOL_RECONNECT_TIMEOUT_SEC", "30")
)
SUPABASE_DB_CONNECT_TIMEOUT_SEC = int(os.getenv("SUPABASE_DB_CONNECT_TIMEOUT_SEC", "8"))

HETZNER_DB_POOL_MIN_SIZE = int(os.getenv("HETZNER_DB_POOL_MIN_SIZE", "1"))
HETZNER_DB_POOL_MAX_SIZE = int(os.getenv("HETZNER_DB_POOL_MAX_SIZE", "4"))
HETZNER_DB_POOL_TIMEOUT_SEC = float(os.getenv("HETZNER_DB_POOL_TIMEOUT_SEC", "12"))
HETZNER_DB_POOL_RECONNECT_TIMEOUT_SEC = float(
    os.getenv("HETZNER_DB_POOL_RECONNECT_TIMEOUT_SEC", "30")
)
HETZNER_DB_CONNECT_TIMEOUT_SEC = int(os.getenv("HETZNER_DB_CONNECT_TIMEOUT_SEC", "8"))

# Prepared statements (psycopg3):
# Em ambientes com pooler (ex.: Supabase/PgBouncer), prepared statements podem
# quebrar de forma intermitente com erro "... does not exist" quando a sessão
# do backend muda entre queries. Por segurança operacional, default = disabled.
SUPABASE_DB_PREPARE_THRESHOLD = _parse_optional_non_negative_int_env(
    "SUPABASE_DB_PREPARE_THRESHOLD",
    "none",
)
HETZNER_DB_PREPARE_THRESHOLD = _parse_optional_non_negative_int_env(
    "HETZNER_DB_PREPARE_THRESHOLD",
    "none",
)


# ============================================================
# RAG â€” ConfiguraÃ§Ãµes de busca e retrieval
# ============================================================

# Tipo de busca padrÃ£o. Controla como os chunks sÃ£o recuperados.
# OpÃ§Ãµes disponÃ­veis:
#   "vector"       â†’ busca apenas por similaridade de cosseno (embedding)
#   "text"         â†’ busca apenas por full-text search (FTS, palavras-chave)
#   "hybrid_rrf"   â†’ combina vector + text com Reciprocal Rank Fusion
#   "hybrid_union" â†’ combina vector + text com uniÃ£o simples (para comparaÃ§Ã£o)
#
# RecomendaÃ§Ã£o por fase:
#   Fase 1 (inÃ­cio):     "vector"     â€” simples, funciona bem para 80% dos casos
#   Fase 2 (refinamento): "hybrid_rrf" â€” melhora busca por termos exatos (legislaÃ§Ã£o)
DEFAULT_SEARCH_TYPE = os.getenv("DEFAULT_SEARCH_TYPE", "vector")

# Quantidade de chunks retornados pela busca.
# 5 Ã© um bom equilÃ­brio: contexto suficiente sem sobrecarregar o LLM.
# Para respostas mais detalhadas, aumente para 8-10.
# Para respostas mais concisas, reduza para 3.
DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))

# Limiar mÃ­nimo de similaridade para incluir um chunk nos resultados.
# Score = 1 - distÃ¢ncia_cosseno. Varia de 0 (nada similar) a 1 (idÃªntico).
# 0.3 Ã© conservador â€” inclui chunks "razoavelmente" relevantes.
# 0.5 Ã© mais restritivo â€” sÃ³ chunks bem relevantes.
# None = sem limiar (retorna os top K independente do score).
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.3")) if os.getenv("MATCH_THRESHOLD") else None


# ============================================================
# RERANKING â€” ReordenaÃ§Ã£o dos resultados (opcional)
# ============================================================

# Reranker a usar apÃ³s a busca. O reranker recebe os top N candidatos
# e reordena por relevÃ¢ncia real em relaÃ§Ã£o Ã  pergunta.
# OpÃ§Ãµes:
#   "none"   â†’ sem reranking (usa a ordem da busca diretamente)
#   "cohere" â†’ usa o modelo rerank-english-v3.0 da Cohere
#
# RecomendaÃ§Ã£o: comece com "none". Ative "cohere" nos agentes 3 e 5
# (regulatÃ³rios e defeitos) quando tiver dados reais para comparar.
RERANKER = os.getenv("RERANKER", "none")

# Chave da API do Cohere. NecessÃ¡ria apenas se RERANKER="cohere".
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# Quantidade de candidatos a enviar para o reranker.
# O reranker recebe RERANK_CANDIDATES chunks e retorna os top DEFAULT_K.
# Quanto maior, mais chances de encontrar o chunk certo, mas mais lento e caro.
# 20-30 Ã© um bom equilÃ­brio.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "24"))


# ============================================================
# HyDE â€” Hypothetical Document Embeddings (opcional)
# ============================================================

# HyDE gera um "documento hipotÃ©tico" a partir da pergunta do usuÃ¡rio
# e usa esse documento para buscar chunks similares.
# Em vez de buscar por "como fabricar mussarela?" (uma pergunta),
# ele gera algo como "A fabricaÃ§Ã£o de mussarela envolve coagulaÃ§Ã£o,
# filagem..." (um parÃ¡grafo tÃ©cnico) e busca por esse texto.
# Isso melhora o recall para perguntas vagas ou mal formuladas.
#
# Trade-off: adiciona 1 chamada extra ao LLM por busca (~0.5s + custo).
# RecomendaÃ§Ã£o: desativado no inÃ­cio. Ativar na Fase 4 se o retrieval
# estiver fraco em perguntas abertas.
USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"

# Modelo usado para gerar o documento hipotÃ©tico (HyDE).
# gpt-4o-mini Ã© suficiente â€” o documento nÃ£o precisa ser perfeito,
# sÃ³ precisa estar no "espaÃ§o semÃ¢ntico" certo.
HYDE_LLM_MODEL = os.getenv("HYDE_LLM_MODEL", "gpt-4o-mini")


# ============================================================
# QUERY REWRITING â€” Expansao de consulta (opcional)
# ============================================================
# Gera variacoes tecnicas da pergunta antes da busca no RAG.
# Trade-off: melhora recall para perguntas mal formuladas, mas adiciona
# 1 chamada LLM por busca e multiplica o numero de buscas no banco.
USE_QUERY_REWRITE = os.getenv("USE_QUERY_REWRITE", "false").strip().lower() == "true"
QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", "gpt-4o-mini")
# Quantidade de variacoes alem da query original.
QUERY_REWRITE_VARIANTS = int(os.getenv("QUERY_REWRITE_VARIANTS", "2"))

# Filtros de qualidade no retrieval (evita chunk lixo tipo "." ou tabela quebrada)
RAG_MIN_CHUNK_CHARS = int(os.getenv("RAG_MIN_CHUNK_CHARS", "60"))
RAG_MIN_ALNUM_RATIO = float(os.getenv("RAG_MIN_ALNUM_RATIO", "0.25"))

# Segunda passada de retrieval (fallback de cobertura).
# Quando ativada, e a primeira busca vier fraca, roda uma segunda busca
# mais ampla para aumentar recall antes de devolver os chunks finais.
RAG_SECOND_PASS_ENABLED = os.getenv("RAG_SECOND_PASS_ENABLED", "false").strip().lower() == "true"
RAG_SECOND_PASS_MIN_RESULTS = int(os.getenv("RAG_SECOND_PASS_MIN_RESULTS", "3"))
RAG_SECOND_PASS_MIN_KEYWORD_HITS = int(os.getenv("RAG_SECOND_PASS_MIN_KEYWORD_HITS", "1"))
RAG_SECOND_PASS_EXPAND_FACTOR = float(os.getenv("RAG_SECOND_PASS_EXPAND_FACTOR", "2.5"))
RAG_SECOND_PASS_MAX_K = int(os.getenv("RAG_SECOND_PASS_MAX_K", "20"))
RAG_SECOND_PASS_FORCE_HYBRID = os.getenv("RAG_SECOND_PASS_FORCE_HYBRID", "true").strip().lower() == "true"
RAG_SECOND_PASS_DISABLE_THRESHOLD = os.getenv("RAG_SECOND_PASS_DISABLE_THRESHOLD", "true").strip().lower() == "true"
RAG_SECOND_PASS_USE_QUERY_REWRITE = os.getenv("RAG_SECOND_PASS_USE_QUERY_REWRITE", "true").strip().lower() == "true"

# Early-skip conservador para buscas sabidamente fracas.
# Objetivo: cortar apenas retries caros e pouco promissores, sem mexer
# no recall dos especialistas. Deve ser habilitado por rollout controlado.
RAG_EARLY_SKIP_WEAK_SEARCH_ENABLED = os.getenv(
    "RAG_EARLY_SKIP_WEAK_SEARCH_ENABLED", "false"
).strip().lower() == "true"
RAG_EARLY_SKIP_WEAK_AGENT_IDS = {
    int(part.strip())
    for part in os.getenv("RAG_EARLY_SKIP_WEAK_AGENT_IDS", "0").split(",")
    if part.strip().isdigit()
}
RAG_EARLY_SKIP_WEAK_MIN_QUERY_KEYWORDS = int(
    os.getenv("RAG_EARLY_SKIP_WEAK_MIN_QUERY_KEYWORDS", "3")
)
RAG_EARLY_SKIP_WEAK_MIN_TOP_KEYWORD_HITS = int(
    os.getenv("RAG_EARLY_SKIP_WEAK_MIN_TOP_KEYWORD_HITS", "2")
)
RAG_EARLY_SKIP_WEAK_HYBRID_MAX_SCORE = float(
    os.getenv("RAG_EARLY_SKIP_WEAK_HYBRID_MAX_SCORE", "0.059")
)

# Fallback final do orquestrador: busca em base geral unificada (multi-tabela).
ENABLE_GENERAL_INDEX_FALLBACK = os.getenv("ENABLE_GENERAL_INDEX_FALLBACK", "true").strip().lower() == "true"
GENERAL_INDEX_FALLBACK_SEARCH_TYPE = os.getenv("GENERAL_INDEX_FALLBACK_SEARCH_TYPE", "hybrid_rrf").strip().lower()
GENERAL_INDEX_FALLBACK_PER_TABLE_K = int(os.getenv("GENERAL_INDEX_FALLBACK_PER_TABLE_K", "3"))
GENERAL_INDEX_FALLBACK_FINAL_K = int(os.getenv("GENERAL_INDEX_FALLBACK_FINAL_K", "6"))
GENERAL_INDEX_FALLBACK_MIN_RESULTS = int(os.getenv("GENERAL_INDEX_FALLBACK_MIN_RESULTS", "2"))
GENERAL_INDEX_FALLBACK_MAX_TABLES = int(os.getenv("GENERAL_INDEX_FALLBACK_MAX_TABLES", "7"))
GENERAL_INDEX_FALLBACK_ONLY_ON_WEAK = os.getenv("GENERAL_INDEX_FALLBACK_ONLY_ON_WEAK", "true").strip().lower() == "true"
GENERAL_INDEX_FALLBACK_REQUIRE_DAIRY_SIGNAL = os.getenv("GENERAL_INDEX_FALLBACK_REQUIRE_DAIRY_SIGNAL", "true").strip().lower() == "true"

# Fallback final na internet (última camada, com whitelist).
ENABLE_WEB_FALLBACK = os.getenv("ENABLE_WEB_FALLBACK", "true").strip().lower() == "true"
WEB_FALLBACK_PROVIDER = os.getenv("WEB_FALLBACK_PROVIDER", "duckduckgo").strip().lower()
WEB_FALLBACK_TIMEOUT_SEC = float(os.getenv("WEB_FALLBACK_TIMEOUT_SEC", "8"))
WEB_FALLBACK_MAX_RESULTS = int(os.getenv("WEB_FALLBACK_MAX_RESULTS", "6"))
WEB_FALLBACK_MAX_SOURCES = int(os.getenv("WEB_FALLBACK_MAX_SOURCES", "3"))
WEB_FALLBACK_ONLY_ON_WEAK = os.getenv("WEB_FALLBACK_ONLY_ON_WEAK", "true").strip().lower() == "true"
WEB_FALLBACK_REQUIRE_DAIRY_SIGNAL = os.getenv("WEB_FALLBACK_REQUIRE_DAIRY_SIGNAL", "true").strip().lower() == "true"
WEB_FALLBACK_REQUIRE_GENERAL_FALLBACK_FIRST = os.getenv(
    "WEB_FALLBACK_REQUIRE_GENERAL_FALLBACK_FIRST", "true"
).strip().lower() == "true"
WEB_FALLBACK_FETCH_FULLTEXT = os.getenv("WEB_FALLBACK_FETCH_FULLTEXT", "false").strip().lower() == "true"
WEB_FALLBACK_MAX_PAGE_CHARS = int(os.getenv("WEB_FALLBACK_MAX_PAGE_CHARS", "2800"))
WEB_FALLBACK_MAX_SNIPPET_CHARS = int(os.getenv("WEB_FALLBACK_MAX_SNIPPET_CHARS", "420"))
WEB_FALLBACK_ALLOWED_DOMAINS = [
    item.strip().lower()
    for item in os.getenv(
        "WEB_FALLBACK_ALLOWED_DOMAINS",
        (
            "gov.br,anvisa.gov.br,agricultura.gov.br,in.gov.br,planalto.gov.br,"
            "fao.org,who.int,codexalimentarius.fao.org,mercosur.int,iso.org,"
            "novonesis.com,chr-hansen.com,dsm.com,dsm-firmenich.com,"
            "ncbi.nlm.nih.gov,pmc.ncbi.nlm.nih.gov,sciencedirect.com"
        ),
    ).split(",")
    if item.strip()
]


# ============================================================
# CHAT MEMORY â€” ConfiguraÃ§Ãµes de memÃ³ria de conversa
# ============================================================

# Quantas mensagens anteriores carregar do histÃ³rico por sessÃ£o.
# 10 = Ãºltimas 5 perguntas + 5 respostas.
# Mais histÃ³rico = mais contexto para o LLM, mas mais tokens consumidos.
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "10"))


# ============================================================
# SERVIDOR â€” ConfiguraÃ§Ãµes do FastAPI
# ============================================================

# Host e porta onde o servidor escuta.
# "0.0.0.0" = aceita conexÃµes de qualquer IP (necessÃ¡rio em produÃ§Ã£o/Docker).
# "127.0.0.1" = aceita apenas conexÃµes locais (desenvolvimento).
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
# CORS em producao: lista separada por virgula
CORS_ALLOW_ORIGINS = [
    item.strip()
    for item in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000").split(",")
    if item.strip()
]

# Seguranca dos endpoints /webhook
ENFORCE_WEBHOOK_API_KEY = os.getenv("ENFORCE_WEBHOOK_API_KEY", "true").strip().lower() == "true"
WEBHOOK_API_KEY_HEADER = os.getenv("WEBHOOK_API_KEY_HEADER", "X-API-Key").strip() or "X-API-Key"
WEBHOOK_API_KEYS = [
    item.strip()
    for item in os.getenv("WEBHOOK_API_KEYS", "").split(",")
    if item.strip()
]


# ============================================================
# INGESTÃƒO â€” ConfiguraÃ§Ãµes do pipeline de documentos
# ============================================================

# EstratÃ©gia de chunking padrÃ£o para ingestÃ£o.
# OpÃ§Ãµes: "fixed", "markdown", "semantic"
#   "fixed"    â†’ corta por tamanho fixo (chunk_size) com overlap
#   "markdown" â†’ corta por cabeÃ§alhos Markdown (##, ###)
#   "semantic" â†’ corta por mudanÃ§a de significado (usa embeddings)
#
# "markdown" Ã© o melhor para documentos tÃ©cnicos e legislaÃ§Ã£o
# (preserva a estrutura de seÃ§Ãµes e artigos).
DEFAULT_CHUNK_STRATEGY = os.getenv("DEFAULT_CHUNK_STRATEGY", "markdown")

# Tamanhos de chunk por tipo de documento.
# Esses valores sÃ£o usados quando a estratÃ©gia Ã© "fixed" ou como
# fallback quando "markdown" gera seÃ§Ãµes muito grandes.
# Formato: { tipo_documento: (chunk_size, chunk_overlap) }
CHUNK_SIZES = {
    "legislacao":    (600, 100),   # Artigos curtos, precisÃ£o alta
    "manual":        (1200, 250),  # SeÃ§Ãµes tÃ©cnicas longas, contexto amplo
    "artigo":        (1000, 200),  # Papers acadÃªmicos, parÃ¡grafos mÃ©dios
    "faq":           (500, 50),    # Perguntas e respostas curtas
    "glossario":     (220, 20),    # Entradas curtas: granularidade por termo
    "formulacao":    (800, 150),   # Receitas e fÃ³rmulas
    "ficha_tecnica": (600, 100),   # Fichas de ingredientes
}

# Tamanho padrÃ£o se o tipo do documento nÃ£o estiver no dicionÃ¡rio acima.
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))


# ============================================================
# QUALITY GATE DE INGESTAO
# ============================================================
# Evita ingerir arquivos extraidos com baixa qualidade (OCR ruim,
# texto muito curto, encoding quebrado etc.).
INGEST_BLOCK_LOW_QUALITY = os.getenv("INGEST_BLOCK_LOW_QUALITY", "true").strip().lower() == "true"
INGEST_MIN_TEXT_CHARS = int(os.getenv("INGEST_MIN_TEXT_CHARS", "400"))
INGEST_MIN_WORDS = int(os.getenv("INGEST_MIN_WORDS", "80"))
INGEST_MAX_GARBLED_RATIO = float(os.getenv("INGEST_MAX_GARBLED_RATIO", "0.08"))
INGEST_MIN_QUALITY_SCORE = float(os.getenv("INGEST_MIN_QUALITY_SCORE", "60"))
# Overrides para documentos de glossario (curtos/tabulares)
INGEST_MIN_TEXT_CHARS_GLOSSARIO = int(
    os.getenv("INGEST_MIN_TEXT_CHARS_GLOSSARIO", "250")
)
INGEST_MIN_WORDS_GLOSSARIO = int(os.getenv("INGEST_MIN_WORDS_GLOSSARIO", "30"))
# Mínimo de caracteres alfanuméricos por chunk (filtra chunks degenerados pós-split)
INGEST_MIN_CHUNK_ALNUM = int(os.getenv("INGEST_MIN_CHUNK_ALNUM", "15"))


# ============================================================
# VALIDAÃ‡ÃƒO â€” Verifica configuraÃ§Ãµes crÃ­ticas no startup
# ============================================================

def validate_config():
    """Verifica se as configuraÃ§Ãµes mÃ­nimas estÃ£o presentes.
    
    Chamada uma vez no startup do servidor (webapp.py).
    Levanta exceÃ§Ã£o se faltar algo crÃ­tico, em vez de falhar
    silenciosamente na primeira request.
    """
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY nÃ£o configurada")
    
    if not SUPABASE_DB_URL:
        errors.append("SUPABASE_DB_URL nÃ£o configurada")
    
    if not HETZNER_DB_URL:
        errors.append("HETZNER_DB_URL não configurada")

    if RERANKER == "cohere" and not COHERE_API_KEY:
        errors.append("RERANKER=cohere mas COHERE_API_KEY não configurada")

    if ENFORCE_WEBHOOK_API_KEY and not WEBHOOK_API_KEYS:
        errors.append("ENFORCE_WEBHOOK_API_KEY=true mas WEBHOOK_API_KEYS nao configurada")

    if errors:
        msg = "ConfiguraÃ§Ã£o incompleta:\n" + "\n".join(f"  - {e}" for e in errors)
        raise EnvironmentError(msg)
    
    return True

