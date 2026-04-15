-- ============================================================
-- 03_app_tables.sql — Tabelas operacionais no Postgres Hetzner
--
-- Executar no Hetzner:
--   psql $HETZNER_DB_URL < sql/03_app_tables.sql
--
-- Estas tabelas são as mesmas que o N8N já usa (criadas pelo
-- init-database.sql que geramos anteriormente). Se já existirem,
-- os CREATE TABLE IF NOT EXISTS não fazem nada.
--
-- Separamos dos scripts do Supabase porque rodam em bancos
-- diferentes. Quando migrar para banco único, rode tudo junto.
-- ============================================================


-- ============================================================
-- 1. CHAT MEMORY
-- ============================================================
-- Armazena o histórico de conversa por sessão.
-- Usado tanto pelo N8N (nó Postgres Chat Memory) quanto pelo
-- LangGraph (db/memory.py → save_memory/load_memory).
--
-- Formato compatível: ambos os backends escrevem e leem
-- da mesma tabela, com o mesmo formato. Conversas iniciadas
-- no N8N aparecem no LangGraph e vice-versa.

CREATE TABLE IF NOT EXISTS chat_memories (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    role VARCHAR(10) NOT NULL,          -- "human" ou "ai"
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índice para busca rápida por session_id
-- (toda request carrega o histórico do session_id)
CREATE INDEX IF NOT EXISTS idx_chat_memories_session
    ON chat_memories (session_id);

-- Índice para ordenação por data (load_memory usa ORDER BY created_at)
CREATE INDEX IF NOT EXISTS idx_chat_memories_created
    ON chat_memories (created_at DESC);


-- ============================================================
-- 2. LOGS DE INTERAÇÃO (analytics)
-- ============================================================
-- Registra cada interação para analytics e debug.
-- Equivalente ao nó "Log Interaction" do N8N.

CREATE TABLE IF NOT EXISTS interaction_logs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    agent_id INTEGER NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    user_message TEXT NOT NULL,
    agent_response TEXT,
    response_time_ms INTEGER,
    feedback VARCHAR(20),               -- "positive", "negative", NULL
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_logs_session
    ON interaction_logs (session_id);

CREATE INDEX IF NOT EXISTS idx_logs_agent
    ON interaction_logs (agent_id);

CREATE INDEX IF NOT EXISTS idx_logs_created
    ON interaction_logs (created_at DESC);

-- Índice parcial: só indexa linhas com feedback (economia de espaço)
CREATE INDEX IF NOT EXISTS idx_logs_feedback
    ON interaction_logs (feedback)
    WHERE feedback IS NOT NULL;


-- ============================================================
-- 3. RASTREABILIDADE DE DOCUMENTOS INGERIDOS
-- ============================================================
-- Registra quais documentos foram ingeridos, quando, e quantos
-- chunks geraram. Permite auditar e re-ingerir se necessário.

CREATE TABLE IF NOT EXISTS ingested_documents (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,   -- tabela destino no Supabase
    agent_id INTEGER NOT NULL,
    source_filename VARCHAR(500) NOT NULL,
    doc_type VARCHAR(50),
    chunk_count INTEGER,
    status VARCHAR(20) DEFAULT 'ingested',
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_docs_table
    ON ingested_documents (table_name);


-- ============================================================
-- 4. TABELAS DO APP MOBILE (futuro)
-- ============================================================
-- Usadas pelo backend do React Native para autenticação,
-- gerenciamento de sessões e perfil do usuário.

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',
    plan VARCHAR(50) DEFAULT 'free',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    agent_id INTEGER NOT NULL,
    title VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON user_sessions (user_id);

CREATE INDEX IF NOT EXISTS idx_sessions_session_id
    ON user_sessions (session_id);


-- ============================================================
-- 5. VIEWS DE ANALYTICS
-- ============================================================

-- Uso por agente (quantas interações por agente por dia)
CREATE OR REPLACE VIEW v_agent_usage AS
SELECT
    agent_id,
    agent_name,
    COUNT(*) as total_interactions,
    COUNT(CASE WHEN feedback = 'positive' THEN 1 END) as positive,
    COUNT(CASE WHEN feedback = 'negative' THEN 1 END) as negative,
    ROUND(AVG(response_time_ms)) as avg_response_ms,
    DATE_TRUNC('day', created_at) as day
FROM interaction_logs
GROUP BY agent_id, agent_name, DATE_TRUNC('day', created_at)
ORDER BY day DESC, total_interactions DESC;

-- Atividade por usuário
CREATE OR REPLACE VIEW v_user_activity AS
SELECT
    u.id as user_id,
    u.name,
    u.email,
    COUNT(DISTINCT us.session_id) as total_sessions,
    COUNT(il.id) as total_messages,
    MAX(il.created_at) as last_activity
FROM users u
LEFT JOIN user_sessions us ON u.id = us.user_id
LEFT JOIN interaction_logs il ON us.session_id = il.session_id
GROUP BY u.id, u.name, u.email;
