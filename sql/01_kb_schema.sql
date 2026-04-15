-- ============================================================
-- 01_kb_schema.sql — Tabelas de embeddings no Supabase
-- 
-- Este script cria as 6 tabelas de embeddings (uma por agente)
-- e a tabela de staging (kb_docs) no Supabase.
--
-- Executar no Supabase:
--   psql $SUPABASE_DB_URL < sql/01_kb_schema.sql
--
-- No projeto original do curso, existe apenas 1 tabela (kb_chunks)
-- com um campo "empresa" para filtrar por tenant. Aqui temos
-- 6 tabelas separadas — uma por agente — porque:
--   1. Isolamento: busca do agente de queijos NUNCA retorna
--      chunks de legislação (mesmo sem filtro WHERE)
--   2. Performance: cada tabela tem seu próprio índice HNSW,
--      que é mais eficiente quando os vetores são do mesmo domínio
--   3. Manutenção: pode re-ingerir a base de um agente sem
--      afetar os outros (DROP TABLE + recriar)
--
-- Pré-requisito: extensão pgvector habilitada
--   CREATE EXTENSION IF NOT EXISTS vector;
--   (No Supabase, habilite via Dashboard → Database → Extensions)
-- ============================================================

-- Garante que pgvector está habilitado
CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- Tabela template para embeddings de cada agente
-- ============================================================
-- Todas as 6 tabelas têm a mesma estrutura. Os campos são:
--
-- id: 
--   Identificador auto-gerado (BIGSERIAL = auto-incremento 64-bit).
--   Usado internamente pelo Postgres, não aparece nas buscas.
--
-- content:
--   Texto do chunk. Ex: "A filagem da mussarela deve ser feita
--   entre 78-82°C com pH de 5.2-5.4..."
--   Tamanho varia: 200 a 1500 caracteres dependendo do doc_type.
--
-- embedding:
--   Vetor de 1536 dimensões gerado pelo text-embedding-3-small.
--   Armazenado como tipo vector do pgvector.
--   É o campo usado na busca por similaridade de cosseno.
--
-- metadata:
--   JSON com metadados do chunk:
--   {
--     "agent_id": 1,
--     "source": "manual-fabricacao-mussarela.pdf",
--     "doc_type": "manual",
--     "chunk_index": 3,
--     "strategy": "markdown",
--     "ingested_at": "2026-04-08T14:30:00"
--   }
--   Permite filtros futuros (ex: buscar só chunks de legislação).
--
-- content_hash:
--   Hash SHA-256 truncado do content (16 caracteres).
--   Usado para idempotência: ON CONFLICT (content_hash) DO UPDATE.
--   Evita duplicatas quando o mesmo documento é re-ingerido.
--
-- fts:
--   Campo de Full-Text Search gerado automaticamente pelo trigger.
--   Contém o texto tokenizado para busca por palavras-chave.
--   Usado pela função kb_text_search.
--
-- created_at / updated_at:
--   Timestamps para rastreabilidade.


-- ============================================================
-- Agente 1: Tecnologia de Queijos
-- ============================================================
-- ============================================================
-- Agente 0: Base Geral Dairy (transversal)
-- ============================================================
CREATE TABLE IF NOT EXISTS embeddings_agente_0_base_geral (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    content_hash VARCHAR(16),
    fts tsvector,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE embeddings_agente_0_base_geral
    ADD CONSTRAINT uq_agente_0_hash UNIQUE (content_hash);


-- ============================================================
-- Agente 1: Tecnologia de Queijos
-- ============================================================
CREATE TABLE IF NOT EXISTS embeddings_agente_1_queijos (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    content_hash VARCHAR(16),
    fts tsvector,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Constraint de unicidade no hash (para idempotência do upsert)
-- Se o hash já existir, o INSERT faz UPDATE em vez de duplicar
ALTER TABLE embeddings_agente_1_queijos
    ADD CONSTRAINT uq_agente_1_hash UNIQUE (content_hash);


-- ============================================================
-- Agente 2: Fermentados
-- ============================================================
CREATE TABLE IF NOT EXISTS embeddings_agente_2_fermentados (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    content_hash VARCHAR(16),
    fts tsvector,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE embeddings_agente_2_fermentados
    ADD CONSTRAINT uq_agente_2_hash UNIQUE (content_hash);


-- ============================================================
-- Agente 3: Regulatórios por País
-- ============================================================
CREATE TABLE IF NOT EXISTS embeddings_agente_3_regulatorios (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    content_hash VARCHAR(16),
    fts tsvector,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE embeddings_agente_3_regulatorios
    ADD CONSTRAINT uq_agente_3_hash UNIQUE (content_hash);


-- ============================================================
-- Agente 4: Qualidade do Leite
-- ============================================================
CREATE TABLE IF NOT EXISTS embeddings_agente_4_qualidade_leite (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    content_hash VARCHAR(16),
    fts tsvector,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE embeddings_agente_4_qualidade_leite
    ADD CONSTRAINT uq_agente_4_hash UNIQUE (content_hash);


-- ============================================================
-- Agente 5: Diagnóstico de Defeitos
-- ============================================================
CREATE TABLE IF NOT EXISTS embeddings_agente_5_defeitos (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    content_hash VARCHAR(16),
    fts tsvector,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE embeddings_agente_5_defeitos
    ADD CONSTRAINT uq_agente_5_hash UNIQUE (content_hash);


-- ============================================================
-- Agente 6: Formulação e Desenvolvimento
-- ============================================================
CREATE TABLE IF NOT EXISTS embeddings_agente_6_formulacao (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    content_hash VARCHAR(16),
    fts tsvector,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE embeddings_agente_6_formulacao
    ADD CONSTRAINT uq_agente_6_hash UNIQUE (content_hash);


-- ============================================================
-- Triggers para atualização automática do FTS
-- ============================================================
-- Quando um chunk é inserido ou atualizado, o campo fts é
-- atualizado automaticamente com o texto tokenizado em português.
--
-- to_tsvector('portuguese', content) converte o texto em tokens:
--   "A filagem da mussarela" → 'filag':2 'mussarel':4
-- Esses tokens são usados pela busca FTS (operador @@).
--
-- No original (sql/kb/01_init.sql), o trigger é criado para
-- a tabela kb_chunks. Aqui criamos para cada uma das 6 tabelas.

-- Função genérica de trigger (reutilizada por todas as tabelas)
CREATE OR REPLACE FUNCTION update_fts_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fts := to_tsvector('portuguese', COALESCE(NEW.content, ''));
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para cada tabela
CREATE OR REPLACE TRIGGER trg_fts_agente_1
    BEFORE INSERT OR UPDATE OF content ON embeddings_agente_1_queijos
    FOR EACH ROW EXECUTE FUNCTION update_fts_column();

CREATE OR REPLACE TRIGGER trg_fts_agente_0
    BEFORE INSERT OR UPDATE OF content ON embeddings_agente_0_base_geral
    FOR EACH ROW EXECUTE FUNCTION update_fts_column();

CREATE OR REPLACE TRIGGER trg_fts_agente_2
    BEFORE INSERT OR UPDATE OF content ON embeddings_agente_2_fermentados
    FOR EACH ROW EXECUTE FUNCTION update_fts_column();

CREATE OR REPLACE TRIGGER trg_fts_agente_3
    BEFORE INSERT OR UPDATE OF content ON embeddings_agente_3_regulatorios
    FOR EACH ROW EXECUTE FUNCTION update_fts_column();

CREATE OR REPLACE TRIGGER trg_fts_agente_4
    BEFORE INSERT OR UPDATE OF content ON embeddings_agente_4_qualidade_leite
    FOR EACH ROW EXECUTE FUNCTION update_fts_column();

CREATE OR REPLACE TRIGGER trg_fts_agente_5
    BEFORE INSERT OR UPDATE OF content ON embeddings_agente_5_defeitos
    FOR EACH ROW EXECUTE FUNCTION update_fts_column();

CREATE OR REPLACE TRIGGER trg_fts_agente_6
    BEFORE INSERT OR UPDATE OF content ON embeddings_agente_6_formulacao
    FOR EACH ROW EXECUTE FUNCTION update_fts_column();
