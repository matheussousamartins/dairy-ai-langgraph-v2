-- ============================================================
-- 05_agent0_base_geral.sql
--
-- Migração incremental para adicionar a base transversal:
--   embeddings_agente_0_base_geral
--
-- Pode ser executada em ambiente já existente sem reset.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

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

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_agente_0_hash'
    ) THEN
        ALTER TABLE embeddings_agente_0_base_geral
            ADD CONSTRAINT uq_agente_0_hash UNIQUE (content_hash);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_agente_0_embedding
    ON embeddings_agente_0_base_geral
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_agente_0_fts
    ON embeddings_agente_0_base_geral
    USING gin (fts);

CREATE OR REPLACE FUNCTION update_fts_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fts := to_tsvector('portuguese', COALESCE(NEW.content, ''));
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_fts_agente_0
    BEFORE INSERT OR UPDATE OF content ON embeddings_agente_0_base_geral
    FOR EACH ROW EXECUTE FUNCTION update_fts_column();

