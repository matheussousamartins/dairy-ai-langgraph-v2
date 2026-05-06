-- =============================================================================
-- Migration 001 — Consolidação de tabelas de embeddings
--
-- Antes: 7 tabelas por agente (embeddings_agente_0..6), só 2 com dados
-- Depois: 2 tabelas semânticas (embeddings_especialista, embeddings_regulatorios)
--
-- As tabelas originais são RENOMEADAS (não deletadas) como backup.
-- Para rollback completo: renomear de volta e apagar as novas.
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- PASSO 1: Criar tabela embeddings_especialista
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS embeddings_especialista (
    id            BIGSERIAL PRIMARY KEY,
    content       TEXT        NOT NULL,
    embedding     vector(1536),
    metadata      JSONB,
    content_hash  VARCHAR,
    fts           TSVECTOR,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_especialista_hash
    ON embeddings_especialista (content_hash);

CREATE INDEX IF NOT EXISTS idx_especialista_embedding
    ON embeddings_especialista
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_especialista_fts
    ON embeddings_especialista
    USING GIN (fts);

-- ---------------------------------------------------------------------------
-- PASSO 2: Criar tabela embeddings_regulatorios
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS embeddings_regulatorios (
    id            BIGSERIAL PRIMARY KEY,
    content       TEXT        NOT NULL,
    embedding     vector(1536),
    metadata      JSONB,
    content_hash  VARCHAR,
    fts           TSVECTOR,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_regulatorios_hash
    ON embeddings_regulatorios (content_hash);

CREATE INDEX IF NOT EXISTS idx_regulatorios_embedding
    ON embeddings_regulatorios
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_regulatorios_fts
    ON embeddings_regulatorios
    USING GIN (fts);

-- ---------------------------------------------------------------------------
-- PASSO 3: Copiar dados das tabelas originais para as novas
--
-- agente_1 (queijos) → embeddings_especialista
-- agente_3 (regulatorios) → embeddings_regulatorios
--
-- ON CONFLICT DO NOTHING garante idempotência (pode rodar mais de uma vez)
-- ---------------------------------------------------------------------------

INSERT INTO embeddings_especialista
    (content, embedding, metadata, content_hash, fts, created_at, updated_at)
SELECT
    content, embedding, metadata, content_hash, fts, created_at, updated_at
FROM embeddings_agente_1_queijos
ON CONFLICT (content_hash) DO NOTHING;

INSERT INTO embeddings_regulatorios
    (content, embedding, metadata, content_hash, fts, created_at, updated_at)
SELECT
    content, embedding, metadata, content_hash, fts, created_at, updated_at
FROM embeddings_agente_3_regulatorios
ON CONFLICT (content_hash) DO NOTHING;

-- ---------------------------------------------------------------------------
-- PASSO 4: Verificação de contagem antes de commitar
-- (falha o bloco se as contagens não baterem)
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    src_especialista  INT;
    dst_especialista  INT;
    src_regulatorios  INT;
    dst_regulatorios  INT;
BEGIN
    SELECT COUNT(*) INTO src_especialista  FROM embeddings_agente_1_queijos;
    SELECT COUNT(*) INTO dst_especialista  FROM embeddings_especialista;
    SELECT COUNT(*) INTO src_regulatorios  FROM embeddings_agente_3_regulatorios;
    SELECT COUNT(*) INTO dst_regulatorios  FROM embeddings_regulatorios;

    IF dst_especialista < src_especialista THEN
        RAISE EXCEPTION
            'FALHA: embeddings_especialista tem % chunks, esperado >= % (agente_1)',
            dst_especialista, src_especialista;
    END IF;

    IF dst_regulatorios < src_regulatorios THEN
        RAISE EXCEPTION
            'FALHA: embeddings_regulatorios tem % chunks, esperado >= % (agente_3)',
            dst_regulatorios, src_regulatorios;
    END IF;

    RAISE NOTICE 'OK: especialista=% regulatorios=%', dst_especialista, dst_regulatorios;
END $$;

-- ---------------------------------------------------------------------------
-- PASSO 5: Renomear tabelas originais para backup (não deletar)
-- ---------------------------------------------------------------------------

ALTER TABLE embeddings_agente_0_base_geral    RENAME TO _bkp_embeddings_agente_0;
ALTER TABLE embeddings_agente_1_queijos       RENAME TO _bkp_embeddings_agente_1;
ALTER TABLE embeddings_agente_2_fermentados   RENAME TO _bkp_embeddings_agente_2;
ALTER TABLE embeddings_agente_3_regulatorios  RENAME TO _bkp_embeddings_agente_3;
ALTER TABLE embeddings_agente_4_qualidade_leite RENAME TO _bkp_embeddings_agente_4;
ALTER TABLE embeddings_agente_5_defeitos      RENAME TO _bkp_embeddings_agente_5;
ALTER TABLE embeddings_agente_6_formulacao    RENAME TO _bkp_embeddings_agente_6;
ALTER TABLE embeddings_teste_smoke            RENAME TO _bkp_embeddings_teste_smoke;

COMMIT;

-- =============================================================================
-- ROLLBACK (executar manualmente se necessário)
-- =============================================================================
--
-- BEGIN;
-- ALTER TABLE _bkp_embeddings_agente_0      RENAME TO embeddings_agente_0_base_geral;
-- ALTER TABLE _bkp_embeddings_agente_1      RENAME TO embeddings_agente_1_queijos;
-- ALTER TABLE _bkp_embeddings_agente_2      RENAME TO embeddings_agente_2_fermentados;
-- ALTER TABLE _bkp_embeddings_agente_3      RENAME TO embeddings_agente_3_regulatorios;
-- ALTER TABLE _bkp_embeddings_agente_4      RENAME TO embeddings_agente_4_qualidade_leite;
-- ALTER TABLE _bkp_embeddings_agente_5      RENAME TO embeddings_agente_5_defeitos;
-- ALTER TABLE _bkp_embeddings_agente_6      RENAME TO embeddings_agente_6_formulacao;
-- ALTER TABLE _bkp_embeddings_teste_smoke   RENAME TO embeddings_teste_smoke;
-- DROP TABLE IF EXISTS embeddings_especialista;
-- DROP TABLE IF EXISTS embeddings_regulatorios;
-- COMMIT;
--
-- =============================================================================
-- LIMPEZA DE BACKUP (só após validação em produção — sem rollback depois disso)
-- =============================================================================
--
-- DROP TABLE IF EXISTS _bkp_embeddings_agente_0;
-- DROP TABLE IF EXISTS _bkp_embeddings_agente_1;
-- DROP TABLE IF EXISTS _bkp_embeddings_agente_2;
-- DROP TABLE IF EXISTS _bkp_embeddings_agente_3;
-- DROP TABLE IF EXISTS _bkp_embeddings_agente_4;
-- DROP TABLE IF EXISTS _bkp_embeddings_agente_5;
-- DROP TABLE IF EXISTS _bkp_embeddings_agente_6;
-- DROP TABLE IF EXISTS _bkp_embeddings_teste_smoke;
