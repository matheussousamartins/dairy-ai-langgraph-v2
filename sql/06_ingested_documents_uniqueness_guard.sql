-- ============================================================
-- 06_ingested_documents_uniqueness_guard.sql
--
-- Blindagem anti-duplicacao para ingestao via web/API.
-- Garante unicidade de file_hash para registros ativos.
--
-- Registros ativos:
--   - processing
--   - ingested
--
-- Assim evitamos corrida de concorrencia:
-- duas requisicoes simultaneas com o mesmo documento nao conseguem
-- reservar o mesmo hash ao mesmo tempo.
-- ============================================================

-- 0) Pre-requisitos
ALTER TABLE IF EXISTS ingested_documents
    ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64);

ALTER TABLE IF EXISTS ingested_documents
    ADD COLUMN IF NOT EXISTS source_size_bytes INTEGER;

ALTER TABLE IF EXISTS ingested_documents
    ADD COLUMN IF NOT EXISTS status_detail TEXT;

-- 1) Limpa duplicados ativos por file_hash, preservando o mais recente
WITH ranked AS (
    SELECT
        id,
        file_hash,
        status,
        ingested_at,
        ROW_NUMBER() OVER (
            PARTITION BY file_hash
            ORDER BY ingested_at DESC, id DESC
        ) AS rn
    FROM ingested_documents
    WHERE file_hash IS NOT NULL
      AND status IN ('processing', 'ingested')
)
DELETE FROM ingested_documents d
USING ranked r
WHERE d.id = r.id
  AND r.rn > 1;

-- 2) Indice unico parcial para registros ativos
CREATE UNIQUE INDEX IF NOT EXISTS uq_ingested_file_hash_active
    ON ingested_documents (file_hash)
    WHERE file_hash IS NOT NULL
      AND status IN ('processing', 'ingested');

-- 3) Indice auxiliar para consulta por hash
CREATE INDEX IF NOT EXISTS idx_ingested_file_hash
    ON ingested_documents (file_hash);

