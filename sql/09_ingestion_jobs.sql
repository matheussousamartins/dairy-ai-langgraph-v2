-- 09_ingestion_jobs.sql
-- Rastreamento de jobs de ingestão de documentos (PDF/DOCX/MD).
-- Executar no banco Hetzner (operacional).

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id                  BIGSERIAL    PRIMARY KEY,
    job_id              TEXT         UNIQUE NOT NULL,
    agent_id            INT          NOT NULL,
    agent_name          TEXT         NOT NULL,
    table_name          TEXT         NOT NULL,
    original_filename   TEXT         NOT NULL,
    doc_type            TEXT         NOT NULL DEFAULT 'manual',
    -- queued | converting | processing | completed | failed
    status              TEXT         NOT NULL DEFAULT 'queued',
    error_detail        TEXT,
    chunks_created      INT,
    chunks_inserted     INT,
    chunks_updated      INT,
    pages_detected      INT,
    file_size_bytes     BIGINT,
    processing_time_ms  INT,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status  ON ingestion_jobs (status);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_agent   ON ingestion_jobs (agent_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_created ON ingestion_jobs (created_at DESC);
