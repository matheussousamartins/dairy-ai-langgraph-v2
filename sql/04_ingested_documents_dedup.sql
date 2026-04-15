-- ============================================================
-- 04_ingested_documents_dedup.sql
--
-- Migração incremental (não destrutiva) para rastreabilidade e
-- deduplicação em nível de documento.
--
-- Pode ser executada no mesmo banco que contém `ingested_documents`.
-- Exemplo:
--   python -c "import os, psycopg; from dotenv import load_dotenv; load_dotenv(); sql=open('sql/04_ingested_documents_dedup.sql','r',encoding='utf-8').read(); conn=psycopg.connect(os.getenv('HETZNER_DB_URL') or os.getenv('SUPABASE_DB_URL')); conn.autocommit=True; cur=conn.cursor(); cur.execute(sql); conn.close(); print('04_ingested_documents_dedup.sql OK')"
-- ============================================================

ALTER TABLE IF EXISTS ingested_documents
    ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64);

ALTER TABLE IF EXISTS ingested_documents
    ADD COLUMN IF NOT EXISTS source_size_bytes INTEGER;

ALTER TABLE IF EXISTS ingested_documents
    ADD COLUMN IF NOT EXISTS status_detail TEXT;

CREATE INDEX IF NOT EXISTS idx_docs_agent_hash
    ON ingested_documents (agent_id, file_hash);

CREATE INDEX IF NOT EXISTS idx_docs_ingested_at
    ON ingested_documents (ingested_at DESC);

