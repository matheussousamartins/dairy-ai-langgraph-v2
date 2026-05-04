"""
ingestion_jobs.py — CRUD para rastreamento de jobs de ingestão no Hetzner.

Cada upload de documento cria um job com status inicial 'queued'.
O worker de background atualiza o status conforme avança:
  queued → converting → processing → completed | failed
"""

import logging
import uuid
from typing import Any, Dict, Optional

from app.db.connection import get_hetzner_conn

_log = logging.getLogger(__name__)


def create_ingestion_job(
    agent_id: int,
    agent_name: str,
    table_name: str,
    filename: str,
    doc_type: str,
    file_size_bytes: int,
) -> str:
    """Cria um job com status 'queued'. Retorna o job_id (UUID string)."""
    job_id = str(uuid.uuid4())
    with get_hetzner_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_jobs
                    (job_id, agent_id, agent_name, table_name,
                     original_filename, doc_type, file_size_bytes, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'queued')
                """,
                (job_id, agent_id, agent_name, table_name,
                 filename, doc_type, file_size_bytes),
            )
        conn.commit()
    _log.info("ingestion_job criado: %s | arquivo=%s | agente=%d", job_id, filename, agent_id)
    return job_id


def update_ingestion_job(job_id: str, status: str, **kwargs: Any) -> None:
    """Atualiza status e campos opcionais de um job.

    Campos aceitos via kwargs:
      error_detail, chunks_created, chunks_inserted,
      chunks_updated, pages_detected, processing_time_ms
    """
    _ALLOWED = {
        "error_detail", "chunks_created", "chunks_inserted",
        "chunks_updated", "pages_detected", "processing_time_ms",
    }
    set_clauses = ["status = %s", "updated_at = now()"]
    values: list = [status]

    for key, val in kwargs.items():
        if key in _ALLOWED:
            set_clauses.append(f"{key} = %s")
            values.append(val)

    values.append(job_id)
    sql = f"UPDATE ingestion_jobs SET {', '.join(set_clauses)} WHERE job_id = %s"

    try:
        with get_hetzner_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
            conn.commit()
    except Exception as exc:
        _log.error("Erro ao atualizar ingestion_job %s → %s: %s", job_id, status, exc)


def get_ingestion_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Retorna dados do job pelo job_id, ou None se não encontrado."""
    sql = """
        SELECT job_id, agent_id, agent_name, table_name, original_filename,
               doc_type, status, error_detail, chunks_created, chunks_inserted,
               chunks_updated, pages_detected, file_size_bytes, processing_time_ms,
               created_at, updated_at
        FROM ingestion_jobs
        WHERE job_id = %s
    """
    try:
        with get_hetzner_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (job_id,))
                row = cur.fetchone()
                if not row:
                    return None
                cols = [d[0] for d in cur.description]
                result = dict(zip(cols, row))
                # Serializar timestamps para string ISO
                for field in ("created_at", "updated_at"):
                    if result.get(field) is not None:
                        result[field] = result[field].isoformat()
                return result
    except Exception as exc:
        _log.error("Erro ao buscar ingestion_job %s: %s", job_id, exc)
        return None
