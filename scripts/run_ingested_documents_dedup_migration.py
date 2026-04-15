"""Executa a migração de deduplicação em ingested_documents.

Uso:
  python scripts/run_ingested_documents_dedup_migration.py
"""

from pathlib import Path
import os

import psycopg
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    db_url = os.getenv("HETZNER_DB_URL") or os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError(
            "Defina HETZNER_DB_URL ou SUPABASE_DB_URL no .env antes de rodar."
        )

    sql_path = Path("sql/04_ingested_documents_dedup.sql")
    if not sql_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {sql_path}")

    sql = sql_path.read_text(encoding="utf-8")
    with psycopg.connect(db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)

    print("Migração aplicada: sql/04_ingested_documents_dedup.sql")


if __name__ == "__main__":
    main()

