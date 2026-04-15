"""Executa a migração da base transversal (agente 0).

Uso:
  python scripts/run_agent0_base_migration.py
"""

from pathlib import Path
import os

import psycopg
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("HETZNER_DB_URL")
    if not db_url:
        raise RuntimeError(
            "Defina SUPABASE_DB_URL ou HETZNER_DB_URL no .env antes de rodar."
        )

    sql_path = Path("sql/05_agent0_base_geral.sql")
    if not sql_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {sql_path}")

    sql = sql_path.read_text(encoding="utf-8")
    with psycopg.connect(db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)

    print("Migração aplicada: sql/05_agent0_base_geral.sql")


if __name__ == "__main__":
    main()

