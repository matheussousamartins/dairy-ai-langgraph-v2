#!/usr/bin/env python3
"""
Ingestão dos documentos do Agente 1 — Tecnologia de Queijos.

Lê todos os arquivos .md em docs/agente-1-queijos/md/,
aplica chunking markdown com chunk_size=1200 (doc_type=manual),
gera embeddings e faz upsert em embeddings_agente_1_queijos.

Uso:
    python scripts/ingest_agent1.py
    python scripts/ingest_agent1.py --dry-run   # mostra chunks sem ingerar
"""

import argparse
import sys
import time
from pathlib import Path

# Garante que o root do projeto está no path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.db.connection import init_pools, close_pools
from app.rag.ingest import ingest_text
from app.rag.loaders import split_by_doc_type

MD_DIR = Path("docs/agente-1-queijos/md")
TABLE_NAME = "embeddings_agente_1_queijos"
AGENT_ID = 1
DOC_TYPE = "manual"
STRATEGY = "markdown"


def dry_run(only_file: str = None):
    if only_file:
        target = MD_DIR / only_file
        if not target.exists():
            print(f"ERRO: Arquivo não encontrado: {target}")
            return
        files = [target]
    else:
        files = sorted(MD_DIR.glob("*.md"))
        if not files:
            print(f"Nenhum .md encontrado em {MD_DIR}")
            return

    total_chunks = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks, strategy = split_by_doc_type(text, DOC_TYPE, STRATEGY)
        print(f"\n{f.name}")
        print(f"  chars: {len(text):,} | chunks: {len(chunks)} | strategy: {strategy}")
        for i, c in enumerate(chunks[:3]):
            preview = c[:120].replace("\n", " ")
            print(f"  [{i}] {preview}...")
        if len(chunks) > 3:
            print(f"  ... +{len(chunks)-3} chunks restantes")
        total_chunks += len(chunks)

    print(f"\nTotal estimado: {total_chunks} chunks de {len(files)} arquivo(s)")
    print("(dry-run: nenhuma ingestão realizada)")


def run(only_file: str = None):
    if only_file:
        target = MD_DIR / only_file
        if not target.exists():
            print(f"ERRO: Arquivo não encontrado: {target}")
            sys.exit(1)
        files = [target]
    else:
        files = sorted(MD_DIR.glob("*.md"))
        if not files:
            print(f"ERRO: Nenhum .md encontrado em {MD_DIR}")
            sys.exit(1)

    print(f"Iniciando ingestão: {len(files)} arquivo(s) → {TABLE_NAME}")
    print(f"Estratégia: {STRATEGY} | doc_type: {DOC_TYPE}\n")

    init_pools()
    total_chunks = 0
    errors = []
    t0 = time.time()

    for f in files:
        text = f.read_text(encoding="utf-8")
        print(f"  {f.name} ({len(text):,} chars) ...", end=" ", flush=True)
        t1 = time.time()

        result = ingest_text(
            text=text,
            table_name=TABLE_NAME,
            agent_id=AGENT_ID,
            source=f.name,
            doc_type=DOC_TYPE,
            strategy=STRATEGY,
        )

        elapsed = int((time.time() - t1) * 1000)

        if result.get("skipped_duplicate"):
            print(f"IGNORADO (já ingerido) [{elapsed}ms]")
        elif result.get("success"):
            ins = result.get("chunks_inserted", 0)
            upd = result.get("chunks_updated", 0)
            tot = result.get("chunks_created", 0)
            print(f"OK — {tot} chunks (inseridos={ins}, atualizados={upd}) [{elapsed}ms]")
            total_chunks += tot
        else:
            err = result.get("error", "erro desconhecido")
            print(f"ERRO — {err}")
            errors.append(f"{f.name}: {err}")

    close_pools()

    elapsed_total = int(time.time() - t0)
    print(f"\n{'='*50}")
    print(f"Concluído em {elapsed_total}s")
    print(f"Total de chunks ingeridos: {total_chunks}")
    if errors:
        print(f"Erros ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestão Agente 1 — Queijos")
    parser.add_argument("--dry-run", action="store_true", help="Preview sem ingerar")
    parser.add_argument("--file", type=str, default=None, help="Nome do arquivo .md (ex: 'DAIRY_MUSSARELA_COMPLETO.md')")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(only_file=args.file)
    else:
        run(only_file=args.file)
