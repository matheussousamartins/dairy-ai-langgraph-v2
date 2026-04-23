# Estrutura de Documentos para Ingestão RAG

Padrão único para todos os agentes:

- `docs/agente-X.../pdf/`: arquivos originais.
- `docs/agente-X.../md/`: arquivos convertidos para ingestão.

Use sempre a pasta `md/` como entrada da ingestão.

## Pastas por agente

- `docs/agente-0-base-geral/{pdf,md}`
- `docs/agente-1-queijos/{pdf,md}`
- `docs/agente-2-fermentados/{pdf,md}`
- `docs/agente-3-regulatorios/{pdf,md}`
- `docs/agente-4-qualidade-leite/{pdf,md}`
- `docs/agente-5-defeitos/{pdf,md}`
- `docs/agente-6-formulacao/{pdf,md}`

## Conversão PDF -> Markdown (Docling)

Exemplo (agente 4):

```powershell
docling "docs/agente-4-qualidade-leite/pdf" --from pdf --to md --output "docs/agente-4-qualidade-leite/md"
```

## Ingestão de Markdown

Exemplo (agente 4):

```powershell
python -c "from app.db.connection import init_pools, close_pools; from app.rag.ingest import ingest_directory; init_pools(); print(ingest_directory(base_dir='docs/agente-4-qualidade-leite/md', table_name='embeddings_agente_4_qualidade_leite', agent_id=4, doc_type='legislacao')); close_pools()"
```

Ou com Makefile:

```powershell
make ingest DIR=docs/agente-4-qualidade-leite/md AGENT=4 TYPE=legislacao
```

Exemplo (agente base geral transversal):

```powershell
python -c "from app.db.connection import init_pools, close_pools; from app.rag.ingest import ingest_directory; init_pools(); print(ingest_directory(base_dir='docs/agente-0-base-geral/md', table_name='embeddings_agente_0_base_geral', agent_id=0, doc_type='glossario')); close_pools()"
```

Exemplo recomendado (agente 2 - fermentados, melhor granularidade):

```powershell
python -c "from app.db.connection import init_pools, close_pools; from app.rag.ingest import ingest_directory; init_pools(); print(ingest_directory(base_dir='docs/agente-2-fermentados/md', table_name='embeddings_agente_2_fermentados', agent_id=2, doc_type='artigo')); close_pools()"
```
