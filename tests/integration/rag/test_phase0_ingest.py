"""
tests/integration/rag/test_phase0_ingest.py — Fase 0: Smoke test de ingestão

Verifica que o pipeline de ingestão funciona ponta a ponta:
  1. Cria um texto de teste (simula documento de queijos)
  2. Faz chunking
  3. Gera embeddings
  4. Insere na tabela de teste do Supabase
  5. Verifica que os chunks foram salvos

Este é o PRIMEIRO teste a rodar. Se falhar, nada mais funciona.
Equivalente ao test_kb_ingest_smoke.py do original.

Uso: make rag_phase0
"""

import pytest

pytestmark = pytest.mark.phase0


SAMPLE_TEXT = """
# Fabricação de Queijo Mussarela

## Recepção e Pasteurização
O leite deve ser recebido com acidez Dornic entre 15-18°D. 
A pasteurização é feita a 72°C por 15 segundos (HTST).

## Coagulação
Adicionar cloreto de cálcio (40mL/100L de solução a 50%).
Adicionar coalho na dosagem recomendada pelo fabricante.
Temperatura de coagulação: 32-35°C. Tempo: 40-50 minutos.

## Corte e Mexedura
Cortar a coalhada em cubos de 1-2 cm.
Mexer lentamente por 15-20 minutos, elevando a temperatura para 40-42°C.

## Filagem
O pH da massa deve estar entre 5.2-5.4 para iniciar a filagem.
A água de filagem deve estar entre 78-82°C.
Trabalhar a massa até obter textura lisa e elástica.

## Salga
Salgar em salmoura com concentração de 20% de NaCl a 10-12°C.
Tempo de salga: proporcional ao peso da peça.
""".strip()


def test_ingest_smoke(db_supabase, require_openai):
    """Testa ingestão básica: texto → chunks → embeddings → Supabase."""
    import psycopg
    from app.rag.ingest import ingest_text
    
    TABLE = "embeddings_teste_smoke"
    
    # Cria tabela de teste (se não existir)
    with psycopg.connect(db_supabase) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    content_hash VARCHAR(16),
                    fts tsvector,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute(f"""
                ALTER TABLE {TABLE}
                ADD CONSTRAINT IF NOT EXISTS uq_smoke_hash 
                UNIQUE (content_hash)
            """)
    
    # Executa ingestão
    result = ingest_text(
        text=SAMPLE_TEXT,
        table_name=TABLE,
        agent_id=0,
        source="test_smoke",
        doc_type="manual",
        strategy="fixed",
    )
    
    assert result["success"] is True, f"Ingestão falhou: {result}"
    assert result["chunks_created"] > 0, "Nenhum chunk criado"
    
    print(f"\n[Phase 0] Ingestão OK: {result['chunks_created']} chunks criados")
    print(f"  Tabela: {TABLE}")
    print(f"  Tempo: {result.get('processing_time_ms', '?')}ms")
    
    # Verifica no banco
    with psycopg.connect(db_supabase) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
            count = cur.fetchone()[0]
    
    assert count > 0, "Tabela vazia após ingestão"
    assert count == result["chunks_created"], "Count no banco difere do retornado"
    
    print(f"  Verificado no banco: {count} chunks")
    
    # Limpa tabela de teste
    with psycopg.connect(db_supabase) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
    
    print("  Tabela de teste removida.")


def test_chunking_strategies(require_openai):
    """Testa que as 3 estratégias de chunking funcionam."""
    from app.rag.loaders import split_text
    
    for strategy in ["fixed", "markdown"]:
        chunks, resolved = split_text(SAMPLE_TEXT, strategy=strategy)
        assert len(chunks) > 0, f"Estratégia '{strategy}' não gerou chunks"
        assert resolved == strategy
        print(f"\n[Phase 0] Chunking '{strategy}': {len(chunks)} chunks")
        for i, c in enumerate(chunks[:3]):
            print(f"  Chunk {i}: {c[:80]}...")


def test_chunk_sizes_by_doc_type():
    """Verifica que o chunking adaptativo usa tamanhos corretos."""
    from app.rag.loaders import split_by_doc_type
    
    # Texto longo para testar chunking
    long_text = SAMPLE_TEXT * 5
    
    # Legislação: chunks menores (600 chars)
    chunks_leg, _ = split_by_doc_type(long_text, "legislacao", strategy="fixed")
    
    # Manual: chunks maiores (1200 chars)
    chunks_man, _ = split_by_doc_type(long_text, "manual", strategy="fixed")
    
    # Legislação deve gerar MAIS chunks (chunks menores = mais pedaços)
    assert len(chunks_leg) > len(chunks_man), (
        f"Legislação ({len(chunks_leg)} chunks) deveria ter mais chunks "
        f"que manual ({len(chunks_man)} chunks)"
    )
    
    print(f"\n[Phase 0] Chunking adaptativo:")
    print(f"  legislacao: {len(chunks_leg)} chunks")
    print(f"  manual: {len(chunks_man)} chunks")
