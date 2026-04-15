-- ============================================================
-- 02_kb_indexes.sql — Índices para busca vetorial e textual
--
-- Executar no Supabase DEPOIS de 01_kb_schema.sql:
--   psql $SUPABASE_DB_URL < sql/02_kb_indexes.sql
--
-- Este script cria 2 tipos de índice para cada tabela:
--
-- 1. HNSW (Hierarchical Navigable Small World) para busca vetorial
--    - Índice aproximado de vizinhos mais próximos
--    - Busca em ~13 passos para qualquer quantidade de vetores
--    - Operador: vector_cosine_ops (similaridade de cosseno)
--    - Sem HNSW: busca brute-force O(n) — lenta para >1000 chunks
--    - Com HNSW: busca O(log n) — rápida até milhões de chunks
--
-- 2. GIN (Generalized Inverted Index) para busca textual (FTS)
--    - Índice invertido: mapeia cada palavra → lista de chunks
--    - Operador: tsvector_ops (Full-Text Search)
--    - Sem GIN: busca sequencial (lê todos os chunks) — lenta
--    - Com GIN: busca por índice (lookup direto) — instantânea
--
-- No projeto original (sql/kb/02_indexes.sql), os índices são
-- criados para a tabela única kb_chunks. Aqui criamos para cada
-- uma das 6 tabelas.
--
-- IMPORTANTE: Criar índices HNSW em tabelas vazias é instantâneo.
-- Criar em tabelas com dados pode levar segundos a minutos
-- (depende do volume). Por isso, rode este script ANTES de ingerir.
--
-- Parâmetros do HNSW:
--   m = 16: Número de conexões por nó no grafo. Mais conexões =
--           busca mais precisa mas índice maior. 16 é o padrão
--           recomendado pelo pgvector.
--   ef_construction = 64: Tamanho do conjunto de candidatos durante
--           construção do índice. Maior = índice melhor mas mais
--           lento para construir. 64 é bom para <100K vetores.
-- ============================================================


-- ============================================================
-- Agente 1: Tecnologia de Queijos
-- ============================================================
-- ============================================================
-- Agente 0: Base Geral Dairy (transversal)
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_agente_0_embedding
    ON embeddings_agente_0_base_geral
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_agente_0_fts
    ON embeddings_agente_0_base_geral
    USING gin (fts);


-- ============================================================
-- Agente 1: Tecnologia de Queijos
-- ============================================================

-- Índice HNSW para busca vetorial (similaridade de cosseno)
CREATE INDEX IF NOT EXISTS idx_agente_1_embedding
    ON embeddings_agente_1_queijos
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Índice GIN para busca textual (Full-Text Search)
CREATE INDEX IF NOT EXISTS idx_agente_1_fts
    ON embeddings_agente_1_queijos
    USING gin (fts);


-- ============================================================
-- Agente 2: Fermentados
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_agente_2_embedding
    ON embeddings_agente_2_fermentados
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_agente_2_fts
    ON embeddings_agente_2_fermentados
    USING gin (fts);


-- ============================================================
-- Agente 3: Regulatórios por País
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_agente_3_embedding
    ON embeddings_agente_3_regulatorios
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_agente_3_fts
    ON embeddings_agente_3_regulatorios
    USING gin (fts);


-- ============================================================
-- Agente 4: Qualidade do Leite
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_agente_4_embedding
    ON embeddings_agente_4_qualidade_leite
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_agente_4_fts
    ON embeddings_agente_4_qualidade_leite
    USING gin (fts);


-- ============================================================
-- Agente 5: Diagnóstico de Defeitos
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_agente_5_embedding
    ON embeddings_agente_5_defeitos
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_agente_5_fts
    ON embeddings_agente_5_defeitos
    USING gin (fts);


-- ============================================================
-- Agente 6: Formulação e Desenvolvimento
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_agente_6_embedding
    ON embeddings_agente_6_formulacao
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_agente_6_fts
    ON embeddings_agente_6_formulacao
    USING gin (fts);
