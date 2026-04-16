# DairyApp AI — Contexto do Projeto

Arquivo lido automaticamente pelo Claude Code. Contém arquitetura, decisões,
estado atual e convenções para que qualquer nova sessão já entenda o projeto.

---

## Visão Geral

Sistema multi-agente de IA para tecnologia de laticínios.
Cliente corporativo. Lançamento previsto: **junho de 2026**.

- **7 agentes especialistas** + **1 orquestrador**
- **App mobile** (React Native, dev separado) + **web** (Next.js, console de teste)
- **Backend dual**: N8N (MVP validado) + LangGraph/FastAPI (produção, este repo)
- Ambos backends expõem o **mesmo contrato de API** — o app aponta para qualquer um

## Agentes

| ID | Nome | Tabela | Status |
|----|------|--------|--------|
| 0 | Base Geral Dairy | `embeddings_agente_0_base_geral` | ✅ ingerido, 85% E2E |
| 1 | Tecnologia de Queijos | `embeddings_agente_1_queijos` | ⏳ aguardando documentos |
| 2 | Fermentados | `embeddings_agente_2_fermentados` | ⏳ aguardando documentos |
| 3 | Regulatórios por País | `embeddings_agente_3_regulatorios` | ✅ ingerido, 85% E2E |
| 4 | Qualidade do Leite | `embeddings_agente_4_qualidade_leite` | ✅ ingerido, 85% E2E |
| 5 | Diagnóstico de Defeitos | `embeddings_agente_5_defeitos` | ⏳ aguardando documentos |
| 6 | Formulação e Desenvolvimento | `embeddings_agente_6_formulacao` | ⏳ aguardando documentos |

O **Agente 0** é transversal: sempre incluído pelo orquestrador para garantir
coerência terminológica (glossário + verdades absolutas).

O **Agente 3** também é incluído por padrão em perguntas de laticínios para
garantir verificação regulatória.

Os agentes **1 a 6** têm acesso a **ferramentas de cálculo determinísticas**
(`app/tools/calculations.py`) — o agente 0 não tem (é transversal, não opera
cálculos técnicos).

---

## Arquitetura

### Stack
- **LangGraph** para grafos de agentes (ReAct pattern)
- **FastAPI** para servidor HTTP (com streaming SSE)
- **OpenAI**: `gpt-4o-mini` (chat), `text-embedding-3-small` (embeddings, 1536 dims)
- **Postgres + pgvector** para vector store (Supabase)
- **Postgres** separado para memory + logs (Hetzner)
- **Autenticação**: header `X-API-Key` (controlado por `WEBHOOK_API_KEYS`)

### Estrutura de Pastas
```
app/
├── config.py              # Configurações centralizadas (lidas de .env)
├── graphs.py              # Exports para LangGraph Studio
├── agents/
│   ├── agent_config.py    # Metadados dos 7 agentes
│   ├── base_agent.py      # Builder de grafos ReAct (com calculations tools)
│   ├── orchestrator.py    # Orquestrador multi-agente PARALELO
│   └── prompts.py         # System prompts dos 7 agentes
├── rag/
│   ├── ingest.py          # Pipeline: quality gate → chunks → embeddings → upsert
│   ├── loaders.py         # Splitters: fixed, markdown, semantic
│   ├── search.py          # Busca: vector, text, hybrid_rrf + HyDE
│   └── rerank.py          # Reranking Cohere (opcional)
├── tools/
│   ├── __init__.py
│   └── calculations.py    # Tools determinísticas de cálculo (5 tools)
├── server/
│   └── webapp.py          # FastAPI: 10+ endpoints
└── db/
    ├── connection.py      # Pools de conexão (Supabase + Hetzner)
    └── memory.py          # Chat memory + interaction logs

sql/
├── 01_kb_schema.sql                      # 7 tabelas de embeddings
├── 02_kb_indexes.sql                     # HNSW (vector) + GIN (FTS)
├── 03_app_tables.sql                     # chat_memories, logs, users
├── 04_ingested_documents_dedup.sql       # Dedup por file_hash
├── 05_agent0_base_geral.sql              # Tabela do agente 0
└── 06_ingested_documents_uniqueness_guard.sql  # Anti-race condition

tests/
├── conftest.py                           # Fixtures pytest
├── fixtures/rag/
│   ├── rag_queries.yaml                  # Dataset de 440+ perguntas
│   └── experiments.template.yaml         # Combinações a testar
└── integration/rag/
    ├── test_phase0_ingest.py             # Smoke test
    ├── test_phase1_retrieval.py          # Hit@k + LLM judge
    ├── test_phase2_strategies.py         # Compara estratégias
    └── test_phase3_agent_e2e.py          # Agente completo

scripts/
├── benchmark_latency.py                  # Mede p50/p95/p99
├── rag_experiments_runner.py             # Runner de combinações
├── consume_rag_queries.py                # Roda queries de teste
├── clean_agent_md_quality.py             # Limpa markdown
└── run_*_migration.py                    # Migrações pontuais

docs/
├── CONTRATO_API_MOBILE_WEB.md            # Contrato oficial com o app mobile
├── GUIA_INGESTAO_WEB_MOBILE.md           # Guia de ingestão para o front
├── MESSAGE_SERVICE_CONTRACT.md           # Integração com serviço de mensagens
├── PAYLOAD_INGESTAO_FRONT.md             # Exemplos de payload
├── formulas_mapeadas.md                  # Inventário de fórmulas por agente
└── agente-{0-6}/                         # Documentos fonte por agente

frontend/                                  # Console de teste (Next.js)
```

### Fluxo de Request (agente direto)
```
POST /webhook/agente-{id}
  ↓
[Carrega histórico do chat_memories]
  ↓
[Grafo ReAct do agente]
  prepare → agent (LLM com tools) → tools (kb_search + calc) → agent → resposta
  ↓
[Sanitiza LaTeX → texto legível mobile]
  ↓
[Salva em chat_memories + interaction_logs]
  ↓
Response: { response, agent_id, agent_name }
```

### Fluxo do Orquestrador (multi-agente paralelo)
```
POST /webhook/orquestrador
  ↓
[classify] — tenta fastpath (regras determinísticas), senão usa LLM
              → lista de agent_ids (sempre inclui 0 e 3 em perguntas dairy)
  ↓
[route] — conversa geral vs consultar agentes
  ↓
[execute] — asyncio.gather chama N agentes em PARALELO
  ↓
[consolidate] — LLM funde as N respostas em uma resposta coerente
  ↓
[sanitize] — remove LaTeX, ajusta formatação para mobile
  ↓
Response: { response, agent_id=primary, agent_name }
```

**Latência** = tempo do agente mais lento (não a soma).
**Timeout por agente** = 12s (`AGENT_TIMEOUT`).
**Cache de classificação** = 256 entradas (`CLASSIFICATION_CACHE_SIZE`).

### Fluxo de Ingestão (com Quality Gate)
```
POST /webhook/ingestao ou /webhook/ingestao-arquivo
  ↓
[_assess_text_quality] — valida qualidade mínima do texto
  - chars >= INGEST_MIN_TEXT_CHARS (400)
  - palavras >= INGEST_MIN_WORDS (80)
  - garbled ratio <= INGEST_MAX_GARBLED_RATIO (0.08)
  - score >= INGEST_MIN_QUALITY_SCORE (60)
  ↓
[_reserve_ingestion_slot] — reserva slot no banco (status=processing)
  ↓ (anti-race condition via unique index parcial)
[chunking adaptativo] — tamanho varia por doc_type
  ↓
[embeddings batch] — 1 chamada OpenAI para N chunks
  ↓
[upsert no Supabase] — idempotente por content_hash
  ↓
[_update_ingestion_status] — marca como ingested (ou failed)
  ↓
Response: { success, chunks_created, table_name, ... }
```

### Endpoints Disponíveis
```
POST /webhook/agente-{id}              # Chat com agente especialista
POST /webhook/agente-{id}/stream       # Chat com SSE (streaming tokens)
POST /webhook/orquestrador             # Chat com orquestrador multi-agente
POST /webhook/orquestrador/stream      # Orquestrador com SSE
POST /webhook/ingestao                 # Ingere texto (JSON body)
POST /webhook/ingestao-arquivo         # Ingere arquivo (.md/.txt multipart)
GET  /health                           # Health check
```

Autenticação: header `X-API-Key` com chave válida em `WEBHOOK_API_KEYS`.

---

## Ferramentas de Cálculo (app/tools/calculations.py)

Para evitar que o LLM faça "conta mental" e erre fórmulas complexas,
agentes 1-6 têm acesso a tools determinísticas:

- **`calcular_expressao`** — avalia expressão aritmética com variáveis (via AST, seguro)
- **`resolver_equacao_linear`** — resolve equação linear para uma incógnita
- Mais 3 tools: inventário de fórmulas, normalização, etc.

O LLM extrai os valores da pergunta, chama a tool com a fórmula, e recebe o
resultado exato. Resolve o problema de LLMs com matemática.

Quando uma fórmula nova entra na base, ela aparece no `docs/formulas_mapeadas.md`
(gerado automaticamente por script).

---

## Estado Atual

### Progresso
- **Código**: 95% (toda infraestrutura pronta, incluindo calculations e quality gate)
- **Dados**: 43% (3 de 7 agentes com documentos ingeridos)
- **Testes**: 85% de acerto E2E nos agentes 0, 3, 4
- **Deploy**: ainda não está em produção

### Bloqueadores
- ~~$5 OpenAI credits~~ ✅ resolvido
- Documentos dos agentes 1, 2, 5, 6 — aguardando curadoria do cliente
- pgvector no Hetzner — pendente (Fabiano trocar imagem Docker)
  - Workaround atual: Supabase como vector store (pode ir para produção assim)

### Decisões de Arquitetura Tomadas
- **Multi-agente mantido** — cliente disse que 80% das perguntas são multi-domínio.
  Considerar arquitetura híbrida (base única com filtros por `agent_id`) na v2
  pós-lançamento, quando tivermos dados reais.
- **Dois bancos** (Supabase + Hetzner) temporariamente. Código preparado para
  migrar para banco único (muda 2 URLs no `.env`).
- **Orquestrador paralelo** (asyncio.gather) — latência = agente mais lento.
- **Fastpath de roteamento** — regras determinísticas antes do LLM. Perguntas
  com termos como "laboratório", "análise", "IN", "RDC" são classificadas sem
  chamar o LLM (mais rápido, 100% preciso para casos óbvios).
- **Quality gate na ingestão** — bloqueia documentos com OCR ruim/texto curto
  para não poluir a base.
- **Anti-duplicação via unique index parcial** — resolve race condition em
  ingestões concorrentes no nível do banco.
- **Sanitização de LaTeX** — respostas do LLM com `\[ x = y \]` são limpas
  antes de retornar (LaTeX não renderiza bem no mobile).
- **Tools de cálculo determinísticas** — evita erro de LLM em fórmulas.
- **Sem staging na ingestão** — texto chega via API já processado.
- **Sem intents** — todas interações são RAG, não há CRUD.

---

## Próximos Passos Priorizados

### Curto prazo (para 85% → 92%+)
1. **Ativar reranking Cohere** — código existe, só ativar `RERANKER=cohere` + `COHERE_API_KEY`
2. **Implementar query rewriting** — LLM reescreve pergunta em 2-3 variações antes de buscar
3. **Adicionar few-shot no prompt do orquestrador** — 10-15 exemplos de classificação

### Médio prazo (deploy)
1. Criar `Dockerfile` + `docker-compose.yml` para Hetzner
2. Configurar Cloudflare DNS (`api.dairyapp.ai` → servidor Hetzner)
3. Testar integração com app React Native
4. Ativar feedback loop (thumbs up/down no app)

### Longo prazo (pós-lançamento)
1. Avaliar migração para arquitetura híbrida (base única com filtros)
2. Parent-child chunking (re-ingestão)
3. Fine-tuning do classificador (após 2-3 meses de dados reais)
4. Expandir inventário de fórmulas em `calculations.py` conforme novos docs

---

## Configurações Importantes (.env)

```bash
# OpenAI (obrigatório)
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Banco (obrigatório)
SUPABASE_DB_URL=postgresql://...
HETZNER_DB_URL=postgresql://...

# RAG
DEFAULT_SEARCH_TYPE=vector          # ou hybrid_rrf
DEFAULT_K=5
MATCH_THRESHOLD=0.3

# Reranking (inativo por padrão)
RERANKER=none                       # Ativar: cohere
COHERE_API_KEY=                     # Necessário se RERANKER=cohere

# HyDE (inativo por padrão)
USE_HYDE=false

# Orquestrador
CLASSIFIER_TEMPERATURE=0
CONSOLIDATION_TEMPERATURE=0.3
DIRECT_TEMPERATURE=0.5
CLASSIFICATION_CACHE_SIZE=256
ORCHESTRATOR_FASTPATH=true
AGENT_TIMEOUT=12

# Quality Gate de Ingestão
INGEST_BLOCK_LOW_QUALITY=true
INGEST_MIN_TEXT_CHARS=400
INGEST_MIN_WORDS=80
INGEST_MAX_GARBLED_RATIO=0.08
INGEST_MIN_QUALITY_SCORE=60

# Autenticação
ENFORCE_WEBHOOK_API_KEY=true
WEBHOOK_API_KEY_HEADER=X-API-Key
WEBHOOK_API_KEYS=chave1,chave2      # Múltiplas chaves aceitas

# Servidor
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
CORS_ALLOW_ORIGINS=*                # Restringir em produção
```

## Infraestrutura

- **Servidor**: Hetzner `manager01` — IP `5.161.236.220`
- **N8N**: rodando em `https://webhooks.dairyapp.ai` (porta 5678)
- **FastAPI**: a ser deployado na porta 8000 no mesmo servidor
- **Domínio**: `dairyapp.ai` (gerenciado via Cloudflare)
- **Supabase**: projeto `aeicuprnutblrbphdxqs`, Session Pooler em `aws-1-us-west-2.pooler.supabase.com`
- **Console de teste**: deployado na Vercel

## Pessoas Envolvidas

- **Matheus** (dev principal)
- **Fabiano** (infraestrutura — Hetzner, Docker, DNS, Cloudflare)
- **Dev mobile** (separado — React Native)
- **Cliente** (fornece documentos para curadoria dos agentes)

---

## Convenções de Código

- Python 3.11+
- Type hints em todas as funções públicas
- Docstrings com propósito, parâmetros e exemplos
- Comentários em português (projeto é 100% em PT-BR)
- Nomes de variáveis em inglês (convenção Python)
- Estrutura modular: `config.py` é fonte única de verdade
- Testes: pytest com markers `phase0`, `phase1`, `phase2`, `phase3`
- SQL idempotente (`CREATE IF NOT EXISTS`, `ON CONFLICT DO UPDATE`)
- Tools com `@tool` decorator do LangChain, usando `ast.parse` em vez de `eval`
- Toda ingestão passa pelo quality gate

## Comandos Úteis (Makefile)

```bash
make setup                 # Cria venv e instala dependências
make db_setup              # Aplica SQL no Supabase e Hetzner
make run                   # Inicia servidor FastAPI (porta 8000)
make run_dev               # Inicia com hot reload

make rag_phase0            # Smoke test de ingestão
make rag_phase1            # Avaliação de retrieval (hit@k + LLM judge)
make rag_phase2_fast       # Compara estratégias (10 queries)
make rag_phase3_fast       # Avaliação E2E do agente (10 queries)
make rag_all_fast          # Todas as fases (modo rápido)

make rag_experiments_fast  # Runner de experimentos (salva CSV/JSON)

make ingest DIR=./docs/agente-1 AGENT=1 TYPE=manual  # Ingere docs
```

---

## Padrões ao Adicionar Código

**Novo agente**: entrada em `AGENTS` (agent_config.py) + prompt em `_AGENT_PROMPTS_COMPACT` (prompts.py) + nova tabela SQL (arquivo novo, incremental). Agente NÃO-0 recebe tools de cálculo automaticamente.

**Nova tool genérica**: criar em `app/tools/`. Usar `@tool` do LangChain. Se for de cálculo, adicionar em `calculations.py` e incluir em `get_calculation_tools()`.

**Nova tool específica de agente**: registrar em `base_agent.py` na construção do grafo daquele agente.

**Novo endpoint**: adicionar em `webapp.py`, seguir padrão dos existentes (auth + Pydantic + async + sanitização de math).

**Nova estratégia de busca**: adicionar em `search.py`, atualizar `search_knowledge_base()` para rotear.

**Mudança no schema SQL**: criar novo arquivo `NN_description.sql` (não editar os anteriores). Garantir idempotência. Considerar blindagem contra race conditions via unique index parcial.

**Nova fórmula na base**: rodar script que atualiza `docs/formulas_mapeadas.md`. O inventário é importante para que as tools saibam quais fórmulas existem.

**Nova regra de roteamento**: adicionar em `_rule_based_route()` no `orchestrator.py`. Fastpath rules são mais rápidas e precisas que LLM para casos óbvios.

**Novo contrato com o front**: documentar em `docs/`. O front tem 4 contratos oficiais (API, ingestão, payload, message service). Alterações no contrato precisam atualizar a documentação.
