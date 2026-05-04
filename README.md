# DairyApp AI — Sistema Multi-Agente com LangGraph

> Console interno de agentes de IA especializados em tecnologia de laticínios.  
> Backend: FastAPI + LangGraph · Frontend: Next.js 15 · RAG: Supabase pgvector

---

## Visão Geral

Sistema multi-agente com **7 especialistas** e um **orquestrador** que classifica cada pergunta, consulta os agentes relevantes em paralelo e consolida a resposta. Inclui console web (Next.js) para uso interno do cliente.

### Agentes

| ID | Nome | Domínio |
|----|------|---------|
| 0 | Base Geral Dairy | Glossário, produtos, fabricantes, ingredientes, distribuidores |
| 1 | Tecnologia de Queijos | Fabricação, maturação, defeitos, cura |
| 2 | Fermentados e Culturas | Iogurtes, kefir, culturas starters, fermentação |
| 3 | Regulatórios por País | Legislação, normas, INs, RDCs |
| 4 | Qualidade do Leite | CCS, CBT, adulteration, plataforma |
| 5 | Diagnóstico de Defeitos | Análise sensorial, causas, soluções |
| 6 | Formulação e P&D | Desenvolvimento de produtos, ingredientes funcionais |

### Orquestrador

- **Agentes 0 e 3 são sempre consultados** para qualquer pergunta de laticínios
- Classifica a pergunta e adiciona o especialista relevante
- Executa todos os agentes **em paralelo** (`asyncio.gather`)
- Latência total = tempo do agente mais lento (não a soma)
- Consolida as respostas em uma única resposta coerente

---

## Arquitetura

```
Pergunta do usuário
      │
      ▼
  [classify]  ──── regras rápidas (ORCHESTRATOR_FASTPATH=true)
      │              └── se não resolver → LLM classifier
      ▼
  [route]
   ├── off-topic/saudação → [respond_direct] → [consolidate] → END
   └── laticínios         → [execute]        → [consolidate] → END
                               │
                    asyncio.gather (paralelo)
                    ├── Agente 0 (sempre)
                    ├── Agente 3 (sempre)
                    └── Agente X (se domínio específico)
```

**Por agente (fluxo interno):**
```
[prepare] → força tool call direto (sem LLM de decisão)
[tools]   → kb_search no Supabase (pgvector)
[agent]   → LLM gera resposta com os chunks
```

---

## Stack

### Backend
- **Python 3.11+**
- **FastAPI** — API HTTP com streaming SSE
- **LangGraph** — orquestração de grafos de agentes
- **LangChain OpenAI** — integração GPT-4o-mini
- **Supabase (pgvector)** — vector store para RAG
- **PostgreSQL (Hetzner)** — memória de chat e logs

### Frontend (Console Interno)
- **Next.js 15** + **React 19** + **TypeScript**
- **Tailwind CSS** — estilização
- **SSE streaming** — respostas em tempo real estilo GPT
- **Autenticação por passkey** — acesso restrito

---

## Estrutura de Pastas

```
dairy-ai-langgraph-v2/
│
├── app/
│   ├── config.py              # Todas as variáveis de ambiente
│   ├── agents/
│   │   ├── base_agent.py      # Grafo base: prepare → tools → agent
│   │   ├── orchestrator.py    # Orquestrador paralelo com cache de classificação
│   │   ├── prompts.py         # System prompts dos 7 agentes
│   │   └── agent_config.py    # Configuração por agente (tabela, keywords, search)
│   ├── rag/
│   │   ├── search.py          # Busca: vetorial, textual, híbrida RRF, HyDE
│   │   ├── ingest.py          # Ingestão: chunking → embeddings → upsert
│   │   ├── loaders.py         # Estratégias de chunking: fixed, markdown, semantic
│   │   └── rerank.py          # Reranking Cohere (opcional)
│   ├── server/
│   │   └── webapp.py          # FastAPI: endpoints REST + SSE streaming
│   └── db/
│       ├── connection.py      # Pool de conexões Postgres
│       └── memory.py          # Chat memory (carregar/salvar histórico)
│
├── frontend/                  # Console Next.js
│   └── src/
│       ├── app/               # Routes Next.js (App Router)
│       ├── components/        # UI: ChatPane, TopBar, Sidebar, HistoryList
│       ├── state/             # useGenesisUI (estado global + streaming)
│       └── lib/               # dairy-backend, agent-catalog, thread-store
│
├── sql/                       # Scripts SQL
│   ├── 01_kb_schema.sql
│   ├── 02_kb_indexes.sql      # HNSW (vector) + GIN (FTS)
│   ├── 03_kb_functions.sql    # kb_vector_search, kb_hybrid_search
│   └── 04_app_tables.sql      # chat_memories, interaction_logs
│
├── tests/
├── docs/
├── langgraph.json             # Config LangGraph Studio
├── requirements.txt
└── .env.example
```

---

## Configuração

```bash
# 1. Clonar e criar ambiente virtual
git clone https://github.com/matheussousamartins/dairy-ai-langgraph-v2.git
cd dairy-ai-langgraph-v2
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 2. Configurar variáveis de ambiente
cp .env.example .env
# Preencher .env com as chaves abaixo
```

### Variáveis do `.env`

```env
# OpenAI
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

# Temperaturas
AGENT_TEMPERATURE=0.3
CONSOLIDATION_TEMPERATURE=0.3
DIRECT_TEMPERATURE=0.5
CLASSIFIER_TEMPERATURE=0

# Bancos de dados
SUPABASE_DB_URL=postgresql://...
HETZNER_DB_URL=postgresql://...
SUPABASE_DB_PREPARE_THRESHOLD=none
HETZNER_DB_PREPARE_THRESHOLD=none

# RAG
DEFAULT_SEARCH_TYPE=vector
DEFAULT_K=5
RERANKER=none

# Orquestrador
ORCHESTRATOR_FASTPATH=true
CLASSIFICATION_CACHE_SIZE=256
AGENT_TIMEOUT=12

# Frontend (Next.js)
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
CONSOLE_PASSKEY=sua-senha-aqui
```

---

## Rodando Localmente

### Backend

```bash
uvicorn app.server.webapp:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Acesse http://localhost:3000
```

---

## Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/webhook/agente-{0..6}` | Chama agente específico (síncrono) |
| POST | `/webhook/agente-{0..6}/stream` | Streaming SSE por agente |
| POST | `/webhook/orquestrador` | Orquestrador multi-agente (síncrono) |
| POST | `/webhook/orquestrador/stream` | Orquestrador com streaming SSE |
| POST | `/webhook/ingestao` | Ingestão de texto |
| POST | `/webhook/ingestao-arquivo` | Ingestão via upload `.md`/`.txt` |
| GET | `/health` | Status do sistema e bancos |

**Request padrão:**
```json
{
  "message": "Qual o limite de coliformes para queijo minas frescal?",
  "session_id": "user-abc-123",
  "user_profile": { "knowledgeLevel": "INTERMEDIATE", "role": "técnico" }
}
```

**Response padrão:**
```json
{
  "response": "Conforme a IN 60/2019...",
  "agent_id": 3,
  "agent_name": "Regulatórios por País"
}
```

---

## Pipeline RAG

```
Documento → chunking (fixed/markdown/semantic)
          → embeddings (text-embedding-3-small)
          → upsert Supabase (pgvector)

Pergunta  → embedding da query
          → kb_hybrid_search (vector + FTS + RRF)  [opcional: HyDE]
          → top K chunks                            [opcional: Cohere rerank]
          → LLM gera resposta com contexto
```

**Progressão de estratégias (via `.env`, sem alterar código):**

| Fase | Configuração | Quando usar |
|------|-------------|-------------|
| 1 | `DEFAULT_SEARCH_TYPE=vector` | Início, boa base |
| 2 | `DEFAULT_SEARCH_TYPE=hybrid_rrf` | Melhor para termos exatos (legislação) |
| 3 | `RERANKER=cohere` | Mais precisão, mais custo |
| 4 | `USE_HYDE=true` | Perguntas vagas ou mal formuladas |
