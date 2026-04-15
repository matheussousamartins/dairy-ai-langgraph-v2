# DairyApp AI — Versão LangGraph

## Visão Geral

Este projeto é a versão em código (Python) do sistema multi-agente de laticínios.
É funcionalmente equivalente à versão N8N — mesmos 7 endpoints, mesmo contrato de API,
mesma base no Supabase — mas com controle total sobre RAG, reranking, streaming e lógica.

### Por que duas versões?

| Aspecto | N8N | LangGraph |
|---------|-----|-----------|
| Velocidade de setup | Rápido (visual) | Mais lento (código) |
| Controle sobre RAG | Limitado (nós pré-feitos) | Total (funções SQL, reranking, HyDE) |
| RAG Híbrido com RRF | Workaround via HTTP | Nativo (funções SQL) |
| Reranking | Toggle simples | Cohere, customizável |
| Streaming (SSE) | Não suporta | Suporta nativamente |
| Manutenção por não-dev | Fácil (interface visual) | Requer dev Python |
| Escalabilidade | Limitada | Horizontal (FastAPI + workers) |

A versão N8N é o MVP para validação rápida.
A versão LangGraph é o caminho para produção enterprise.

---

## Estrutura de Pastas

```
dairy-ai-langgraph/
│
├── app/                          # Código principal
│   ├── __init__.py
│   ├── config.py                 # Configurações e variáveis de ambiente
│   │
│   ├── agents/                   # Os 6 agentes + orquestrador
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Classe base: cria ReAct executor com RAG tool
│   │   ├── orchestrator.py       # Orquestrador: classifica → roteia → consolida
│   │   ├── prompts.py            # System prompts dos 7 agentes
│   │   └── agent_config.py       # Configuração por agente (tabela, nome, descrição)
│   │
│   ├── rag/                      # Pipeline RAG completo
│   │   ├── __init__.py
│   │   ├── ingest.py             # Ingestão: staging → chunking → embeddings → upsert
│   │   ├── loaders.py            # Splitters: fixed, markdown, semantic
│   │   ├── search.py             # Busca: vetorial, textual, híbrida RRF, HyDE
│   │   └── rerank.py             # Reranking: Cohere (opcional)
│   │
│   ├── server/                   # API HTTP
│   │   ├── __init__.py
│   │   └── webapp.py             # FastAPI com 7 endpoints + ingestão + health
│   │
│   └── db/                       # Banco de dados
│       ├── __init__.py
│       ├── connection.py         # Pool de conexões Postgres
│       └── memory.py             # Chat memory (salvar/carregar histórico)
│
├── sql/                          # Scripts SQL
│   ├── 01_kb_schema.sql          # Tabelas: kb_chunks, kb_docs
│   ├── 02_kb_indexes.sql         # Índices: HNSW (vector), GIN (FTS)
│   ├── 03_kb_functions.sql       # Funções: kb_vector_search, kb_hybrid_search
│   └── 04_app_tables.sql         # Tabelas do app: chat_memories, logs, users
│
├── tests/                        # Testes
│   ├── test_agents.py
│   ├── test_rag.py
│   └── test_api.py
│
├── langgraph.json                # Config do LangGraph Studio
├── requirements.txt              # Dependências Python
├── .env.example                  # Template de variáveis de ambiente
├── Makefile                      # Comandos úteis (setup, test, run)
└── README.md                     # Este arquivo
```

---

## Fluxo de uma Pergunta (passo a passo)

### Cenário: usuário na aba "Queijos" pergunta "Como fabricar mussarela?"

```
1. App envia POST /webhook/agente-1
   Body: { message: "Como fabricar mussarela?", session_id: "user-abc-aba-1" }

2. FastAPI recebe, identifica agente_id=1

3. Carrega histórico do chat_memories (últimas 10 msgs do session_id)

4. Chama o grafo do Agente 1 (ReAct executor):
   a. System prompt do agente queijos é injetado
   b. LLM decide chamar a tool kb_search com query="fabricar mussarela"
   c. kb_search executa:
      - Gera embedding da query via OpenAI
      - Chama kb_hybrid_search no Supabase (busca vetorial + textual + RRF)
      - Retorna top 5 chunks mais relevantes
   d. LLM recebe os chunks como contexto
   e. LLM gera a resposta final baseada nos chunks

5. Salva mensagem + resposta no chat_memories

6. Registra no interaction_logs (agente, tempo, session)

7. Retorna { response: "A mussarela é...", agent_id: 1, agent_name: "Tecnologia de Queijos" }
```

### Cenário: usuário na aba "Geral" pergunta "Qual a legislação para queijo minas?"

```
1. App envia POST /webhook/orquestrador
   Body: { message: "Qual a legislação para queijo minas?", session_id: "user-abc-aba-geral" }

2. FastAPI recebe, identifica como orquestrador

3. Carrega histórico do session_id

4. Chama o grafo do Orquestrador:
   a. Nó classify: LLM classifica a pergunta → domínio "regulatorios" + "queijos"
   b. Nó route: decide chamar Agente 3 (regulatórios) como principal
   c. Nó execute: invoca o sub-grafo do Agente 3
      - Agente 3 faz kb_search na tabela embeddings_agente_3_regulatorios
      - Retorna chunks de legislação relevantes
   d. Nó consolidate: LLM consolida a resposta do sub-agente

5. Salva no chat_memories + interaction_logs

6. Retorna { response: "A IN 30/2001 define...", agent_id: 3, agent_name: "Regulatórios" }
```

---

## Componentes Detalhados

### 1. config.py — Configurações

Centraliza todas as variáveis de ambiente. Cada variável tem um valor padrão
para desenvolvimento, mas DEVE ser configurada em produção.

```python
# O que cada variável controla:
OPENAI_API_KEY          # Chave da OpenAI (embeddings + LLM)
LLM_MODEL               # Modelo do chat (ex: gpt-4o-mini)
EMBEDDING_MODEL         # Modelo de embeddings (text-embedding-3-small)
SUPABASE_DB_URL         # Connection string do Supabase (vector store)
HETZNER_DB_URL          # Connection string do Postgres Hetzner (memory + logs)
DEFAULT_SEARCH_TYPE     # Tipo de busca padrão: "vector", "hybrid", "hybrid_rrf"
DEFAULT_K               # Quantidade de chunks retornados (padrão: 5)
RERANKER                # Reranker: "none" ou "cohere"
COHERE_API_KEY          # Chave do Cohere (se reranker=cohere)
USE_HYDE                # Ativar HyDE (True/False)
```

### 2. agent_config.py — Configuração por Agente

Define os 6 agentes com seus metadados. Cada agente tem:
- ID e nome
- Tabela de embeddings no Supabase
- Descrição para o orquestrador (quando rotear para este agente)
- Parâmetros de busca customizáveis (search_type, k, reranker)

### 3. base_agent.py — Classe Base do Agente

Usa o padrão do `new_react.py` do projeto original: cria um executor ReAct
com uma tool de busca no KB configurada para a tabela específica do agente.

Cada agente é um grafo LangGraph com 3 nós:
- prepare: injeta o system prompt
- agent: chama o LLM com as tools disponíveis
- tools: executa as tools chamadas pelo LLM

O loop ReAct continua até o LLM decidir responder sem chamar mais tools.

### 4. orchestrator.py — Orquestrador

Grafo LangGraph com 4 nós:
- classify: classifica a pergunta em 1+ domínios
- route: decide qual(is) agente(s) chamar
- execute: invoca o sub-grafo do agente escolhido
- consolidate: consolida a resposta (se múltiplos agentes, agrega)

### 5. search.py — Busca RAG

Reutiliza as funções SQL do projeto original:
- kb_vector_search: busca por similaridade de cosseno
- kb_text_search: busca por full-text search (FTS)
- kb_hybrid_search: combina ambas com Reciprocal Rank Fusion (RRF)

A progressão de estratégias é por configuração:
- Fase 1: search_type="vector" (só semântica)
- Fase 2: search_type="hybrid_rrf" (semântica + FTS + RRF)
- Fase 3: reranker="cohere" (adiciona reranking)
- Fase 4: use_hyde=True (adiciona query expansion)

Cada fase é ativada mudando uma variável de ambiente, sem tocar no código.

### 6. webapp.py — Servidor FastAPI

Expõe os mesmos endpoints que o N8N:
- POST /webhook/agente-{1..6} → chama o agente correspondente
- POST /webhook/orquestrador → chama o orquestrador
- POST /webhook/ingestao → ingestão de documentos
- GET /health → status do sistema

O contrato é IDÊNTICO ao do N8N — o app React Native funciona
com qualquer backend sem mudar uma linha.

---

## Diferenças em relação ao projeto original (ai-agent-sales)

| Original (CRM) | Adaptado (Laticínios) |
|----------------|----------------------|
| 1 agente principal + sub-grafos | 6 agentes + 1 orquestrador |
| Intents de CRM (lead_criar, etc.) | Intents de domínio (queijos, regulatórios, etc.) |
| Tools de CRM (create_lead, etc.) | Tools de RAG (kb_search por agente) |
| 1 tabela kb_chunks (filtro por empresa) | 6 tabelas embeddings (1 por agente) |
| gpt-5-nano | Configurável (gpt-4o-mini padrão) |
| Busca por empresa/client_id | Busca por agent_id/table_name |

### O que foi mantido intacto:
- `new_react.py` → executor ReAct genérico
- `loaders.py` → 3 estratégias de chunking
- `search.py` (ex tools.py) → busca vetorial/textual/híbrida + reranking + HyDE
- `sql/kb/03_functions.sql` → funções SQL de busca
- `ingest.py` → pipeline de ingestão staging → chunks → embeddings

---

## Como Rodar

```bash
# 1. Clonar e instalar
git clone <repo>
cd dairy-ai-langgraph
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\Activate.ps1 no Windows
pip install -r requirements.txt

# 2. Configurar
cp .env.example .env
# Editar .env com suas chaves (OpenAI, Supabase, Hetzner)

# 3. Criar tabelas no banco
psql $SUPABASE_DB_URL < sql/01_kb_schema.sql
psql $SUPABASE_DB_URL < sql/02_kb_indexes.sql
psql $SUPABASE_DB_URL < sql/03_kb_functions.sql
psql $HETZNER_DB_URL  < sql/04_app_tables.sql

# 4. Ingerir documentos (exemplo)
python -m app.rag.ingest --dir ./docs/agente-1-queijos --agent-id 1

# 5. Rodar servidor
uvicorn app.server.webapp:app --host 0.0.0.0 --port 8000

# 6. Testar
curl -X POST http://localhost:8000/webhook/agente-1 \
  -H "Content-Type: application/json" \
  -d '{"message": "Como fabricar mussarela?", "session_id": "teste-1"}'
```

---

## Compatibilidade com N8N

As duas versões (N8N e LangGraph) são intercambiáveis:
- Mesmos endpoints (POST /webhook/agente-{1..6}, POST /webhook/orquestrador)
- Mesmo formato de request (message, session_id, user_profile)
- Mesmo formato de response (response, agent_id, agent_name)
- Mesma base de dados (Supabase para embeddings, Hetzner para memory/logs)

O app React Native não precisa saber qual backend está rodando.
Para trocar, basta mudar a Base URL de `webhooks.dairyapp.ai` para o
endereço do servidor FastAPI.
