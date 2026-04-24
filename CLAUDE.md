# DairyApp AI - Handoff Completo (Claude Code)

Última atualização: 2026-04-23
Escopo: estado atual do projeto inteiro + evolução do orquestrador + benchmark + próximos passos de produção.

---

## 1) Resumo Executivo

- Projeto: backend multiagente para laticínios (FastAPI + LangGraph + RAG em Supabase pgvector).
- Situação geral: base técnica sólida, orquestrador já profissionalizado em várias camadas, ainda com ajustes finais de precisão/latência para go-live robusto.
- Estado de conhecimento:
  - Agentes 0, 1, 2, 3, 4 com base e dataset de avaliação.
  - Agentes 5 e 6 sem base de conhecimento completa (impacta roteamento e benchmark).
- Principal frente atual: elevar qualidade de roteamento (especialmente confusões 1/2/3/4) e estabilizar latência sob carga real.

---

## 2) Arquitetura Atual (Produção)

- Backend: `FastAPI` + `LangGraph`
- LLM: configurável por `.env` (`LLM_MODEL`)
- Embeddings: `text-embedding-3-small`
- Vector Store: Supabase
- Memória/logs: Postgres operacional (mesma estrutura de `chat_memories`, `interaction_logs`, `routing_logs`)

Fluxo de alto nível:
1. `classify` (fast-path + classificador estruturado)
2. `route` (plano por bucket de confiança)
3. `execute` (multiagente paralelo)
4. `fallback_reclassify` (2ª passada quando aplicável)
5. `consolidate` (consolidação com guardrails de evidência)

---

## 3) Estado dos Agentes e Dados

Pastas de documentos:
- `docs/agente-0-base-geral/...`
- `docs/agente-1-queijos/...`
- `docs/agente-2-fermentados/...`
- `docs/agente-3-regulatorios/...`
- `docs/agente-4-qualidade-leite/...`
- `docs/agente-5-defeitos/...`
- `docs/agente-6-formulacao/...`

Dataset de roteamento:
- Arquivo: `tests/fixtures/rag/rag_queries.yaml`
- Total de queries válidas: **1184**
- Distribuição:
  - agent 0: 34
  - agent 1: 209
  - agent 2: 228
  - agent 3: 399
  - agent 4: 314

Observação:
- Atualmente o benchmark oficial usa agentes esperados `[0,1,2,3,4]`.
- Sem base real para 5/6, porém eles podem aparecer como confusão do classificador em perguntas ambíguas.

---

## 4) O que já foi implementado no Orquestrador (melhorias reais)

Arquivo principal: `app/agents/orchestrator.py`

### 4.1 Classificação e planejamento
- Fast-path determinístico reforçado (regras de alta precisão).
- Classificador estruturado com:
  - `agent_ids`
  - `confidence`
  - `reason`
  - `alternatives`
- Recalibração de confiança e mapeamento para bucket:
  - high / medium / low
- Planner de execução por bucket (`execution_plan`).
- Seleção de `primary_agent_id` mais orientada ao domínio técnico.

### 4.2 Guardrails e desambiguação
- Guardrails de domínio para reduzir competição indevida entre especialistas.
- Regras curtas para intents críticas (ex.: dornic/acidez, iogurte/fermentação, pizza/browning, rotulagem).
- Preferência por resposta factual de especialista em perguntas objetivas.

### 4.3 Fallback de roteamento e evidência
- `fallback_reclassify` formal (segunda passada com vizinhança de domínios).
- Coleta de candidatos de fallback baseada no plano/chosen atual.
- Critérios para disparo de fallback por evidência fraca/conflito.

### 4.4 Fallback em base geral (feature flag)
- Camada de fallback em índice geral multi-tabela (última camada interna de KB):
  - `ENABLE_GENERAL_INDEX_FALLBACK`
  - parâmetros de k/tabelas/critério fraco no `config.py`.

### 4.5 Fallback web (feature flag + whitelist)
- Implementado fallback final na web com domínios permitidos:
  - `app/tools/web_fallback.py`
  - provider atual: DuckDuckGo HTML
  - whitelist de domínios confiáveis
  - fonte citada na resposta quando fallback web é usado.
- Flags:
  - `ENABLE_WEB_FALLBACK`
  - `WEB_FALLBACK_ALLOWED_DOMAINS`
  - `WEB_FALLBACK_REQUIRE_GENERAL_FALLBACK_FIRST`
  - `WEB_FALLBACK_ONLY_ON_WEAK`
  - `WEB_FALLBACK_REQUIRE_DAIRY_SIGNAL`

### 4.6 Observabilidade de roteamento
- Tabela `routing_logs` e view `v_routing_quality_daily` no SQL:
  - arquivo: `sql/03_app_tables.sql`
- Persistência estruturada em `app/db/memory.py` (`save_routing_log`):
  - confidence, bucket, chosen, execution_plan, fallback trigger, etc.

### 4.7 Hints especialistas derivados do dataset real
- Gerador: `scripts/build_routing_specialist_hints.py`
- Saída: `docs/orchestrator/day1/ROUTING_SPECIALIST_HINTS.yaml`
- Loader no orquestrador via `_load_specialist_strong_hints()`

### 4.8 Benchmark dedicado de roteamento
- Script: `scripts/benchmark_routing.py`
- Métricas:
  - Routing@1
  - Routing@3
  - fallback_rate
  - p50/p95/p99
  - confusions
- Artefatos:
  - `docs/orchestrator/day1/routing_benchmark_latest.json`
  - `docs/orchestrator/day1/routing_benchmark_quota_check.json`
  - `docs/orchestrator/day1/routing_benchmark_smoke_after_patch.json`

---

## 5) Resultados de Benchmark (estado atual)

Arquivo de referência atual: `docs/orchestrator/day1/routing_benchmark_latest.json`

Resumo:
- `total_cases`: 100
- `routing_at_1`: **61%**
- `routing_at_3`: **77%**
- `fallback_rate`: **15%**
- `latency_p50`: ~13.8s
- `latency_p95`: ~37.7s
- `latency_p99`: ~39.4s

Confusões dominantes:
- `3 -> 1`
- `1 -> 5`
- `0 -> 1`
- `1 -> 4`
- `2 -> 4`

Leitura correta:
- Qualidade de roteamento evoluiu bastante vs início.
- Ainda abaixo da régua final de produção premium.
- Latência ficou degradada por gargalo de conexão com pool do Supabase em algumas rodadas.

---

## 6) Incidentes recentes e mitigação aplicada

### 6.1 Saturação de conexão Supabase (EMAXCONNSESSION)
Sintoma:
- `max clients reached in session mode - pool_size: 15`
- conexões BAD, reconexão longa, p95 inflado.

Mitigação aplicada no código:
- `app/config.py`:
  - novos parâmetros de pool por ambiente.
- `app/db/connection.py`:
  - `min_size/max_size` configuráveis
  - `timeout/reconnect_timeout` configuráveis
  - `connect_timeout` por conexão
- `.env.example` atualizado com defaults conservadores.

Config recomendada:
- `SUPABASE_DB_POOL_MIN_SIZE=1`
- `SUPABASE_DB_POOL_MAX_SIZE=3` (ou 4)
- `SUPABASE_DB_POOL_TIMEOUT_SEC=12`
- `SUPABASE_DB_POOL_RECONNECT_TIMEOUT_SEC=30`
- `SUPABASE_DB_CONNECT_TIMEOUT_SEC=8`

---

## 7) Percentual de Evolução (estimativa operacional)

### Orquestrador (roteamento + execução + observabilidade)
- **~80%** pronto para produção.

Justificativa:
- Já possui arquitetura profissional (confidence/buckets/planner/fallback/logs/benchmark/flags).
- Falta fechar precisão e latência em nível de go-live premium.

### Projeto como um todo
- **~83%**.

Justificativa:
- Backend e contratos bem avançados.
- Base de conhecimento e calibração final ainda pendentes para 5/6 e ajustes finais de roteamento.

---

## 8) O que falta para “go-live robusto”

Prioridade P0:
1. Estabilizar benchmark sem saturação de pool (infra + execução).
2. Subir `Routing@1` para faixa alvo com pacote cirúrgico nas confusões (1/2/3/4).
3. Reduzir `p95` real (além de otimizar roteamento, evitar gargalos de DB/rede).

Prioridade P1:
1. Completar base de conhecimento dos agentes 5 e 6.
2. Recalibrar hints/few-shots com novos dados e rodar benchmark comparativo.
3. Definir política final de web fallback em produção (ligado/desligado por tenant e guardrails).

Prioridade P2:
1. Expandir views de observabilidade (ex.: taxa de fallback por trigger).
2. Rodar homologação formal com template em `docs/homologacao`.

---

## 9) Onde paramos exatamente

Última etapa concluída:
- Aplicadas melhorias finais cirúrgicas de roteamento + fallback + pool config.
- Usuário executou benchmark de 100 casos e reportou:
  - `R@1 61%`, `R@3 77%`, `fallback 15%`, `p95 ~37.7s`
  - com logs de saturação de conexão no Supabase durante execução.

Próxima ação imediata sugerida:
1. Ajustar `.env` com pool conservador.
2. Rodar benchmark limpo (idealmente com API parada para não competir por conexões).
3. Comparar com `routing_benchmark_latest.json`.
4. Aplicar ajustes finais de confusão por eixo:
   - `3<->1`, `1->5`, `2->4`, `0->1`.

---

## 10) Arquivos-chave para continuidade (Claude)

- Orquestrador:
  - `app/agents/orchestrator.py`
- Busca/fallback:
  - `app/rag/search.py`
  - `app/tools/web_fallback.py`
- Config:
  - `app/config.py`
  - `.env.example`
- Conexão DB:
  - `app/db/connection.py`
  - `app/db/memory.py`
- SQL observabilidade:
  - `sql/03_app_tables.sql`
- Benchmark e tuning:
  - `scripts/benchmark_routing.py`
  - `scripts/build_routing_specialist_hints.py`
  - `docs/orchestrator/day1/ROUTING_SPECIALIST_HINTS.yaml`
  - `docs/orchestrator/day1/routing_benchmark_latest.json`
  - `docs/orchestrator/day1/ROUTING_CONFIDENCE_POLICY.md`
  - `docs/orchestrator/day1/AGENT_ROUTING_TAXONOMY.yaml`

---

## 11) Estado de Git no momento deste handoff

Último commit relevante:
- `c73b9b5 feat: improve orchestrator routing with confidence planner, fallback and structured routing logs`

Há mudanças locais importantes não commitadas nesta etapa (incluindo ajustes mais novos de orquestrador/benchmark/fallback/pool). Validar `git status` antes de publicar.

