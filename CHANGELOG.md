# DairyApp AI — Histórico de Mudanças

Documento de referência: arquitetura, decisões técnicas e evolução do sistema desde o MVP até o estado atual.

---

## Ponto de partida: versão GitHub (MVP)

**Repositório:** `matheussousamartins/dairy-ai-langgraph-v2`

Orquestrador monolítico em arquivo único (~600 linhas). Sem módulos separados, sem controle de qualidade de resposta, sem filtragem determinística de evidências.

**Grafo original:**
```
classify → execute (paralelo) → consolidate → END
        ↘ respond_direct ────────────────────↗
```

**Agentes ativos:** 1 (Queijos) e 3 (Regulatórios). Agentes 0, 2, 4, 5, 6 sem KB carregada.

**Comportamento do consolidador (MVP):** LLM recebia toda a evidência bruta dos agentes e fundia livremente. Simples e previsível — menos código, menos superfície para bugs.

---

## Por que as respostas pioraram após as primeiras expansões

Durante as expansões de robustez foram introduzidos filtros de evidência (specialist_rag, _build_factual_response_candidates, fast-paths de retorno direto) que causaram dois problemas encadeados:

**Bug 1 — Agente 3 descartado silenciosamente**
Quando o especialista (agente 1) tinha RAG real e o regulatório (agente 3) respondia via [CONHECIMENTO GERAL] (sem KB para aquela query), o filtro descartava a resposta do agente 3 porque `answer_source = "general_knowledge"`. O agente 3 sumia do `successful` sem nenhum aviso.

**Bug 2 — Fast-path bypassa o consolidador LLM**
Com o agente 3 dropado, `successful` ficava com 1 item. O fast-path `len(successful) == 1` então retornava a resposta do especialista direto — sem consolidação LLM, sem complemento regulatório.

**Bug 3 — LLM misturava chunks com memória de treinamento**
O LLM recebia chunks do Parmesão com "16-18°C" mas respondia "10-18°C" sem usar [CONHECIMENTO GERAL] — simplesmente misturava valores corretos da KB com memória de treinamento. Todos os filtros de detecção de tag eram inúteis para esse caso porque a tag nunca aparecia.

**Causa raiz dos bugs 1 e 2:** fast-paths de retorno antecipado que bypasavam o LLM de consolidação. A versão MVP não tinha esses fast-paths — passava tudo pelo consolidador.

**Causa raiz do bug 3:** instrução de ancoragem insuficiente no agente. O system prompt permitia [CONHECIMENTO GERAL] em certos contextos, mas o LLM ocasionalmente ignorava a tag e misturava sem sinalizar.

---

## Fixes aplicados (em ordem cronológica)

### Fix 1 — Circuit breaker não conta timeout de infra como falha de agente

**Arquivo:** `app/resilience.py`

**Problema:** Timeouts causados por pool do Supabase saturado em testes consecutivos abriam o circuit breaker do agente 1 após 3 falhas, bloqueando-o por 60 segundos. O agente nunca havia falhado — foi o banco que demorou.

**Fix:** `record_failure(is_infra_timeout=True)` — timeouts de rede/pool não contam para o threshold. Só erros reais do agente (exception do LLM, código quebrado) abrem o circuito.

**Em `orchestrator.py`:** `call_one` agora passa `is_infra_timeout=True` explicitamente no `asyncio.TimeoutError`.

---

### Fix 2 — Ancoragem obrigatória no agente quando há chunks RAG

**Arquivo:** `app/agents/base_agent.py` — função `call_model`

**Problema:** O LLM recebia chunks via ToolMessage mas misturava com conhecimento de treinamento sem usar [CONHECIMENTO GERAL]. Valores errados apareciam na resposta.

**Fix:** Quando `_extract_tool_results()` retorna chunks reais, injeta um `SystemMessage` de ancoragem explícita imediatamente antes do LLM responder:

```python
anchoring = (
    "\n\nATENCAO — ANCORAGEM OBRIGATORIA: use EXCLUSIVAMENTE os trechos "
    "recuperados abaixo. Nao use conhecimento geral nem complete com memoria "
    "de treinamento..."
    f"Trechos recuperados:\n{chunk_summary}"
)
prompt_text = prompt_text + anchoring
```

O LLM vê os chunks duas vezes: via ToolMessage (estrutura LangGraph) e via SystemMessage direto no contexto. Isso elimina a ambiguidade que causava mistura.

---

### Fix 3 — Remoção de todos os fast-paths de retorno antecipado no consolidador

**Arquivo:** `app/agents/orchestrator.py` — função `consolidate`

**Problema:** Os fast-paths `len(successful) == 1` retornavam a resposta do especialista direto, bypassando o LLM de consolidação e descartando o complemento regulatório.

**Fix:** O `consolidate` atual não tem nenhum retorno antecipado com evidência. Todo caminho com evidência passa pelo LLM de síntese via `_build_consolidation_prompt`. Só há retorno antecipado nos dois casos extremos:
- `successful` vazio → `_build_web_last_resort_response`
- `_factual_candidates` vazio após filtragem → `_build_web_last_resort_response`

---

### Fix 4 — Preservação do agente regulatório mesmo sem RAG

**Arquivo:** `app/agents/orchestrator.py` — filtro `specialist_rag`

**Problema:** O agente 3 era descartado quando usava [CONHECIMENTO GERAL], deixando o consolidador sem o complemento regulatório.

**Fix atual no código:**
```python
specialist_rag = [
    r for r in specialist_candidates
    if r.get("answer_source") in {"rag", "rag_evidence", "raw_fallback"}
]
if specialist_rag:
    specialist_candidates = specialist_rag
# regulatory_candidate é separado antes desta filtragem — preservado sempre
```

O `regulatory_candidate` é extraído antes do filtro `specialist_rag` e nunca é descartado por esse filtro.

---

## Arquitetura atual (estado do código)

### Grafo V1 — Orquestrador multi-agente

```
classify → execute (paralelo) → consolidate → END
        ↘ respond_direct ────────────────────↗
```

4 nós. Sem `ask_clarification`, sem `fallback_reclassify`, sem edges condicionais após `execute`.

**Pior caso: 4 chamadas LLM** — classifier + agente 1 + agente 3 + consolidação.

### Módulos do orquestrador

| Arquivo | Responsabilidade |
|---|---|
| `orchestrator.py` | Nós do grafo, consolidação, web fallback last-resort |
| `orch_schema.py` | `OrchestratorState`, `ClassificationResult` |
| `orch_text.py` | Normalização, extração factual, detecção de incerteza — sem deps LangChain |
| `orch_signals.py` | Sinais de intenção por agente, padrões regulatórios, detecção de saudação |
| `orch_routing.py` | Fast-path rule-based, guardrails de domínio, cache LRU, confidence bucketing |
| `orch_models.py` | Lazy init de modelos com floor constraints (gpt-4o mínimo para consolidação) |
| `orch_quality.py` | 7 tipos de pergunta, instruções de formato por tipo, `ResponseQuality`, strip de tags |
| `orch_fewshot.py` | Bloco de few-shot injetado no classificador |
| `orch_warmup.py` | Pré-carregamento de cache na inicialização |
| `evidence_reducer.py` | Redução determinística de chunks: filtra por relevância, detecta perguntas métricas |
| `prompts.py` | System prompts dos agentes 1 e 3 + prompt do classificador |

### Consolidador atual — fluxo determinístico

```
successful (agentes bem-sucedidos)
    ↓
_build_factual_response_candidates()   — filtra: fora-de-escopo, sem overlap lexical
    ↓
_build_local_primary_regulatory_evidence()  — injeta fatos legais críticos (IN 76 CCS/CBT)
    ↓
raw_fallback                           — se filtros não produziram candidatos, usa texto bruto
    ↓
se ainda vazio → _build_web_last_resort_response()
    ↓
separa specialist_candidates vs regulatory_candidate
    ↓
_build_consolidation_prompt()          — prompt hierárquico por tipo de pergunta (7 tipos)
    ↓
LLM de síntese (sempre)
    ↓
_postprocess_consolidated_answer()     — strip de tags, fórmula Dornic, deduplicação
    ↓
_append_missing_regulatory_numeric_complement()  — garante que limites numéricos não somem
    ↓
retorno final
```

### Prompt de síntese — regras invariantes (R1–R9)

O `_build_consolidation_prompt` tem 9 regras base aplicadas em todos os caminhos:

- **R1** — Ancoragem total: proibido completar com memória de treinamento
- **R2** — Preservação numérica INCONDICIONAL: todo número das evidências vai na resposta
- **R3** — Técnico e legal coexistem: apresenta ambos quando presentes, distinguidos claramente
- **R4** — Completude sem excesso: sem ressalvas genéricas ou parágrafos não solicitados
- **R5** — Prosa coesa: sem headers de seção, bullets soltos ou metadados de chunk
- **R6** — Identidade invisível: sem menções a agentes, ferramentas ou bases internas
- **R7** — Tom técnico direto sem hedging ("parece que", "provavelmente")
- **R8** — Encerramento limpo no último dado técnico
- **R9** — Proibição de falsa ausência: nunca escreve "não há especificação" quando evidência tem número

Formato varia por tipo de pergunta: FACTUAL_SHORT (1-3 frases), REGULATORY (norma + artigo), PROCESS (etapas numeradas com parâmetros), TROUBLESHOOTING (DEFEITO/CAUSA/AÇÃO), COMPARATIVE (tabela simétrica), CALCULATIVE (fórmula → resultado → unidade), GENERAL (prosa técnica).

---

### Grafo V2 — Single-Agent Pipeline

**Controlado por:** `RAG_ARCHITECTURE=single_agent` no `.env`

```
analyze_query → retrieve_context → generate_answer → validate_response → END
     ↓ (saudação)
validate_response → END
```

**Pior caso: 2 chamadas LLM** — classifier LLM (apenas quando keyword é ambígua) + generate_answer.

| Nó | Responsabilidade |
|---|---|
| `analyze_query` | Classifica intenção via keyword/few-shot, resolve anáfora. Saudações retornam direto. |
| `retrieve_context` | Busca em múltiplas tabelas em paralelo, embedding pré-computado uma vez. Busca regulatória complementar com threshold de score mínimo. |
| `generate_answer` | Uma chamada LLM com todos os chunks consolidados. |
| `validate_response` | Strip de frases proibidas + pós-processamento + classificação de qualidade. |

### Comparação V1 vs V2

| Aspecto | V1 Orquestrador | V2 Single-Agent |
|---|---|---|
| Chamadas LLM típicas | 3–4 | 1–2 |
| Chamadas LLM pior caso | 4 | 2 |
| Especialização por domínio | 2 agentes especializados | 1 agente generalista |
| Complemento regulatório | Agente 3 sempre presente | Busca regulatória complementar com score mínimo |
| Hierarquia técnico/regulatório | Explícita no prompt de síntese | Única chamada LLM integra tudo |
| Contexto conversacional | Anáfora via `contextualize_query_for_rag` | Idem |
| Qualidade em perguntas complexas | Alta (2 especialistas + síntese hierárquica) | Boa |
| Latência estimada | 4–8s | 2–4s |

---

## Outros componentes relevantes

### `app/agents/base_agent.py`
- Grafo ReAct individual por agente: `prepare_search` → `kb_search` (ToolNode) → `call_model`
- Máximo 2 buscas por agente (1 forçada + 1 retry voluntário)
- Ancoragem obrigatória injetada em `call_model` quando há chunks RAG reais
- Agente 3: se resposta insuficiente, tenta `_build_regulatory_general_rule_answer` como fallback interno

### `app/rag/search.py`
- `contextualize_query_for_rag`: resolve anáfora em follow-ups antes de embedar
- Reranker cross-encoder após busca vetorial
- Suporte a hybrid_rrf, semantic, text

### `app/rag/conversation_resolver.py`
- Resolução de anáfora conservadora: só reescreve quando há referente explícito no contexto

### `app/resilience.py`
- Circuit breaker por agente com distinção infra-timeout vs falha real
- Timeout adaptativo por agente (aprende latências históricas)

### `app/observability.py`
- Logging estruturado com trace IDs, NodeTimer, LLMSlot
- Todos os nós emitem eventos com métricas de performance

---

## Variáveis de configuração críticas

```env
# Arquitetura ativa
RAG_ARCHITECTURE=orchestrator          # ou: single_agent

# V1 — Circuit breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD=3    # falhas reais (não infra) para abrir circuito
CIRCUIT_BREAKER_RECOVERY_SEC=60        # tempo de recuperação

# V1 — Timeouts de agente
AGENT_TIMEOUT=18                       # timeout base em segundos
AGENT_TIMEOUT_MAX_SEC=45
AGENT_TIMEOUT_MIN_SEC=8

# Pool do banco
SUPABASE_DB_POOL_MAX_SIZE=4            # aumentar em produção (mínimo 8)
SUPABASE_DB_POOL_TIMEOUT_SEC=12

# V2 — Single-agent
SINGLE_AGENT_MAX_TABLES=2
SINGLE_AGENT_K_PER_TABLE=5
SINGLE_AGENT_SEARCH_TYPE=hybrid_rrf
SINGLE_AGENT_REGULATORY_K=2
SINGLE_AGENT_REGULATORY_MIN_SCORE=0.015

# Web fallback (V1)
ENABLE_WEB_FALLBACK=true
```

---

## Lições aprendidas

**Fast-paths de retorno antecipado são armadilhas.** O MVP não tinha — passava tudo pelo LLM de consolidação. As adições de "otimização" (retornar resposta de 1 agente direto) eliminaram o complemento regulatório e criaram respostas incompletas. A versão atual remove todos esses atalhos.

**Filtragem determinística de evidências precisa de fallback raw.** Quando os filtros lexicais descartam tudo (query ambígua, KB esparsa), o sistema deve usar o texto bruto do agente como candidato — o LLM de síntese é mais tolerante a ruído do que os filtros heurísticos.

**POOL_MAX_SIZE=4 é insuficiente para testes intensivos.** Com 2 agentes em paralelo por request e sessões de teste consecutivas, o pool satura. Em produção recomenda-se mínimo 8–12 conexões.
