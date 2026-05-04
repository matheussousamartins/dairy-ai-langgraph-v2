# Contrato API - Mobile/Web (Integracao)

Data: 2026-04-15  
Base URL (local): `http://192.168.1.31:8000`

## Autenticacao

Header obrigatorio em todos os endpoints `POST /webhook/*`:

- `X-API-Key: <WEBHOOK_API_KEY>`

---

## 1) Chat principal (recomendado)

### Endpoint

- `POST /webhook/orquestrador/stream`

### Request

Headers:
- `Content-Type: application/json`
- `X-API-Key: <chave>`

Body:

```json
{
  "message": "texto da pergunta",
  "session_id": "mobile-session-001",
  "user_profile": {
    "knowledgeLevel": "INTERMEDIATE",
    "role": "tecnico"
  }
}
```

Observacoes:
- `user_profile` e opcional.

### Response (SSE)

`Content-Type: text/event-stream`

Eventos emitidos em `data: {...}`:
- `{"event":"chunk","text":"..."}`
- `{"event":"final","agent_id":4,"agent_name":"Qualidade do Leite"}`
- `{"event":"error","detail":"..."}`

---

## 2) Chat por agente especifico (opcional)

Use quando quiser forcar um especialista em vez do orquestrador.

Endpoints:
- `POST /webhook/agente-1/stream` (Tecnologia de Queijos)
- `POST /webhook/agente-2/stream` (Fermentados e Culturas)
- `POST /webhook/agente-3/stream` (Regulatorios)
- `POST /webhook/agente-4/stream` (Qualidade do Leite)
- `POST /webhook/agente-5/stream` (Diagnostico de Defeitos)
- `POST /webhook/agente-6/stream` (Formulacao e P&D)

Observacao:
- `POST /webhook/agente-0/stream` nao e necessario para o fluxo normal.

Payload e eventos SSE:
- Mesmos do `POST /webhook/orquestrador/stream`.

---

## 3) Ingestao via arquivo (web)

### Endpoint

- `POST /webhook/ingestao-arquivo`

### Request

Headers:
- `X-API-Key: <chave>`
- `Content-Type: multipart/form-data`

Campos `form-data`:
- `file` (obrigatorio): arquivo `.md` ou `.txt` em UTF-8
- `agent_id` (obrigatorio): inteiro de `0` a `6`
- `doc_type` (opcional): default `manual`
- `table_name` (recomendado): deve corresponder ao `agent_id`

### Response 200 (sucesso)

```json
{
  "success": true,
  "chunks_created": 45,
  "chunks_processed": 45,
  "chunks_inserted": 45,
  "chunks_updated": 0,
  "table_name": "embeddings_agente_4_qualidade_leite",
  "agent_id": 4,
  "source": "arquivo.md",
  "doc_type": "manual",
  "strategy": "markdown",
  "processing_time_ms": 3200,
  "file_hash": "...",
  "quality_gate_passed": true,
  "quality_score": 98.3,
  "garbled_ratio": 0.0,
  "resolved_agent_id": 4,
  "resolved_agent_name": "Qualidade do Leite",
  "resolved_table_name": "embeddings_agente_4_qualidade_leite"
}
```

### Response 200 (arquivo duplicado)

```json
{
  "success": true,
  "skipped_duplicate": true,
  "chunks_created": 0,
  "chunks_processed": 0,
  "chunks_inserted": 0,
  "chunks_updated": 0,
  "table_name": "embeddings_agente_4_qualidade_leite",
  "agent_id": 4,
  "source": "arquivo.md",
  "doc_type": "manual",
  "strategy": "markdown",
  "processing_time_ms": 120,
  "duplicate_of": {
    "id": 123,
    "source_filename": "arquivo.md",
    "chunk_count": 45,
    "ingested_at": "...",
    "status": "ingested"
  },
  "file_hash": "...",
  "resolved_agent_id": 4,
  "resolved_agent_name": "Qualidade do Leite",
  "resolved_table_name": "embeddings_agente_4_qualidade_leite"
}
```

Observacao:
- Duplicado **nao** retorna erro HTTP. Retorna `200 OK` com `skipped_duplicate: true`.

Mapeamento recomendado de `agent_id -> table_name`:
- `0 -> embeddings_agente_0_base_geral`
- `1 -> embeddings_agente_1_queijos`
- `2 -> embeddings_agente_2_fermentados`
- `3 -> embeddings_agente_3_regulatorios`
- `4 -> embeddings_agente_4_qualidade_leite`
- `5 -> embeddings_agente_5_defeitos`
- `6 -> embeddings_agente_6_formulacao`

---

## 4) Healthcheck

### Endpoint

- `GET /health`

Sem API key.

Resposta esperada:

```json
{
  "status": "ok",
  "agents": 7,
  "version": "1.0.0",
  "databases": {
    "supabase": "connected",
    "hetzner": "connected"
  }
}
```

---

## 5) Erros esperados

- `401`: missing/invalid API key
- `404`: agente invalido ou `agent_id` invalido
- `400`: arquivo invalido (tipo, vazio, encoding)
- `422`: body invalido
- `500`: erro interno

---

## 6) Fluxo recomendado no app

1. Usar `POST /webhook/orquestrador/stream` como fluxo padrao.
2. Usar `POST /webhook/agente-{id}/stream` apenas quando quiser forcar especialista.
3. Usar `POST /webhook/ingestao-arquivo` para inclusao de documentos.
