# Guia Completo de Ingestao (Web/Mobile)

Data: 2026-04-15  
Projeto: Dairy AI LangGraph v2

## Objetivo

Este documento define o contrato oficial para ingestao de documentos via web/mobile, incluindo:
- payload esperado
- metadados obrigatorios
- respostas de sucesso e erro
- criterios de qualidade do Markdown
- testes de homologacao

## Endpoint oficial

- Metodo: `POST`
- URL: `/webhook/ingestao-arquivo`
- Tipo de envio: `multipart/form-data`

## Configuracao obrigatoria no front

Definir no app web/mobile:

- `BACKEND_BASE_URL` = URL base da API  
  Exemplo local: `http://192.168.1.31:8000`
- `WEBHOOK_API_KEY_HEADER` = `X-API-Key`
- `WEBHOOK_API_KEY` = chave valida cadastrada no backend (`WEBHOOK_API_KEYS`)

Exemplo de `.env` no front:

```env
BACKEND_BASE_URL=http://192.168.1.31:8000
WEBHOOK_API_KEY_HEADER=X-API-Key
WEBHOOK_API_KEY=COLE_A_CHAVE_AQUI
```

## Autenticacao

Enviar header com API key:

```http
X-API-Key: <WEBHOOK_API_KEY>
```

Se faltar ou estiver invalida, retorna `401`.

## Campos do payload (multipart/form-data)

- `file` (obrigatorio): arquivo `.md` ou `.txt` em UTF-8
- `agent_id` (obrigatorio): inteiro de `0` a `6`
- `doc_type` (opcional): default `manual`
- `table_name` (recomendado): tabela correta do `agent_id`

## Mapeamento oficial agent_id -> table_name

- `0` -> `embeddings_agente_0_base_geral`
- `1` -> `embeddings_agente_1_queijos`
- `2` -> `embeddings_agente_2_fermentados`
- `3` -> `embeddings_agente_3_regulatorios`
- `4` -> `embeddings_agente_4_qualidade_leite`
- `5` -> `embeddings_agente_5_defeitos`
- `6` -> `embeddings_agente_6_formulacao`

Se `table_name` divergir do `agent_id`, retorna `400`.

## Exemplo de chamada (cURL)

```bash
curl -X POST "http://SEU_BACKEND:8000/webhook/ingestao-arquivo" \
  -H "X-API-Key: SUA_CHAVE" \
  -F "file=@/caminho/documento.md" \
  -F "agent_id=4" \
  -F "doc_type=manual" \
  -F "table_name=embeddings_agente_4_qualidade_leite"
```

## Exemplo de chamada (TypeScript)

```ts
type AgentId = 0 | 1 | 2 | 3 | 4 | 5 | 6;

const TABLE_BY_AGENT: Record<AgentId, string> = {
  0: "embeddings_agente_0_base_geral",
  1: "embeddings_agente_1_queijos",
  2: "embeddings_agente_2_fermentados",
  3: "embeddings_agente_3_regulatorios",
  4: "embeddings_agente_4_qualidade_leite",
  5: "embeddings_agente_5_defeitos",
  6: "embeddings_agente_6_formulacao",
};

export async function ingestFile(params: {
  baseUrl: string;
  apiKey: string;
  file: File;
  agentId: AgentId;
  docType?: string;
}) {
  const { baseUrl, apiKey, file, agentId, docType = "manual" } = params;
  const fd = new FormData();
  fd.append("file", file);
  fd.append("agent_id", String(agentId));
  fd.append("doc_type", docType);
  fd.append("table_name", TABLE_BY_AGENT[agentId]);

  const res = await fetch(`${baseUrl}/webhook/ingestao-arquivo`, {
    method: "POST",
    headers: { "X-API-Key": apiKey },
    body: fd,
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data?.detail || `Erro HTTP ${res.status}`);
  }
  return data;
}
```

## Respostas esperadas da API

### 1) Sucesso de ingestao

HTTP `200`

```json
{
  "success": true,
  "chunks_created": 51,
  "chunks_processed": 51,
  "chunks_inserted": 51,
  "chunks_updated": 0,
  "table_name": "embeddings_agente_4_qualidade_leite",
  "agent_id": 4,
  "source": "IN 68 -  METODOS QUALITATIVOS.md",
  "doc_type": "manual",
  "strategy": "markdown",
  "processing_time_ms": 13239,
  "file_hash": "5ad4bc...",
  "quality_gate_passed": true,
  "quality_score": 100.0,
  "text_chars": 33867,
  "word_count": 4335,
  "garbled_ratio": 0.0,
  "thresholds": {
    "min_text_chars": 400,
    "min_words": 80,
    "max_garbled_ratio": 0.08,
    "min_quality_score": 60.0
  },
  "quality_issues": [],
  "resolved_agent_id": 4,
  "resolved_agent_name": "Qualidade do Leite",
  "resolved_table_name": "embeddings_agente_4_qualidade_leite"
}
```

### 2) Duplicado detectado (conteudo igual)

HTTP `200`

```json
{
  "success": true,
  "skipped_duplicate": true,
  "chunks_created": 0,
  "chunks_processed": 0,
  "chunks_inserted": 0,
  "chunks_updated": 0,
  "duplicate_of": {
    "id": 29,
    "table_name": "embeddings_agente_4_qualidade_leite",
    "agent_id": 4,
    "source_filename": "IN 68 -  METODOS QUALITATIVOS.md",
    "chunk_count": 51,
    "status": "ingested"
  },
  "file_hash": "5ad4bc..."
}
```

### 3) Bloqueado por quality gate

HTTP `200`

```json
{
  "success": false,
  "error": "Documento bloqueado pelo quality gate de ingestao.",
  "chunks_created": 0,
  "quality_gate_passed": false,
  "quality_score": 31.4,
  "quality_issues": [
    "Texto muito curto (...)",
    "Poucas palavras (...)"
  ]
}
```

## Erros HTTP possiveis

### 400 - Validacao de request

Formato:

```json
{ "detail": "mensagem de erro" }
```

Causas comuns:
- `table_name` divergente do `agent_id`
- extensao diferente de `.md`/`.txt`
- arquivo sem UTF-8
- arquivo vazio

### 401 - API key

Formato:

```json
{ "detail": "Missing API key" }
```

ou

```json
{ "detail": "Invalid API key" }
```

### 404 - Agent ID invalido

Formato:

```json
{ "detail": "Agente X nao encontrado. IDs validos: 0 a 6." }
```

### 500 - Erro interno

Formato:

```json
{ "detail": "Erro na ingestao de arquivo: ..." }
```

## Metadados que o front deve registrar em log

Registrar no front (para auditoria e debug):
- `source` (nome do arquivo enviado)
- `agent_id`
- `resolved_agent_id`
- `resolved_table_name`
- `file_hash`
- `quality_gate_passed`
- `quality_score`
- `chunks_created`
- `chunks_inserted`
- `chunks_updated`
- `skipped_duplicate` (quando existir)
- `processing_time_ms`

## Requisitos de qualidade para PDF -> Markdown

O conversor da web/mobile deve gerar Markdown com:
- codificacao UTF-8
- acentos corretos (sem caracteres quebrados)
- estrutura preservada (`#`, `##`, listas, tabelas)
- unidades e formulas legiveis
- sem lixo de OCR em excesso

## Regras de deduplicacao (comportamento esperado)

- Dedup e por `file_hash` do conteudo normalizado no backend.
- Mesmo conteudo -> `skipped_duplicate=true`.
- Nome de arquivo diferente NAO burla dedup se o conteudo for igual.
- Conteudo diferente (mesmo nome) pode entrar como nova versao de conhecimento.

## Bateria minima de homologacao

1. Subir 1 arquivo novo do agente correto.
2. Validar retorno com `success=true` e `quality_gate_passed=true`.
3. Reenviar o mesmo arquivo.
4. Validar `skipped_duplicate=true` e `chunks_created=0`.
5. Fazer 2 perguntas de negocio no front para validar que o conteudo esta recuperavel via RAG.

## Troubleshooting rapido

- Erro `curl: (26) Failed to open/read local data from file/application`
  - caminho do arquivo invalido no cliente

- `There was an error parsing the body`
  - payload/form-data malformado

- `Invalid API key`
  - chave incorreta ou header errado

- `table_name divergente do agent_id`
  - mapear corretamente `agent_id -> table_name`

## Regra operacional recomendada

Para ambiente de producao:
- usar sempre endpoint oficial acima
- usar sempre `table_name` explicita
- manter um unico pipeline de conversao PDF->MD
- acompanhar diariamente logs de `quality_score`, `skipped_duplicate` e falhas
