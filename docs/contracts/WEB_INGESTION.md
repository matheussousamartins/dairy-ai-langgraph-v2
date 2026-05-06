# DairyApp AI — Guia de Ingestão de Documentos (Web)

**Versão:** 1.0  
**Atualizado:** 2026-05-04  

---

## Visão Geral

O cliente (administrador) pode fazer upload de documentos para a base de conhecimento dos agentes diretamente pelo app web. O pipeline de ingestão é **assíncrono**: o upload retorna imediatamente com um `job_id` e o processamento (conversão → chunking → embedding → indexação) ocorre em background.

O fluxo é sempre:

```
1. POST /webhook/ingestao-arquivo  →  recebe job_id (HTTP 202)
2. GET  /webhook/ingestao-status/{job_id}  →  polling até completed | failed
```

---

## Autenticação

Todas as requisições precisam do header:

```
X-API-Key: <chave-configurada-no-backend>
```

Se a chave estiver errada ou ausente, a resposta é `401 Unauthorized`.

> **Nota:** obtenha a chave com o Matheus. Não hardcode — use variável de ambiente.

---

## Endpoint 1 — Upload de Documento

### `POST /webhook/ingestao-arquivo`

Envia um arquivo para ingestão. Retorna `HTTP 202` imediatamente com o `job_id`.

#### Request

O body é `multipart/form-data` (não JSON).

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `file` | arquivo | ✅ | O documento a ingerir |
| `agent_id` | inteiro | ✅ | ID do agente que receberá o documento — veja tabela abaixo |
| `doc_type` | string | ❌ | Tipo do documento — veja tabela abaixo. Se omitido, o backend detecta automaticamente |

#### Agentes disponíveis (`agent_id`)

| `agent_id` | Base de conhecimento |
|-----------|----------------------|
| `0` | Base Geral (glossário, diretrizes) |
| `1` | Tecnologia de Queijos |
| `2` | Fermentados Lácteos |
| `3` | Legislação e Regulatório |
| `4` | Qualidade do Leite |
| `5` | Diagnóstico de Defeitos |
| `6` | Formulação e Desenvolvimento |

#### Tipos de documento (`doc_type`)

| Valor | Quando usar |
|-------|-------------|
| `manual` | Manuais, procedimentos, guias técnicos (padrão se omitido) |
| `legislacao` | INs, RDCs, portarias, decretos, normas |
| `glossario` | Glossários técnicos |
| `faq` | Perguntas e respostas frequentes |
| `formulacao` | Fichas técnicas de formulação |
| `tabela_nutricional` | Tabelas de composição nutricional |

> **Dica:** se o `doc_type` for omitido, o backend analisa o nome do arquivo e o conteúdo para detectar automaticamente o tipo correto. Para documentos simples, omitir é suficiente. Para legislação e glossários, informar explicitamente garante chunking otimizado.

#### Formatos aceitos

`.pdf` `.docx` `.md` `.txt`

Tamanho máximo: **50 MB** por arquivo.

#### Exemplo de request (JavaScript / fetch)

```js
async function uploadDocument(file, agentId, docType = null) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('agent_id', agentId);
  if (docType) formData.append('doc_type', docType);

  const response = await fetch(`${BASE_URL}/webhook/ingestao-arquivo`, {
    method: 'POST',
    headers: {
      'X-API-Key': process.env.DAIRY_API_KEY,
      // Não definir Content-Type — o browser define automaticamente com o boundary
    },
    body: formData,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Upload falhou (${response.status}): ${detail}`);
  }

  return await response.json(); // { job_id, status, filename, agent_id, ... }
}
```

#### Response `202 Accepted`

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "queued",
  "filename": "IN_76_leite_cru.pdf",
  "agent_id": 3,
  "agent_name": "Legislação e Regulatório",
  "table_name": "embeddings_agente_3_regulatorios",
  "doc_type": "legislacao",
  "file_size_bytes": 204800,
  "message": "Documento recebido. Acompanhe o progresso em /webhook/ingestao-status/{job_id}"
}
```

#### Erros HTTP

| Status | Motivo | O que fazer |
|--------|--------|-------------|
| `400` | Formato não suportado / arquivo vazio | Validar extensão antes do upload |
| `401` | API Key inválida | Verificar configuração |
| `404` | `agent_id` não existe | Usar IDs 0 a 6 |
| `413` | Arquivo maior que 50 MB | Reduzir tamanho ou dividir o arquivo |
| `500` | Erro interno ao registrar o job | Exibir mensagem e permitir nova tentativa |

---

## Endpoint 2 — Status do Job

### `GET /webhook/ingestao-status/{job_id}`

Consulta o status de um job de ingestão. Usar em polling até `completed` ou `failed`.

#### Request

```http
GET /webhook/ingestao-status/f47ac10b-58cc-4372-a567-0e02b2c3d479
X-API-Key: sua-chave-aqui
```

#### Response `200 OK`

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "agent_id": 3,
  "agent_name": "Legislação e Regulatório",
  "original_filename": "IN_76_leite_cru.pdf",
  "doc_type": "legislacao",
  "status": "completed",
  "pages_detected": 12,
  "chunks_created": 47,
  "chunks_inserted": 45,
  "chunks_updated": 2,
  "file_size_bytes": 204800,
  "processing_time_ms": 8320,
  "error_detail": null,
  "created_at": "2026-05-04T14:30:00.000Z",
  "updated_at": "2026-05-04T14:30:08.320Z"
}
```

#### Ciclo de status

```
queued → converting → processing → completed
                                 ↘ failed
```

| Status | Significado |
|--------|-------------|
| `queued` | Na fila, aguardando processamento |
| `converting` | Convertendo PDF/DOCX para texto |
| `processing` | Gerando chunks, embeddings e indexando |
| `completed` | Ingestão concluída com sucesso |
| `failed` | Falha — ver campo `error_detail` |

#### Erros HTTP

| Status | Motivo |
|--------|--------|
| `404` | `job_id` não encontrado |

---

## Fluxo Completo com Polling

```js
async function ingestAndWait(file, agentId, docType, onProgress) {
  // 1. Enviar arquivo
  const job = await uploadDocument(file, agentId, docType);
  onProgress({ status: 'queued', job });

  // 2. Polling até finalizar
  const POLL_INTERVAL_MS = 3000;
  const TIMEOUT_MS = 5 * 60 * 1000; // 5 minutos
  const startedAt = Date.now();

  while (true) {
    await sleep(POLL_INTERVAL_MS);

    if (Date.now() - startedAt > TIMEOUT_MS) {
      throw new Error('Timeout aguardando processamento do documento.');
    }

    const status = await getJobStatus(job.job_id);
    onProgress(status);

    if (status.status === 'completed') return status;
    if (status.status === 'failed') {
      throw new Error(`Ingestão falhou: ${status.error_detail}`);
    }
  }
}

async function getJobStatus(jobId) {
  const response = await fetch(`${BASE_URL}/webhook/ingestao-status/${jobId}`, {
    headers: { 'X-API-Key': process.env.DAIRY_API_KEY },
  });
  if (!response.ok) throw new Error(`Status check falhou: ${response.status}`);
  return await response.json();
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
```

---

## Exemplo de UI Recomendada

```
[Selecionar arquivo]  →  [Escolher agente]  →  [Enviar]

Após envio:
  ⏳ Na fila...
  ⚙️  Convertendo PDF...
  🔄 Processando (indexando chunks)...
  ✅ Concluído — 47 trechos indexados em 8,3s
  ❌ Falha: [error_detail]
```

**Intervalo de polling recomendado:** 3 a 5 segundos. Documentos grandes (PDFs com muitas páginas) podem levar até 2–3 minutos.

---

## Campos de Resultado (`completed`)

| Campo | Descrição |
|-------|-----------|
| `chunks_created` | Total de trechos gerados pelo chunking |
| `chunks_inserted` | Trechos novos inseridos na base |
| `chunks_updated` | Trechos existentes atualizados (re-upload do mesmo documento) |
| `pages_detected` | Páginas detectadas (PDFs); `1` para DOCX/MD/TXT |
| `processing_time_ms` | Tempo total de processamento em milissegundos |

> **Re-upload:** se o mesmo documento for enviado novamente, o sistema detecta automaticamente e atualiza os trechos existentes em vez de duplicar.

---

## Checklist de Integração

- [ ] Configurar `BASE_URL` por ambiente (dev / produção)
- [ ] Configurar `X-API-Key` via variável de ambiente (nunca hardcode)
- [ ] Validar extensão do arquivo antes do upload (`.pdf`, `.docx`, `.md`, `.txt`)
- [ ] Validar tamanho antes do upload (máximo 50 MB)
- [ ] Implementar polling com intervalo de 3–5 segundos
- [ ] Exibir progresso de status para o usuário (queued → converting → processing → completed)
- [ ] Tratar `failed` com exibição do `error_detail`
- [ ] Implementar timeout no polling (sugestão: 5 minutos)
- [ ] Não definir `Content-Type` manualmente no upload — deixar o browser definir com boundary

---
