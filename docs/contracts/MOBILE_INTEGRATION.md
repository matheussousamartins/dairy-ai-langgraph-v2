# DairyApp AI — Guia de Integração Mobile

**Versão:** 1.2  
**Atualizado:** 2026-05-04  
---

## Visão Geral

O backend é um sistema multiagente RAG especializado em tecnologia de laticínios. Para o app mobile, existe **um único endpoint**:

```
POST /webhook/orquestrador/stream
```

Esse endpoint recebe a pergunta do usuário, roteia internamente para os agentes corretos (Tecnologia de Queijos e/ou Regulatório [por enquanto], depois os outros agentes serão disponibilizados), consolida a resposta e transmite os tokens em tempo real via SSE — padrão de mercado para apps de chat com LLM.

Não é necessário conhecer os agentes internos. O orquestrador cuida disso.

---

## Base URL

```
https://<seu-dominio>/
```

Em desenvolvimento local o padrão é `http://localhost:8000`.

---

## Autenticação

Todas as requisições precisam do header:

```
X-API-Key: <chave-configurada-no-backend>
```

Se a chave estiver errada ou ausente, a resposta é `401 Unauthorized`.

> **Nota:** obtenha a chave com o Matheus. Não hardcode — use variável de ambiente no app.

---

## Endpoint de Chat — Streaming SSE

### `POST /webhook/orquestrador/stream`

#### Request

```http
POST /webhook/orquestrador/stream
Content-Type: application/json
X-API-Key: sua-chave-aqui
```

```json
{
  "message": "Qual é o teor mínimo de gordura do queijo minas frescal?",
  "session_id": "a3f7c2d1-4e5b-4c8a-9f2b-1d3e5f7a9b0c",
  "user_profile": {
    "knowledgeLevel": "INTERMEDIATE",
    "role": "técnico de laticínios"
  }
}
```

#### Campos do Request

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `message` | string | ✅ | Pergunta do usuário |
| `session_id` | string | ✅ | UUID da conversa — veja [Gerenciamento de Sessão](#gerenciamento-de-sessão) |
| `user_profile` | object | ❌ | Perfil do usuário — personaliza tom da resposta |
| `model` | string | ❌ | Omitir — usa o modelo padrão do backend |

#### Campos de `user_profile`

| Campo | Valores aceitos | Padrão |
|-------|-----------------|--------|
| `knowledgeLevel` | `BASIC`, `INTERMEDIATE`, `ADVANCED` | `INTERMEDIATE` |
| `role` | texto livre (`"técnico"`, `"gerente"`, etc.) | não informado |

---

#### Eventos SSE

O servidor emite eventos no formato `data: {...}\n\n`. Cada linha é um JSON com o campo `event`.

**`chunk`** — fragmento da resposta. Concatenar em ordem para montar o texto completo:
```json
{"event": "chunk", "text": "De acordo com a IN 73/2019,"}
{"event": "chunk", "text": " o queijo Minas Frescal deve ter..."}
```

**`final`** — stream encerrado. Contém metadados do agente que respondeu:
```json
{"event": "final", "agent_id": 3, "agent_name": "Regulatório"}
```

**`error`** — erro durante a geração:
```json
{"event": "error", "detail": "mensagem de erro"}
```

**`trace`** — eventos de observabilidade internos (grafo, RAG). **Ignorar no app:**
```json
{"event": "trace", "type": "node_start", "node": "classify", "ts": 1714830000000}
```

---

#### Exemplo de consumo

```js
async function sendMessage(message, sessionId, userProfile) {
  const response = await fetch(`${BASE_URL}/webhook/orquestrador/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': process.env.DAIRY_API_KEY,
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      user_profile: userProfile,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = '';
  let agentId = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const lines = decoder.decode(value).split('\n');
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;

      let event;
      try {
        event = JSON.parse(line.slice(6));
      } catch {
        continue;
      }

      if (event.event === 'chunk') {
        fullText += event.text;
        onChunk(fullText); // atualizar UI progressivamente

      } else if (event.event === 'final') {
        agentId = event.agent_id;
        break;

      } else if (event.event === 'error') {
        throw new Error(event.detail);
      }
      // ignorar 'trace'
    }
  }

  return { text: fullText, agentId };
}
```

---

#### Erros HTTP

| Status | Motivo | O que fazer |
|--------|--------|-------------|
| `401` | API Key ausente ou inválida | Verificar configuração |
| `422` | `session_id` não informado | Bug de integração — corrigir |
| `500` | Erro interno | Exibir mensagem abaixo e permitir reenvio |

Em caso de `500`, o backend retorna a mensagem diretamente no body — exibir ao usuário sem modificar:

```
Não foi possível processar sua pergunta no momento. Por favor, tente novamente.
```

---

## Gerenciamento de Sessão

O `session_id` controla a memória da conversa. É o campo mais crítico da integração.

**Regras:**

1. **Gere um UUID v4 ao iniciar cada conversa nova** — nunca reutilize entre conversas diferentes.
2. **Use o mesmo `session_id` em todas as mensagens da mesma conversa** — o backend carrega e salva o histórico por esse ID.
3. **Persista o `session_id`** no storage do app para que a conversa sobreviva a reinicializações se necessário.

```js
import { v4 as uuidv4 } from 'uuid';

// Ao iniciar nova conversa
const sessionId = uuidv4(); // "a3f7c2d1-4e5b-..."
await storage.set('currentSessionId', sessionId);

// Em todas as mensagens subsequentes da mesma conversa
const sessionId = await storage.get('currentSessionId');
await sendMessage(text, sessionId);
```

---

## Comportamento Garantido pelo Backend

- **Nunca retorna vazio:** se a base de conhecimento não tiver informação, o sistema tenta fallbacks antes de admitir ausência de dados.
- **Resposta sempre em português.**
- **Sem LaTeX:** fórmulas são convertidas para texto simples antes de chegar ao app.
- **Sem menções à arquitetura interna:** a resposta nunca expõe agentes, bases ou ferramentas internas.

---

## Latência Esperada

| Cenário | Estimativa |
|---------|-----------|
| Pergunta simples (fast-path) | 3–8s |
| Pergunta técnica (1 agente) | 8–15s |
| Pergunta híbrida (2 agentes + consolidação) | 15–25s |
| Com fallback web | +5–10s adicionais |

Com streaming, o usuário vê os primeiros tokens em 2–4s independente da latência total — a percepção de velocidade é muito melhor que esperar a resposta completa.

---

## Health Check

```http
GET /health
```

Sem autenticação. Use na inicialização do app ou antes de exibir a tela de chat.

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

| `status` | HTTP |
|----------|------|
| `"ok"` | 200 |
| `"degraded"` | 503 |

---

## Fluxo Completo

```
Usuário abre nova conversa
  → gerar sessionId (UUID v4)
  → persistir no storage

Usuário envia mensagem
  → POST /webhook/orquestrador/stream
  → body: { message, session_id, user_profile }
  → mostrar indicador de digitação / loading

Backend processa e emite SSE
  → eventos chunk: acumular e exibir progressivamente
  → evento final: marcar resposta como completa
  → evento error: exibir mensagem de erro

Usuário inicia nova conversa
  → gerar novo sessionId
  → descartar o anterior
```

---

## Endpoints de Stream Disponíveis

O app tem **dois modos de chat**, cada um com seu endpoint.

---

### Modo 1 — Assistente Geral (padrão)

```
POST /webhook/orquestrador/stream
```

Roteamento inteligente: o backend decide quais agentes consultar e consolida a resposta. Use para chat livre, sem domínio fixo. Lida com perguntas que cruzam domínios (ex: técnico + regulatório).

---

### Modo 2 — Agente Especialista Direto

```
POST /webhook/agente-{agent_id}/stream
```

Acessa um especialista diretamente, sem orquestrador. Use quando o usuário seleciona explicitamente um domínio no app (ex: "Falar com especialista em Queijos").

#### Agentes disponíveis

| `agent_id` | Especialidade |
|-----------|---------------|
| `1` | Tecnologia de Queijos |
| `2` | Fermentados Lácteos |
| `3` | Legislação e Regulatório |
| `4` | Qualidade do Leite |
| `5` | Diagnóstico de Defeitos |
| `6` | Formulação e Desenvolvimento |

#### Request — formato idêntico ao orquestrador

```http
POST /webhook/agente-1/stream
Content-Type: application/json
X-API-Key: sua-chave-aqui
```

```json
{
  "message": "Qual o pH ideal na filagem da mussarela?",
  "session_id": "a3f7c2d1-4e5b-4c8a-9f2b-1d3e5f7a9b0c",
  "user_profile": {
    "knowledgeLevel": "ADVANCED",
    "role": "queijeiro"
  }
}
```

#### Eventos SSE — idênticos ao orquestrador

Mesmos eventos: `chunk`, `final`, `error`, `trace`. O evento `final` inclui o `agent_id` do agente que respondeu.

#### Quando usar cada modo

| Situação | Endpoint |
|----------|----------|
| Chat livre / pergunta sem domínio definido | `/webhook/orquestrador/stream` |
| Usuário seleciona um especialista no app | `/webhook/agente-{id}/stream` |
| Pergunta cruza dois domínios (técnico + legal) | `/webhook/orquestrador/stream` |
| Domínio fixo e menor latência é prioridade | `/webhook/agente-{id}/stream` |

> **Atenção:** agentes individuais não têm roteamento. Uma pergunta fora do domínio selecionado será respondida pelo especialista escolhido de qualquer forma — sem redirecionamento automático. O orquestrador é mais resiliente a perguntas ambíguas.

---

## O que NÃO usar no App

Os endpoints `/webhook/agente-{0..6}` e `/webhook/orquestrador` **sem `/stream`** existem mas são para uso interno e testes. O app deve usar **sempre a versão com `/stream`**.

---

## Checklist de Integração

- [ ] Configurar `BASE_URL` por ambiente (dev / produção)
- [ ] Configurar `X-API-Key` via variável de ambiente (nunca hardcode)
- [ ] Implementar geração de UUID v4 por conversa
- [ ] Persistir `session_id` no storage do app
- [ ] Implementar consumo de SSE com acumulação de chunks (orquestrador e agentes)
- [ ] Ignorar eventos `trace` no SSE
- [ ] Exibir resposta progressivamente conforme chunks chegam
- [ ] Tratar evento `error` e HTTP `500` com mensagem amigável
- [ ] Implementar `/health` check na inicialização
- [ ] Enviar `user_profile` quando o app tiver o perfil do usuário
- [ ] Definir qual `agent_id` mapeia para cada tela/modo de especialista no app

---
