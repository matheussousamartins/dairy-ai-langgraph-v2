# Persistência de Trace com Baixo Impacto

## Objetivo

Persistir os traces de execução do chat sem bagunçar a arquitetura atual do cliente.

A proposta é separar:

- conteúdo da conversa
- metadados de execução
- traces/logs

Assim, o histórico principal continua limpo, enquanto os traces passam a existir como dado durável e auditável.

---

## Princípio

O `trace` não deve ser tratado como parte central da mensagem.

Ele deve ser tratado como:

- dado auxiliar de observabilidade
- log de execução
- metadado associado a uma rodada (`turn`)

---

## Modelagem Mínima

## 1. Conversa

Manter a estrutura atual de mensagens:

- `chat_memories`

Campos conceituais:

- `id`
- `session_id`
- `role`
- `content`
- `created_at`
- `turn_id` *(recomendado)*

---

## 2. Trace

Criar uma estrutura separada:

- `chat_execution_traces`

Campos sugeridos:

- `id`
- `session_id`
- `turn_id`
- `agent_id`
- `agent_name`
- `model_id`
- `trace_payload`
- `latency_ms`
- `created_at`

---

## Por que `turn_id` é importante

Sem `turn_id`, a associação entre:

- mensagem assistant
- trace correspondente

fica mais frágil.

Com `turn_id`:

- a rodada fica identificada do começo ao fim
- a associação entre resposta e trace fica explícita
- a leitura no front fica mais segura

---

## Fluxo de Escrita

1. usuário envia uma mensagem
2. o sistema gera um `turn_id`
3. o backend acumula `traceEvents`
4. o backend salva a mensagem do usuário
5. o backend salva a resposta assistant
6. o backend salva o trace em `chat_execution_traces`

---

## Fluxo de Leitura

1. buscar mensagens da sessão
2. buscar traces da sessão
3. mapear os traces por `turn_id`
4. enriquecer as mensagens assistant com `trace` quando existir
5. devolver ao front o payload já pronto

---

## Endpoints Mínimos

### `POST /api/threads/:id/stream`

Mantém o comportamento atual de streaming, mas ao final:

- salva mensagem
- salva trace separado

### `GET /api/threads/:id`

Retorna:

- mensagens
- trace associado, quando existir

### `GET /api/threads/:id/traces` *(opcional)*

Pode ser usado no futuro para:

- abrir logs sob demanda
- reduzir peso do payload principal

---

## Estrutura de Payload no Front

Exemplo:

```json
{
  "id": "msg-1",
  "role": "assistant",
  "content": "Resposta...",
  "turn_id": "turn-123",
  "agentId": "orquestrador",
  "modelId": "gpt-4.1",
  "trace": [
    {
      "type": "node_start",
      "node": "route",
      "ts": 1710000000
    }
  ]
}
```

---

## Benefícios

- não polui a estrutura principal de conversa
- preserva a arquitetura atual
- deixa o botão de log confiável
- permite auditoria posterior
- prepara terreno para avaliação de agentes
- prepara terreno para sistema de testes

---

## Impacto Esperado de Latência

## Na experiência do usuário

Baixo, se implementado corretamente.

Motivos:

- o streaming da resposta continua igual
- o trace pode ser salvo apenas ao final
- o usuário recebe os tokens em tempo real do mesmo jeito

## No backend

Há um pequeno custo adicional de persistência:

- mais uma escrita por rodada
- eventualmente uma leitura extra no carregamento do histórico

Na prática, o impacto costuma ser:

- baixo
- previsível
- controlável

## Como minimizar esse impacto

- salvar o trace apenas no final da execução
- usar tabela/estrutura separada
- não serializar/deserializar trace desnecessariamente durante o stream
- limitar o tamanho do payload se necessário
- no futuro, carregar trace sob demanda se o volume crescer

---

## Estratégia Recomendada

### Fase 1

- persistir trace separado
- enriquecer histórico com trace
- manter tudo simples

### Fase 2

- adicionar `GET /traces` sob demanda
- reduzir payload do histórico principal

### Fase 3

- usar traces também em avaliação automática
- comparação entre execuções
- sistema de regressão

---

## Recomendação Final

Sim, esta é uma solução profissional e de baixo impacto.

Ela:

- preserva a arquitetura atual
- resolve o problema do botão de log
- melhora observabilidade
- prepara a base para um test harness desacoplado

