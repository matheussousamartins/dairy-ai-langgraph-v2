# Test Evaluations — Plano de Persistência em Supabase

## Objetivo

Transformar a feature de avaliações de teste do console em um ativo durável de produto, **sem mexer na arquitetura dos agentes do cliente**.

Hoje:
- a tela de testes já funciona no front
- as avaliações já podem ser feitas no console
- o score da sessão já é calculado
- a persistência inicial foi construída de forma desacoplada na camada do shell/Next

Próximo passo recomendado:
- ativar a persistência real em **Supabase/Postgres**
- mantendo a camada de testes **desacoplada** do backend dos agentes

---

## O que isso resolve

Com Supabase, passamos a ter:

- histórico durável das sessões de teste
- avaliações preservadas entre reinícios e deploys
- base para relatórios
- base para análise de qualidade dos agentes
- base futura para melhoria de prompts, roteamento e RAG

---

## O que não muda

Esta implantação **não precisa alterar**:

- backend dos agentes do cliente
- orquestrador
- RAG
- ferramentas
- contratos principais de chat

Ou seja:
- continuamos com a lógica de testes em uma camada paralela
- só trocamos a persistência local por persistência real

---

## Modelo recomendado

### 1. `test_sessions`

Representa uma sessão de teste feita por um tester em uma thread.

Campos principais:
- `id`
- `thread_id`
- `thread_title`
- `status`
- `started_at`
- `ended_at`
- `created_at`
- `updated_at`
- métricas consolidadas:
  - `evaluated_count`
  - `correct_count`
  - `partial_count`
  - `incorrect_count`
  - `score_percent`

### 2. `test_evaluations`

Representa a avaliação de uma resposta específica do agente.

Campos principais:
- `id`
- `session_id`
- `thread_id`
- `message_id`
- `turn_id`
- `verdict`
- `score`
- `question`
- `answer`
- `agent_id`
- `model_id`
- `comment`
- `metadata`
- `created_at`
- `updated_at`

---

## Por que esse desenho é profissional

Porque separa claramente:

- conversa do usuário
- execução do agente
- avaliação do tester

Isso é o padrão mais saudável para:
- observabilidade
- auditoria
- evolução do produto
- reaproveitamento futuro em outros clientes

---

## Fluxo de gravação recomendado

### Durante o uso do console
1. tester conversa normalmente
2. ao avaliar uma resposta:
   - `Correta`
   - `Parcial`
   - `Incorreta`
3. a camada `/api/tests/*` salva no Supabase

### Ao finalizar a sessão
1. a sessão é marcada como `completed`
2. as métricas consolidadas ficam registradas
3. a sessão pode ser exibida na tela `/tests`

---

## Benefícios para reunião com cliente

Essa evolução permite dizer com segurança que:

- o sistema já não serve só para “conversar”
- ele passa a servir também para **avaliar qualidade**
- e cria uma base confiável para melhoria contínua dos agentes

Em linguagem executiva:

> “A camada de testes passa a ser persistida separadamente, sem interferir no backend dos agentes, permitindo histórico, score, auditoria e melhoria contínua.”

---

## O que já foi implementado

### Front e UX
- tela `/tests` criada e funcional
- avaliação por resposta com:
  - `Correta`
  - `Parcial`
  - `Incorreta`
- score por sessão
- resumo consolidado da sessão
- comentário opcional por avaliação
- navegação para a área de testes integrada ao front

### Camada de dados e API
- store local inicial criada para demonstração
- camada de repositório desacoplada criada em:
  - `frontend/src/lib/test-evaluations-repository.ts`
- suporte a dois modos de persistência:
  - `memory`
  - `supabase`
- rotas `/api/tests/*` refatoradas para usar essa abstração
- tratamento de erro das rotas padronizado com `try/catch` e retorno JSON consistente

### Supabase
- schema SQL criado em:
  - `sql/07_test_evaluations_schema.sql`
- tabelas já criadas no banco:
  - `test_sessions`
  - `test_evaluations`
- view criada:
  - `v_test_session_quality_daily`

### Qualidade técnica
- dependência `@supabase/supabase-js` instalada no `frontend`
- client server-side do Supabase implementado em:
  - `frontend/src/lib/supabase-server.ts`
- `npm run lint` passou
- `npm run build` passou

---

## Estado atual da implementação

Neste momento, o projeto já está **pronto no código** para usar Supabase como persistência da feature de testes.

O que já está pronto:
- UI de testes
- score da sessão
- comentário opcional
- API `/api/tests/*`
- repositório com modo `supabase`
- tabelas no Supabase

O que ainda está bloqueando o uso real:
- a variável `SUPABASE_URL` do `frontend/.env.local` está apontando para um host que **não está resolvendo por DNS**

Erro identificado:

```text
getaddrinfo ENOTFOUND aeicuprnutblrpbhdxqs.supabase.co
```

Isso significa:
- não é erro de tabela
- não é erro do SQL
- não é erro da lógica da feature
- é erro de **configuração da URL do projeto Supabase**

---

## Próximo passo imediato

Corrigir a configuração do `frontend/.env.local`:

```env
TEST_EVALUATIONS_STORAGE=supabase
SUPABASE_URL=https://SEU-PROJECT-REF.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
```

Observações:
- `SUPABASE_URL` deve ser a base do projeto
- sem `/rest/v1/`
- sem `db.`
- a chave pode ser:
  - `SUPABASE_SERVICE_ROLE_KEY`
  - ou `SUPABASE_SECRET_KEY`

Depois:
1. reiniciar o `frontend`
2. testar avaliação
3. testar comentário
4. finalizar sessão
5. abrir `/tests`
6. dar refresh e confirmar persistência

---

## Ordem de implantação recomendada

### Etapa 1 — Banco
- criar tabelas no Supabase
- criar índices
- criar view simples de acompanhamento

### Etapa 2 — Shell/Next
- trocar `test-store.ts` de memória para leitura/gravação no Supabase
- manter os endpoints `/api/tests/*` como fachada do front

### Etapa 2.5 — Ativação controlada
- manter `memory` como fallback seguro
- ativar Supabase por flag:
  - `TEST_EVALUATIONS_STORAGE=memory`
  - depois:
  - `TEST_EVALUATIONS_STORAGE=supabase`

### Etapa 3 — Refino
- filtros por período
- filtros por agente/modelo
- exportação de resultados
- relatórios por cliente/tenant

---

## Observação importante

Para a **demonstração e uso imediato**, não era obrigatório criar essas tabelas.

Mas para:
- persistência real
- uso interno sério
- melhoria contínua
- histórico entre deploys

**sim, eu recomendo a criação no Supabase**.

Essa etapa já foi executada no banco. O único bloqueio restante no momento é a correção da `SUPABASE_URL` no `frontend/.env.local`.

---

## Arquivos principais envolvidos

- `sql/07_test_evaluations_schema.sql`
- `frontend/src/lib/test-store.ts`
- `frontend/src/lib/test-evaluations-repository.ts`
- `frontend/src/lib/supabase-server.ts`
- `frontend/src/app/api/tests/sessions/route.ts`
- `frontend/src/app/api/tests/sessions/[sessionId]/finalize/route.ts`
- `frontend/src/app/api/tests/threads/[threadId]/route.ts`
- `frontend/src/app/api/tests/threads/[threadId]/evaluate/route.ts`
- `frontend/src/state/useThreadTesting.ts`
- `frontend/src/components/app/ChatPane.tsx`
- `frontend/src/components/app/TestSessionsView.tsx`
- `frontend/src/app/tests/page.tsx`

---

## Handoff operacional para continuar no Claude Code

### Estado atual
- a feature de testes já está implementada no front
- a persistência em `memory` funciona
- a persistência em `supabase` já está implementada no código
- as tabelas já foram criadas no Supabase
- `lint` e `build` passaram com a implementação atual

### O problema atual
- a gravação no Supabase ainda não está funcionando porque a `SUPABASE_URL` configurada no `frontend/.env.local` não resolve por DNS
- erro identificado:

```text
getaddrinfo ENOTFOUND aeicuprnutblrpbhdxqs.supabase.co
```

### O que precisa ser feito agora
1. confirmar a `Project URL` correta no painel do Supabase
2. atualizar `frontend/.env.local` com a URL correta
3. reiniciar o frontend com `npm run dev`
4. validar o fluxo completo da feature de testes

### Variáveis esperadas no `frontend/.env.local`

```env
TEST_EVALUATIONS_STORAGE=supabase
SUPABASE_URL=https://SEU-PROJECT-REF.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
```

Alternativamente, o código também aceita:

```env
SUPABASE_SECRET_KEY=...
```

### Checklist de validação após corrigir a URL
1. abrir o console
2. fazer uma pergunta
3. avaliar a resposta como:
   - `Correta`
   - `Parcial`
   - ou `Incorreta`
4. salvar uma observação
5. finalizar a sessão
6. abrir `/tests`
7. atualizar a página
8. confirmar que os dados continuam visíveis

### O que validar no Supabase
- em `test_sessions`:
  - criação da sessão
  - `status`
  - `score_percent`
  - contadores consolidados
- em `test_evaluations`:
  - `thread_id`
  - `message_id`
  - `turn_id`
  - `verdict`
  - `score`
  - `comment`
  - `agent_id`
  - `model_id`

### Se ainda der erro
Próximos pontos para inspeção:
- valor exato da `SUPABASE_URL`
- resolução DNS da máquina
- conteúdo retornado pelas rotas:
  - `/api/tests/threads/[threadId]`
  - `/api/tests/threads/[threadId]/evaluate`
  - `/api/tests/sessions`

### Importante
- nada disso mexe no backend dos agentes do cliente
- nada disso altera o orquestrador
- a camada de testes continua desacoplada e em linha com padrão de mercado
