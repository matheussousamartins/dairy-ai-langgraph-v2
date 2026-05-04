# Test Harness Desacoplado para Avaliação de Agentes

## Objetivo

Criar uma camada de testes e avaliação de agentes que seja:

- desacoplada do backend atual do cliente
- reutilizável para outros clientes no futuro
- capaz de executar baterias de testes
- capaz de consolidar resultados
- capaz de comparar execuções entre agentes, modelos e versões

O ponto central desta proposta é **não alterar a arquitetura atual dos agentes do cliente**. O backend existente continua sendo tratado como um alvo de execução, e o sistema de testes passa a ser uma camada independente.

---

## Princípio Arquitetural

O sistema de testes não deve conhecer detalhes internos da arquitetura do cliente além do necessário para enviar entradas e receber saídas.

Em vez de acoplar testes ao backend principal, a proposta é criar um:

- **Test Harness**
- **Evaluation Layer**
- **Adapter/Connector por cliente**

Assim:

- a lógica de testes fica genérica
- a integração com cada cliente fica isolada
- a reutilização futura fica viável

---

## O Problema que Esta Arquitetura Resolve

Hoje, se a funcionalidade de testes for construída dentro do backend atual do cliente:

- o domínio do cliente fica poluído com lógica de avaliação
- a manutenção cresce
- o risco operacional aumenta
- a solução fica difícil de reaproveitar em outros projetos

Com a abordagem desacoplada:

- o sistema do cliente segue intacto
- o módulo de testes vira um produto reutilizável
- novas integrações podem ser feitas via adapters
- o front de testes pode ser reaproveitado para diferentes clientes

---

## Visão Geral da Arquitetura

Separar a solução em 3 blocos:

### 1. Sistema do Cliente

Responsável por:

- agentes
- orquestração
- modelos
- ferramentas
- regras de negócio

Esse bloco **não deve ser alterado estruturalmente** para suportar testes.

### 2. Test Harness / Evaluation Layer

Responsável por:

- suites de teste
- casos de teste
- execuções
- avaliações
- score final
- relatórios
- comparação histórica

Esse bloco é genérico e reaproveitável.

### 3. Adapter / Connector

Responsável por:

- conectar o módulo de testes ao backend do cliente
- enviar prompts/inputs
- escolher agente/modelo quando necessário
- receber respostas
- normalizar o resultado

Esse bloco é o único que precisa conhecer o “alvo”.

---

## Princípio do Adapter

O adapter é a peça que permite reaproveitar o sistema para qualquer outro cliente.

Exemplo conceitual:

- `DairyAppAdapter`
- `ClienteBAdapter`
- `ClienteCAdapter`

Cada adapter implementa a mesma interface lógica:

- enviar entrada
- escolher contexto de execução
- receber resultado
- converter resposta para um formato padrão

Formato conceitual:

```ts
runTestCase(targetConfig, testCaseInput) => normalizedExecutionResult
```

O sistema de testes trabalha sempre com esse resultado padronizado, sem depender do backend real por trás.

---

## Benefício Estratégico

Essa arquitetura transforma a funcionalidade de testes em um ativo reaproveitável:

- serve para este cliente
- serve para outros clientes
- serve para outros agentes
- serve para outros modelos
- serve para comparação entre versões

Ou seja: em vez de construir uma feature local, você constrói uma base de produto.

---

## Capacidades do Sistema de Testes

O módulo deve permitir:

- criar suites de teste
- criar casos de teste
- executar testes individualmente
- executar baterias completas
- comparar execuções
- pontuar respostas
- consolidar score final
- armazenar histórico
- exportar relatórios

---

## Entidades Principais

## 1. Target

Representa um sistema alvo.

Exemplos:

- DairyApp
- Cliente B
- Cliente C

Campos sugeridos:

- `id`
- `name`
- `adapter_type`
- `base_url`
- `auth_config`
- `default_agent`
- `default_model`
- `metadata`

---

## 2. Test Suite

Representa uma bateria de testes.

Campos sugeridos:

- `id`
- `name`
- `description`
- `target_id`
- `version`
- `status`
- `created_at`
- `updated_at`

Exemplos:

- Regressão do Orquestrador
- Validação de Respostas Técnicas
- Suite de Segurança e Não-Alucinação

---

## 3. Test Case

Representa um caso individual dentro de uma suite.

Campos sugeridos:

- `id`
- `suite_id`
- `title`
- `description`
- `prompt`
- `context`
- `expected_output`
- `expected_format`
- `tags`
- `weight`
- `priority`
- `evaluation_type`

Possíveis tipos:

- `manual`
- `rule_based`
- `llm_judge`
- `hybrid`

---

## 4. Test Run

Representa a execução de uma suite.

Campos sugeridos:

- `id`
- `suite_id`
- `target_id`
- `agent_id`
- `model_id`
- `status`
- `started_at`
- `finished_at`
- `summary_score`
- `notes`

Status possíveis:

- `pending`
- `running`
- `completed`
- `failed`
- `canceled`

---

## 5. Test Case Run

Representa a execução real de um caso de teste.

Campos sugeridos:

- `id`
- `run_id`
- `test_case_id`
- `input_payload`
- `output_payload`
- `raw_response`
- `latency_ms`
- `status`
- `error_message`
- `score`
- `cost_estimate`

---

## 6. Evaluation

Representa a avaliação de uma resposta.

Campos sugeridos:

- `id`
- `test_case_run_id`
- `evaluation_method`
- `instruction_adherence_score`
- `factuality_score`
- `completeness_score`
- `clarity_score`
- `format_score`
- `safety_score`
- `tool_usage_score`
- `final_score`
- `rationale`
- `reviewer`

---

## 7. Report

Representa o relatório consolidado de uma execução.

Campos sugeridos:

- `id`
- `run_id`
- `overall_score`
- `pass_rate`
- `critical_failures`
- `average_latency_ms`
- `average_cost`
- `summary`
- `recommendations`

---

## Tipos de Avaliação Recomendados

## 1. Avaliação Manual

Útil para:

- qualidade técnica
- consistência
- tom
- utilidade real da resposta

Vantagem:

- alta precisão humana

Desvantagem:

- pouco escalável

---

## 2. Avaliação por Regras

Útil para:

- checagem de formato
- regex
- presença/ausência de termos
- JSON válido
- campos obrigatórios

Vantagem:

- objetiva e barata

Desvantagem:

- limitada semanticamente

---

## 3. LLM as Judge

Útil para:

- aderência à instrução
- completude
- clareza
- comparação com resposta esperada

Vantagem:

- escalável e mais inteligente

Desvantagem:

- precisa de boa rubric
- pode introduzir variação

---

## 4. Avaliação Híbrida

Recomendação principal para o produto.

Combina:

- validação por regra
- avaliação por LLM juiz
- revisão manual quando necessário

Essa tende a ser a melhor solução para confiabilidade e escala.

---

## Critérios de Avaliação

Critérios recomendados:

- aderência à instrução
- corretude factual
- completude
- clareza
- segurança
- ausência de alucinação
- conformidade com formato esperado
- uso correto de ferramentas
- latência
- custo

Nem todo caso precisa usar todos os critérios. Cada caso pode ter sua rubric própria.

---

## Fluxo de Uso do Produto

Fluxo recomendado:

1. selecionar um target
2. criar uma suite
3. adicionar casos de teste
4. escolher agente e modelo
5. rodar a bateria
6. armazenar respostas
7. avaliar resultados
8. gerar score consolidado
9. comparar com execuções anteriores

---

## Fluxo Técnico de Execução

1. o usuário inicia um `Test Run`
2. o sistema carrega a suite e os casos
3. para cada caso:
   - o adapter envia a entrada para o target
   - recebe a resposta
   - normaliza o resultado
   - salva métricas
4. a camada de avaliação pontua os resultados
5. o relatório final é consolidado

---

## Interface do Adapter

Sugestão conceitual:

```python
class BaseTargetAdapter:
    def run_test_case(self, target_config, test_case, execution_config):
        raise NotImplementedError
```

Retorno normalizado sugerido:

```python
{
  "status": "completed",
  "response_text": "...",
  "raw_payload": {...},
  "latency_ms": 1234,
  "model_id": "gpt-4o",
  "agent_id": "orquestrador",
  "tool_trace": [],
  "error": None,
}
```

---

## Exemplo de Adapter para o Cliente Atual

O adapter do cliente atual pode:

- chamar o endpoint atual de mensagens
- enviar thread, agente, modelo e prompt
- esperar resposta
- medir tempo
- devolver formato padronizado

Importante:

- ele consome o backend existente
- não altera o backend existente
- não precisa mudar a arquitetura dos agentes

---

## Proposta de Estrutura no Mesmo Repositório

Se a decisão for começar no mesmo repo, mas isolado:

```text
frontend/
  src/
    app/
      tests/
    components/
      tests/
    lib/
      test-harness/

app/
  test_harness/
    domain/
    services/
    repositories/
    evaluators/
    adapters/
      dairyapp_adapter.py
      base_adapter.py
```

Essa organização permite:

- isolamento lógico
- reaproveitamento
- baixo impacto no sistema atual

---

## Alternativa de Evolução Futura

Se isso crescer, o caminho natural é separar em um produto próprio:

- frontend próprio
- backend próprio
- adapters por cliente

Mas para começar, manter no mesmo repositório com isolamento arquitetural já é um bom equilíbrio.

---

## MVP Recomendado

Fase 1:

- cadastro de target
- cadastro de suite
- cadastro de casos
- execução de testes
- salvamento de resposta
- avaliação manual
- score final

Esse MVP já entrega valor real sem tocar no backend principal do cliente.

---

## Fase 2

- avaliação por regra
- juiz automático por LLM
- comparação entre execuções
- histórico de regressão
- exportação de relatório
- filtros por agente/modelo

---

## Fase 3

- aprovação mínima por suite
- baseline oficial por cliente
- comparação entre versões de agentes
- ranking entre modelos
- alertas de regressão
- dashboards

---

## Proposta de UI

Nova área no produto:

- `Console`
- `Histórico`
- `Testes`

Dentro de `Testes`:

- lista de suites
- botão `Nova Suite`
- detalhe da suite
- tabela de casos
- botão `Rodar Bateria`
- visão de resultados
- relatório final

---

## Exemplo de Telas

### Tela 1. Lista de Suites

- nome
- target
- agente
- modelo
- data
- score mais recente
- ações

### Tela 2. Detalhe da Suite

- informações gerais
- lista de casos
- pesos
- status

### Tela 3. Execução

- progresso da bateria
- status por caso
- latência
- falhas

### Tela 4. Relatório Final

- score geral
- score por critério
- taxa de aprovação
- falhas críticas
- comparação com run anterior

---

## Regras de Isolamento

Para manter a solução desacoplada:

- não criar dependência do domínio do cliente dentro do harness
- não colocar tabelas de teste misturadas com tabelas do produto principal, se possível
- não misturar regras de avaliação com regras de negócio do cliente
- não modificar o fluxo principal dos agentes para acomodar testes

O sistema de testes deve consumir o cliente como um alvo externo, mesmo estando no mesmo repositório.

---

## Recomendação Final

Sim, é possível criar isso sem mexer na arquitetura principal do cliente.

A melhor abordagem é:

- construir um **Test Harness desacoplado**
- integrar via **Adapter**
- tratar o backend atual apenas como **Target**
- modelar a solução para reutilização futura em outros clientes

Essa abordagem preserva o projeto atual e abre caminho para um sistema de avaliação reutilizável, profissional e escalável.

---

## Próximo Passo Recomendado

Próxima etapa ideal:

1. definir o modelo de dados do MVP
2. definir a interface do adapter
3. desenhar a primeira versão da UI de `Testes`
4. implementar o fluxo:
   - suite
   - caso
   - run
   - score

Se esta direção for aprovada, o passo seguinte é produzir uma especificação de MVP técnico com:

- entidades
- rotas
- componentes
- fluxo de execução
- ordem de implementação

