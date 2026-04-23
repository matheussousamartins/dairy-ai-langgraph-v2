# Politica de Confianca de Roteamento (Dia 1)

Esta e a politica oficial de roteamento do orquestrador.

## Objetivo

Balancear velocidade, custo e corretude para uma arquitetura com 6 especialistas.

## Camadas de Decisao

1. Camada deterministica (fast path):
- Se uma regra de alta confianca casar, rotear sem classifier LLM.
- Saidas tipicas: `[0,3]`, `[0,3,X]`, `[0,3,X,Y]`.

2. Camada de classificacao (LLM):
- Acionada apenas quando a camada deterministica estiver ambigua.
- Deve retornar saida estruturada com:
  - `agent_ids`
  - `confidence` (0.0 a 1.0)
  - `reason`

3. Camada de execucao:
- A quantidade de agentes executados depende do bucket de confianca.

## Buckets de Confianca

- Alta confianca (`>= 0.85`):
  - Executar: especialista principal + agentes transversais obrigatorios.
  - Conjunto tipico: `[0,3,X]`.

- Media confianca (`0.60 a 0.84`):
  - Executar: top-2 especialistas + agentes transversais obrigatorios.
  - Conjunto tipico: `[0,3,X,Y]`.

- Baixa confianca (`< 0.60`):
  - Executar: top-3 especialistas + agentes transversais obrigatorios.
  - Conjunto tipico: `[0,3,X,Y,Z]` (maximo de 5 especialistas no total dentro do limite atual da arquitetura).

## Regras de Fallback

- Se todos os especialistas retornarem evidencia fraca ou "nao encontrado":
  - Rodar uma passada de fallback com dominios vizinhos (nearest neighbor domains).
- Se um especialista retornar resposta factual e os demais apenas incerteza:
  - Manter a resposta factual e remover blocos apenas de incerteza da consolidacao.
- Se forem detectados conflitos de formulas ou limites:
  - Aplicar primeiro a regra canonica do dominio com base na fonte especialista.

## Metas Operacionais de SLO

- `Routing@1 >= 90%`
- `Routing@3 >= 97%`
- `Fallback Rate <= 12%`
- `Cross-Agent Conflict Rate <= 3%`
- `P95 Latency <= 4.5s`

## Contrato de Logging (obrigatorio)

Por request, registrar:

- hash da pergunta do usuario
- `agent_ids` selecionados
- bucket de confianca
- fallback usado (`true/false`)
- agente primario final
- latencia (ms)
- estimativa de tokens/custo

Esse logging e obrigatorio para calibracao semanal de roteamento.
