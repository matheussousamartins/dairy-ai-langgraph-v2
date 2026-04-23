# Notas de Execucao do Dia 1 (Programa de Roteamento)

Esta pasta contem os artefatos de nivel enterprise do Dia 1 para endurecimento do orquestrador.

## Artefatos

- `ROUTING_BASELINE_REPORT.md`
  - Gerado a partir do dataset atual.
  - Inclui distribuicao por agente/tabela/grupo e contrato de KPIs.

- `AGENT_ROUTING_TAXONOMY.yaml`
  - Taxonomia canonica de dominios e mapa de gatilhos para todos os especialistas.
  - Inclui pares de confusao e dicas de desambiguacao.

- `ROUTING_CONFIDENCE_POLICY.md`
  - Buckets oficiais de confianca e politica de fallback.
  - Define metas de SLO e logs obrigatorios.

## Como regenerar o baseline

Execute:

```powershell
$env:PYTHONPATH="."
python scripts/build_routing_baseline.py
```

## Criterios de aceite do Dia 1

- O relatorio de baseline existe e e reproduzivel.
- A taxonomia cobre os agentes 1..6 e a politica transversal.
- A politica de confianca tem limiares numericos e contrato de fallback.
- O time pode iniciar o Dia 2 sem ambiguidades.
