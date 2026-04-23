# Guia de Homologação (RAG + Orquestração)

## Arquivo de registro
- Preencha: `docs/homologacao/TESTES_RAG_ORQUESTRACAO_TEMPLATE.csv`

## Como executar (rápido)
1. Rode 10 perguntas objetivas (1 domínio por pergunta).
2. Para cada pergunta, registre:
- agente esperado vs agente retornado
- top-3 chunks (fonte e trecho)
- se a resposta final está factual e sem ruído
3. Marque causa raiz quando falhar.

## Critério de aprovação sugerido
- Roteamento correto: >= 90%
- Resposta factual correta: >= 90%
- Ruído/disclaimer indevido: <= 10%
- Alucinação crítica: 0%

## Causa raiz (padrão)
- `orquestracao`: agente errado ou chamada excessiva de agentes
- `retrieval`: chunks sem evidência direta
- `consolidacao`: mistura de resposta correta com ruído
- `prompt`: extrapolação, conselho extra, ou linguagem fora do escopo
- `dados`: documento/chunk realmente não contém o que foi perguntado

## Ações recomendadas por causa
- `orquestracao`: reforçar regras determinísticas + few-shot de roteamento
- `retrieval`: query rewrite, filtro por agente/doc_type, rerank
- `consolidacao`: descartar agente sem evidência útil na resposta final
- `prompt`: “responder apenas com evidência recuperada” + sem extrapolação
- `dados`: revisar/normalizar markdown e reingestão

## SQL rápido de sanidade de ingestão/chunking
```sql
-- 1) Detectar possível mojibake (ajuste tabela/source)
select
  count(*) as total,
  count(*) filter (
    where content like '%Ã%'
       or content like '%Â%'
       or content like '%ï¿%'
  ) as possivel_mojibake
from embeddings_agente_3_regulatorios
where metadata->>'source' = 'INSTRUCAO_NORMATIVA_65_2020_Ricota.md';
```

```sql
-- 2) Detectar chunk “somente título” (ideal: 0)
select id, metadata->>'chunk_index' as chunk_index, content
from embeddings_agente_3_regulatorios
where metadata->>'source' = 'INSTRUCAO_NORMATIVA_65_2020_Ricota.md'
  and content !~ E'\\n'
  and (
    content like '% > %'
    or content ~ '^Art\\.'
    or content ~ '^§'
  )
order by id;
```

```sql
-- 3) Ver chunks da fonte para inspeção manual
select id, metadata->>'chunk_index' as chunk_index, left(content, 240) as preview
from embeddings_agente_3_regulatorios
where metadata->>'source' = 'INSTRUCAO_NORMATIVA_65_2020_Ricota.md'
order by id;
```
