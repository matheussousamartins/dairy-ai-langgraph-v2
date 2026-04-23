# Relatorio de Baseline de Roteamento (Dia 1)

- Gerado em (UTC): `2026-04-20 15:23:57Z`
- Arquivo de dataset: `C:/Users/User/Documents/dairy_ai_langgraph_v2/tests/fixtures/rag/rag_queries.yaml`
- Total de consultas: **304**
- Agentes cobertos no dataset: **4 / 7**
- Consultas com `expected_all` multi-termo: **0**
- Consultas sem campo `expected`: **0**

## Distribuicao por Agente

| agent_id | agent_name | queries | share |
|---:|---|---:|---:|
| 0 | Base Geral Dairy | 20 | 6.6% |
| 1 | Tecnologia de Queijos | 80 | 26.3% |
| 3 | Regulatórios por País | 105 | 34.5% |
| 4 | Qualidade do Leite | 99 | 32.6% |

- `agent_ids` ausentes no dataset: `[2, 5, 6]`

## Distribuicao por Tabela

| table_name | queries | share |
|---|---:|---:|
| embeddings_agente_3_regulatorios | 105 | 34.5% |
| embeddings_agente_4_qualidade_leite | 99 | 32.6% |
| embeddings_agente_1_queijos | 80 | 26.3% |
| embeddings_agente_0_base_geral | 20 | 6.6% |

## Principais Grupos de Consulta

| group | queries |
|---|---:|
| riispoa_decreto_9_013_2017 | 15 |
| in_68_metodos_quantitativos | 15 |
| in_68_caracteristicas_sensoriais_e_preparo_de_amostras | 12 |
| in_68_metodos_qualitativos | 12 |
| glossario_dairy_rag | 10 |
| verdades_absolutas_rag | 10 |
| edi_o_138_queijos_duros_com_heran_a_italiana | 10 |
| edicao_152_browning_pizza | 10 |
| edi_o_159_160_cottage_um_queijo_simples_com_ingredientes_naturais | 10 |
| edi_o_162_inova_o_no_cultivo_que_transforma_a_rela_o_rendimento_e_produtividade_em_queijos_prensados | 10 |
| edi_o_164_quanto_vale_o_sabor_cultivos_adjuntos_em_queijos_maturados | 10 |
| edicao_166_explorando_a_rentabilidade_na_produ_o_de_queijos | 10 |
| edicao_169_parte_1_defeitos_mais_comuns_em_queijos | 10 |
| edicao_170_parte_2_defeitos_mais_comuns_em_queijos | 10 |
| instrucao_normativa_65_2020_ricota | 10 |
| instrucao_normativa_66_2020_minas_padrao | 10 |
| instrucao_normativa_71_2020_cream_cheese | 10 |
| instrucao_normativa_72_2020_sobremesa_lactea | 10 |
| instrucao_normativa_73_2020_provolone | 10 |
| instrucao_normativa_74_2020_minas_meia_cura | 10 |
| mp_772_2017_multas | 10 |
| rdc_53_2014_lista_de_enzimas | 10 |
| rdc_54_2012_informacao_nutricional_complementar | 10 |
| in_68_metodos_analiticos_oficiais_fisico_quimicos_para_controle_de_leite | 10 |
| in_68_solucoes_indicadora | 10 |

## Distribuicao de Grupos por Agente

### Agent 0 - Base Geral Dairy

| group | queries |
|---|---:|
| glossario_dairy_rag | 10 |
| verdades_absolutas_rag | 10 |

### Agent 1 - Tecnologia de Queijos

| group | queries |
|---|---:|
| edi_o_138_queijos_duros_com_heran_a_italiana | 10 |
| edicao_152_browning_pizza | 10 |
| edi_o_159_160_cottage_um_queijo_simples_com_ingredientes_naturais | 10 |
| edi_o_162_inova_o_no_cultivo_que_transforma_a_rela_o_rendimento_e_produtividade_em_queijos_prensados | 10 |
| edi_o_164_quanto_vale_o_sabor_cultivos_adjuntos_em_queijos_maturados | 10 |
| edicao_166_explorando_a_rentabilidade_na_produ_o_de_queijos | 10 |
| edicao_169_parte_1_defeitos_mais_comuns_em_queijos | 10 |
| edicao_170_parte_2_defeitos_mais_comuns_em_queijos | 10 |

### Agent 3 - Regulatórios por País

| group | queries |
|---|---:|
| riispoa_decreto_9_013_2017 | 15 |
| instrucao_normativa_65_2020_ricota | 10 |
| instrucao_normativa_66_2020_minas_padrao | 10 |
| instrucao_normativa_71_2020_cream_cheese | 10 |
| instrucao_normativa_72_2020_sobremesa_lactea | 10 |
| instrucao_normativa_73_2020_provolone | 10 |
| instrucao_normativa_74_2020_minas_meia_cura | 10 |
| mp_772_2017_multas | 10 |
| rdc_53_2014_lista_de_enzimas | 10 |
| rdc_54_2012_informacao_nutricional_complementar | 10 |

### Agent 4 - Qualidade do Leite

| group | queries |
|---|---:|
| in_68_metodos_quantitativos | 15 |
| in_68_caracteristicas_sensoriais_e_preparo_de_amostras | 12 |
| in_68_metodos_qualitativos | 12 |
| in_68_metodos_analiticos_oficiais_fisico_quimicos_para_controle_de_leite | 10 |
| in_68_solucoes_indicadora | 10 |
| in_68_solucoes_padroes | 10 |
| in_68_solucoes_tampoes | 10 |
| in_68_recomendacoes_gerais | 10 |
| qualidade_leite | 10 |

## Contrato de Metricas de Baseline (Dia 1)

Acompanhar estas metricas como KPIs oficiais de roteamento a partir do Dia 2:

- `Routing@1`: agente primario selecionado coincide com o dominio primario esperado.
- `Routing@3`: dominio esperado aparece entre os 3 primeiros agentes selecionados.
- `Fallback Rate`: % de requests que precisaram de segunda tentativa de roteamento.
- `Cross-Agent Conflict Rate`: % de respostas com conflito entre especialistas.
- `Answer Accuracy`: avaliada contra a resposta esperada no dataset.
- `P95 Latency`: tempo ponta a ponta da resposta.
- `Cost per Request`: custo de modelo + retrieval.

Faixas iniciais de meta (enterprise-ready, para validar no Dia 2):

- `Routing@1 >= 90%`
- `Routing@3 >= 97%`
- `Fallback Rate <= 12%`
- `Cross-Agent Conflict Rate <= 3%`
- `P95 Latency <= 4.5s` (orchestrator stream)
