"""Exports de grafos para LangGraph Studio.

O LangGraph carrega grafos por referência `modulo:variavel`.
Por isso, expomos aqui variáveis já compiladas para cada agente.
"""

from app.agents.base_agent import build_agent_graph
from app.agents.orchestrator import get_orchestrator_graph
from app.db.connection import init_pools
from app.rag.ingest import graph as rag_ingest

# Garante pools de banco inicializados quando os grafos são carregados
# via LangGraph Studio (fora do ciclo de vida do FastAPI).
init_pools()


agente_1_queijos = build_agent_graph(1)
agente_2_fermentados = build_agent_graph(2)
agente_3_regulatorios = build_agent_graph(3)
agente_4_qualidade_leite = build_agent_graph(4)
agente_5_defeitos = build_agent_graph(5)
agente_6_formulacao = build_agent_graph(6)
agente_0_base_geral = build_agent_graph(0)
orquestrador = get_orchestrator_graph()
