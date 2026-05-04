"""
orch_schema.py — State schema e structured output do orquestrador.

Separado de orchestrator.py para eliminar dependência circular:
  orch_models.py usa ClassificationResult → importa de cá
  orchestrator.py usa ambos → importa de cá
"""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


# ============================================================
# Estado do orquestrador
# ============================================================

class OrchestratorState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    llm_model: str
    chosen_agent_ids: List[int]
    chosen_agent_names: List[str]
    execution_plan: List[int]
    agent_responses: List[Dict[str, Any]]
    final_response: str
    primary_agent_id: int
    primary_agent_name: str
    user_profile: Optional[Dict[str, Any]]
    routing_confidence: float
    routing_bucket: str
    routing_reason: str
    routing_alternatives: List[int]
    fallback_used: bool
    fallback_attempts: int
    fallback_trigger: str
    previous_agent_responses: List[Dict[str, Any]]
    general_index_fallback_used: bool
    web_fallback_used: bool
    web_fallback_sources: List[Dict[str, str]]
    needs_clarification: bool


# ============================================================
# Schema de classificação (structured output do LLM)
# ============================================================

class ClassificationResult(BaseModel):
    """
    agent_ids: Lista de IDs relevantes, ordenada por relevância.
               Deve SEMPRE incluir 0 e 3 para perguntas de laticínios.
               [] apenas para saudações ou tópicos fora do setor.
    confidence: Grau de confiança do roteamento (0.0 a 1.0).
    reason/reasoning: Justificativa breve (para debug).
    alternatives: IDs alternativos relevantes para fallback/planner.
    """
    agent_ids: List[int]
    confidence: float = 0.50
    reason: str = ""
    alternatives: List[int] = []
    reasoning: str = ""
