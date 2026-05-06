"""
orch_models.py — Lazy initialization dos modelos LLM do orquestrador.

Separa a criação e cache de instâncias ChatOpenAI do restante do
orchestrator.py para facilitar testes e troca de provider.
"""

from typing import Any, Dict

from langchain_openai import ChatOpenAI

from app.config import (
    LLM_MODEL,
    CLASSIFIER_TEMPERATURE,
    CLASSIFIER_MAX_TOKENS,
    CONSOLIDATION_TEMPERATURE,
    CONSOLIDATION_MAX_TOKENS,
    DIRECT_TEMPERATURE,
    DIRECT_MAX_TOKENS,
)
from app.agents.orch_schema import ClassificationResult, OrchestratorState


# Caches por nome de modelo (evita recriar instâncias a cada chamada)
_classifier_models: Dict[str, Any] = {}
_consolidation_models: Dict[str, Any] = {}
_direct_models: Dict[str, Any] = {}

# Modelos que não têm capacidade de síntese factual confiável.
# O consolidador nunca deve usar esses modelos — ele recebe evidências
# brutas e precisa preservar valores numéricos com precisão.
_WEAK_CONSOLIDATION_MODELS = {"gpt-4o-mini", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct"}

# Modelo mínimo garantido para consolidação quando o modelo da UI é fraco.
_CONSOLIDATION_FLOOR_MODEL = "gpt-4o"


def _resolve_state_model(state: OrchestratorState) -> str:
    return str(state.get("llm_model") or LLM_MODEL)


def _resolve_consolidation_model(state: OrchestratorState) -> str:
    """Retorna o modelo para consolidação — nunca abaixo do floor de capacidade."""
    requested = _resolve_state_model(state)
    if requested in _WEAK_CONSOLIDATION_MODELS:
        return _CONSOLIDATION_FLOOR_MODEL
    return requested


def _get_classifier(model_name: str):
    if model_name not in _classifier_models:
        _classifier_models[model_name] = ChatOpenAI(
            model=model_name,
            temperature=CLASSIFIER_TEMPERATURE,
            max_tokens=CLASSIFIER_MAX_TOKENS,
        ).with_structured_output(
            ClassificationResult,
            method="function_calling",
        )
    return _classifier_models[model_name]


def _get_consolidation_model(model_name: str):
    if model_name not in _consolidation_models:
        _consolidation_models[model_name] = ChatOpenAI(
            model=model_name,
            temperature=CONSOLIDATION_TEMPERATURE,
            max_tokens=CONSOLIDATION_MAX_TOKENS,
        )
    return _consolidation_models[model_name]


def _get_direct_model(model_name: str):
    if model_name not in _direct_models:
        _direct_models[model_name] = ChatOpenAI(
            model=model_name,
            temperature=DIRECT_TEMPERATURE,
            max_tokens=DIRECT_MAX_TOKENS,
        )
    return _direct_models[model_name]
