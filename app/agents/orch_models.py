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
    CONSOLIDATION_TEMPERATURE,
    DIRECT_TEMPERATURE,
)
from app.agents.orch_schema import ClassificationResult, OrchestratorState


# Caches por nome de modelo (evita recriar instâncias a cada chamada)
_classifier_models: Dict[str, Any] = {}
_consolidation_models: Dict[str, Any] = {}
_direct_models: Dict[str, Any] = {}


def _resolve_state_model(state: OrchestratorState) -> str:
    return str(state.get("llm_model") or LLM_MODEL)


def _get_classifier(model_name: str):
    if model_name not in _classifier_models:
        _classifier_models[model_name] = ChatOpenAI(
            model=model_name,
            temperature=CLASSIFIER_TEMPERATURE,
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
        )
    return _consolidation_models[model_name]


def _get_direct_model(model_name: str):
    if model_name not in _direct_models:
        _direct_models[model_name] = ChatOpenAI(
            model=model_name,
            temperature=DIRECT_TEMPERATURE,
        )
    return _direct_models[model_name]
