"""
agents/orchestrator.py â€” Orquestrador multi-agente com execuÃ§Ã£o paralela

Fluxo do grafo:
  classify â†’ route â†’ execute (paralelo) â†’ consolidate â†’ END
                â†˜ respond_direct â†’ consolidate â†’ END

Agentes 0 (Base Geral) e 3 (Regulatórios) são SEMPRE incluídos
para qualquer pergunta sobre laticÃ­nios â€” o classificador Ã© instruÃ­do
a retorná-los obrigatoriamente.

Execução paralela:
  Todos os agentes rodam ao mesmo tempo via asyncio.gather + ainvoke.
  Latência total = tempo do agente mais lento (não a soma).
"""

import asyncio
import os
import re
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from app.config import (
    LLM_MODEL,
    CLASSIFIER_TEMPERATURE,
    CONSOLIDATION_TEMPERATURE,
    DIRECT_TEMPERATURE,
    ORCHESTRATOR_FASTPATH,
    CLASSIFICATION_CACHE_SIZE,
)
from app.agents.prompts import get_orchestrator_prompt
from app.agents.agent_config import AGENTS, get_agent_by_id
from app.agents.base_agent import get_agent_graph

# Tempo máximo de espera por agente (segundos)
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "12"))
_SPECIALISTS_DESC = "".join(
    f"  {agent['agent_id']} = {agent['name']}\n"
    for agent in AGENTS
    if agent["agent_id"] not in (0, 3)
)

_CLASSIFICATION_CACHE: "OrderedDict[str, List[int]]" = OrderedDict()
_MAX_CLASSIFICATION_CACHE = max(0, CLASSIFICATION_CACHE_SIZE)
_GREETINGS = {
    "oi", "ola", "olá", "bom dia", "boa tarde", "boa noite",
    "e ai", "e aí", "tudo bem", "blz", "beleza",
}
_DAIRY_TERMS = {
    "leite", "lacteo", "laticinio", "laticinios", "queijo",
    "iogurte", "fermentado", "ricota", "requeijao", "mussarela",
    "coalhada", "soro", "pasteurizacao", "ccs", "cbt", "rtiq", "rdc",
}


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _strip_profile_suffix(text: str) -> str:
    if "\n[Perfil" in text:
        return text.split("\n[Perfil", 1)[0]
    return text


def _build_keyword_sets() -> Dict[int, set]:
    keyword_sets: Dict[int, set] = {}
    for agent in AGENTS:
        aid = agent["agent_id"]
        if aid in (0, 3):
            continue
        raw_keywords = agent.get("keywords", []) or []
        words = {
            _normalize_text(str(k))
            for k in raw_keywords
            if isinstance(k, str) and len(_normalize_text(k)) >= 4
        }
        keyword_sets[aid] = words
    return keyword_sets


_SPECIALIST_KEYWORDS = _build_keyword_sets()


def _cache_get(cache_key: str) -> Optional[List[int]]:
    if _MAX_CLASSIFICATION_CACHE <= 0:
        return None
    cached = _CLASSIFICATION_CACHE.get(cache_key)
    if cached is None:
        return None
    _CLASSIFICATION_CACHE.move_to_end(cache_key)
    return list(cached)


def _cache_set(cache_key: str, agent_ids: List[int]) -> None:
    if _MAX_CLASSIFICATION_CACHE <= 0:
        return
    _CLASSIFICATION_CACHE[cache_key] = list(agent_ids)
    _CLASSIFICATION_CACHE.move_to_end(cache_key)
    while len(_CLASSIFICATION_CACHE) > _MAX_CLASSIFICATION_CACHE:
        _CLASSIFICATION_CACHE.popitem(last=False)


def _looks_like_greeting_only(text_norm: str) -> bool:
    if not text_norm:
        return False
    if text_norm in _GREETINGS:
        return True
    if len(text_norm.split()) <= 4 and any(text_norm.startswith(g) for g in _GREETINGS):
        return True
    return False


def _contains_dairy_signal(text_norm: str) -> bool:
    if any(term in text_norm for term in _DAIRY_TERMS):
        return True
    if re.search(r"\b(in|rdc|rtiq)\s*\d{1,4}\b", text_norm):
        return True
    return False


def _rule_based_route(user_text: str) -> Optional[List[int]]:
    text = _normalize_text(_strip_profile_suffix(user_text))
    if not text:
        return []

    if _looks_like_greeting_only(text):
        return []

    specialist_scores: List[tuple[int, int]] = []
    for aid, keywords in _SPECIALIST_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw and kw in text)
        if score > 0:
            specialist_scores.append((aid, score))

    specialist_scores.sort(key=lambda x: x[1], reverse=True)

    # Alta confiança: 2+ keywords do mesmo especialista.
    high_conf = [aid for aid, score in specialist_scores if score >= 2]
    if high_conf:
        ids = [0, 3] + high_conf[:3]
        return ids

    # Domínio dairy evidente, mas sem especialista forte -> baseline [0, 3].
    if _contains_dairy_signal(text):
        return [0, 3]

    # Baixa confiança: deixar o classificador LLM decidir.
    return None


# ============================================================
# Estado do orquestrador
# ============================================================

class OrchestratorState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    chosen_agent_ids: List[int]
    chosen_agent_names: List[str]
    agent_responses: List[Dict[str, Any]]
    final_response: str
    primary_agent_id: int
    primary_agent_name: str
    user_profile: Optional[Dict[str, Any]]


# ============================================================
# Schema de classificação
# ============================================================

class ClassificationResult(BaseModel):
    """
    agent_ids: Lista de IDs relevantes, ordenada por relevância.
               Deve SEMPRE incluir 0 e 3 para perguntas de laticínios.
               [] apenas para saudações ou tópicos fora do setor.
    reasoning: Justificativa breve (para debug).
    """
    agent_ids: List[int]
    reasoning: str


# ============================================================
# Lazy init dos modelos
# ============================================================

_classifier_model = None
_consolidation_model = None
_direct_model = None


def _get_classifier():
    global _classifier_model
    if _classifier_model is None:
        _classifier_model = ChatOpenAI(model=LLM_MODEL, temperature=CLASSIFIER_TEMPERATURE).with_structured_output(
            ClassificationResult
        )
    return _classifier_model


def _get_consolidation_model():
    global _consolidation_model
    if _consolidation_model is None:
        _consolidation_model = ChatOpenAI(model=LLM_MODEL, temperature=CONSOLIDATION_TEMPERATURE)
    return _consolidation_model


def _get_direct_model():
    global _direct_model
    if _direct_model is None:
        _direct_model = ChatOpenAI(model=LLM_MODEL, temperature=DIRECT_TEMPERATURE)
    return _direct_model


# ============================================================
# Nó CLASSIFY
# ============================================================

def _get_last_user_text(messages: List[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _build_classification_state(agent_ids: List[int]) -> OrchestratorState:
    if not agent_ids:
        return {
            "chosen_agent_ids": [],
            "chosen_agent_names": [],
            "primary_agent_id": 0,
            "primary_agent_name": "Assistente Geral",
            "agent_responses": [],
            "final_response": "",
        }

    agent_names = []
    for aid in agent_ids:
        cfg = get_agent_by_id(aid)
        agent_names.append(cfg["name"] if cfg else f"Agente {aid}")

    return {
        "chosen_agent_ids": agent_ids,
        "chosen_agent_names": agent_names,
        "primary_agent_id": agent_ids[0],
        "primary_agent_name": agent_names[0],
        "agent_responses": [],
        "final_response": "",
    }


async def classify(state: OrchestratorState) -> OrchestratorState:
    """Identifica quais agentes devem ser consultados.

    Agentes 0 e 3 são SEMPRE obrigatórios para qualquer pergunta
    de laticÃ­nios â€” o prompt instrui o LLM explicitamente.
    """
    messages = state.get("messages", [])
    user_text = _get_last_user_text(messages)

    if not user_text:
        return _build_classification_state([])

    route_text = _strip_profile_suffix(user_text)
    cache_key = _normalize_text(route_text)

    cached_ids = _cache_get(cache_key)
    if cached_ids is not None:
        return _build_classification_state(cached_ids)

    if ORCHESTRATOR_FASTPATH:
        fast_ids = _rule_based_route(route_text)
        if fast_ids is not None:
            _cache_set(cache_key, fast_ids)
            return _build_classification_state(fast_ids)

    system_prompt = get_orchestrator_prompt()

    classification_instruction = f"""

Com base na pergunta do usuário, identifique quais agentes devem ser consultados.

REGRA OBRIGATÃ“RIA:
- Para QUALQUER pergunta relacionada a laticínios (produtos, processos,
  ingredientes, fabricantes, distribuidores, equipamentos, normas, qualidade,
  defeitos, formulação, legislação), SEMPRE inclua os agentes 0 e 3 na lista.
- Agente 0 (Base Geral Dairy): glossário, produtos, fabricantes, ingredientes,
  distribuidores, equipamentos â€” base de conhecimento transversal.
- Agente 3 (Regulatórios por País): normas, legislação, requisitos legais.

ESPECIALISTAS (adicione apenas se a pergunta for claramente desse domínio):
{_SPECIALISTS_DESC}
FORMATO DA RESPOSTA:
- SaudaÃ§Ã£o / off-topic (sem relaÃ§Ã£o com laticÃ­nios) â†’ []
- Pergunta de laticÃ­nios sem especialidade clara â†’ [0, 3]
- Pergunta com especialidade clara â†’ [0, 3, X]
- Pergunta com mÃºltiplas especialidades â†’ [0, 3, X, Y] (mÃ¡x 5 IDs)
- Ordene por relevância: o agente mais relevante primeiro.
"""

    classifier = _get_classifier()
    result = await classifier.ainvoke([
        SystemMessage(content=system_prompt + classification_instruction),
        HumanMessage(content=user_text),
    ])

    # Valida IDs (0-6), preserva ordem, remove duplicatas
    seen = set()
    agent_ids: List[int] = []
    for aid in result.agent_ids:
        if 0 <= aid <= 6 and aid not in seen:
            seen.add(aid)
            agent_ids.append(aid)

    if not agent_ids:
        _cache_set(cache_key, [])
        return _build_classification_state([])

    _cache_set(cache_key, agent_ids)
    return _build_classification_state(agent_ids)

# ============================================================
# Roteamento condicional
# ============================================================

def route(state: OrchestratorState) -> str:
    return "respond_direct" if not state.get("chosen_agent_ids") else "execute"


# ============================================================
# NÃ³ EXECUTE â€” execuÃ§Ã£o paralela
# ============================================================

async def execute(state: OrchestratorState) -> OrchestratorState:
    """Invoca todos os agentes em PARALELO via asyncio.gather.

    LatÃªncia total â‰ˆ tempo do agente mais lento (nÃ£o a soma).
    Cada agente tem timeout individual de AGENT_TIMEOUT segundos.
    """
    agent_ids = state.get("chosen_agent_ids", [])
    agent_names = state.get("chosen_agent_names", [])

    user_text = _get_last_user_text(state.get("messages", []))

    if not user_text:
        return {"agent_responses": []}

    async def call_one(agent_id: int, agent_name: str) -> Dict[str, Any]:
        try:
            graph = get_agent_graph(agent_id)
            result = await asyncio.wait_for(
                graph.ainvoke({"messages": [HumanMessage(content=user_text)]}),
                timeout=AGENT_TIMEOUT,
            )
            agent_msgs = result.get("messages", [])
            agent_text = ""
            for msg in reversed(agent_msgs):
                if isinstance(msg, AIMessage):
                    content = msg.content
                    if isinstance(content, list):
                        agent_text = "\n".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        )
                    elif isinstance(content, str):
                        agent_text = content
                    if agent_text:
                        break
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "response": agent_text,
                "success": bool(agent_text),
            }
        except asyncio.TimeoutError:
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "response": f"{agent_name}: timeout ao consultar base de conhecimento.",
                "success": False,
            }
        except Exception as e:
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "response": f"Erro ao consultar {agent_name}: {e}",
                "success": False,
            }

    # Dispara todos os agentes ao mesmo tempo
    responses = await asyncio.gather(
        *[call_one(aid, name) for aid, name in zip(agent_ids, agent_names)]
    )

    return {"agent_responses": list(responses)}


# ============================================================
# NÃ³ RESPOND_DIRECT â€” saudaÃ§Ãµes e off-topic
# ============================================================

async def respond_direct(state: OrchestratorState) -> OrchestratorState:
    """Resposta direta para saudações e mensagens off-topic (sem RAG)."""
    user_text = _get_last_user_text(state.get("messages", []))

    system = (
        "Você é o assistente geral do DairyApp AI, especializado em tecnologia "
        "de laticínios. Responda de forma amigável. Quando pertinente, sugira que "
        "o usuário faça perguntas técnicas sobre queijos, fermentados, regulatórios, "
        "qualidade do leite, diagnóstico de defeitos ou formulação. "
        "Responda em português brasileiro."
    )

    response = await _get_direct_model().ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=user_text),
    ])

    final_text = response.content or ""
    return {
        "agent_responses": [],
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
    }


# ============================================================
# NÃ³ CONSOLIDATE â€” fusÃ£o das respostas
# ============================================================

async def consolidate(state: OrchestratorState) -> OrchestratorState:
    """Funde as respostas dos agentes em uma resposta coerente.

    1 agente bem-sucedido â†’ repassa direto (sem chamada LLM extra).
    2+ agentes â†’ LLM funde preservando todos os dados tÃ©cnicos.
    """
    # Veio de respond_direct: já tem final_response
    if not state.get("chosen_agent_ids") and state.get("final_response"):
        final_text = state.get("final_response") or ""
        if final_text:
            msgs = state.get("messages", [])
            if msgs and isinstance(msgs[-1], AIMessage) and (msgs[-1].content or "") == final_text:
                return {}
            return {"messages": [AIMessage(content=final_text)]}
        return {}

    successful = [
        r for r in state.get("agent_responses", [])
        if r.get("success") and r.get("response")
    ]

    if not successful:
        final_text = (
            "Não foi possível obter uma resposta no momento. "
            "Por favor, tente reformular sua pergunta."
        )
        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
        }

    # 1 agente: repassa direto (econômico)
    if len(successful) == 1:
        final_text = successful[0]["response"]
        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
        }

    # 2+ agentes: consolida com LLM
    user_text = _get_last_user_text(state.get("messages", []))

    responses_text = "".join(
        f"\n--- {r['agent_name']} ---\n{r['response']}\n"
        for r in successful
    )

    consolidation_prompt = (
        "Você é o assistente geral do DairyApp AI. Recebeu respostas de múltiplos "
        "especialistas para a pergunta do usuário. Sua tarefa:\n"
        "- Fundir em UMA resposta coerente e completa\n"
        "- Preservar TODOS os dados técnicos (temperaturas, pHs, normas, prazos)\n"
        "- Não perder informação de nenhum especialista\n"
        "- Não mencionar que consultou múltiplos agentes internos\n"
        "- Tom técnico e profissional em português brasileiro\n\n"
        f"PERGUNTA: {user_text}\n\n"
        f"RESPOSTAS DOS ESPECIALISTAS:{responses_text}\n"
        "Resposta unificada:"
    )

    try:
        response = await _get_consolidation_model().ainvoke(
            [HumanMessage(content=consolidation_prompt)]
        )
        final_text = response.content or ""
    except Exception:
        final_text = "\n\n".join(r["response"] for r in successful)

    return {
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
    }


# ============================================================
# Montagem e compilação do grafo
# ============================================================

def build_orchestrator_graph() -> Any:
    graph = StateGraph(OrchestratorState)

    graph.add_node("classify", classify)
    graph.add_node("execute", execute)
    graph.add_node("respond_direct", respond_direct)
    graph.add_node("consolidate", consolidate)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route,
        {"execute": "execute", "respond_direct": "respond_direct"},
    )

    graph.add_edge("execute", "consolidate")
    graph.add_edge("respond_direct", "consolidate")
    graph.add_edge("consolidate", END)

    return graph.compile()


# ============================================================
# Instância global (lazy cache)
# ============================================================

_orchestrator_graph = None


def get_orchestrator_graph() -> Any:
    global _orchestrator_graph
    if _orchestrator_graph is None:
        _orchestrator_graph = build_orchestrator_graph()
    return _orchestrator_graph

