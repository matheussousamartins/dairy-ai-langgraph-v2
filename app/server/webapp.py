"""
server/webapp.py â€” Servidor FastAPI com os 7 endpoints

Este Ã© o ponto de entrada HTTP do sistema. ExpÃµe os mesmos endpoints
que o N8N, com o mesmo contrato de API â€” o app React Native funciona
com qualquer backend sem mudar uma linha.

No projeto original do curso (app/server/webapp.py), o servidor tem
apenas 2 endpoints:
  - POST /chat â†’ chama o agente principal
  - GET /stream â†’ streaming via SSE

Aqui temos:
  - POST /webhook/agente-{id} â†’ chama o agente especialista (6 endpoints)
  - POST /webhook/orquestrador â†’ chama o orquestrador (1 endpoint)
  - POST /webhook/ingestao â†’ ingestÃ£o de documentos
  - GET /health â†’ status do sistema

Os URLs usam /webhook/ para manter compatibilidade com o N8N.
O app React Native jÃ¡ estÃ¡ configurado para chamar esses endpoints.
Para trocar de N8N para LangGraph, basta mudar a Base URL no app.

Request/Response:
  Os mesmos que documentamos no API-Laticinios-AI-v1.1.docx:
  
  Request:
    { "message": "...", "session_id": "...", "user_profile": {...} }
  
  Response:
    { "response": "...", "agent_id": 1, "agent_name": "Tecnologia de Queijos" }

DiferenÃ§a do original:
  O original usa result.get("structured_response") para extrair a
  resposta, que Ã© o formato especÃ­fico do CRM. Aqui usamos o formato
  do contrato de API de laticÃ­nios (response, agent_id, agent_name).
"""

import asyncio
import json
import re
import hashlib
import secrets
import time
import os
import traceback
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage

from app.config import (
    SERVER_HOST,
    SERVER_PORT,
    CORS_ALLOW_ORIGINS,
    ENFORCE_WEBHOOK_API_KEY,
    WEBHOOK_API_KEY_HEADER,
    WEBHOOK_API_KEYS,
    ORCHESTRATOR_CONTEXT_MEMORY_ENABLED,
    ORCHESTRATOR_CONTEXT_MAX_MESSAGES,
    ORCHESTRATOR_CONTEXT_MAX_CHARS,
    ORCHESTRATOR_CONTEXT_TRIGGER_MAX_CHARS,
    validate_config,
)
from app.db.connection import init_pools, close_pools
from app.db.memory import load_memory, save_chat_turn, save_routing_log
from app.agents.base_agent import get_agent_graph, get_all_agent_graphs
from app.agents.orchestrator import get_orchestrator_graph
from app.agents.agent_config import get_agent_by_id
from app.llm.model_selector import resolve_chat_model
from app.llm.model_selector import get_allowed_chat_models
from app.rag.ingest import ingest_text


# ============================================================
# Lifecycle: startup e shutdown
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida do servidor.
    
    Startup (antes de aceitar requests):
      1. Valida configuraÃ§Ã£o (chaves, URLs)
      2. Inicializa pools de conexÃ£o com os bancos
      3. PrÃ©-compila os grafos dos agentes (aquece o cache)
    
    Shutdown (ao parar o servidor):
      1. Fecha pools de conexÃ£o (libera recursos no banco)
    
    No original, nÃ£o hÃ¡ lifecycle â€” as conexÃµes sÃ£o abertas
    sob demanda e nunca fechadas explicitamente. Aqui Ã© mais
    robusto: pools sÃ£o abertos no inÃ­cio e fechados no fim.
    
    O @asynccontextmanager Ã© o padrÃ£o moderno do FastAPI para
    lifecycle (substituiu os eventos on_startup/on_shutdown).
    Tudo antes do yield Ã© startup, tudo depois Ã© shutdown.
    """
    # --- STARTUP ---
    print("[server] Validando configuraÃ§Ã£o...")
    validate_config()
    
    print("[server] Inicializando pools de conexÃ£o...")
    init_pools()
    
    print("[server] PrÃ©-compilando grafos dos agentes...")
    get_all_agent_graphs()      # Compila os 6 agentes
    get_orchestrator_graph()     # Compila o orquestrador
    
    print("[server] Servidor pronto!")
    print(f"[server] Endpoints disponÃ­veis:")
    print(f"  POST /webhook/agente-{{0..6}}")
    print(f"  POST /webhook/orquestrador")
    print(f"  POST /webhook/ingestao")
    print(f"  POST /webhook/ingestao-arquivo")
    print(f"  GET  /health")
    
    yield  # Servidor rodando e aceitando requests
    
    # --- SHUTDOWN ---
    print("[server] Fechando pools de conexÃ£o...")
    close_pools()
    print("[server] Servidor encerrado.")


# ============================================================
# App FastAPI
# ============================================================

app = FastAPI(
    title="DairyApp AI",
    description="Sistema multi-agente para tecnologia de laticÃ­nios",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: aceita requests de qualquer origem (necessÃ¡rio para o
# console de teste na Vercel e para o app React Native).
# Em produÃ§Ã£o, restrinja para os domÃ­nios do app.
#
# No original, nÃ£o hÃ¡ configuraÃ§Ã£o de CORS. Aqui adicionamos
# porque o console de teste faz requests cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Schemas de request/response (Pydantic)
# ============================================================

class ChatRequest(BaseModel):
    """Schema do body do request de chat.
    
    IdÃªntico ao contrato documentado no API-Laticinios-AI-v1.1.docx.
    O Pydantic valida automaticamente: se faltar "message", retorna
    HTTP 422 com mensagem de erro clara.
    
    No original (webapp.py), o body Ã© lido como dict genÃ©rico:
      body = await req.json()
      text = body.get("input")
    Aqui usamos Pydantic para validaÃ§Ã£o automÃ¡tica.
    """
    message: str
    # Convencao de integracao: session_id deve ser o chatId do app.
    session_id: Optional[str] = None
    # Aceita chat_id/chatId para facilitar integracao com Message Service.
    chat_id: Optional[str] = None
    chatId: Optional[str] = None
    model: Optional[str] = None
    user_profile: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Schema do body da response de chat.
    
    Campos:
      response: Texto da resposta do agente (para exibir ao usuÃ¡rio).
      agent_id: ID do agente que respondeu (0 = base/orquestrador; 1-6 = especialistas).
      agent_name: Nome legÃ­vel do agente.
    """
    response: str
    agent_id: int
    agent_name: str


class IngestRequest(BaseModel):
    """Schema do body do request de ingestÃ£o.
    
    IdÃªntico ao contrato do pipeline de ingestÃ£o do N8N.
    """
    text: str
    table_name: str
    agent_id: int = 0
    source: str = "upload"
    doc_type: str = "manual"


def _verify_webhook_api_key(
    x_api_key: Optional[str] = Header(default=None, alias=WEBHOOK_API_KEY_HEADER),
):
    if not ENFORCE_WEBHOOK_API_KEY:
        return
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if not any(secrets.compare_digest(x_api_key, key) for key in WEBHOOK_API_KEYS):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _history_to_messages(history: list[dict[str, str]]) -> list:
    messages = []
    for msg in history:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages




def _load_history_safe(session_id: str) -> list[dict[str, str]]:
    """Carrega historico sem derrubar request em falha transiente de banco."""
    try:
        return load_memory(session_id)
    except Exception as e:
        print(f"[memory] Aviso: falha ao carregar historico ({session_id}): {e}")
        return []


def _resolve_session_id(request: ChatRequest) -> str:
    """Resolve o identificador de conversa sem transformar o valor.

    Regra:
    - usar `session_id` quando presente
    - fallback para `chat_id`/`chatId`
    - ambos devem representar o chatId do app (UUID/string estavel)
    """
    session_id = (request.session_id or request.chat_id or request.chatId or "").strip()
    if not session_id:
        raise HTTPException(
            status_code=422,
            detail="session_id (ou chat_id) e obrigatorio",
        )
    return session_id

def _inject_user_profile(message: str, user_profile: Optional[Dict[str, Any]]) -> str:
    if not user_profile:
        return message
    profile_text = (
        f"\n[Perfil do usuÃ¡rio: "
        f"nÃ­vel={user_profile.get('knowledgeLevel', 'INTERMEDIATE')}, "
        f"funÃ§Ã£o={user_profile.get('role', 'nÃ£o informado')}]"
    )
    return message + profile_text


def _strip_profile_suffix(text: str) -> str:
    if "\n[Perfil" in text:
        return text.split("\n[Perfil", 1)[0]
    return text


def _normalize_context_probe(text: str) -> str:
    cleaned = _strip_profile_suffix(text or "")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _looks_like_context_dependent_followup(message: str) -> bool:
    if not ORCHESTRATOR_CONTEXT_MEMORY_ENABLED:
        return False

    text = _normalize_context_probe(message)
    if not text:
        return False

    # Follow-ups realmente dependentes do histórico costumam ser curtos e
    # anafóricos. Mensagens longas/independentes seguem no fast-path sem memória.
    if len(text) > max(40, int(ORCHESTRATOR_CONTEXT_TRIGGER_MAX_CHARS)):
        return False

    strong_phrases = (
        "no caso anterior",
        "no contexto anterior",
        "do que falamos",
        "o que falamos antes",
        "o que conversamos",
        "sobre o que conversamos",
        "conversamos recentemente",
        "falamos recentemente",
        "me explique sobre o que conversamos",
        "me lembre do que conversamos",
        "resuma o que conversamos",
        "resuma o que falamos",
        "retome o que falamos",
        "continue de onde paramos",
        "que voce falou",
        "que você falou",
        "compare com",
        "comparando com",
        "em relacao ao anterior",
        "em relação ao anterior",
        "sobre isso",
        "sobre o anterior",
        "nesse caso",
        "neste caso",
        "mesmo caso",
        "mesma coisa",
        "isso muda",
        "isso vale",
        "isso se aplica",
    )
    if any(phrase in text for phrase in strong_phrases):
        return True

    followup_prefixes = (
        "e no caso",
        "e quanto",
        "e para",
        "e se",
        "e no ",
        "agora",
        "entao",
        "então",
        "nesse caso",
        "neste caso",
        "sobre isso",
        "compare",
        "comparando",
    )
    if any(text.startswith(prefix) for prefix in followup_prefixes):
        return True

    # Curto + demonstrativo costuma indicar anáfora real.
    demonstratives = ("isso", "isto", "esse", "essa", "aquele", "aquela", "anterior")
    return any(token in text for token in demonstratives) and len(text.split()) <= 12


def _truncate_memory_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def _select_orchestrator_context_lines(history: list[dict[str, str]]) -> list[str]:
    max_messages = max(0, int(ORCHESTRATOR_CONTEXT_MAX_MESSAGES))
    char_budget = max(0, int(ORCHESTRATOR_CONTEXT_MAX_CHARS))
    if max_messages == 0 or char_budget == 0:
        return []

    per_message_cap = min(320, max(120, char_budget // max(1, max_messages)))
    selected: list[str] = []
    used_chars = 0

    for item in reversed(history or []):
        role = str(item.get("role", "")).strip().lower()
        if role not in {"human", "ai"}:
            continue

        content = str(item.get("content", "")).strip()
        if not content:
            continue

        if role == "human":
            content = _strip_profile_suffix(content)

        label = "Usuario" if role == "human" else "Dairy AI"
        line = f"{label}: {_truncate_memory_text(content, per_message_cap)}"
        projected = used_chars + len(line) + 1

        if selected and projected > char_budget:
            break
        if not selected and projected > char_budget:
            line = _truncate_memory_text(line, char_budget)
            projected = len(line) + 1

        selected.append(line)
        used_chars = projected

        if len(selected) >= max_messages:
            break

    selected.reverse()
    return selected


def _build_orchestrator_input_messages(
    session_id: str,
    message: str,
    user_profile: Optional[Dict[str, Any]],
) -> list:
    current_message = _inject_user_profile(message, user_profile)
    if not _looks_like_context_dependent_followup(message):
        return [HumanMessage(content=current_message)]

    history = _load_history_safe(session_id)
    context_lines = _select_orchestrator_context_lines(history)
    if not context_lines:
        return [HumanMessage(content=current_message)]

    contextual_message = "\n".join(
        [
            "[Contexto recente da conversa]",
            *context_lines,
            "",
            "[Pergunta atual]",
            current_message,
        ]
    ).strip()
    return [HumanMessage(content=contextual_message)]


def _sanitize_math_for_ui(text: str) -> str:
    """Converte trechos matematicos em LaTeX para texto simples amigavel ao front."""
    if not text:
        return text

    out = str(text)
    out = out.replace("\\n", "\n")
    out = out.replace("\t", " ")

    # Delimitadores comuns de math mode (incluindo quando vierem incompletos).
    out = re.sub(r"\\\[(.*?)\\\]", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\\\((.*?)\\\)", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\$\$(.*?)\$\$", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\$(.*?)\$", r"\1", out, flags=re.DOTALL)
    out = out.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")

    # Comandos latex usuais em respostas de calculo.
    out = out.replace("\\times", "x").replace("\\cdot", "x").replace("\\,", " ")
    out = re.sub(r"\\text\{([^}]*)\}", r"\1", out)

    # Limpeza de comandos residuais sem mexer em acentuacao/UTF-8.
    out = re.sub(r"\\[a-zA-Z]+", "", out)
    out = out.replace("{", "").replace("}", "")

    # Remove wrappers [ ... ] usados em blocos matematicos.
    cleaned_lines = []
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]") and len(s) >= 2:
            s = s[1:-1].strip()
        cleaned_lines.append(s if s else line.strip())
    out = "\n".join(cleaned_lines)

    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def _log_server_error(tag: str, exc: Exception) -> None:
    """Loga stack trace completa no servidor sem expor detalhes ao cliente."""
    print(f"[{tag}] {exc}")
    print(traceback.format_exc())


def _backend_failure_response(message: str, status_code: int = 500) -> PlainTextResponse:
    """Resposta simples para facilitar o consumo pelo proxy/frontend."""
    return PlainTextResponse(message, status_code=status_code)


def _safe_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            aid = int(item)
        except (TypeError, ValueError):
            continue
        if aid not in out:
            out.append(aid)
    return out


def _build_query_hash(message: str) -> str:
    normalized = re.sub(r"\s+", " ", (message or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _estimate_routing_cost_usd(execution_plan: list[int], fallback_attempts: int) -> float:
    # Estimativa simples e estável para observabilidade operacional.
    classifier_cost = 0.0005
    per_agent_call_cost = 0.0015
    consolidation_cost = 0.0008
    planned_agents = max(1, len(execution_plan))
    passes = 1 + max(0, int(fallback_attempts or 0))
    estimate = classifier_cost + consolidation_cost + (planned_agents * passes * per_agent_call_cost)
    return round(estimate, 6)


async def _run_bg_db_with_retry(
    tag: str,
    func: Any,
    *args: Any,
    attempts: int = 3,
    base_delay_sec: float = 0.12,
) -> None:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            await run_in_threadpool(func, *args)
            return
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            await asyncio.sleep(base_delay_sec * attempt)

    if last_exc is not None:
        _log_server_error(f"{tag} attempts={attempts}", last_exc)


async def _bg_save_chat_turn(
    session_id: str,
    agent_id: int,
    agent_name: str,
    user_message: str,
    response_text: str,
    elapsed_ms: int,
) -> None:
    """Persiste o turno de chat em background, sem bloquear o response ao usuário."""
    await _run_bg_db_with_retry(
        f"bg-save chat_turn {session_id}",
        save_chat_turn,
        session_id,
        agent_id,
        agent_name,
        user_message,
        response_text,
        elapsed_ms,
    )


async def _bg_save_routing_log(payload: Dict[str, Any]) -> None:
    """Persiste o routing_log em background, sem bloquear o response ao usuário."""
    await _run_bg_db_with_retry(
        f"bg-save routing_log {payload.get('session_id', '?')}",
        save_routing_log,
        payload["session_id"],
        payload["user_message"],
        payload["response_time_ms"],
        payload["query_hash"],
        payload["selected_agent_ids"],
        payload["chosen_agent_ids"],
        payload["execution_plan"],
        payload["primary_agent_id"],
        payload["primary_agent_name"],
        payload["routing_confidence"],
        payload["routing_bucket"],
        payload["routing_reason"],
        payload["routing_alternatives"],
        payload["fallback_used"],
        payload["fallback_attempts"],
        payload["fallback_trigger"],
        payload["cost_estimate_usd"],
    )


def _extract_routing_payload(
    request_message: str,
    orchestrator_output: Dict[str, Any],
    elapsed_ms: int,
    default_agent_id: int,
    default_agent_name: str,
) -> Dict[str, Any]:
    chosen_ids = _safe_int_list(orchestrator_output.get("chosen_agent_ids"))
    execution_plan = _safe_int_list(orchestrator_output.get("execution_plan"))
    selected_ids = execution_plan or chosen_ids
    routing_alternatives = _safe_int_list(orchestrator_output.get("routing_alternatives"))
    fallback_attempts = int(orchestrator_output.get("fallback_attempts", 0) or 0)
    primary_agent_id = int(orchestrator_output.get("primary_agent_id", default_agent_id) or default_agent_id)
    primary_agent_name = str(orchestrator_output.get("primary_agent_name", default_agent_name) or default_agent_name)
    routing_confidence = float(orchestrator_output.get("routing_confidence", 0.0) or 0.0)
    routing_bucket = str(orchestrator_output.get("routing_bucket", "unknown") or "unknown")
    routing_reason = str(orchestrator_output.get("routing_reason", "") or "")
    fallback_used = bool(orchestrator_output.get("fallback_used", False))
    fallback_trigger = str(orchestrator_output.get("fallback_trigger", "") or "")
    cost_estimate_usd = _estimate_routing_cost_usd(execution_plan=selected_ids, fallback_attempts=fallback_attempts)

    return {
        "session_id": "",
        "user_message": request_message,
        "response_time_ms": int(elapsed_ms),
        "query_hash": _build_query_hash(request_message),
        "selected_agent_ids": selected_ids,
        "chosen_agent_ids": chosen_ids,
        "execution_plan": execution_plan,
        "primary_agent_id": primary_agent_id,
        "primary_agent_name": primary_agent_name,
        "routing_confidence": routing_confidence,
        "routing_bucket": routing_bucket,
        "routing_reason": routing_reason,
        "routing_alternatives": routing_alternatives,
        "fallback_used": fallback_used,
        "fallback_attempts": fallback_attempts,
        "fallback_trigger": fallback_trigger,
        "cost_estimate_usd": cost_estimate_usd,
    }


# ============================================================
# Endpoint: POST /webhook/agente-{agent_id}
# ============================================================

@app.post("/webhook/agente-{agent_id}")
async def chat_agent(
    agent_id: int,
    request: ChatRequest,
    _api_key: Optional[str] = Header(default=None, alias=WEBHOOK_API_KEY_HEADER),
) -> ChatResponse:
    """Endpoint dos agentes especialistas.
    
    Recebe a pergunta, carrega o histÃ³rico da sessÃ£o, chama o
    grafo do agente, salva a nova mensagem no histÃ³rico, registra
    o log, e retorna a resposta.
    
    Este endpoint faz o que o N8N faz com 8 nÃ³s:
    Webhook â†’ AI Agent â†’ Postgres Chat Memory â†’ Log â†’ Respond
    
    Aqui Ã© tudo em cÃ³digo, mas o fluxo Ã© idÃªntico:
    1. Valida o agent_id
    2. Carrega histÃ³rico (load_memory)
    3. Chama o grafo (get_agent_graph â†’ invoke)
    4. Salva histÃ³rico (save_memory x2)
    5. Registra log (save_interaction_log)
    6. Retorna response
    
    ParÃ¢metros da URL:
      agent_id: 0 a 6 (validado contra agent_config.py)
    
    Body: ChatRequest (message, session_id, user_profile)
    
    Response: ChatResponse (response, agent_id, agent_name)
    
    Erros:
      404: agent_id não existe
      500: erro interno no grafo
    """
    _verify_webhook_api_key(_api_key)
    start_time = time.time()
    session_id = _resolve_session_id(request)
    try:
        resolved_model = resolve_chat_model(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    # ---- 1. Validar agent_id ----
    agent_config = get_agent_by_id(agent_id)
    if not agent_config:
        raise HTTPException(
            status_code=404,
            detail=f"Agente {agent_id} nÃ£o encontrado. IDs vÃ¡lidos: 0 a 6.",
        )
    
    agent_name = agent_config["name"]
    
    # ---- 2. Carregar histórico da sessão ----
    # Mesma tabela chat_memories que o N8N usa
    history = await run_in_threadpool(_load_history_safe, session_id)
    messages = _history_to_messages(history)
    messages.append(
        HumanMessage(
            content=_inject_user_profile(request.message, request.user_profile)
        )
    )
    
    # ---- 3. Chamar o grafo do agente ----
    try:
        graph = get_agent_graph(agent_id, resolved_model)
        result = await graph.ainvoke({"messages": messages, "llm_model": resolved_model})
        
        # Extrai a resposta (Ãºltima AIMessage)
        response_text = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and not hasattr(msg, "tool_calls"):
                response_text = msg.content
                break
            # AIMessage com content mas sem tool_calls = resposta final
            if hasattr(msg, "content") and hasattr(msg, "tool_calls"):
                if not msg.tool_calls:
                    response_text = msg.content
                    break
        
    except Exception as e:
        _log_server_error(f"agent-{agent_id}", e)
        return _backend_failure_response(
            "Não foi possível processar sua pergunta no momento. Por favor, tente novamente.",
            status_code=500,
        )
    
    response_text = _sanitize_math_for_ui(response_text)

    # ---- 4. Salvar no histórico (background — não bloqueia o response) ----
    elapsed_ms = int((time.time() - start_time) * 1000)
    asyncio.create_task(_bg_save_chat_turn(
        session_id, agent_id, agent_name, request.message, response_text, elapsed_ms
    ))

    # ---- 6. Retornar response ----
    return ChatResponse(
        response=response_text,
        agent_id=agent_id,
        agent_name=agent_name,
    )


# ============================================================
# Endpoint: POST /webhook/orquestrador
# ============================================================

@app.post("/webhook/orquestrador")
async def chat_orchestrator(
    request: ChatRequest,
    _api_key: Optional[str] = Header(default=None, alias=WEBHOOK_API_KEY_HEADER),
) -> ChatResponse:
    """Endpoint do Assistente Geral (orquestrador) â€” multi-agente.
    
    O orquestrador pode consultar 1 a 3 agentes por pergunta.
    O response inclui o agent_id do agente PRINCIPAL (mais relevante).
    Se consultou Agente 3 + Agente 1, o agent_id no response Ã© 3
    (o primeiro da lista ordenada por relevância).
    Se foi conversa geral, agent_id Ã© 0.
    """
    _verify_webhook_api_key(_api_key)
    start_time = time.time()
    session_id = _resolve_session_id(request)
    orchestrator_output: Dict[str, Any] = {}
    try:
        resolved_model = resolve_chat_model(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    # Memória seletiva do orquestrador:
    # - perguntas independentes seguem no fast-path sem I/O de memória
    # - follow-ups reais recebem apenas um contexto recente e compacto
    messages = _build_orchestrator_input_messages(
        session_id=session_id,
        message=request.message,
        user_profile=request.user_profile,
    )
    
    # Chama o orquestrador
    try:
        graph = get_orchestrator_graph()
        result = await graph.ainvoke({
            "messages": messages,
            "llm_model": resolved_model,
            "user_profile": request.user_profile,
        })
        orchestrator_output = dict(result or {})
        
        response_text = _sanitize_math_for_ui(result.get("final_response", ""))
        # primary_agent_id Ã© o agente mais relevante da lista
        agent_id = result.get("primary_agent_id", 0)
        agent_name = result.get("primary_agent_name", "Assistente Geral")
        
    except Exception as e:
        _log_server_error("orquestrador", e)
        return _backend_failure_response(
            "Não foi possível processar sua pergunta no momento. Por favor, tente novamente.",
            status_code=500,
        )
    
    # Salva histórico e log em background — não bloqueia o response ao usuário.
    elapsed_ms = int((time.time() - start_time) * 1000)
    asyncio.create_task(_bg_save_chat_turn(
        session_id, agent_id, agent_name, request.message, response_text, elapsed_ms
    ))
    try:
        payload = _extract_routing_payload(
            request_message=request.message,
            orchestrator_output=orchestrator_output,
            elapsed_ms=elapsed_ms,
            default_agent_id=agent_id,
            default_agent_name=agent_name,
        )
        payload["session_id"] = session_id
        asyncio.create_task(_bg_save_routing_log(payload))
    except Exception as e:
        _log_server_error("routing-log payload", e)

    return ChatResponse(
        response=response_text,
        agent_id=agent_id,
        agent_name=agent_name,
    )


# ============================================================
# Endpoint: POST /webhook/agente-{agent_id}/stream  (SSE)
# ============================================================

@app.post("/webhook/agente-{agent_id}/stream")
async def chat_agent_stream(
    agent_id: int,
    request: ChatRequest,
    _api_key: Optional[str] = Header(default=None, alias=WEBHOOK_API_KEY_HEADER),
):
    """Endpoint de streaming SSE para agentes especialistas.

    Emite tokens conforme o LLM gera a resposta â€” estilo ChatGPT.
    Usa graph.astream_events() para capturar chunks do modelo.

    Eventos SSE emitidos:
      data: {"event": "chunk", "text": "..."}   â€” token(s) da resposta
      data: {"event": "final", "agent_id": N}   â€” sinaliza fim do stream
      data: {"event": "error", "detail": "..."}  â€” erro durante geraÃ§Ã£o
    """
    _verify_webhook_api_key(_api_key)
    start_time = time.time()
    session_id = _resolve_session_id(request)
    try:
        resolved_model = resolve_chat_model(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    agent_config = get_agent_by_id(agent_id)
    if not agent_config:
        raise HTTPException(
            status_code=404,
            detail=f"Agente {agent_id} nÃ£o encontrado. IDs vÃ¡lidos: 0 a 6.",
        )

    agent_name = agent_config["name"]
    history = await run_in_threadpool(_load_history_safe, session_id)
    messages = _history_to_messages(history)
    messages.append(
        HumanMessage(
            content=_inject_user_profile(request.message, request.user_profile)
        )
    )

    graph = get_agent_graph(agent_id, resolved_model)

    async def generate():
        accumulated_raw = ""
        emitted_clean = ""
        try:
            async for event in graph.astream_events({"messages": messages, "llm_model": resolved_model}, version="v2"):
                ev = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")
                ts = int(time.time() * 1000)

                # Tokens da resposta final
                if ev == "on_chat_model_stream" and node == "agent":
                    chunk = event["data"]["chunk"]
                    content = chunk.content if isinstance(chunk.content, str) else ""
                    tool_calls = getattr(chunk, "tool_call_chunks", [])
                    if content and not tool_calls:
                        accumulated_raw += content
                        cleaned = _sanitize_math_for_ui(accumulated_raw)
                        if len(cleaned) > len(emitted_clean):
                            delta = cleaned[len(emitted_clean):]
                            emitted_clean = cleaned
                            if delta:
                                yield f"data: {json.dumps({'event': 'chunk', 'text': delta})}\n\n"

                # TransiÃ§Ãµes de nÃ³ â€” apenas eventos de nÃ­vel raiz do LangGraph
                # (event["name"] == node filtra sub-chains internas)
                elif ev == "on_chain_start" and node and node != "__start__" and event.get("name") == node:
                    yield f"data: {json.dumps({'event': 'trace', 'type': 'node_start', 'node': node, 'ts': ts})}\n\n"
                elif ev == "on_chain_end" and node and node != "__start__" and event.get("name") == node:
                    yield f"data: {json.dumps({'event': 'trace', 'type': 'node_end', 'node': node, 'ts': ts})}\n\n"

                # Chamada de ferramenta (RAG)
                elif ev == "on_tool_start":
                    tool_name = event.get("name", "tool")
                    raw_input = event["data"].get("input", {})
                    query = raw_input.get("query", str(raw_input)) if isinstance(raw_input, dict) else str(raw_input)
                    yield f"data: {json.dumps({'event': 'trace', 'type': 'tool_call', 'tool': tool_name, 'input': query[:400], 'ts': ts})}\n\n"

                # Resultado da ferramenta â€” extrai chunks do JSON
                elif ev == "on_tool_end":
                    tool_name = event.get("name", "tool")
                    raw_output = event["data"].get("output", "")
                    # Extrai conteÃºdo do ToolMessage (formato: content='[{...}]')
                    output_str = raw_output.content if hasattr(raw_output, "content") else str(raw_output)
                    try:
                        chunks = json.loads(output_str) if isinstance(output_str, str) else output_str
                        if isinstance(chunks, list):
                            snippets = []
                            for c in chunks[:3]:  # top 3 chunks
                                content = c.get("content", "")[:200] if isinstance(c, dict) else str(c)[:200]
                                score = c.get("score", "") if isinstance(c, dict) else ""
                                source = (c.get("metadata", {}) or {}).get("source", "") if isinstance(c, dict) else ""
                                snippets.append({"content": content, "score": round(score, 4) if score else None, "source": source})
                            output_str = json.dumps(snippets, ensure_ascii=False)
                        else:
                            output_str = str(output_str)[:600]
                    except Exception:
                        output_str = str(output_str)[:600]
                    yield f"data: {json.dumps({'event': 'trace', 'type': 'tool_result', 'tool': tool_name, 'output': output_str, 'ts': ts})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"
            return

        accumulated = _sanitize_math_for_ui(accumulated_raw)
        if len(accumulated) > len(emitted_clean):
            delta = accumulated[len(emitted_clean):]
            if delta:
                yield f"data: {json.dumps({'event': 'chunk', 'text': delta})}\n\n"

        elapsed_ms = int((time.time() - start_time) * 1000)
        # Salva em background: o evento final chega ao usuário imediatamente.
        asyncio.create_task(_bg_save_chat_turn(
            session_id, agent_id, agent_name, request.message, accumulated, elapsed_ms
        ))
        yield f"data: {json.dumps({'event': 'final', 'agent_id': agent_id, 'agent_name': agent_name})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ============================================================
# Endpoint: POST /webhook/orquestrador/stream  (SSE)
# ============================================================

@app.post("/webhook/orquestrador/stream")
async def chat_orchestrator_stream(
    request: ChatRequest,
    _api_key: Optional[str] = Header(default=None, alias=WEBHOOK_API_KEY_HEADER),
):
    """Endpoint de streaming SSE para o orquestrador."""
    _verify_webhook_api_key(_api_key)
    start_time = time.time()
    session_id = _resolve_session_id(request)
    try:
        resolved_model = resolve_chat_model(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Mesmo racional do endpoint não-streaming: memória curta só entra quando
    # a mensagem atual parece depender do contexto anterior.
    messages = _build_orchestrator_input_messages(
        session_id=session_id,
        message=request.message,
        user_profile=request.user_profile,
    )

    graph = get_orchestrator_graph()

    async def generate():
        accumulated_raw = ""
        last_sanitized = ""
        chunks_sent = 0
        agent_id = 0
        agent_name = "Assistente Geral"
        orchestrator_output: Dict[str, Any] = {}
        RESPONSE_NODES = {"respond_direct", "consolidate"}
        fallback_response = ""

        try:
            async for event in graph.astream_events(
                {"messages": messages, "user_profile": request.user_profile, "llm_model": resolved_model},
                version="v2",
            ):
                ev = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")
                ts = int(time.time() * 1000)

                # Tokens da resposta final — streaming token-a-token para o front.
                # A sanitizacao de LaTeX opera sobre o texto acumulado completo e
                # emite apenas o delta (texto novo), garantindo que sequencias LaTeX
                # multi-token sejam substituidas corretamente antes de chegar ao front.
                if ev == "on_chat_model_stream" and node in RESPONSE_NODES:
                    chunk = event["data"]["chunk"]
                    content = chunk.content if isinstance(chunk.content, str) else ""
                    tool_calls = getattr(chunk, "tool_call_chunks", [])
                    if content and not tool_calls:
                        accumulated_raw += content
                        sanitized_full = _sanitize_math_for_ui(accumulated_raw)
                        delta = sanitized_full[len(last_sanitized):]
                        if delta:
                            last_sanitized = sanitized_full
                            chunks_sent += 1
                            yield f"data: {json.dumps({'event': 'chunk', 'text': delta})}\n\n"

                # Captura agent_id final e fallback quando nÃ£o houve streaming
                # (ex: consolidate com 1 agente retorna direto, sem chamar LLM)
                elif ev == "on_chain_end" and event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    orchestrator_output = dict(output or {})
                    agent_id = output.get("primary_agent_id", 0)
                    agent_name = output.get("primary_agent_name", "Assistente Geral")
                    if not accumulated_raw:
                        fallback_response = output.get("final_response", "")

                # TransiÃ§Ãµes de nÃ³ â€” apenas nÃ­vel raiz
                elif ev == "on_chain_start" and node and node != "__start__" and event.get("name") == node:
                    yield f"data: {json.dumps({'event': 'trace', 'type': 'node_start', 'node': node, 'ts': ts})}\n\n"
                elif ev == "on_chain_end" and node and node != "__start__" and event.get("name") == node:
                    yield f"data: {json.dumps({'event': 'trace', 'type': 'node_end', 'node': node, 'ts': ts})}\n\n"

                # Chamadas de ferramenta (RAG dos sub-agentes)
                elif ev == "on_tool_start":
                    tool_name = event.get("name", "tool")
                    raw_input = event["data"].get("input", {})
                    query = raw_input.get("query", str(raw_input)) if isinstance(raw_input, dict) else str(raw_input)
                    yield f"data: {json.dumps({'event': 'trace', 'type': 'tool_call', 'tool': tool_name, 'input': query[:400], 'ts': ts})}\n\n"

                elif ev == "on_tool_end":
                    tool_name = event.get("name", "tool")
                    raw_output = event["data"].get("output", "")
                    output_str = raw_output.content if hasattr(raw_output, "content") else str(raw_output)
                    try:
                        chunks = json.loads(output_str) if isinstance(output_str, str) else output_str
                        if isinstance(chunks, list):
                            snippets = []
                            for c in chunks[:6]:
                                content = c.get("content", "") if isinstance(c, dict) else str(c)
                                score = c.get("score", "") if isinstance(c, dict) else ""
                                source = (c.get("metadata", {}) or {}).get("source", "") if isinstance(c, dict) else ""
                                snippets.append({"content": content, "score": round(score, 4) if score else None, "source": source})
                            output_str = json.dumps(snippets, ensure_ascii=False)
                        else:
                            output_str = str(output_str)
                    except Exception:
                        output_str = str(output_str)
                    yield f"data: {json.dumps({'event': 'trace', 'type': 'tool_result', 'tool': tool_name, 'output': output_str, 'ts': ts})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"
            return

        # Resolve texto final para persistencia no DB (sanitizado, sem LaTeX).
        if not accumulated_raw and fallback_response:
            accumulated_raw = fallback_response
        accumulated = _sanitize_math_for_ui(accumulated_raw) if accumulated_raw else ""

        # Se nenhum token foi emitido em tempo real (ex: consolidate sem LLM,
        # respond_direct retornando direto da chain), envia o bloco completo agora.
        if chunks_sent == 0 and accumulated:
            yield f"data: {json.dumps({'event': 'chunk', 'text': accumulated})}\n\n"

        elapsed_ms = int((time.time() - start_time) * 1000)
        # Salva em background: o evento final chega ao usuário imediatamente.
        asyncio.create_task(_bg_save_chat_turn(
            session_id, agent_id, agent_name, request.message, accumulated, elapsed_ms
        ))
        try:
            payload = _extract_routing_payload(
                request_message=request.message,
                orchestrator_output=orchestrator_output,
                elapsed_ms=elapsed_ms,
                default_agent_id=agent_id,
                default_agent_name=agent_name,
            )
            payload["session_id"] = session_id
            asyncio.create_task(_bg_save_routing_log(payload))
        except Exception as e:
            _log_server_error("routing-log payload", e)

        yield f"data: {json.dumps({'event': 'final', 'agent_id': agent_id, 'agent_name': agent_name})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ============================================================
# Endpoint: POST /webhook/ingestao
# ============================================================

@app.post("/webhook/ingestao")
async def ingest_document(
    request: IngestRequest,
    _api_key: Optional[str] = Header(default=None, alias=WEBHOOK_API_KEY_HEADER),
):
    """Endpoint de ingestÃ£o de documentos.
    
    Recebe texto jÃ¡ processado (Markdown limpo) e executa:
    chunking â†’ embeddings â†’ upsert no Supabase â†’ log no Hetzner.
    
    Mesmo contrato que o pipeline de ingestÃ£o do N8N.
    O form de ingestÃ£o (N8N) ou o app web podem chamar este endpoint.
    
    Body: IngestRequest (text, table_name, agent_id, source, doc_type)
    
    Response: estatÃ­sticas da ingestÃ£o
    """
    _verify_webhook_api_key(_api_key)
    try:
        result = ingest_text(
            text=request.text,
            table_name=request.table_name,
            agent_id=request.agent_id,
            source=request.source,
            doc_type=request.doc_type,
        )
        return result
    except Exception as e:
        print(f"[ingestao] Erro: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro na ingestÃ£o: {str(e)}",
        )


@app.post("/webhook/ingestao-arquivo")
async def ingest_document_file(
    file: UploadFile = File(...),
    agent_id: int = Form(...),
    doc_type: str = Form("manual"),
    table_name: Optional[str] = Form(default=None),
    _api_key: Optional[str] = Header(default=None, alias=WEBHOOK_API_KEY_HEADER),
):
    """Ingestao via upload de arquivo (multipart/form-data).

    Fluxo inicial para o webapp:
      - aceita `.md` e `.txt` diretamente
      - mapeia `agent_id` para a tabela de embeddings correta
      - executa deduplicacao por hash + ingestao

    Para PDF/DOCX, a recomendacao atual e converter para Markdown
    antes de enviar.
    """
    _verify_webhook_api_key(_api_key)
    agent_config = get_agent_by_id(agent_id)
    if not agent_config:
        raise HTTPException(
            status_code=404,
            detail=f"Agente {agent_id} nao encontrado. IDs validos: 0 a 6.",
        )

    resolved_table_name = agent_config["table_name"]
    if table_name and table_name != resolved_table_name:
        raise HTTPException(
            status_code=400,
            detail=(
                "table_name divergente do agent_id informado. "
                f"Para agent_id={agent_id}, use table_name='{resolved_table_name}'."
            ),
        )

    filename = file.filename or "upload"
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if extension not in {"md", "txt"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "Formato nao suportado no upload direto. "
                "Envie .md/.txt ou converta para Markdown antes da ingestao."
            ),
        )

    try:
        raw = await file.read()
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Arquivo deve estar em UTF-8.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao ler arquivo: {e}",
        )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Arquivo vazio.",
        )

    try:
        result = ingest_text(
            text=text,
            table_name=resolved_table_name,
            agent_id=agent_id,
            source=filename,
            doc_type=doc_type,
        )
        result["resolved_agent_id"] = agent_id
        result["resolved_agent_name"] = agent_config["name"]
        result["resolved_table_name"] = resolved_table_name
        return result
    except Exception as e:
        print(f"[ingestao-arquivo] Erro: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro na ingestao de arquivo: {str(e)}",
        )


# ============================================================
# Endpoint: GET /health
# ============================================================

@app.get("/health")
async def health():
    """Endpoint de health check.
    
    Verifica se o servidor estÃ¡ rodando e os bancos estÃ£o acessÃ­veis.
    Usado por load balancers, monitoring, e Docker health checks.
    
    Retorna:
      { "status": "ok", "agents": 7, "version": "1.0.0" }
    
    NÃ£o existe no projeto original (o original nÃ£o tem health check).
    """
    # VerificaÃ§Ã£o bÃ¡sica: tenta conectar nos dois bancos
    status = "ok"
    details = {}
    
    try:
        from app.db.connection import get_supabase_conn
        with get_supabase_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        details["supabase"] = "connected"
    except Exception as e:
        status = "degraded"
        details["supabase"] = "unavailable"
        _log_server_error("health supabase", e)
    
    try:
        from app.db.connection import get_hetzner_conn
        with get_hetzner_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        details["hetzner"] = "connected"
    except Exception as e:
        status = "degraded"
        details["hetzner"] = "unavailable"
        _log_server_error("health hetzner", e)

    payload = {
        "status": status,
        "agents": 7,
        "version": "1.0.0",
        "databases": details,
    }
    return JSONResponse(status_code=200 if status == "ok" else 503, content=payload)


@app.get("/console/models/status")
async def console_models_status(
    _api_key: Optional[str] = Header(default=None, alias=WEBHOOK_API_KEY_HEADER),
):
    """Status real dos modelos permitidos no backend para consumo do console Next."""
    _verify_webhook_api_key(_api_key)

    allowed_models = get_allowed_chat_models()
    has_openai_key = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    has_compatible_gateway = any(
        (os.getenv(var_name) or "").strip()
        for var_name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENROUTER_BASE_URL")
    )

    items = []
    for model_id in allowed_models:
        normalized = model_id.lower()
        provider = "OpenAI"
        if "claude" in normalized:
            provider = "Anthropic"
        elif "llama" in normalized:
            provider = "Meta"

        is_openai_model = provider == "OpenAI"
        is_ready = has_openai_key if is_openai_model else has_compatible_gateway

        if is_openai_model:
            compatibility_message = (
                "Pronto no backend atual."
                if is_ready
                else "Atencao: falta configurar OPENAI_API_KEY no backend."
            )
            setup_hint = (
                "Backend pronto: OPENAI_API_KEY configurada."
                if is_ready
                else "Para liberar este modelo, configure OPENAI_API_KEY no backend."
            )
        else:
            compatibility_message = (
                "Pronto via gateway compativel configurado no backend."
                if is_ready
                else "Requer gateway compativel no backend para este provider."
            )
            setup_hint = (
                "Backend pronto: gateway compativel configurado para providers externos."
                if is_ready
                else "Para liberar este modelo, configure um gateway compativel no backend, como OPENAI_BASE_URL, OPENAI_API_BASE ou OPENROUTER_BASE_URL."
            )

        items.append(
            {
                "id": model_id,
                "provider": provider,
                "compatibility_status": "ready" if is_ready else "requires_adapter",
                "compatibility_message": compatibility_message,
                "setup_hint": setup_hint,
                "selectable": is_ready,
            }
        )

    return {
        "models": items,
        "default_model": os.getenv("LLM_MODEL", ""),
        "has_openai_key": has_openai_key,
        "has_compatible_gateway": has_compatible_gateway,
    }


# ============================================================
# Para rodar diretamente: python -m app.server.webapp
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.server.webapp:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,  # Hot reload durante desenvolvimento
    )

