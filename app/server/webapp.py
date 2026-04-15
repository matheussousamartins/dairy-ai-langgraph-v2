"""
server/webapp.py — Servidor FastAPI com os 7 endpoints

Este é o ponto de entrada HTTP do sistema. Expõe os mesmos endpoints
que o N8N, com o mesmo contrato de API — o app React Native funciona
com qualquer backend sem mudar uma linha.

No projeto original do curso (app/server/webapp.py), o servidor tem
apenas 2 endpoints:
  - POST /chat → chama o agente principal
  - GET /stream → streaming via SSE

Aqui temos:
  - POST /webhook/agente-{id} → chama o agente especialista (6 endpoints)
  - POST /webhook/orquestrador → chama o orquestrador (1 endpoint)
  - POST /webhook/ingestao → ingestão de documentos
  - GET /health → status do sistema

Os URLs usam /webhook/ para manter compatibilidade com o N8N.
O app React Native já está configurado para chamar esses endpoints.
Para trocar de N8N para LangGraph, basta mudar a Base URL no app.

Request/Response:
  Os mesmos que documentamos no API-Laticinios-AI-v1.1.docx:
  
  Request:
    { "message": "...", "session_id": "...", "user_profile": {...} }
  
  Response:
    { "response": "...", "agent_id": 1, "agent_name": "Tecnologia de Queijos" }

Diferença do original:
  O original usa result.get("structured_response") para extrair a
  resposta, que é o formato específico do CRM. Aqui usamos o formato
  do contrato de API de laticínios (response, agent_id, agent_name).
"""

import json
import secrets
import time
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage

from app.config import (
    SERVER_HOST,
    SERVER_PORT,
    CORS_ALLOW_ORIGINS,
    ENFORCE_WEBHOOK_API_KEY,
    WEBHOOK_API_KEY_HEADER,
    WEBHOOK_API_KEYS,
    validate_config,
)
from app.db.connection import init_pools, close_pools
from app.db.memory import load_memory, save_chat_turn
from app.agents.base_agent import get_agent_graph, get_all_agent_graphs
from app.agents.orchestrator import get_orchestrator_graph
from app.agents.agent_config import get_agent_by_id
from app.rag.ingest import ingest_text


# ============================================================
# Lifecycle: startup e shutdown
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida do servidor.
    
    Startup (antes de aceitar requests):
      1. Valida configuração (chaves, URLs)
      2. Inicializa pools de conexão com os bancos
      3. Pré-compila os grafos dos agentes (aquece o cache)
    
    Shutdown (ao parar o servidor):
      1. Fecha pools de conexão (libera recursos no banco)
    
    No original, não há lifecycle — as conexões são abertas
    sob demanda e nunca fechadas explicitamente. Aqui é mais
    robusto: pools são abertos no início e fechados no fim.
    
    O @asynccontextmanager é o padrão moderno do FastAPI para
    lifecycle (substituiu os eventos on_startup/on_shutdown).
    Tudo antes do yield é startup, tudo depois é shutdown.
    """
    # --- STARTUP ---
    print("[server] Validando configuração...")
    validate_config()
    
    print("[server] Inicializando pools de conexão...")
    init_pools()
    
    print("[server] Pré-compilando grafos dos agentes...")
    get_all_agent_graphs()      # Compila os 6 agentes
    get_orchestrator_graph()     # Compila o orquestrador
    
    print("[server] Servidor pronto!")
    print(f"[server] Endpoints disponíveis:")
    print(f"  POST /webhook/agente-{{0..6}}")
    print(f"  POST /webhook/orquestrador")
    print(f"  POST /webhook/ingestao")
    print(f"  POST /webhook/ingestao-arquivo")
    print(f"  GET  /health")
    
    yield  # Servidor rodando e aceitando requests
    
    # --- SHUTDOWN ---
    print("[server] Fechando pools de conexão...")
    close_pools()
    print("[server] Servidor encerrado.")


# ============================================================
# App FastAPI
# ============================================================

app = FastAPI(
    title="DairyApp AI",
    description="Sistema multi-agente para tecnologia de laticínios",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: aceita requests de qualquer origem (necessário para o
# console de teste na Vercel e para o app React Native).
# Em produção, restrinja para os domínios do app.
#
# No original, não há configuração de CORS. Aqui adicionamos
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
    
    Idêntico ao contrato documentado no API-Laticinios-AI-v1.1.docx.
    O Pydantic valida automaticamente: se faltar "message", retorna
    HTTP 422 com mensagem de erro clara.
    
    No original (webapp.py), o body é lido como dict genérico:
      body = await req.json()
      text = body.get("input")
    Aqui usamos Pydantic para validação automática.
    """
    message: str
    session_id: str
    user_profile: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Schema do body da response de chat.
    
    Campos:
      response: Texto da resposta do agente (para exibir ao usuário).
      agent_id: ID do agente que respondeu (0 = base/orquestrador; 1-6 = especialistas).
      agent_name: Nome legível do agente.
    """
    response: str
    agent_id: int
    agent_name: str


class IngestRequest(BaseModel):
    """Schema do body do request de ingestão.
    
    Idêntico ao contrato do pipeline de ingestão do N8N.
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

def _inject_user_profile(message: str, user_profile: Optional[Dict[str, Any]]) -> str:
    if not user_profile:
        return message
    profile_text = (
        f"\n[Perfil do usuário: "
        f"nível={user_profile.get('knowledgeLevel', 'INTERMEDIATE')}, "
        f"função={user_profile.get('role', 'não informado')}]"
    )
    return message + profile_text


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
    
    Recebe a pergunta, carrega o histórico da sessão, chama o
    grafo do agente, salva a nova mensagem no histórico, registra
    o log, e retorna a resposta.
    
    Este endpoint faz o que o N8N faz com 8 nós:
    Webhook → AI Agent → Postgres Chat Memory → Log → Respond
    
    Aqui é tudo em código, mas o fluxo é idêntico:
    1. Valida o agent_id
    2. Carrega histórico (load_memory)
    3. Chama o grafo (get_agent_graph → invoke)
    4. Salva histórico (save_memory x2)
    5. Registra log (save_interaction_log)
    6. Retorna response
    
    Parâmetros da URL:
      agent_id: 0 a 6 (validado contra agent_config.py)
    
    Body: ChatRequest (message, session_id, user_profile)
    
    Response: ChatResponse (response, agent_id, agent_name)
    
    Erros:
      404: agent_id não existe
      500: erro interno no grafo
    """
    _verify_webhook_api_key(_api_key)
    start_time = time.time()
    
    # ---- 1. Validar agent_id ----
    agent_config = get_agent_by_id(agent_id)
    if not agent_config:
        raise HTTPException(
            status_code=404,
            detail=f"Agente {agent_id} não encontrado. IDs válidos: 0 a 6.",
        )
    
    agent_name = agent_config["name"]
    
    # ---- 2. Carregar histórico da sessão ----
    # Mesma tabela chat_memories que o N8N usa
    history = await run_in_threadpool(_load_history_safe, request.session_id)
    messages = _history_to_messages(history)
    messages.append(
        HumanMessage(
            content=_inject_user_profile(request.message, request.user_profile)
        )
    )
    
    # ---- 3. Chamar o grafo do agente ----
    try:
        graph = get_agent_graph(agent_id)
        result = await graph.ainvoke({"messages": messages})
        
        # Extrai a resposta (última AIMessage)
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
        # Erro no grafo: retorna mensagem amigável
        # Em produção, logar o erro completo (não print)
        print(f"[agent-{agent_id}] Erro: {e}")
        response_text = (
            "Não foi possível processar sua pergunta no momento. "
            "Por favor, tente novamente."
        )
    
    # ---- 4. Salvar no histórico ----
    elapsed_ms = int((time.time() - start_time) * 1000)
    await run_in_threadpool(
        save_chat_turn,
        request.session_id,
        agent_id,
        agent_name,
        request.message,
        response_text,
        elapsed_ms,
    )
    
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
    """Endpoint do Assistente Geral (orquestrador) — multi-agente.
    
    O orquestrador pode consultar 1 a 3 agentes por pergunta.
    O response inclui o agent_id do agente PRINCIPAL (mais relevante).
    Se consultou Agente 3 + Agente 1, o agent_id no response é 3
    (o primeiro da lista ordenada por relevância).
    Se foi conversa geral, agent_id é 0.
    """
    _verify_webhook_api_key(_api_key)
    start_time = time.time()
    
    # Carrega histórico
    history = await run_in_threadpool(_load_history_safe, request.session_id)
    messages = _history_to_messages(history)
    messages.append(
        HumanMessage(
            content=_inject_user_profile(request.message, request.user_profile)
        )
    )
    
    # Chama o orquestrador
    try:
        graph = get_orchestrator_graph()
        result = await graph.ainvoke({
            "messages": messages,
            "user_profile": request.user_profile,
        })
        
        response_text = result.get("final_response", "")
        # primary_agent_id é o agente mais relevante da lista
        agent_id = result.get("primary_agent_id", 0)
        agent_name = result.get("primary_agent_name", "Assistente Geral")
        
    except Exception as e:
        print(f"[orquestrador] Erro: {e}")
        response_text = (
            "Não foi possível processar sua pergunta no momento. "
            "Por favor, tente novamente."
        )
        agent_id = 0
        agent_name = "Assistente Geral"
    
    # Salva histórico e log
    elapsed_ms = int((time.time() - start_time) * 1000)
    await run_in_threadpool(
        save_chat_turn,
        request.session_id,
        agent_id,
        agent_name,
        request.message,
        response_text,
        elapsed_ms,
    )
    
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

    Emite tokens conforme o LLM gera a resposta — estilo ChatGPT.
    Usa graph.astream_events() para capturar chunks do modelo.

    Eventos SSE emitidos:
      data: {"event": "chunk", "text": "..."}   — token(s) da resposta
      data: {"event": "final", "agent_id": N}   — sinaliza fim do stream
      data: {"event": "error", "detail": "..."}  — erro durante geração
    """
    _verify_webhook_api_key(_api_key)
    start_time = time.time()

    agent_config = get_agent_by_id(agent_id)
    if not agent_config:
        raise HTTPException(
            status_code=404,
            detail=f"Agente {agent_id} não encontrado. IDs válidos: 0 a 6.",
        )

    agent_name = agent_config["name"]
    history = await run_in_threadpool(_load_history_safe, request.session_id)
    messages = _history_to_messages(history)
    messages.append(
        HumanMessage(
            content=_inject_user_profile(request.message, request.user_profile)
        )
    )

    graph = get_agent_graph(agent_id)

    async def generate():
        accumulated = ""
        try:
            async for event in graph.astream_events({"messages": messages}, version="v2"):
                ev = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")
                ts = int(time.time() * 1000)

                # Tokens da resposta final
                if ev == "on_chat_model_stream" and node == "agent":
                    chunk = event["data"]["chunk"]
                    content = chunk.content if isinstance(chunk.content, str) else ""
                    tool_calls = getattr(chunk, "tool_call_chunks", [])
                    if content and not tool_calls:
                        accumulated += content
                        yield f"data: {json.dumps({'event': 'chunk', 'text': content})}\n\n"

                # Transições de nó — apenas eventos de nível raiz do LangGraph
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

                # Resultado da ferramenta — extrai chunks do JSON
                elif ev == "on_tool_end":
                    tool_name = event.get("name", "tool")
                    raw_output = event["data"].get("output", "")
                    # Extrai conteúdo do ToolMessage (formato: content='[{...}]')
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

        elapsed_ms = int((time.time() - start_time) * 1000)
        await run_in_threadpool(
            save_chat_turn,
            request.session_id,
            agent_id,
            agent_name,
            request.message,
            accumulated,
            elapsed_ms,
        )

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

    history = await run_in_threadpool(_load_history_safe, request.session_id)
    messages = _history_to_messages(history)
    messages.append(
        HumanMessage(
            content=_inject_user_profile(request.message, request.user_profile)
        )
    )

    graph = get_orchestrator_graph()

    async def generate():
        accumulated = ""
        agent_id = 0
        agent_name = "Assistente Geral"
        # Nós que geram a resposta final visível ao usuário.
        # "classify" é excluído pois emite JSON interno de roteamento.
        RESPONSE_NODES = {"respond_direct", "consolidate"}
        # Captura o final_response quando consolidate retorna direto (sem LLM)
        fallback_response = ""

        try:
            async for event in graph.astream_events(
                {"messages": messages, "user_profile": request.user_profile},
                version="v2",
            ):
                ev = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")
                ts = int(time.time() * 1000)

                # Tokens da resposta final (apenas nós de resposta)
                if ev == "on_chat_model_stream" and node in RESPONSE_NODES:
                    chunk = event["data"]["chunk"]
                    content = chunk.content if isinstance(chunk.content, str) else ""
                    tool_calls = getattr(chunk, "tool_call_chunks", [])
                    if content and not tool_calls:
                        accumulated += content
                        yield f"data: {json.dumps({'event': 'chunk', 'text': content})}\n\n"

                # Captura agent_id final e fallback quando não houve streaming
                # (ex: consolidate com 1 agente retorna direto, sem chamar LLM)
                elif ev == "on_chain_end" and event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    agent_id = output.get("primary_agent_id", 0)
                    agent_name = output.get("primary_agent_name", "Assistente Geral")
                    if not accumulated:
                        fallback_response = output.get("final_response", "")

                # Transições de nó — apenas nível raiz
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
                            for c in chunks[:3]:
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

        # Fallback: consolidate retornou direto (1 agente, sem LLM).
        # Nenhum chunk foi emitido — envia o texto completo como um único chunk
        # para que o frontend e o route.ts consigam capturá-lo.
        if not accumulated and fallback_response:
            accumulated = fallback_response
            yield f"data: {json.dumps({'event': 'chunk', 'text': fallback_response})}\n\n"

        elapsed_ms = int((time.time() - start_time) * 1000)
        await run_in_threadpool(
            save_chat_turn,
            request.session_id,
            agent_id,
            agent_name,
            request.message,
            accumulated,
            elapsed_ms,
        )

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
    """Endpoint de ingestão de documentos.
    
    Recebe texto já processado (Markdown limpo) e executa:
    chunking → embeddings → upsert no Supabase → log no Hetzner.
    
    Mesmo contrato que o pipeline de ingestão do N8N.
    O form de ingestão (N8N) ou o app web podem chamar este endpoint.
    
    Body: IngestRequest (text, table_name, agent_id, source, doc_type)
    
    Response: estatísticas da ingestão
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
            detail=f"Erro na ingestão: {str(e)}",
        )


@app.post("/webhook/ingestao-arquivo")
async def ingest_document_file(
    file: UploadFile = File(...),
    agent_id: int = Form(...),
    doc_type: str = Form("manual"),
    _api_key: Optional[str] = Header(default=None, alias=WEBHOOK_API_KEY_HEADER),
):
    """Ingestão via upload de arquivo (multipart/form-data).

    Fluxo inicial para o webapp:
      - aceita `.md` e `.txt` diretamente
      - mapeia `agent_id` para a tabela de embeddings correta
      - executa deduplicação por hash + ingestão

    Para PDF/DOCX, a recomendação atual é converter para Markdown
    antes de enviar.
    """
    _verify_webhook_api_key(_api_key)
    agent_config = get_agent_by_id(agent_id)
    if not agent_config:
        raise HTTPException(
            status_code=404,
            detail=f"Agente {agent_id} não encontrado. IDs válidos: 0 a 6.",
        )

    filename = file.filename or "upload"
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if extension not in {"md", "txt"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "Formato não suportado no upload direto. "
                "Envie .md/.txt ou converta para Markdown antes da ingestão."
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
            table_name=agent_config["table_name"],
            agent_id=agent_id,
            source=filename,
            doc_type=doc_type,
        )
        return result
    except Exception as e:
        print(f"[ingestao-arquivo] Erro: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro na ingestão de arquivo: {str(e)}",
        )


# ============================================================
# Endpoint: GET /health
# ============================================================

@app.get("/health")
async def health():
    """Endpoint de health check.
    
    Verifica se o servidor está rodando e os bancos estão acessíveis.
    Usado por load balancers, monitoring, e Docker health checks.
    
    Retorna:
      { "status": "ok", "agents": 7, "version": "1.0.0" }
    
    Não existe no projeto original (o original não tem health check).
    """
    # Verificação básica: tenta conectar nos dois bancos
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
        details["supabase"] = f"error: {e}"
    
    try:
        from app.db.connection import get_hetzner_conn
        with get_hetzner_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        details["hetzner"] = "connected"
    except Exception as e:
        status = "degraded"
        details["hetzner"] = f"error: {e}"
    
    return {
        "status": status,
        "agents": 7,
        "version": "1.0.0",
        "databases": details,
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
