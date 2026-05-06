"""
db/memory.py — Chat Memory (histórico de conversa)

Este módulo salva e carrega o histórico de mensagens de cada sessão de chat.
É o equivalente ao nó "Postgres Chat Memory" que configuramos no N8N.

No N8N, o nó Chat Memory faz tudo automaticamente: recebe o session_id,
busca as últimas N mensagens, e injeta no contexto do AI Agent.

Aqui fazemos o mesmo, mas em código:
  1. Antes de chamar o agente, load_memory() busca as últimas mensagens
  2. As mensagens são injetadas no estado do grafo LangGraph
  3. Depois que o agente responde, save_memory() salva a pergunta e resposta
  4. Na próxima mensagem com o mesmo session_id, o histórico está disponível

A tabela usada é chat_memories no Postgres da Hetzner:
  - id: SERIAL (auto-incremento)
  - session_id: VARCHAR — identifica a conversa (ex: "user-abc-aba-1")
  - role: VARCHAR — "human" ou "ai"
  - content: TEXT — texto da mensagem
  - created_at: TIMESTAMP — quando foi salva

Por que não usar o MemorySaver do LangGraph?
O projeto original do curso usa MemorySaver (linha 1163 do workflow.py):
    compiled_graph = graph.compile(checkpointer=MemorySaver())

O MemorySaver salva o estado do grafo inteiro em memória RAM.
Problemas para produção:
  - Se o servidor reiniciar, todo o histórico é perdido (RAM é volátil)
  - Se tiver múltiplos servidores (load balancer), cada um tem seu próprio
    histórico (não compartilham memória)
  - Não há como consultar o histórico externamente (analytics, debug)

Salvando no Postgres:
  - Histórico persiste entre reinícios
  - Múltiplos servidores compartilham o mesmo banco
  - Você pode consultar via SQL (ex: "quantas mensagens o user X enviou?")
  - É o mesmo banco que o N8N já usa (chat_memories)
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging
import time
import psycopg

from app.config import (
    MEMORY_WINDOW,
    MEMORY_SUMMARIZATION_ENABLED,
    MEMORY_SUMMARIZATION_THRESHOLD,
    MEMORY_SUMMARIZATION_KEEP_RECENT,
)
from app.db.connection import get_hetzner_conn

_log = logging.getLogger(__name__)


def save_memory(
    session_id: str,
    role: str,
    content: str,
) -> None:
    """Salva uma mensagem no histórico da conversa.
    
    Chamada duas vezes por request:
      1. Antes de chamar o agente: save_memory(session_id, "human", pergunta)
      2. Depois da resposta: save_memory(session_id, "ai", resposta)
    
    Parâmetros:
        session_id: Identificador da sessão (ex: "user-abc-aba-1").
                    Mesmo formato que o app React Native envia.
        role: "human" para mensagens do usuário, "ai" para respostas do agente.
              Esses nomes seguem a convenção do LangChain
              (HumanMessage, AIMessage).
        content: Texto da mensagem.
    
    A query usa INSERT simples — cada mensagem é um novo registro.
    Não há UPDATE porque o histórico é append-only (nunca editamos
    mensagens anteriores).
    """
    # Abre uma conexão do pool do Hetzner
    # O "with" garante que a conexão é devolvida ao pool ao final
    with get_hetzner_conn() as conn:
        # Abre um cursor (objeto que executa queries)
        # O cursor também é um context manager — fecha automaticamente
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_memories (session_id, role, content, created_at)
                VALUES (%s, %s, %s, %s)
                """,
                # %s são placeholders — o psycopg substitui pelos valores
                # de forma segura (previne SQL injection).
                # NUNCA use f-strings ou .format() com SQL!
                (session_id, role, content, datetime.utcnow()),
            )


def load_memory(
    session_id: str,
    limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Carrega as últimas mensagens da sessão, incluindo o resumo comprimido se existir.

    Retorna as N mensagens mais recentes (excluindo role="summary") ordenadas por
    tempo crescente, precedidas de um entry {"role": "summary", "content": "..."} se
    houver um resumo de compressão salvo para a sessão.

    O resumo é carregado na mesma transação que as mensagens recentes para consistência.
    Possui retry para falhas transientes de conexão no Postgres.
    """
    window = limit or MEMORY_WINDOW

    rows: list[tuple[str, str]] = []
    summary_content: Optional[str] = None

    for attempt in range(2):
        try:
            with get_hetzner_conn() as conn:
                with conn.cursor() as cur:
                    # Mensagens recentes não-summary, ordem cronológica
                    cur.execute(
                        """
                        SELECT role, content
                        FROM (
                            SELECT role, content, created_at
                            FROM chat_memories
                            WHERE session_id = %s AND role != 'summary'
                            ORDER BY created_at DESC
                            LIMIT %s
                        ) sub
                        ORDER BY created_at ASC
                        """,
                        (session_id, window),
                    )
                    rows = cur.fetchall()
                    # Resumo mais recente (se existir)
                    cur.execute(
                        """
                        SELECT content FROM chat_memories
                        WHERE session_id = %s AND role = 'summary'
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (session_id,),
                    )
                    summary_row = cur.fetchone()
                    if summary_row:
                        summary_content = summary_row[0]
            break
        except psycopg.OperationalError:
            if attempt == 1:
                raise
            # o pool descarta conexoes BAD; tenta novamente com nova conexao
            time.sleep(0.15)

    result = [{"role": row[0], "content": row[1]} for row in rows]
    if summary_content:
        result = [{"role": "summary", "content": summary_content}] + result
    return result

def clear_memory(session_id: str) -> int:
    """Apaga todo o histórico de uma sessão.
    
    Uso: quando o usuário clica em "nova conversa" na mesma aba,
    ou para limpar dados de teste.
    
    Retorna: quantidade de mensagens apagadas.
    
    Na prática, o app React Native cria um novo session_id para
    "nova conversa" (ex: user-abc-aba-1-1712345678), então o
    histórico antigo fica intocado. Mas essa função existe para
    limpeza manual ou via API admin.
    """
    with get_hetzner_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_memories WHERE session_id = %s",
                (session_id,),
            )
            # rowcount retorna quantas linhas foram afetadas pelo DELETE
            return cur.rowcount


def save_interaction_log(
    session_id: str,
    agent_id: int,
    agent_name: str,
    user_message: str,
    agent_response: str,
    response_time_ms: int,
) -> None:
    """Registra uma interação no log de analytics.
    
    Equivalente ao nó "Log Interaction" do N8N.
    Chamada uma vez por request, DEPOIS que o agente respondeu.
    
    Esses logs são usados para:
      - Analytics: qual agente é mais usado? quantas perguntas por dia?
      - Debug: quando uma resposta é ruim, consultar o log para ver
        o que o agente recebeu e respondeu
      - Feedback: o campo feedback (adicionado depois via endpoint separado)
        permite medir satisfação do usuário
    
    Parâmetros:
        session_id: ID da sessão
        agent_id: ID do agente que respondeu (1-6, ou 0 para orquestrador)
        agent_name: Nome do agente (ex: "Tecnologia de Queijos")
        user_message: Pergunta do usuário
        agent_response: Resposta gerada pelo agente
        response_time_ms: Tempo de resposta em milissegundos
    """
    with get_hetzner_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO interaction_logs
                    (session_id, agent_id, agent_name, user_message,
                     agent_response, response_time_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    agent_id,
                    agent_name,
                    user_message,
                    agent_response,
                    response_time_ms,
                    datetime.utcnow(),
                ),
            )


def save_chat_turn(
    session_id: str,
    agent_id: int,
    agent_name: str,
    user_message: str,
    agent_response: str,
    response_time_ms: int,
) -> None:
    """Salva a rodada completa (human + ai + interaction_log) em uma única conexão.

    Mantém o histórico da conversa e também alimenta a tabela de analytics
    usada pelo console/testes. Assim, todos os caminhos HTTP que persistem
    um turno de chat também registram a interação correspondente.
    """
    now = datetime.utcnow()
    with get_hetzner_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_memories (session_id, role, content, created_at)
                VALUES
                    (%s, %s, %s, %s),
                    (%s, %s, %s, %s)
                """,
                (
                    session_id, "human", user_message, now,
                    session_id, "ai", agent_response, now,
                ),
            )
            cur.execute(
                """
                INSERT INTO interaction_logs
                    (session_id, agent_id, agent_name, user_message,
                     agent_response, response_time_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    agent_id,
                    agent_name,
                    user_message,
                    agent_response,
                    int(response_time_ms),
                    now,
                ),
            )


def maybe_summarize_memory(session_id: str) -> bool:
    """Comprime o histórico antigo se a sessão exceder MEMORY_SUMMARIZATION_THRESHOLD.

    Fluxo:
        1. Conta mensagens não-summary da sessão (query barata).
        2. Se count ≤ THRESHOLD, retorna False imediatamente (caminho feliz).
        3. Calcula quantas comprimir: total - MEMORY_SUMMARIZATION_KEEP_RECENT.
        4. Em uma conexão: carrega as mensagens mais antigas + resumo existente.
        5. Chama o LLM para gerar novo resumo cumulativo (fora da conexão DB).
        6. Em outra conexão: deleta as mensagens antigas + resumo antigo,
           insere o novo resumo em uma única transação.

    Fail-safe absoluto: qualquer exceção é logada e retorna False.
    Nunca bloqueia nem levanta exceção para o chamador.

    Returns:
        True se a compressão foi executada com sucesso, False caso contrário.
    """
    from app.rag.summarizer import summarize_conversation  # lazy import

    if not MEMORY_SUMMARIZATION_ENABLED:
        return False

    try:
        # --- Fase 1: verificar threshold (conexão barata) ---
        with get_hetzner_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM chat_memories WHERE session_id = %s AND role != 'summary'",
                    (session_id,),
                )
                total: int = cur.fetchone()[0]

        if total <= MEMORY_SUMMARIZATION_THRESHOLD:
            return False

        to_compress_count = total - MEMORY_SUMMARIZATION_KEEP_RECENT
        if to_compress_count <= 0:
            return False

        # --- Fase 2: carregar dados para sumarização ---
        with get_hetzner_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, role, content FROM chat_memories
                    WHERE session_id = %s AND role != 'summary'
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (session_id, to_compress_count),
                )
                oldest_rows = cur.fetchall()

                cur.execute(
                    """
                    SELECT content FROM chat_memories
                    WHERE session_id = %s AND role = 'summary'
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (session_id,),
                )
                existing_summary_row = cur.fetchone()

        if not oldest_rows:
            return False

        oldest_messages = [{"id": r[0], "role": r[1], "content": r[2]} for r in oldest_rows]

        # Sumarização cumulativa: inclui resumo existente para não perder contexto.
        messages_for_summary: List[Dict[str, Any]] = []
        if existing_summary_row:
            messages_for_summary.append({"role": "summary", "content": existing_summary_row[0]})
        messages_for_summary.extend(
            {"role": m["role"], "content": m["content"]} for m in oldest_messages
        )

        # --- Fase 3: chamada LLM (fora de conexão DB para não segurar pool) ---
        new_summary = summarize_conversation(messages_for_summary)
        if not new_summary:
            _log.warning(
                "maybe_summarize_memory: LLM retornou None — histórico preservado [session=%s]",
                session_id,
            )
            return False

        # --- Fase 4: persiste resultado em transação atômica ---
        ids_to_delete = [m["id"] for m in oldest_messages]
        now = datetime.utcnow()

        with get_hetzner_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_memories WHERE session_id = %s AND id = ANY(%s)",
                    (session_id, ids_to_delete),
                )
                cur.execute(
                    "DELETE FROM chat_memories WHERE session_id = %s AND role = 'summary'",
                    (session_id,),
                )
                cur.execute(
                    "INSERT INTO chat_memories (session_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
                    (session_id, "summary", new_summary, now),
                )

        _log.info(
            "maybe_summarize_memory: %d msgs comprimidas em %d chars [session=%s]",
            len(oldest_messages),
            len(new_summary),
            session_id,
        )
        return True

    except Exception as exc:
        _log.warning(
            "maybe_summarize_memory: falha silenciosa — %s [session=%s]",
            exc,
            session_id,
        )
        return False


def save_routing_log(
    session_id: str,
    user_message: str,
    response_time_ms: int,
    query_hash: str,
    selected_agent_ids: List[int],
    chosen_agent_ids: List[int],
    execution_plan: List[int],
    primary_agent_id: int,
    primary_agent_name: str,
    routing_confidence: float,
    routing_bucket: str,
    routing_reason: str,
    routing_alternatives: List[int],
    fallback_used: bool,
    fallback_attempts: int,
    fallback_trigger: str,
    cost_estimate_usd: float,
) -> None:
    """Registra dados estruturados de roteamento do orquestrador.

    A persistência da interação textual fica concentrada em save_chat_turn().
    Este método grava apenas a telemetria estruturada de roteamento para evitar
    duplicidade e manter responsabilidades separadas.
    """
    now = datetime.utcnow()
    with get_hetzner_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO routing_logs (
                    session_id,
                    user_message,
                    query_hash,
                    selected_agent_ids,
                    chosen_agent_ids,
                    execution_plan,
                    primary_agent_id,
                    primary_agent_name,
                    routing_confidence,
                    routing_bucket,
                    routing_reason,
                    routing_alternatives,
                    fallback_used,
                    fallback_attempts,
                    fallback_trigger,
                    response_time_ms,
                    cost_estimate_usd,
                    created_at
                )
                VALUES (
                    %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s::jsonb,
                    %s, %s, %s, %s, %s,
                    %s::jsonb, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    session_id,
                    user_message,
                    query_hash,
                    json.dumps(selected_agent_ids),
                    json.dumps(chosen_agent_ids),
                    json.dumps(execution_plan),
                    primary_agent_id,
                    primary_agent_name,
                    float(routing_confidence),
                    routing_bucket,
                    routing_reason,
                    json.dumps(routing_alternatives),
                    bool(fallback_used),
                    int(fallback_attempts),
                    fallback_trigger,
                    int(response_time_ms),
                    float(cost_estimate_usd),
                    now,
                ),
            )
