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
import time
import psycopg

from app.config import MEMORY_WINDOW
from app.db.connection import get_hetzner_conn


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
    """Carrega as ultimas mensagens de uma sessao.

    Possui retry para falhas transientes de conexao no Postgres.
    """
    window = limit or MEMORY_WINDOW

    rows: list[tuple[str, str]] = []
    for attempt in range(2):
        try:
            with get_hetzner_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT role, content
                        FROM (
                            SELECT role, content, created_at
                            FROM chat_memories
                            WHERE session_id = %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        ) sub
                        ORDER BY created_at ASC
                        """,
                        (session_id, window),
                    )
                    rows = cur.fetchall()
            break
        except psycopg.OperationalError:
            if attempt == 1:
                raise
            # o pool descarta conexoes BAD; tenta novamente com nova conexao
            time.sleep(0.15)

    return [{"role": row[0], "content": row[1]} for row in rows]

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
    """Salva a rodada completa (human + ai + log) em uma única conexão.

    Reduz overhead de pool/round-trip comparado a 3 chamadas separadas.
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
                    response_time_ms,
                    now,
                ),
            )
