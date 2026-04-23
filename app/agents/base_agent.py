"""
agents/base_agent.py — Classe base para os agentes especialistas

Este módulo cria os grafos LangGraph dos 6 agentes especialistas.
Cada agente é um executor ReAct (Reasoning + Acting) que:
  1. Recebe uma pergunta do usuário
  2. Decide se precisa buscar na base de conhecimento (tool call)
  3. Se sim, chama a tool kb_search e recebe os chunks relevantes
  4. Gera a resposta final baseada nos chunks

O padrão ReAct:
  ReAct é um loop: o LLM "pensa" (reasoning) e depois "age" (acting).
  
  Iteração 1:
    LLM pensa: "O usuário quer saber sobre filagem. Preciso buscar na base."
    LLM age: chama kb_search(query="filagem mussarela")
    Tool retorna: [chunk1: "A filagem é feita a 78-82°C...", chunk2: "..."]
  
  Iteração 2:
    LLM pensa: "Tenho informações suficientes para responder."
    LLM age: gera a resposta final (sem chamar mais tools)
  
  O loop para quando o LLM não chama nenhuma tool (decidiu responder).
  No máximo, roda 3-4 iterações (configurável).

Como se conecta ao projeto original:
  O app/agent/new_react.py do curso implementa create_react_executor(),
  que é exatamente esse padrão. O nosso base_agent.py REUTILIZA essa
  função — não reescrevemos o executor ReAct, apenas configuramos
  para o domínio de laticínios.

  Original (CRM):
    executor = create_react_executor(
        model=model,
        tools=[create_lead, get_lead, search_leads, ...],
        prompt=LEAD_REACT_PROMPT,
    )
  
  Adaptado (Laticínios):
    executor = create_react_executor(
        model=model,
        tools=[kb_search_queijos],  ← apenas 1 tool (busca no KB)
        prompt=AGENT_1_PROMPT,       ← prompt do agente de queijos
    )

  A diferença é que no CRM os agentes têm múltiplas tools (CRUD),
  enquanto no laticínios cada agente tem apenas 1 tool (busca RAG).
  A mecânica do ReAct é idêntica.

Hierarquia:
  agent_config.py (dados) → base_agent.py (constrói grafos) → webapp.py (expõe HTTP)
"""

from typing import Dict, Any, Optional, List, Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import LLM_MODEL, AGENT_TEMPERATURE
from app.agents.agent_config import get_agent_by_id, get_search_config, AGENTS
from app.agents.prompts import get_agent_prompt
from app.rag.search import create_kb_search_tool
from app.tools.calculations import get_calculation_tools


# ============================================================
# Estado do agente (TypedDict)
# ============================================================

class AgentState(TypedDict, total=False):
    """Estado interno do grafo ReAct de cada agente.
    
    No projeto original (workflow.py linhas 176-183), o AgentState
    tem muitos campos (intent, slots, lead_atual, errors, context)
    porque o CRM precisa rastrear estado complexo entre nós.
    
    Aqui, o estado é MUITO mais simples porque o agente faz apenas
    uma coisa: receber pergunta → buscar → responder.
    
    Campos:
        messages: Histórico de mensagens da conversa.
                  Usa add_messages do LangGraph que faz append
                  automático (nova mensagem é adicionada, não substitui).
                  Inclui: SystemMessage (prompt), HumanMessage (pergunta),
                  AIMessage (resposta), ToolMessage (resultado da busca).
        
        agent_prompt: System prompt do agente (injetado no início).
                      Específico por agente (queijos ≠ regulatórios).
    """
    messages: Annotated[List[AnyMessage], add_messages]
    agent_prompt: str


# ============================================================
# Construção do grafo ReAct para um agente
# ============================================================

def build_agent_graph(agent_id: int) -> Any:
    """Constrói o grafo LangGraph completo para um agente especialista.
    
    Esta função é o coração do módulo. Ela:
      1. Lê a configuração do agente (agent_config.py)
      2. Cria a tool de busca no KB (configurada para a tabela do agente)
      3. Cria o modelo LLM com as tools vinculadas
      4. Monta o grafo ReAct com 3 nós (prepare → agent → tools)
      5. Compila e retorna o grafo pronto para execução
    
    O grafo resultante é equivalente ao que o create_react_executor
    do projeto original faz (new_react.py linhas 28-85), mas com
    a configuração específica para laticínios.
    
    Parâmetros:
        agent_id: ID do agente (1 a 6).
    
    Retorna:
        Grafo LangGraph compilado, pronto para .invoke() ou .ainvoke().
    
    Uso:
        graph = build_agent_graph(1)  # Agente de Queijos
        result = graph.invoke({
            "messages": [HumanMessage(content="Como fabricar mussarela?")]
        })
    
    Por que não usar create_react_executor diretamente?
    Poderíamos! A função do original funciona perfeitamente.
    Reconstruímos aqui por 2 motivos:
      1. Clareza didática: cada nó é explicado em detalhe
      2. Customização: adicionamos o nó 'prepare' que injeta
         o system prompt de forma diferente do original
    
    Se preferir usar o original, basta importar:
      from app.agent.new_react import create_react_executor
      graph = create_react_executor(model, tools, prompt)
    """
    # ---- Passo 1: Ler configuração do agente ----
    agent_config = get_agent_by_id(agent_id)
    if not agent_config:
        raise ValueError(f"Agente {agent_id} não encontrado em agent_config.py")
    
    table_name = agent_config["table_name"]
    agent_name = agent_config["name"]
    search_config = get_search_config(agent_id)
    
    # ---- Passo 2: Criar a tool de busca no KB ----
    # create_kb_search_tool é a factory do search.py
    # Retorna uma tool LangChain configurada para a tabela deste agente
    #
    # Exemplo: para agent_id=1, cria uma tool que busca em
    # embeddings_agente_1_queijos com os parâmetros de search_config
    kb_tool = create_kb_search_tool(
        table_name=table_name,
        agent_name=agent_name,
        search_config=search_config,
    )
    tools = [kb_tool]
    # Tools de cálculo disponíveis para todos os agentes especialistas.
    # Assim, qualquer domínio que tenha fórmulas na base pode calcular
    # de forma determinística sem depender de "conta mental" do LLM.
    if agent_id != 0:
        tools.extend(get_calculation_tools())
    
    # ---- Passo 3: Criar o modelo LLM com tools ----
    model = ChatOpenAI(model=LLM_MODEL, temperature=AGENT_TEMPERATURE)
    model_with_tools = model.bind_tools(tools)

    # Nome da tool (usado na chamada forçada do prepare)
    kb_tool_name = kb_tool.name  # "kb_search"

    # ---- Passo 4: Definir os nós do grafo ----

    def prepare(state: AgentState) -> AgentState:
        """Nó PREPARE: injeta o system prompt e dispara a busca diretamente.

        Otimização de latência: em vez de chamar o LLM para decidir
        "devo buscar?", injetamos um AIMessage com a chamada à tool
        já formada. O grafo vai direto para o ToolNode sem passar
        pelo LLM de decisão.

        Fluxo resultante:
          prepare → tools (busca direta) → agent (responde com contexto) → END

        Isso elimina 1 chamada LLM por agente (~0.8-1.5s de economia).
        """
        prompt_text = get_agent_prompt(agent_id)

        # Extrai a última pergunta do usuário para a query de busca
        user_text = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_text = msg.content
                break

        # AIMessage que força a chamada à tool sem passar pelo LLM
        forced_call = AIMessage(
            content="",
            tool_calls=[{
                "id": "direct_search_0",
                "name": kb_tool_name,
                "args": {"query": user_text},
                "type": "tool_call",
            }],
        )

        return {
            "agent_prompt": prompt_text,
            "messages": [forced_call],
        }

    async def call_model(state: AgentState) -> AgentState:
        """Nó AGENT: gera a resposta final com base nos chunks recuperados.

        Neste fluxo optimizado, este nó só é chamado UMA VEZ — após
        o ToolNode já ter executado a busca. O LLM recebe o histórico
        [SystemMessage, HumanMessage, AIMessage(tool_calls), ToolMessage(chunks)]
        e gera a resposta diretamente.

        O conditional edge mantém o loop ReAct como safety net: se o
        LLM precisar de outra busca (raro), ele pode pedir.
        """
        messages = list(state.get("messages", []))

        prompt_text = state.get("agent_prompt", "")
        if prompt_text:
            messages = [SystemMessage(content=prompt_text)] + messages

        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    # ---- Passo 5: Montar o grafo ----
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    graph.add_node("prepare", prepare)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    # Fluxo otimizado:
    #   START → prepare → tools (busca direta) → agent (responde) → END
    #
    # prepare injeta a tool call já formada, pulando o LLM de decisão.
    # O loop ReAct (agent ↔ tools) ainda existe como safety net.
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "tools")  # Busca direta — sem LLM de decisão

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )

    graph.add_edge("tools", "agent")

    return graph.compile()


# ============================================================
# Cache de grafos compilados
# ============================================================
# 
# Compilar um grafo LangGraph é rápido (~1ms), mas não precisamos
# recompilar a cada request. O cache armazena os grafos compilados
# e reutiliza entre requests.
#
# _agent_graphs é um dict: { agent_id: grafo_compilado }
# Preenchido sob demanda (lazy): o grafo é compilado na primeira
# request para aquele agente e reutilizado nas seguintes.

_agent_graphs: Dict[int, Any] = {}


def get_agent_graph(agent_id: int) -> Any:
    """Retorna o grafo compilado de um agente (com cache).
    
    Primeira chamada: compila o grafo e salva no cache.
    Chamadas seguintes: retorna do cache (instantâneo).
    
    Parâmetros:
        agent_id: ID do agente (1 a 6).
    
    Retorna:
        Grafo LangGraph compilado.
    
    Raises:
        ValueError: se agent_id não existir em agent_config.py.
    
    Usado por:
        webapp.py → ao receber POST /webhook/agente-{id}
        orchestrator.py → ao rotear para um sub-agente
    """
    if agent_id not in _agent_graphs:
        _agent_graphs[agent_id] = build_agent_graph(agent_id)
    return _agent_graphs[agent_id]


def get_all_agent_graphs() -> Dict[int, Any]:
    """Compila e retorna todos os 6 grafos de agentes.
    
    Útil para pré-aquecer o cache no startup do servidor,
    em vez de compilar sob demanda na primeira request.
    
    Usado por: webapp.py → no evento de startup.
    """
    for agent_config in AGENTS:
        agent_id = agent_config["agent_id"]
        if agent_id not in _agent_graphs:
            _agent_graphs[agent_id] = build_agent_graph(agent_id)
    return _agent_graphs
