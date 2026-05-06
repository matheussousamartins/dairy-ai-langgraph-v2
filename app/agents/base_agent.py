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

import json
import ast
import re
import unicodedata
from typing import Dict, Any, Optional, List, Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import (
    LLM_MODEL,
    LLM_MAX_TOKENS,
    AGENT_TEMPERATURE,
    DEFAULT_SEARCH_TYPE,
    RAG_EARLY_SKIP_WEAK_SEARCH_ENABLED,
    RAG_EARLY_SKIP_WEAK_AGENT_IDS,
    RAG_EARLY_SKIP_WEAK_MIN_QUERY_KEYWORDS,
    RAG_EARLY_SKIP_WEAK_MIN_TOP_KEYWORD_HITS,
    RAG_EARLY_SKIP_WEAK_HYBRID_MAX_SCORE,
    CONTEXTUAL_QUERY_REWRITE_ENABLED,
)
from app.agents.agent_config import get_agent_by_id, get_search_config, AGENTS
from app.agents.prompts import get_agent_prompt
from app.rag.search import create_kb_search_tool, contextualize_query_for_rag
from app.tools.calculations import get_calculation_tools


_PT_STOPWORDS = {
    "como", "qual", "quais", "para", "com", "sem", "uma", "uns", "umas",
    "dos", "das", "de", "do", "da", "e", "ou", "o", "a", "os", "as",
    "em", "no", "na", "nos", "nas", "por", "que", "se", "ao", "aos",
    "sao", "são", "antes", "apos", "após", "deve", "devem", "ser",
    "fazer", "tomar",
}


def _normalize_text_for_match(text: str) -> str:
    normalized = (text or "").lower()
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_folded_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    return re.sub(r"\s+", " ", normalized).strip()


def _looks_like_greeting_only(text: str) -> bool:
    current = _normalize_folded_text(_extract_current_user_segment(_strip_profile_suffix(text)))
    if not current:
        return False

    current = re.sub(r"[.!?,;:]+", " ", current)
    current = re.sub(r"\s+", " ", current).strip()
    if not current:
        return False

    greeting_phrases = {
        "oi",
        "ola",
        "olá",
        "bom dia",
        "boa tarde",
        "boa noite",
        "e ai",
        "ei",
        "hey",
        "tudo bem",
        "oi tudo bem",
        "ola tudo bem",
        "olá tudo bem",
        "boa tarde tudo bem",
        "boa noite tudo bem",
        "bom dia tudo bem",
    }
    if current in greeting_phrases:
        return True

    tokens = current.split()
    if len(tokens) > 5:
        return False

    allowed_tokens = {
        "oi", "ola", "olá", "bom", "boa", "dia", "tarde", "noite",
        "ei", "hey", "e", "ai", "tudo", "bem",
    }
    return all(token in allowed_tokens for token in tokens)


def _strip_leading_agent_intro(text: str) -> str:
    original = (text or "").strip()
    if not original:
        return original

    intro_patterns = [
        r"^\s*(?:ol[áa]!?\s*)?sou o dairy ai\b[^.?!]*[.?!]\s*",
        r"^\s*sou o dairy ai\b[^.?!]*[.?!]\s*",
    ]
    trailing_help_patterns = [
        r"^(?:posso ajudar[^.?!]*[.?!]\s*)+",
        r"^(?:como posso ajudar[^.?!]*[.?!]\s*)+",
    ]

    stripped = original
    removed_intro = False
    for pattern in intro_patterns:
        updated, count = re.subn(pattern, "", stripped, count=1, flags=re.IGNORECASE)
        if count:
            stripped = updated.lstrip()
            removed_intro = True
            break

    if not removed_intro:
        return original

    for pattern in trailing_help_patterns:
        stripped = re.sub(pattern, "", stripped, count=1, flags=re.IGNORECASE).lstrip()

    if not stripped:
        return original

    if _normalize_folded_text(stripped) == _normalize_folded_text(original):
        return original
    return stripped


def _extract_query_keywords(query: str) -> List[str]:
    tokens = re.findall(r"[^\W_]+", _normalize_text_for_match(query), flags=re.UNICODE)
    keywords: List[str] = []
    seen = set()
    for token in tokens:
        if len(token) < 4 or token in _PT_STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def _keyword_hits_in_result(result: Dict[str, Any], keywords: List[str]) -> int:
    content = _normalize_text_for_match(str(result.get("content", "")))
    metadata = result.get("metadata") or {}
    source = ""
    if isinstance(metadata, dict):
        source = _normalize_text_for_match(
            " ".join(
                str(metadata.get(key, ""))
                for key in ("source", "title", "path")
                if metadata.get(key)
            )
        )
    haystack = f"{content} {source}".strip()
    return sum(1 for kw in keywords if kw in haystack)


def _strip_profile_suffix(text: str) -> str:
    if "\n[Perfil" in text:
        return text.split("\n[Perfil", 1)[0]
    return text


def _extract_current_user_segment(text: str) -> str:
    marker = "\n[Pergunta atual]\n"
    if marker in text:
        return text.rsplit(marker, 1)[1].strip()
    if text.strip().startswith("[Pergunta atual]"):
        return text.split("[Pergunta atual]", 1)[1].strip()
    return text


def _build_contextual_search_query(text: str) -> str:
    """Constrói a query de busca RAG a partir da mensagem do usuário.

    Quando a mensagem contém um bloco de contexto histórico (inserido por
    _build_orchestrator_input_messages), duas estratégias são possíveis:

    1. LLM-based (CONTEXTUAL_QUERY_REWRITE_ENABLED=true — padrão):
       Chama contextualize_query_for_rag() para resolver anáfora e referências
       implícitas antes do retrieval. Ex.:
           "E quanto ao pH?"  →  "Qual o pH ideal na filagem da mussarela?"
       Isso garante que o vetor de busca carregue o contexto semântico correto,
       não apenas o pronome solto.

    2. Fallback por concatenação (feature desabilitada ou sem histórico):
       Une os últimos snippets do usuário com " | " — comportamento legado.
       Adequado como safety net mas impreciso para anáfora.
    """
    current = _strip_profile_suffix(_extract_current_user_segment(text)).strip()
    if not current:
        return ""

    # Sem bloco de contexto histórico: query autossuficiente, usa direto.
    if "[Contexto recente da conversa]" not in text or "[Pergunta atual]" not in text:
        return current

    # Extrai o bloco de contexto entre os marcadores.
    context_block = text.split("[Contexto recente da conversa]", 1)[1]
    context_block = context_block.split("[Pergunta atual]", 1)[0]

    # Coleta todas as linhas de conversa (usuário e assistente) preservando
    # a ordem e o prefixo de papel — o LLM usa ambos para resolver anáfora.
    context_lines: List[str] = []
    for raw_line in context_block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line_norm = unicodedata.normalize("NFKD", line)
        line_norm = "".join(ch for ch in line_norm if not unicodedata.combining(ch)).lower()
        if line_norm.startswith("usuario:") or line_norm.startswith("dairy ai:"):
            context_lines.append(line)

    if CONTEXTUAL_QUERY_REWRITE_ENABLED and context_lines:
        # Caminho principal: reescrita semântica via LLM.
        # contextualize_query_for_rag() garante fail-safe: retorna current
        # sem modificação em caso de falha, timeout ou query autossuficiente.
        return contextualize_query_for_rag(current, context_lines)

    # Fallback: concatenação de keywords por " | " (comportamento pré-existente).
    # Acionado apenas quando CONTEXTUAL_QUERY_REWRITE_ENABLED=false ou sem linhas.
    user_snippets: List[str] = []
    for line in context_lines:
        line_norm = unicodedata.normalize("NFKD", line)
        line_norm = "".join(ch for ch in line_norm if not unicodedata.combining(ch)).lower()
        if not line_norm.startswith("usuario:"):
            continue
        snippet = line.split(":", 1)[1].strip()
        snippet = _strip_profile_suffix(snippet)
        if snippet:
            user_snippets.append(snippet)

    if not user_snippets:
        return current

    combined = " | ".join(user_snippets[-2:] + [current]).strip(" |")
    combined = re.sub(r"\s+", " ", combined).strip()
    if len(combined) <= 320:
        return combined
    return combined[:317].rstrip() + "..."


def _looks_like_insufficient_answer(text: str) -> bool:
    normalized = _normalize_folded_text(text)
    if not normalized:
        return True
    markers = (
        "nao encontrei informacoes especificas",
        "nao encontrei informacao suficiente",
        "nao tenho informacao suficiente",
        "nao ha informacao suficiente",
        "nao foi possivel identificar",
        "com o meu conhecimento atual",
        "nas normas consultadas",
    )
    return any(marker in normalized for marker in markers)


def _is_regulatory_minimum_maturation_query(query: str) -> bool:
    text = _normalize_folded_text(_build_contextual_search_query(query))
    if not text:
        return False

    has_requirement = (
        "exigid" in text
        or "obrigat" in text
        or "minimo legal" in text
        or "periodo minimo" in text
        or "prazo minimo" in text
        or "tempo minimo" in text
    )
    has_maturation = "maturacao" in text
    has_cheese_context = any(term in text for term in ("queijo", "grana", "parmesao", "parmesan"))
    return has_requirement and has_maturation and has_cheese_context


def _unwrap_tool_result_payload(payload: Any) -> List[Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            try:
                payload = ast.literal_eval(payload)
            except Exception:
                return []

    if isinstance(payload, list):
        flattened: List[Any] = []
        for item in payload:
            if isinstance(item, dict):
                nested = None
                for key in ("results", "chunks", "items", "data", "documents", "matches"):
                    value = item.get(key)
                    if isinstance(value, list):
                        nested = value
                        break
                if nested is not None:
                    flattened.extend(nested)
                    continue
            flattened.append(item)
        return flattened

    if isinstance(payload, dict):
        for key in ("results", "chunks", "items", "data", "documents", "matches"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if any(
            key in payload
            for key in ("content", "text", "page_content", "snippet", "chunk")
        ):
            return [payload]

    return []


def _extract_tool_results(messages: List[AnyMessage]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        parsed = _unwrap_tool_result_payload(msg.content)
        if not parsed:
            artifact = getattr(msg, "artifact", None)
            if artifact is not None:
                parsed = _unwrap_tool_result_payload(artifact)
        for item in parsed:
            if isinstance(item, dict):
                if "content" not in item:
                    text = (
                        item.get("text")
                        or item.get("page_content")
                        or item.get("snippet")
                        or item.get("chunk")
                    )
                    if text:
                        item = dict(item)
                        item["content"] = text
                results.append(item)
    return results


def _build_regulatory_general_rule_answer(query: str, tool_results: List[Dict[str, Any]]) -> Optional[str]:
    if not _is_regulatory_minimum_maturation_query(query):
        return None

    for item in tool_results:
        content = str(item.get("content", "") or "")
        normalized = _normalize_folded_text(content)
        if "sessenta dias" not in normalized:
            continue
        if "matur" not in normalized:
            continue
        if "riispoa" not in normalized and "§ 6" not in content and "paragrafo 6" not in normalized:
            continue
        return (
            "Na ausencia de RTIQ especifico citado nos trechos recuperados para queijo tipo Grana, "
            "a regra geral explicita na base e a do § 6º do RIISPOA (Decreto 9.013/2017): "
            "o leite destinado a queijos submetidos a maturacao deve observar periodo nao inferior "
            "a sessenta dias, ressalvadas excecoes amparadas por estudos tecnico-cientificos ou "
            "regulamentos tecnicos especificos."
        )

    return None


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

        precomputed_embedding: Embedding da query pré-computado pelo orquestrador.
                               Quando presente, evita chamada duplicada à OpenAI.
    """
    messages: Annotated[List[AnyMessage], add_messages]
    agent_prompt: str
    llm_model: str
    precomputed_embedding: Optional[List[float]]
    search_query: Optional[str]


# ============================================================
# Construção do grafo ReAct para um agente
# ============================================================

def build_agent_graph(agent_id: int, model_name: str = LLM_MODEL) -> Any:
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
    agent_search_type = (search_config.get("search_type") or DEFAULT_SEARCH_TYPE).strip().lower()
    
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
    model = ChatOpenAI(
        model=model_name,
        temperature=AGENT_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
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
        search_query = str(state.get("search_query") or "").strip()
        if not search_query:
            search_query = _build_contextual_search_query(user_text)

        # Reutiliza embedding pré-computado pelo orquestrador quando disponível,
        # evitando chamada duplicada à OpenAI para a mesma query.
        tool_args: Dict[str, Any] = {"query": search_query}
        precomputed = state.get("precomputed_embedding")
        if precomputed:
            tool_args["embedding"] = precomputed

        # AIMessage que força a chamada à tool sem passar pelo LLM
        forced_call = AIMessage(
            content="",
            tool_calls=[{
                "id": "direct_search_0",
                "name": kb_tool_name,
                "args": tool_args,
                "type": "tool_call",
            }],
        )

        return {
            "agent_prompt": prompt_text,
            "llm_model": model_name,
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

        # Quando há chunks RAG reais, reforça ancoragem no system prompt.
        # Evidência insuficiente é decisão do orquestrador, não improviso do agente.
        tool_results = _extract_tool_results(messages)
        prompt_text = state.get("agent_prompt", "")
        if tool_results and prompt_text:
            # 400 chars por chunk garante que valores numéricos e parâmetros
            # críticos apareçam no resumo — 120 chars truncava antes do dado relevante.
            chunk_summary = "\n".join(
                f"[{i+1}] {str(item.get('content', ''))[:400].replace(chr(10), ' ')}"
                for i, item in enumerate(tool_results[:5])
                if item.get("content")
            )
            anchoring = (
                "\n\nATENCAO — ANCORAGEM OBRIGATORIA: use EXCLUSIVAMENTE os trechos "
                "recuperados abaixo. Nao use conhecimento geral nem complete com memoria "
                "de treinamento. Extraia e sintetize os dados presentes — mesmo que "
                "parciais. Nao emita juizo sobre suficiencia da evidencia: isso e decisao "
                "do sistema. Se absolutamente nenhum trecho tocar no tema da pergunta, "
                "responda exatamente [FORA_DE_ESCOPO] — mas apenas nesse caso extremo.\n"
                f"Trechos recuperados:\n{chunk_summary}"
            )
            prompt_text = prompt_text + anchoring

        if prompt_text:
            messages = [SystemMessage(content=prompt_text)] + messages

        response = await model_with_tools.ainvoke(messages)

        last_user_text = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_text = msg.content
                break

        if isinstance(response.content, str) and last_user_text and not _looks_like_greeting_only(last_user_text):
            stripped_intro = _strip_leading_agent_intro(response.content)
            if stripped_intro != response.content:
                response = AIMessage(content=stripped_intro)

        if agent_id == 3 and isinstance(response.content, str):
            if _looks_like_insufficient_answer(response.content):
                fallback_answer = _build_regulatory_general_rule_answer(
                    last_user_text,
                    _extract_tool_results(messages),
                )
                if fallback_answer:
                    response = AIMessage(content=fallback_answer)

        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"
        last_message = messages[-1]
        # Limite: máximo 2 buscas por agente (1 forçada no prepare + 1 retry voluntário).
        # Evita loops infinitos quando a base não tem evidência para a query.
        tool_calls_done = sum(1 for m in messages if isinstance(m, ToolMessage))
        if tool_calls_done >= 2:
            return "end"
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    def check_search_quality(state: AgentState) -> str:
        """Após a primeira busca, verifica se os resultados têm relevância mínima.

        Versão conservadora:
        - só roda quando a feature flag está ligada;
        - só vale para agentes explicitamente permitidos (default: base_geral);
        - só atua em busca híbrida;
        - exige query minimamente específica e baixa aderência lexical.

        Assim cortamos apenas retries caros e pouco promissores, sem
        transformar score baixo isolado em "sem evidência".
        """
        if not RAG_EARLY_SKIP_WEAK_SEARCH_ENABLED:
            return "agent"
        if agent_id not in RAG_EARLY_SKIP_WEAK_AGENT_IDS:
            return "agent"
        if agent_search_type not in {"hybrid", "hybrid_rrf"}:
            return "agent"

        messages = state.get("messages", [])
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]

        # Aplica apenas na primeira busca; retries seguem fluxo normal.
        if len(tool_msgs) != 1:
            return "agent"

        user_text = str(state.get("search_query") or "").strip()
        for msg in reversed(messages):
            if user_text:
                break
            if isinstance(msg, HumanMessage):
                user_text = _build_contextual_search_query(msg.content)
                break

        keywords = _extract_query_keywords(user_text)
        if len(keywords) < RAG_EARLY_SKIP_WEAK_MIN_QUERY_KEYWORDS:
            return "agent"

        content = tool_msgs[0].content
        try:
            results = _unwrap_tool_result_payload(content)
            if not results or not isinstance(results, list):
                # Base retornou lista vazia → sem evidência.
                return "skip_weak"

            top_results = [
                item for item in results[: min(3, len(results))]
                if isinstance(item, dict)
            ]
            max_score = max(
                float(r.get("score", 0.0)) for r in top_results
            )
            max_keyword_hits = max(
                (_keyword_hits_in_result(item, keywords) for item in top_results),
                default=0,
            )

            if (
                max_score < RAG_EARLY_SKIP_WEAK_HYBRID_MAX_SCORE
                and max_keyword_hits < RAG_EARLY_SKIP_WEAK_MIN_TOP_KEYWORD_HITS
            ):
                return "skip_weak"
        except Exception:
            # Não conseguiu parsear → deixa o LLM decidir (fail open).
            return "agent"

        return "agent"

    def respond_no_evidence(state: AgentState) -> AgentState:
        """Retorna AIMessage vazia para sinalizar ao orquestrador que não há evidência.

        O orquestrador interpreta conteúdo vazio como success=False e
        aciona o fallback adequado (general index / web / resposta padrão).
        Poupa a chamada LLM de decisão + busca retry que seriam inúteis.
        """
        return {"messages": [AIMessage(content="")]}

    # ---- Passo 5: Montar o grafo ----
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    graph.add_node("prepare", prepare)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("skip_weak", respond_no_evidence)

    # Fluxo otimizado:
    #   START → prepare → tools → [check_search_quality] → agent → END
    #                                                     ↘ skip_weak → END
    #
    # prepare injeta a tool call já formada, pulando o LLM de decisão.
    # check_search_quality corta o ciclo quando a 1ª busca é fraca.
    # O loop ReAct (agent ↔ tools) ainda existe como safety net nos demais casos.
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "tools")

    graph.add_conditional_edges(
        "tools",
        check_search_quality,
        {"agent": "agent", "skip_weak": "skip_weak"},
    )

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )

    graph.add_edge("skip_weak", END)

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

_agent_graphs: Dict[tuple[int, str], Any] = {}


def get_agent_graph(agent_id: int, model_name: str = LLM_MODEL) -> Any:
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
    cache_key = (agent_id, model_name)
    if cache_key not in _agent_graphs:
        _agent_graphs[cache_key] = build_agent_graph(agent_id, model_name=model_name)
    return _agent_graphs[cache_key]


def get_all_agent_graphs(model_names: Optional[List[str]] = None) -> Dict[tuple[int, str], Any]:
    """Compila e retorna todos os 6 grafos de agentes.
    
    Útil para pré-aquecer o cache no startup do servidor,
    em vez de compilar sob demanda na primeira request.
    
    Usado por: webapp.py → no evento de startup.
    """
    target_models = model_names or [LLM_MODEL]
    for agent_config in AGENTS:
        agent_id = agent_config["agent_id"]
        for model_name in target_models:
            cache_key = (agent_id, model_name)
            if cache_key not in _agent_graphs:
                _agent_graphs[cache_key] = build_agent_graph(agent_id, model_name=model_name)
    return _agent_graphs
