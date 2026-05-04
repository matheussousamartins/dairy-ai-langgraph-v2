"""
rag/clarification.py — Mecanismo estruturado de clarificação pré-RAG

Detecta queries tão vagas que o RAG dificilmente recuperaria algo útil e,
nesses casos, retorna uma pergunta de esclarecimento ao usuário antes de
acionar o pipeline completo (classificador → agentes → consolidação).

Filosofia: conservadorismo máximo
  A clarificação deve ser exceção, não regra. Uma resposta imperfeita do
  RAG é muito melhor que uma pergunta desnecessária que interrompe o fluxo.
  O LLM recebe instruções explícitas: "na dúvida, não pergunte".

Arquitetura:
  check_needs_clarification() é chamada ANTES de qualquer invocação de grafo.
  Se needs_clarification=True, o pipeline inteiro é bypassado e apenas a
  pergunta de esclarecimento é emitida via SSE — zero custo de agentes RAG.

Proteções:
  - Heurística pré-filtro: queries longas e com termos técnicos nunca chegam
    ao LLM (zero latência adicional para a grande maioria das queries).
  - Loop guard: se o último turno do assistente foi uma clarificação, não
    pede outra (evita ciclo infinito de perguntas).
  - Fail-safe total: qualquer exceção retorna needs_clarification=False,
    nunca bloqueando o pipeline principal.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from app.config import (
    CLARIFICATION_ENABLED,
    CLARIFICATION_MODEL,
    CLARIFICATION_MAX_QUERY_LEN_TO_SKIP,
    CLARIFICATION_TIMEOUT_SEC,
)

_log = logging.getLogger(__name__)

_clarification_model: Optional[ChatOpenAI] = None


# ============================================================
# Singleton do modelo de clarificação
# ============================================================

def _get_clarification_model() -> ChatOpenAI:
    global _clarification_model
    if _clarification_model is None:
        _clarification_model = ChatOpenAI(
            model=CLARIFICATION_MODEL,
            temperature=0,
            max_tokens=200,
            timeout=CLARIFICATION_TIMEOUT_SEC,
        )
    return _clarification_model


# ============================================================
# Resultado (imutável)
# ============================================================

@dataclass(frozen=True)
class ClarificationResult:
    """Resultado imutável da verificação de necessidade de clarificação."""
    needs_clarification: bool
    question: Optional[str]  # preenchido apenas quando needs_clarification=True
    reason: str              # para log e observabilidade


# ============================================================
# Constantes de detecção
# ============================================================

# Termos técnicos de domínio. Presença de qualquer um indica que a query
# tem especificidade suficiente para tentar o RAG diretamente.
_DOMAIN_SIGNALS: frozenset[str] = frozenset({
    # Produtos
    "mussarela", "mozzarela", "prato", "minas", "gorgonzola", "brie",
    "camembert", "grana", "parmesao", "cheddar", "gouda", "ricota",
    "requeijao", "iogurte", "kefir", "coalhada", "coalho", "nata",
    "creme", "manteiga", "soro", "leitelho", "permeado",
    # Matéria-prima
    "leite", "lactose", "caseina", "proteina", "gordura", "lactobacilo",
    "streptococcus", "cultura", "fermento", "enzima", "coalho", "quimosina",
    # Processos
    "pasteurizacao", "esterilizacao", "fermentacao", "filagem", "prensagem",
    "maturacao", "salga", "coagulacao", "homogeneizacao", "ultrafiltracao",
    "microfiltrar", "nanofiltrar", "evaporacao", "spray", "liofilizacao",
    # Parâmetros físico-químicos
    "temperatura", "tempo", "ph", "acidez", "dornic", "brix", "umidade",
    "aw", "atividade", "sal", "nacl", "gordura", "extrato", "cfcs",
    # Regulatório
    "instrucao", "normativa", "portaria", "legislacao", "regulamento",
    "anvisa", "mapa", "riispoa", "rtiq", "rdc", "pbufala", "sbcta",
    # Qualidade e defeitos
    "defeito", "contaminacao", "textura", "sabor", "odor", "aroma",
    "separacao", "sineres", "estufamento", "escurecimento", "ranso",
    "oxidacao", "lipolise", "proteolise", "olhaduras",
    # Formulação
    "formulacao", "receita", "ingredientes", "aditivo", "conservante",
    "corante", "espessante", "estabilizante",
})

# Marcadores de vagueza explícita — indicam que a query provavelmente
# não tem especificidade suficiente para um retrieval útil.
_VAGUE_MARKERS: tuple[str, ...] = (
    "quero saber mais",
    "me fala sobre",
    "me conte sobre",
    "me explica sobre",
    "pode me ajudar",
    "preciso de ajuda com",
    "alguma coisa sobre",
    "algo sobre",
    "o que voce sabe sobre",
    "o que você sabe sobre",
    "como funciona isso",
    "fala sobre",
    "quero entender",
    "me da uma ideia",
    "me dê uma ideia",
    "o que e isso",
    "o que é isso",
)


# ============================================================
# Funções auxiliares
# ============================================================

def _fold_accents(text: str) -> str:
    """Remove acentos para comparação case-insensitive sem dependências externas."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _strip_profile_suffix(text: str) -> str:
    """Remove o bloco [Perfil do usuário: ...] appendado pelo _inject_user_profile."""
    if "\n[Perfil" in text:
        return text.split("\n[Perfil", 1)[0]
    return text


def _strip_context_block(text: str) -> str:
    """Remove o bloco [Contexto recente da conversa] — fica com a pergunta atual."""
    if "[Pergunta atual]" in text:
        return text.rsplit("[Pergunta atual]", 1)[1].strip()
    if "[Contexto recente da conversa]" in text:
        # Bloco sem marcador de pergunta — descarta tudo (improvável).
        return ""
    return text


def _is_clarification_eligible(query: str) -> bool:
    """Heurística rápida (sem LLM): decide se a query PODE precisar de clarificação.

    Retorna False imediatamente quando há evidências fortes de especificidade,
    evitando qualquer chamada LLM para a grande maioria das queries.

    Política:
    - Queries longas (> CLARIFICATION_MAX_QUERY_LEN_TO_SKIP) → nunca clarificar.
    - Presença de termo técnico de domínio → nunca clarificar.
    - Marcadores de vagueza explícita → elegível.
    - Queries muito curtas sem termos de domínio → elegível.
    """
    cleaned = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not cleaned:
        return False

    if len(cleaned) > CLARIFICATION_MAX_QUERY_LEN_TO_SKIP:
        return False

    folded = _fold_accents(cleaned)
    tokens = set(re.findall(r"[^\W_]+", folded, flags=re.UNICODE))
    if tokens & _DOMAIN_SIGNALS:
        return False

    if any(m in cleaned for m in _VAGUE_MARKERS):
        return True

    # Query muito curta e sem termos técnicos → pode ser vaga.
    if len(cleaned.split()) <= 4:
        return True

    return False


def _was_last_ai_turn_a_clarification(history: List[Dict[str, str]]) -> bool:
    """Detecta se o último turno do assistente foi uma pergunta de clarificação.

    Critério conservador: mensagem curta (≤ 260 chars) que termina com "?".
    Respostas RAG reais são sempre bem mais longas.

    Propósito: evitar loop de clarificação — se o usuário já recebeu uma
    pergunta de esclarecimento e está respondendo, não perguntar novamente.
    """
    for msg in reversed(history or []):
        role = msg.get("role", "")
        if role == "ai":
            content = (msg.get("content", "") or "").strip()
            return bool(content) and len(content) <= 260 and content.endswith("?")
        if role == "human":
            # Encontrou outra mensagem humana antes de AI: não é reply a clarificação.
            break
    return False


def _format_history_for_prompt(
    history: List[Dict[str, str]],
    max_turns: int,
) -> str:
    """Formata histórico recente para inclusão no prompt de clarificação.

    Limita a max_turns turnos (pares humano+assistente), trunca mensagens
    longas e remove blocos de perfil e contexto para manter o prompt enxuto.
    """
    if not history:
        return "(sem histórico)"

    lines: List[str] = []
    turns_seen = 0

    for msg in reversed(history):
        role = msg.get("role", "")
        raw = (msg.get("content", "") or "").strip()
        if not raw:
            continue

        raw = _strip_profile_suffix(raw)
        raw = _strip_context_block(raw)
        content = re.sub(r"\s+", " ", raw).strip()
        if not content:
            continue

        if role == "human":
            snippet = content[:220] + "..." if len(content) > 220 else content
            lines.append(f"Usuário: {snippet}")
            turns_seen += 1
        elif role == "ai":
            snippet = content[:160] + "..." if len(content) > 160 else content
            lines.append(f"Assistente: {snippet}")

        if turns_seen >= max_turns:
            break

    lines.reverse()
    return "\n".join(lines) if lines else "(sem histórico)"


# ============================================================
# Função principal
# ============================================================

def check_needs_clarification(
    query: str,
    history: List[Dict[str, str]],
    user_profile: Optional[Dict[str, Any]] = None,
) -> ClarificationResult:
    """Verifica se a query atual precisa de esclarecimento antes do pipeline RAG.

    Fluxo:
        1. Guard de feature flag (retorno imediato se desabilitado).
        2. Guard de loop (não perguntar se o último turno já foi clarificação).
        3. Heurística pré-filtro sem LLM (zero latência para queries específicas).
        4. Chamada LLM com output estruturado JSON.
        5. Validação e sanitização da resposta.

    Fail-safe garantido: qualquer exceção retorna needs_clarification=False.
    Nunca bloqueia nem lança exceção para o chamador.

    Args:
        query:        Pergunta atual do usuário (texto puro, sem bloco de perfil).
        history:      Histórico de turnos [{"role": "human|ai", "content": "..."}].
        user_profile: Perfil do usuário — opcional, melhora contextualização.

    Returns:
        ClarificationResult imutável.
    """
    if not CLARIFICATION_ENABLED:
        return ClarificationResult(False, None, "feature_disabled")

    query_clean = re.sub(r"\s+", " ", _strip_profile_suffix(query or "").strip())
    query_clean = _strip_context_block(query_clean).strip()
    if not query_clean:
        return ClarificationResult(False, None, "empty_query")

    # Guard: usuário está respondendo a uma clarificação anterior.
    if _was_last_ai_turn_a_clarification(history):
        return ClarificationResult(False, None, "answering_previous_clarification")

    # Heurística rápida — bloqueia LLM para a grande maioria das queries.
    if not _is_clarification_eligible(query_clean):
        return ClarificationResult(False, None, "query_specific_enough")

    # --- Chamada LLM com output estruturado ---
    history_text = _format_history_for_prompt(history, max_turns=3)

    profile_note = ""
    if user_profile:
        level = (user_profile.get("knowledgeLevel") or "").strip()
        role = (user_profile.get("role") or "").strip()
        if level or role:
            profile_note = (
                f"\nPerfil do usuário: "
                f"nível={level or 'não informado'}, "
                f"função={role or 'não informado'}"
            )

    prompt = (
        "Você avalia se perguntas sobre tecnologia de laticínios precisam de "
        "esclarecimento antes de serem respondidas por uma base de conhecimento técnica.\n\n"
        "Responda APENAS com JSON válido no formato exato:\n"
        '{"needs_clarification": bool, "question": "string ou null"}\n\n'
        "═══ QUANDO PEDIR ESCLARECIMENTO (raramente) ═══\n"
        "- A pergunta não menciona nenhum produto, processo ou parâmetro de laticínios\n"
        "  e é tão vaga que qualquer busca retornaria resultados irrelevantes.\n"
        "  Exemplos que precisam: 'quero saber mais', 'me fala sobre', 'o que é isso?'\n"
        "- Há ambiguidade radical que mudaria completamente a resposta\n"
        "  (ex: 'como faço?' sem nenhum contexto sobre qual produto).\n\n"
        "═══ NÃO PEDIR ESCLARECIMENTO (na maioria dos casos) ═══\n"
        "- A pergunta menciona qualquer produto, processo ou parâmetro de laticínios.\n"
        "- O histórico recente fornece contexto suficiente para interpretar a query.\n"
        "- O usuário está respondendo a uma pergunta anterior do assistente.\n"
        "- A pergunta é curta mas técnica: 'pH da filagem', 'IN 76', 'tempo de prensagem'.\n"
        "- NA DÚVIDA → NÃO PERGUNTE. Proceda sem esclarecimento. Esta é a regra mais importante.\n\n"
        "Quando pedir esclarecimento, a pergunta deve ser:\n"
        "- Máximo 1 pergunta objetiva e direta.\n"
        "- Orientada ao produto ou processo específico que falta especificar.\n"
        "- Em português brasileiro, tom profissional e acolhedor.\n"
        f"{profile_note}\n"
        f"Histórico recente da conversa:\n{history_text}\n\n"
        f"Pergunta do usuário: {query_clean}\n\n"
        "JSON:"
    )

    try:
        llm = _get_clarification_model()
        resp = llm.invoke(prompt)
        raw = (resp.content or "").strip()

        # Extrai o bloco JSON mesmo que o LLM adicione texto extra ao redor.
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not match:
            _log.warning(
                "check_needs_clarification: JSON não encontrado na resposta: %.200s", raw
            )
            return ClarificationResult(False, None, "llm_invalid_json")

        parsed = json.loads(match.group())
        needs = bool(parsed.get("needs_clarification", False))
        question_raw = parsed.get("question") or None

        if not needs or not question_raw:
            return ClarificationResult(False, None, "llm_not_needed")

        # Sanitiza a pergunta de esclarecimento.
        question = re.sub(r"\s+", " ", str(question_raw)).strip()
        question = re.sub(r'^["\'""`]|["\'""`]$', "", question).strip()
        question = question.lstrip(":").strip()

        if not question:
            return ClarificationResult(False, None, "llm_empty_question")

        if not question.endswith("?"):
            question += "?"

        # Segurança: uma pergunta de clarificação não deve ser uma resposta disfarçada.
        if len(question) > 300:
            _log.warning(
                "check_needs_clarification: pergunta longa demais (%d chars), ignorando.",
                len(question),
            )
            return ClarificationResult(False, None, "llm_question_too_long")

        _log.info(
            "check_needs_clarification: clarificação necessária | query=%r | question=%r",
            query_clean,
            question,
        )
        return ClarificationResult(True, question, "llm_decision")

    except Exception as exc:
        _log.warning(
            "check_needs_clarification: falha — pipeline normal será executado. Erro: %s",
            exc,
        )
        return ClarificationResult(False, None, f"error:{type(exc).__name__}")
