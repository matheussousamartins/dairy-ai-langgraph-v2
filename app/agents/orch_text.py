"""
orch_text.py — Funções puras de processamento de texto do orquestrador.

Contém apenas funções que dependem de stdlib (re, unicodedata) e nada de
LangChain, LangGraph, Supabase ou config. Pode ser importado por qualquer
módulo sem criar dependências circulares.
"""

import re
import unicodedata
from typing import List, Optional


# ============================================================
# Normalização básica
# ============================================================

def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalize_mul_symbols(text: str) -> str:
    return (
        text.replace("×", "x")
        .replace("*", "x")
        .replace("X", "x")
    )


# ============================================================
# Extração de segmentos de contexto
# ============================================================

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


def _extract_recent_context_block(text: str) -> str:
    if "[Contexto recente da conversa]" not in text:
        return ""
    context_block = text.split("[Contexto recente da conversa]", 1)[1]
    if "[Pergunta atual]" in context_block:
        context_block = context_block.split("[Pergunta atual]", 1)[0]
    return context_block.strip()


def _has_recent_context_block(text: str) -> bool:
    return bool(_extract_recent_context_block(text))


def _build_contextual_search_query(text: str) -> str:
    current = _strip_profile_suffix(_extract_current_user_segment(text)).strip()
    if not current:
        return ""

    if "[Contexto recente da conversa]" not in text or "[Pergunta atual]" not in text:
        return current

    context_block = text.split("[Contexto recente da conversa]", 1)[1]
    context_block = context_block.split("[Pergunta atual]", 1)[0]

    user_snippets: List[str] = []
    for raw_line in context_block.splitlines():
        line = raw_line.strip()
        line_norm = unicodedata.normalize("NFKD", line)
        line_norm = "".join(ch for ch in line_norm if not unicodedata.combining(ch))
        line_norm = line_norm.lower()
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


# ============================================================
# Detecção de intenção de recapitulação e perguntas objetivas
# ============================================================

def _is_conversation_recap_request(text_norm: str) -> bool:
    if not text_norm:
        return False

    strong_phrases = (
        "o que conversamos",
        "sobre o que conversamos",
        "conversamos recentemente",
        "o que falamos antes",
        "o que falamos",
        "falamos recentemente",
        "me explique sobre o que conversamos",
        "me lembre do que conversamos",
        "resuma o que conversamos",
        "resuma o que falamos",
        "retome o que falamos",
        "continue de onde paramos",
        "retome a conversa",
    )
    if any(phrase in text_norm for phrase in strong_phrases):
        return True

    return bool(
        re.search(
            r"\b(resuma|retome|relembre|recapitule|continue|explique)\b.*\b(conversa|conversamos|falamos)\b",
            text_norm,
        )
    )


def _is_objective_question(text: str) -> bool:
    q = _normalize_text(text)
    if not q:
        return False
    patterns = (
        r"^(quem e|qual e|quais sao|quanto e|onde fica|onde e|quando|como se chama)\b",
        r"^(quem|qual|quais|quanto|onde|quando)\b",
    )
    return any(re.search(p, q) for p in patterns)


# ============================================================
# Detecção de incerteza
# ============================================================

def _looks_uncertain(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return True
    uncertainty_markers = (
        "nao encontrei informacao suficiente",
        "nao encontrei informacoes suficientes",
        "nao encontrei informacoes especificas",
        "nao tenho informacao suficiente",
        "nao tenho informacoes suficientes",
        "nao ha informacao suficiente",
        "nao ha evidencia suficiente",
        "nao ha evidencias suficientes",
        "nao ha dados suficientes",
        "sem informacao suficiente",
        "informacao insuficiente",
        "faltam evidencias",
        "pode ser",
        "talvez",
        "recomenda-se verificar",
        "aconselhavel verificar",
        "consultar fontes adicionais",
        "com o meu conhecimento atual",
        "com o seu conhecimento atual",
        "nao foi possivel identificar",
        "nao foi possivel encontrar",
        "evidencia insuficiente",
        "nao disponho de informacao",
    )
    return any(marker in t for marker in uncertainty_markers)


def _strip_uncertainty_tail(text: str) -> str:
    """Remove cauda de ressalva genérica quando houver fato já respondido."""
    if not text:
        return ""
    out = str(text).strip()

    m = re.search(r"\b(No entanto|Por[ée]m|Contudo)\b", out, flags=re.IGNORECASE)
    if m:
        out = out[: m.start()].strip()

    m2 = re.search(
        r"\b(a base atual n[ãa]o trouxe|nao trouxe informacao suficiente|"
        r"faltam evidencias|com o meu conhecimento atual|"
        r"recomenda-se verificar|aconselh[aá]vel verificar)\b",
        out,
        flags=re.IGNORECASE,
    )
    if m2:
        out = out[: m2.start()].strip()

    out = re.sub(r"[;,:\-–—]+$", "", out).strip()
    return out


def _strip_leading_uncertainty_prefix(text: str) -> str:
    if not text:
        return ""
    out = str(text).strip()
    out = re.sub(
        r"^\s*(?:Com base no meu conhecimento atual|Com o meu conhecimento atual|Com base nas informacoes disponiveis),\s*",
        "",
        out,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    return out


def _extract_factual_candidate(text: str) -> Optional[str]:
    """Extrai parte factual útil de uma resposta mista (fato + ressalva)."""
    cleaned = _sanitize_math_for_ui(text or "")
    cleaned = _strip_leading_uncertainty_prefix(cleaned)
    if not cleaned:
        return None
    head = _strip_uncertainty_tail(cleaned)
    if not head:
        return None
    if len(head.split()) < 3:
        return None
    if _looks_uncertain(head):
        return None
    return head


# ============================================================
# Limpeza de saída matemática / LaTeX
# ============================================================

def _sanitize_math_for_ui(text: str) -> str:
    """Converte trechos matemáticos em LaTeX para texto simples amigável ao front."""
    if not text:
        return text

    out = str(text)
    out = re.sub(r"\\\[(.*?)\\\]", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\\\((.*?)\\\)", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\$\$(.*?)\$\$", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\$(.*?)\$", r"\1", out, flags=re.DOTALL)

    out = out.replace(r"\times", "x")
    out = out.replace(r"\cdot", "x")
    out = out.replace(r"\,", " ")
    out = out.replace("\\n", "\n")
    out = out.replace("\t", " ")
    out = re.sub(r"\\text\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\\[a-zA-Z]+", "", out)
    out = out.replace("{", "").replace("}", "")

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


def _dedupe_paragraphs(text: str) -> str:
    """Remove parágrafos duplicados mantendo a primeira ocorrência."""
    if not text:
        return text
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    seen = set()
    kept: List[str] = []
    for p in parts:
        key = _normalize_text(p)
        if key in seen:
            continue
        seen.add(key)
        kept.append(p)
    return "\n\n".join(kept).strip()


def _enforce_dornic_canonical_formula(user_text: str, text: str) -> str:
    """Garante forma canônica da fórmula Dornic quando a pergunta for desse tema.

    Fórmula canônica (IN 68): Acidez (Dornic) = V x f x 0,9 x 10
    """
    if not text:
        return text
    q = _normalize_text(user_text)
    if "dornic" not in q and not ("acidez" in q and "titul" in q):
        return text

    out = text
    patterns = [
        r"Acidez\s*\(?°?\s*D(?:ornic)?\)?\s*=\s*V\s*[x\*]\s*f\s*[x\*]\s*10",
        r"Acidez\s*\('?\s*Dornic\s*'?\)\s*=\s*V\s*[x\*]\s*f\s*[x\*]\s*10",
    ]
    for pat in patterns:
        out = re.sub(
            pat,
            "Acidez (Dornic) = V x f x 0,9 x 10",
            out,
            flags=re.IGNORECASE,
        )

    lines = out.splitlines()
    new_lines: List[str] = []
    seen_formula_line = False
    for line in lines:
        ln_norm = _normalize_mul_symbols(line)
        is_dornic_formula = (
            "acidez" in ln_norm.lower()
            and "dornic" in ln_norm.lower()
            and "=" in ln_norm
            and "v" in ln_norm.lower()
            and "f" in ln_norm.lower()
            and "10" in ln_norm
        )
        if is_dornic_formula:
            if seen_formula_line:
                continue
            seen_formula_line = True
        new_lines.append(line)
    return "\n".join(new_lines).strip()


def _postprocess_consolidated_answer(user_text: str, text: str) -> str:
    out = _sanitize_math_for_ui(text or "")
    out = _enforce_dornic_canonical_formula(user_text, out)
    out = _dedupe_paragraphs(out)
    out = _strip_leading_uncertainty_prefix(out)
    stripped = _strip_uncertainty_tail(out)
    if stripped and len(stripped) > 80:
        out = stripped
    out = _sanitize_math_for_ui(out)
    return out
