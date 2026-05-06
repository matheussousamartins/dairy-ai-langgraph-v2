"""Evidence selection and compression for RAG answers.

This module is intentionally deterministic. The LLM may write the final prose,
but this layer decides which recovered snippets are allowed to support it.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.agents.orch_quality import QuestionType, detect_question_type
from app.agents.orch_text import (
    _normalize_text,
    _sanitize_math_for_ui,
    _strip_profile_suffix,
)


_STOPWORDS = {
    "qual", "quais", "quanto", "quantos", "deve", "devem", "ser",
    "considerado", "considerada", "para", "com", "sobre", "apos", "cerca",
    "logo", "mais", "menos", "adequado", "adequada", "desenvolvimento",
    "pergunta", "atual", "contexto", "recente", "usuario",
}

_NUMERIC_UNIT_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:"
    r"l\s*/\s*kg|litros?|kg|%|ph|"
    r"\u00b0\s*c|\u00ba\s*c|\?\s*c|c\b|"
    r"dias?|meses?|anos?|horas?|h\b|min(?:utos?)?|"
    r"ufc\s*/?\s*ml|cs\s*/?\s*ml|celulas?\s*(?:/|por)\s*ml"
    r")",
    re.IGNORECASE,
)

_TEMPERATURE_UNIT_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:\u00b0\s*c|\u00ba\s*c|\?\s*c|c\b)",
    re.IGNORECASE,
)

_LABEL_BOUNDARY_RE = re.compile(
    r"\s+(?=(?:Processo|Rendimento|Regulamento citado|Maturacao|Maturação|"
    r"Temperatura|Sal|Leite|Umidade|Cura):)",
    re.IGNORECASE,
)

_REQUIRED_ANCHOR_GROUPS = (
    (("vacuo", "vácuo"), ("vacuo", "vácuo")),
    (("ejetor", "ejetor de vapor"), ("ejetor", "ejetor de vapor")),
)


@dataclass
class ReducedEvidence:
    text: str
    snippets: List[Dict[str, Any]]
    direct_answer: Optional[str] = None


def _meaningful_terms(text: str) -> set[str]:
    normalized = _normalize_text(_strip_profile_suffix(text))
    return {
        token
        for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        if len(token) >= 4 and token not in _STOPWORDS
    }


def _expects_numeric_answer(user_text: str) -> bool:
    norm = _normalize_text(user_text)
    if any(marker in norm for marker in ("processo", "composicao", "etapas", "fabricacao", "como fabricar", "como produzir")):
        return False
    qtype = detect_question_type(user_text)
    if qtype in {QuestionType.FACTUAL_SHORT, QuestionType.CALCULATIVE, QuestionType.REGULATORY}:
        return True
    return any(
        term in norm
        for term in (
            "rendimento", "temperatura", "faixa", "ph", "umidade", "gordura",
            "proteina", "sal", "limite", "contagem", "ccs", "cbt", "tempo",
            "maturacao", "maturar", "cura",
        )
    ) and any(starter in norm for starter in ("qual", "quanto", "quais", "quando"))


def _metric_terms(user_text: str) -> set[str]:
    norm = _normalize_text(user_text)
    terms: set[str] = set()
    if "rendimento" in norm:
        terms.update({"rendimento", "litro", "litros", "l/kg", "leite por kg"})
    if any(t in norm for t in ("temperatura", "faixa", "maturar", "maturacao", "cura")):
        terms.update({"temperatura", "faixa", "cura", "maturacao", "maturar"})
    if "ph" in norm:
        terms.add("ph")
    if "sal" in norm:
        terms.add("sal")
    if "umidade" in norm:
        terms.add("umidade")
    if "ccs" in norm or "celulas somaticas" in norm or "contagem de celulas" in norm:
        terms.update({"ccs", "celulas somaticas", "celulas", "cs/ml", "por ml"})
    if "cbt" in norm or "contagem bacteriana" in norm:
        terms.update({"cbt", "contagem bacteriana"})
    return terms


def _missing_required_anchor(user_norm: str, sent_norm: str) -> bool:
    for triggers, anchors in _REQUIRED_ANCHOR_GROUPS:
        if any(trigger in user_norm for trigger in triggers):
            return not any(anchor in sent_norm for anchor in anchors)
    return False


def _clean_sentence(text: str) -> str:
    out = str(text or "").strip()
    out = re.sub(r"^Trecho\s+\d+\s*[—-]\s*score\s+[0-9.]+\s*[—-]\s*[^\n]+\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"^\*\*[^*]{3,140}\*\*\s*", "", out)
    out = re.sub(r"^[A-Z0-9][^*]{3,160}\*\*\s*", "", out)
    out = re.sub(r"^\s*[-*]\s*", "", out)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"^(?:Rendimento|Temperatura|Maturacao|Maturação|Cura):\s*", "", out, flags=re.IGNORECASE)
    return out.strip(" ;")


def _split_evidence_sentences(text: str) -> List[str]:
    cleaned = str(text or "")
    cleaned = re.sub(
        r"Trecho\s+\d+\s*[—-]\s*score\s+[0-9.]+\s*[—-]\s*[^\n]+\n",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\|[-:\s|]+\|", " ", cleaned)
    cleaned = _LABEL_BOUNDARY_RE.sub("\n", cleaned)
    cleaned = re.sub(r"\n+", " ", cleaned)

    raw_parts = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences: List[str] = []
    for part in raw_parts:
        part = _clean_sentence(part)
        if not part or len(part) < 18:
            continue
        if part.lower().startswith("trecho "):
            continue
        if part not in sentences:
            sentences.append(part)
    return sentences


def _score_sentence(user_text: str, sentence: str, *, top_score: float) -> int:
    user_norm = _normalize_text(user_text)
    sent_norm = _normalize_text(sentence)
    query_terms = _meaningful_terms(user_text)
    sent_terms = _meaningful_terms(sentence)
    overlap = query_terms & sent_terms

    score = len(overlap) * 2

    metrics = _metric_terms(user_text)
    if metrics and any(metric in sent_norm for metric in metrics):
        score += 4

    if _expects_numeric_answer(user_text) and _NUMERIC_UNIT_RE.search(sentence):
        score += 5

    if "parmesao" in user_norm and "parmesao" in sent_norm:
        score += 3

    # CCS / contagem de células queries: boost sentences with cell counts (por mL / /mL)
    if any(t in user_norm for t in ("ccs", "celulas somaticas", "contagem de celulas")):
        if re.search(r"\d[\d.,]*\s*(?:celulas?|cs|ccs)\s*(?:/|por)\s*ml", sent_norm, re.IGNORECASE):
            score += 5
        elif re.search(r"\d[\d.,]*\s*(?:celulas?|cs|ccs)", sent_norm, re.IGNORECASE):
            score += 3

    for marker in ("salga", "12 meses", "1 ano", "um ano", "cura", "maturacao"):
        if marker in user_norm and marker in sent_norm:
            score += 2

    if top_score >= 0.075:
        score += 2
    elif top_score >= 0.05:
        score += 1

    unrelated_products = {
        "provolone", "minas padrao", "minas meia cura", "ricota", "cream cheese",
    }
    if "parmesao" in user_norm and any(p in sent_norm for p in unrelated_products):
        score -= 6

    # Penaliza sentenças que descrevem valores como inadequados/incorretos.
    # Ex: "entre 5 e 12 °C. Essa faixa é inadequada" — é um contraexemplo,
    # não a resposta. Passa pelo scorer porque tem °C e parmesao, mas não
    # deve liderar a resposta nem receber bônus numérico.
    _NEGATIVE_QUALIFIERS = (
        "inadequada", "inadequado", "impropria", "improprio",
        "nao recomendada", "nao recomendado", "incorreta", "incorreto",
        "nao deve", "nao deveria", "evitar",
    )
    if _NUMERIC_UNIT_RE.search(sentence) and any(q in sent_norm for q in _NEGATIVE_QUALIFIERS):
        score -= 8

    if _missing_required_anchor(user_norm, sent_norm):
        score -= 12

    return score


def reduce_evidence_for_question(
    user_text: str,
    evidence_text: str,
    *,
    agent_id: int,
    top_score: float = 0.0,
    max_sentences: Optional[int] = None,
) -> ReducedEvidence:
    """Select only the snippets that directly support the user's question."""
    sentences = _split_evidence_sentences(evidence_text)
    if not sentences:
        return ReducedEvidence(text="", snippets=[])

    factual = _expects_numeric_answer(user_text)
    limit = max_sentences or (3 if factual else 5)
    threshold = 5 if factual else 4
    if top_score >= 0.075:
        threshold -= 1

    _NEGATIVE_CONTEXT = (
        "inadequada", "inadequado", "impropria", "improprio",
        "nao recomendada", "nao recomendado", "nao deve", "nao deveria",
    )

    scored: List[Dict[str, Any]] = []
    for idx, sentence in enumerate(sentences):
        score = _score_sentence(user_text, sentence, top_score=top_score)
        # Se a sentença seguinte classifica esta como contraexemplo, penaliza.
        # Ex: "...entre 5 e 12 °C." seguida de "Essa faixa é inadequada..."
        if idx + 1 < len(sentences):
            next_norm = _normalize_text(sentences[idx + 1])
            if any(q in next_norm for q in _NEGATIVE_CONTEXT):
                score -= 8
        if score >= threshold:
            scored.append({
                "text": sentence,
                "score": score,
                "agent_id": agent_id,
                "rag_top_score": top_score,
            })

    if not scored:
        return ReducedEvidence(text="", snippets=[])

    scored.sort(key=lambda item: item["score"], reverse=True)
    selected: List[Dict[str, Any]] = []
    seen_norm: set[str] = set()
    for item in scored:
        norm = _normalize_text(item["text"])
        if any(norm in old or old in norm for old in seen_norm):
            continue
        seen_norm.add(norm)
        selected.append(item)
        if len(selected) >= limit:
            break

    # Preserve the original evidence order for readability after scoring.
    selected.sort(key=lambda item: sentences.index(item["text"]))

    text = " ".join(item["text"] for item in selected).strip()
    direct = _build_direct_answer_from_snippets(user_text, selected) if factual else None
    return ReducedEvidence(text=text, snippets=selected, direct_answer=direct)


def _build_direct_answer_from_snippets(
    user_text: str,
    snippets: List[Dict[str, Any]],
) -> Optional[str]:
    if not snippets:
        return None

    texts = [_sanitize_math_for_ui(str(item.get("text", "")).strip()) for item in snippets]
    texts = [_focus_direct_text(user_text, text) for text in texts]
    texts = [text for text in texts if text and _NUMERIC_UNIT_RE.search(text)]
    if not texts:
        return None

    norm = _normalize_text(user_text)
    if "rendimento" in norm:
        selected = [
            text for text in texts
            if any(marker in _normalize_text(text) for marker in ("rendimento", "litro", "l/kg", "leite por kg", "salga", "12 meses", "1 ano"))
        ]
        texts = selected or texts
    elif any(t in norm for t in ("temperatura", "faixa", "maturar", "maturacao", "cura")):
        selected = [
            text for text in texts
            if _TEMPERATURE_UNIT_RE.search(text)
            and any(marker in _normalize_text(text) for marker in ("temperatura", "16", "18", "cura", "maturacao"))
        ]
        texts = selected or texts

    answer_parts: List[str] = []
    max_parts = 1 if any(t in norm for t in ("temperatura", "faixa", "maturar", "maturacao", "cura")) else 2
    for text in texts:
        if text not in answer_parts:
            answer_parts.append(text)
        if len(answer_parts) >= max_parts:
            break

    answer = " ".join(answer_parts).strip()
    if len(answer) > 650:
        return None
    return answer


def _focus_direct_text(user_text: str, text: str) -> str:
    """Keep only the sub-sentence that answers a short factual question."""
    norm = _normalize_text(user_text)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if not parts:
        return text

    if any(t in norm for t in ("temperatura", "faixa", "maturar", "maturacao", "cura")):
        match = re.search(
            r"[^.!?]*temperatura[^.!?]*\d+(?:[.,]\d+)?\s*(?:\u00b0\s*c|\u00ba\s*c|\?\s*c|c\b)[^.!?]*[.!?]?",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(0).strip()
        for part in parts:
            part_norm = _normalize_text(part)
            if _TEMPERATURE_UNIT_RE.search(part) and any(t in part_norm for t in ("temperatura", "recomendada", "cura", "maturacao")):
                return part

    if "rendimento" in norm:
        selected = [
            part for part in parts
            if any(t in _normalize_text(part) for t in ("rendimento", "litro", "l/kg", "leite por kg", "salga", "12 meses", "1 ano"))
            and _NUMERIC_UNIT_RE.search(part)
        ]
        if selected:
            return " ".join(selected[:2])

    return text


def build_direct_answer_from_candidates(
    user_text: str,
    candidates: List[Dict[str, Any]],
) -> Optional[str]:
    """Return a deterministic answer for short factual/numeric questions."""
    if not _expects_numeric_answer(user_text):
        return None

    ordered = sorted(
        candidates,
        key=lambda item: (
            0 if int(item.get("agent_id", -1)) not in {0, 3} else 1,
            -float(item.get("rag_top_score") or 0.0),
        ),
    )
    for item in ordered:
        direct = str(item.get("direct_answer_candidate") or "").strip()
        if direct:
            return _sanitize_math_for_ui(direct)

    return None
