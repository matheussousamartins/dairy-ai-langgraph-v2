"""
orch_text.py — Funções puras de processamento de texto do orquestrador.

Contém apenas funções que dependem de stdlib (re, unicodedata) e nada de
LangChain, LangGraph, Supabase ou config. Pode ser importado por qualquer
módulo sem criar dependências circulares.
"""

import re
import unicodedata
from typing import List, Optional

OUT_OF_SCOPE_TAG = "[FORA_DE_ESCOPO]"
GENERAL_KNOWLEDGE_TAG = "[CONHECIMENTO GERAL]"


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

# Marcadores de AUSÊNCIA de informação — o agente declara explicitamente que não
# encontrou dados. Estes são os únicos sinais confiáveis de resposta inutilizável.
# NÃO incluir verbos modais ("pode", "talvez") — em Português técnico, "pode causar",
# "pode ser arriscado" são afirmações factuais, não hedges de incerteza.
_ABSENCE_MARKERS: frozenset = frozenset({
    # Declarações explícitas de ausência via 1ª pessoa (agente falando de si)
    "nao encontrei informacao suficiente",
    "nao encontrei informacoes suficientes",
    "nao encontrei informacoes especificas",
    "nao tenho informacao suficiente",
    "nao tenho informacoes suficientes",
    "nao disponho de informacao",
    "com o meu conhecimento atual",
    # Declarações de ausência via 3ª pessoa / impessoal
    "nao ha informacao suficiente",
    "nao ha evidencia suficiente",
    "nao ha evidencias suficientes",
    "nao ha dados suficientes",
    "nao ha informacoes disponiveis",
    "nao ha dados disponiveis",
    "nao ha dados especificos",
    "nao ha informacoes especificas",
    "nao ha informacao especifica",
    "nao foram encontrados dados",
    "nao foram encontradas informacoes",
    "nao foram encontrados dados especificos",
    "nao foram encontradas informacoes especificas",
    "sem informacao suficiente",
    "informacao insuficiente",
    "evidencia insuficiente",
    "faltam evidencias",
    "nao foi possivel identificar",
    "nao foi possivel encontrar",
    "base de conhecimento nao contem",
    "nao consta na base",
    # Falsa ausência quando o LLM ignora o número nas evidências
    "nao ha especificacao numerica",
    "nao ha valor especifico",
    "nao ha um valor exato",
    "nao ha um numero especifico",
    "nao apresenta valor especifico",
    "nao especifica um valor",
    "nao especifica numericamente",
    "sem especificacao numerica",
})

# Marcadores de DEFERIMENTO — o agente pede que o usuário consulte outra fonte.
# São sinais válidos de resposta fraca, mas separados dos marcadores de ausência
# para facilitar ajuste fino independente.
_DEFERRAL_MARKERS: frozenset = frozenset({
    "recomenda-se verificar",
    "recomenda-se consultar",
    "aconselhavel verificar",
    "consultar fontes adicionais",
    "consultar estudos ou publicacoes",
    "consulte um especialista",
    "recomendo consultar",
    "para mais informacoes consulte",
    "para informacoes mais detalhadas",
})

# União para uso em _looks_uncertain
_ALL_UNCERTAINTY_MARKERS: frozenset = _ABSENCE_MARKERS | _DEFERRAL_MARKERS

# Prefixos hedging que o LLM às vezes usa no início da resposta antes do conteúdo real.
# Removidos via _strip_leading_uncertainty_prefix (defesa em profundidade além do prompt).
_HEDGE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:"
    r"Com base no meu conhecimento atual"
    r"|Com o meu conhecimento atual"
    r"|Com base nas informacoes disponiveis"
    r"|Com base no que encontrei"
    r"|Com base nas evidencias disponiveis"
    r"|De acordo com a base de conhecimento"
    r"|De acordo com o que foi recuperado"
    r"|Segundo os trechos recuperados"
    r"|A evidencia disponivel indica que"
    r"|As informacoes disponiveis indicam que"
    r"|Os dados encontrados sugerem que"
    r")[,:]?\s*",
    flags=re.IGNORECASE,
)

# Conjunções concessivas que introduzem cauda de ressalva — cortamos SOMENTE se
# o trecho após a conjunção contiver um marcador de ausência. Conjunções legítimas
# em comparações técnicas ("No entanto, em climas tropicais...") são preservadas.
_CONCESSIVE_CONJUNCTION_PATTERN = re.compile(
    r"\b(No entanto|Por[ée]m|Contudo|Entretanto)\b",
    flags=re.IGNORECASE,
)

# Frases de cauda explícita — sempre seguro cortar aqui.
_EXPLICIT_UNCERTAINTY_TAIL_PATTERN = re.compile(
    r"\b("
    r"a base atual n[ãa]o trouxe"
    r"|nao trouxe informacao suficiente"
    r"|faltam evidencias"
    r"|com o meu conhecimento atual"
    r"|recomenda-se verificar"
    r"|recomenda-se consultar"
    r"|aconselh[aá]vel verificar"
    r"|para informa[cç][oõ]es mais detalhadas"
    r"|para mais informa[cç][oõ]es"
    r"|nao ha informacoes suficientes sobre"
    r"|nao ha dados suficientes sobre"
    r")\b",
    flags=re.IGNORECASE,
)


def _has_control_tag(text: str, tag: str) -> bool:
    return tag in (text or "")


def _is_out_of_scope_response(text: str) -> bool:
    return _has_control_tag(text, OUT_OF_SCOPE_TAG)


def _is_general_knowledge_response(text: str) -> bool:
    return _has_control_tag(text, GENERAL_KNOWLEDGE_TAG)


def _strip_control_tags(text: str) -> str:
    out = str(text or "")
    out = out.replace(OUT_OF_SCOPE_TAG, "")
    out = out.replace(GENERAL_KNOWLEDGE_TAG, "")
    return out.strip()


def _looks_uncertain(text: str) -> bool:
    """Retorna True SOMENTE quando o texto declara explicitamente ausência de
    informação ou faz deferimento. Verbos modais ('pode', 'talvez') em afirmações
    técnicas factuais NÃO são considerados incerteza."""
    if _is_out_of_scope_response(text):
        return True
    t = _normalize_text(_strip_control_tags(text))
    if not t:
        return True
    return any(marker in t for marker in _ALL_UNCERTAINTY_MARKERS)


def _strip_uncertainty_tail(text: str) -> str:
    """Remove cauda de ressalva quando o fato principal já foi respondido.

    Conjunções concessivas ('No entanto', 'Porém', etc.) são cortadas SOMENTE
    se o trecho seguinte contiver marcador de ausência — preservando comparações
    técnicas legítimas ('No entanto, em climas tropicais a variação é maior').
    """
    if not text:
        return ""
    out = str(text).strip()

    # Corte condicional em conjunção concessiva: só corta se o tail for incerto.
    m = _CONCESSIVE_CONJUNCTION_PATTERN.search(out)
    if m:
        tail_normalized = _normalize_text(out[m.start():])
        if any(marker in tail_normalized for marker in _ABSENCE_MARKERS):
            out = out[: m.start()].strip()

    # Corte incondicional em frases de cauda explícita — sempre indica ausência.
    m2 = _EXPLICIT_UNCERTAINTY_TAIL_PATTERN.search(out)
    if m2:
        out = out[: m2.start()].strip()

    out = re.sub(r"[;,:\-–—]+$", "", out).strip()
    return out


def _strip_leading_uncertainty_prefix(text: str) -> str:
    """Remove prefixos hedging do início da resposta (defesa em profundidade
    além das regras do prompt — cobre casos em que o LLM ainda os produz)."""
    if not text:
        return ""
    return _HEDGE_PREFIX_PATTERN.sub("", str(text).strip(), count=1).strip()


def _extract_factual_candidate(text: str) -> Optional[str]:
    """Extrai parte factual útil de uma resposta mista (fato + ressalva)."""
    if _is_out_of_scope_response(text):
        return None
    cleaned = _sanitize_math_for_ui(_strip_control_tags(text or ""))
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
    out = re.sub(r"(?<=\w)_(?=[\s,.;:)]|$)", "", out)
    out = re.sub(r"(?<!\w)_(?=\w)", "", out)
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


def _strip_document_section_headers(text: str) -> str:
    """Remove títulos de seção numérica herdados dos chunks do documento.

    Dois formatos tratados:
    A) Título em linha própria:  "13. Durabilidade real...\n"  → linha removida
    B) Título inline em negrito: "**3. Uso de gelo seco** Conteúdo..." → só o
       trecho bold é removido, o conteúdo que segue é preservado

    Linhas de lista legítimas ("1. Diluir todo o conteúdo...") são preservadas
    porque contêm verbos de ação no infinitivo/imperativo.
    """
    if not text:
        return text

    _ACTION_VERBS = re.compile(
        r"\b(?:diluir|dividir|usar|manter|fechar|pesar|adicionar|misturar|"
        r"aquecer|resfriar|pasteurizar|coagular|dessorar|prensar|salgar|"
        r"embalar|armazenar|verificar|controlar|medir|calcular|ajustar|"
        r"inocular|agitar|filtrar|centrifugar|padronizar|homogeneizar|"
        r"esterilizar|higienizar|lavar|secar|etiquetar|"
        r"realize|execute|certifique|garanta|evite|assegure)\b",
        re.IGNORECASE,
    )

    def _is_action_title(title_text: str) -> bool:
        return bool(_ACTION_VERBS.search(title_text))

    def _title_word_count(title_text: str) -> int:
        return len(title_text.split())

    # Formato A: título ocupa a linha inteira (com ou sem negrito)
    # Captura: opcional "**", número de seção, separador, texto do título, opcional "**", fim de linha
    _STANDALONE = re.compile(
        r"^(\*\*)?(\d+(?:\.\d+)*)[\.\s]\s*([A-ZÁÉÍÓÚÀÃÕÂÊÔÜÇ][^\n]{5,120}?)(\*\*)?\s*$",
        re.MULTILINE,
    )

    def _replace_standalone(m: re.Match) -> str:
        title_text = m.group(3)
        if _is_action_title(title_text) or _title_word_count(title_text) < 3:
            return m.group(0)
        return ""

    result = _STANDALONE.sub(_replace_standalone, text)

    # Formato B: título em negrito inline, seguido de conteúdo na mesma linha
    # Ex: "**3. Uso de gelo seco para acidificação parcial do leite** Nos Estados..."
    # Remove apenas a parte bold, preserva o conteúdo que vem depois
    _INLINE_BOLD = re.compile(
        r"\*\*(\d+(?:\.\d+)*)[\.\s]\s*([A-ZÁÉÍÓÚÀÃÕÂÊÔÜÇ][^*\n]{5,120}?)\*\*\s*",
    )

    def _replace_inline_bold(m: re.Match) -> str:
        title_text = m.group(2)
        if _is_action_title(title_text) or _title_word_count(title_text) < 3:
            return m.group(0)
        return ""

    result = _INLINE_BOLD.sub(_replace_inline_bold, result)

    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _strip_internal_artifacts(text: str) -> str:
    """Remove metadados internos que nunca devem aparecer ao usuário final."""
    out = str(text or "")
    out = re.sub(r"\[(?:embeddings_agente|base_geral_unificada)[^\]]*\]\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\bembeddings_agente_\d+_[a-z0-9_]+\b:?\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\bsource_table\s*[:=]\s*[a-z0-9_]+\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(?:Trecho|chunk)\s+\d+\s*[—-]\s*score\s*:?\s*[0-9.]+\s*[—-]\s*[^\n]+\n?", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def _postprocess_consolidated_answer(user_text: str, text: str) -> str:
    out = _sanitize_math_for_ui(_strip_internal_artifacts(_strip_control_tags(text or "")))
    out = _strip_document_section_headers(out)
    out = _enforce_dornic_canonical_formula(user_text, out)
    out = _dedupe_paragraphs(out)
    out = _strip_leading_uncertainty_prefix(out)
    stripped = _strip_uncertainty_tail(out)
    if stripped and len(stripped) > 80:
        out = stripped
    out = _sanitize_math_for_ui(_strip_internal_artifacts(_strip_control_tags(out)))

    # Safety net: validator programático de frases proibidas.
    # Importado aqui (lazy) para evitar dependência circular no módulo.
    try:
        from app.agents.orch_quality import strip_prohibited_phrases
        out = strip_prohibited_phrases(out)
    except ImportError:
        pass

    return out
