"""
orch_quality.py — Guardrails de qualidade de resposta.

Contém:
  1. Detector de tipo de pergunta → instrução de formato para consolidação
  2. Validator programático de frases proibidas (safety net pós-LLM)
  3. Classificador de qualidade de resposta (HIGH/MEDIUM/LOW/UNUSABLE)

Separado do orch_text.py para manter funções puras de texto vs. lógica
de qualidade/formato que depende de heurísticas de domínio.
"""

import re
from typing import List, Optional, Tuple

from app.agents.orch_text import _normalize_text


# ============================================================
# 1. Detecção de tipo de pergunta → formato de resposta
# ============================================================

class QuestionType:
    FACTUAL_SHORT = "factual_short"       # "Qual o pH...", "Quem é..."
    COMPARATIVE = "comparative"            # "Diferença entre X e Y"
    PROCESS = "process"                    # "Como fabricar...", "Quais etapas..."
    TROUBLESHOOTING = "troubleshooting"    # "Por que meu queijo...", "Defeito X..."
    REGULATORY = "regulatory"              # "Qual norma...", "O que diz a IN..."
    CALCULATIVE = "calculative"            # "Calcule...", "Qual o valor de..."
    GENERAL = "general"                    # Default


# Padrões de detecção (aplicados sobre texto normalizado)
_QUESTION_PATTERNS: List[Tuple[str, str]] = [
    # Comparativa — tem mais prioridade que as demais
    (r"\b(diferen[cç]a|compara[cç]|versus|vs\.?|comparando|em rela[cç][aã]o a)\b", QuestionType.COMPARATIVE),
    (r"\b(x\s+vs?\s+y|entre\s+.+\s+e\s+)\b", QuestionType.COMPARATIVE),

    # Troubleshooting / diagnóstico
    (r"\b(defeito|problema|causa|diagnostico|por\s*que\s+.*(esta|apresent|ocorre))\b", QuestionType.TROUBLESHOOTING),
    (r"\b(estufamento|amargor|sinerese|trinca|ranc|mofo|contamin)\b", QuestionType.TROUBLESHOOTING),
    (r"o\s+que\s+pode\s+(ser|causar|estar)", QuestionType.TROUBLESHOOTING),

    # Processo / fabricação
    (r"\b(composi[cç][aã]o\s+e\s+processo|processo\s+b[aá]sico|composi[cç][aã]o\s+b[aá]sica)\b", QuestionType.PROCESS),
    (r"\b(como\s+fabr|etapas?\s+d[eao]|processo\s+de|fluxo\s+de|procedimento)\b", QuestionType.PROCESS),
    (r"\b(como\s+(?:produz|prepar|faz))\b", QuestionType.PROCESS),

    # Regulatório
    (r"\b(norma|regulament|legisla[cç]|decreto|artigo|riispoa|rtiq|rdc|in\s+\d|instrucao\s+normativa)\b", QuestionType.REGULATORY),
    (r"\b(exigid|obrigat|minimo\s+legal|permitid|proibid|limite\s+legal)\b", QuestionType.REGULATORY),

    # Cálculo
    (r"\b(calcul|formula|quanto\s+d[aáe]|qual\s+(?:o\s+)?valor)\b", QuestionType.CALCULATIVE),

    # Factual curta
    (r"^(qual\s+[eé]|quem\s+[eé]|quanto\s+[eé]|onde\s+(?:fica|[eé])|quando)\b", QuestionType.FACTUAL_SHORT),
    (r"^(qual\s+(?:risco|origem|fun[cç][aã]o|motivo|raz[aã]o|causa|papel|efeito|impacto))\b", QuestionType.FACTUAL_SHORT),
    (r"^(qual\s+(?:a|o)\s+(?:temperatura|ph|acidez|teor|concentra|faixa|limite|contagem|rendimento))\b", QuestionType.FACTUAL_SHORT),
]


def detect_question_type(user_text: str) -> str:
    """Detecta o tipo de pergunta para ajustar o formato da resposta.

    Retorna um dos valores de QuestionType.
    """
    normalized = _normalize_text(user_text)
    if not normalized:
        return QuestionType.GENERAL

    for pattern, qtype in _QUESTION_PATTERNS:
        if re.search(pattern, normalized):
            return qtype

    return QuestionType.GENERAL


# Instruções de formato por tipo — injetadas no consolidation prompt
_FORMAT_INSTRUCTIONS: dict[str, str] = {
    QuestionType.FACTUAL_SHORT: (
        "FORMATO: Responda de forma direta e concisa (1-3 frases). "
        "Inclua o valor numérico/dado solicitado logo no início. "
        "Não adicione contexto extra desnecessário."
    ),
    QuestionType.COMPARATIVE: (
        "FORMATO: Organize a comparação em lista ou tabela com eixos claros. "
        "Cada item comparado deve ter os mesmos critérios. "
        "Destaque as diferenças principais primeiro."
    ),
    QuestionType.PROCESS: (
        "FORMATO: Estruture em etapas numeradas com parâmetros críticos "
        "(temperatura, pH, tempo, concentração) em cada etapa. "
        "Seja técnico e sequencial."
    ),
    QuestionType.TROUBLESHOOTING: (
        "FORMATO: Use a estrutura DEFEITO → CAUSA PROVÁVEL → AÇÃO CORRETIVA. "
        "Ordene as causas por probabilidade (mais provável primeiro). "
        "Inclua análises recomendadas para confirmar a causa."
    ),
    QuestionType.REGULATORY: (
        "FORMATO: Cite norma, artigo e parágrafo quando disponíveis. "
        "Apresente limites em formato objetivo. "
        "Se houver atualização/revogação, mencione."
    ),
    QuestionType.CALCULATIVE: (
        "FORMATO: Apresente fórmula → valores substituídos → resultado → unidade. "
        "Use texto simples (sem LaTeX). "
        "Se houver variantes ou observações, liste após o resultado."
    ),
    QuestionType.GENERAL: "",
}


def get_format_instruction(question_type: str) -> str:
    """Retorna a instrução de formato para o tipo de pergunta."""
    return _FORMAT_INSTRUCTIONS.get(question_type, "")


# ============================================================
# 2. Validator de frases proibidas (safety net pós-LLM)
# ============================================================

# Frases que NUNCA devem aparecer na resposta final ao usuário.
# O prompt já instrui o LLM a evitá-las, mas LLMs eventualmente
# escapam — esta camada é o safety net programático.
_PROHIBITED_PREAMBLES: List[str] = [
    "a evidencia disponivel indica que",
    "com base nas informacoes disponiveis",
    "de acordo com a base de conhecimento",
    "segundo os trechos recuperados",
    "os dados encontrados sugerem que",
    "as informacoes disponiveis indicam que",
    "com base no que encontrei",
    "de acordo com o que foi recuperado",
    "com base nos trechos recuperados",
    "segundo as informacoes disponveis",
    "de acordo com os dados encontrados",
    "a partir dos trechos analisados",
    "conforme os dados recuperados",
    "com base na evidencia coletada",
]

_PROHIBITED_DISCLAIMERS: List[str] = [
    "no entanto, nao ha informacoes especificas sobre",
    "nao foi possivel fornecer detalhes sobre",
    "nao foram encontradas informacoes adicionais sobre",
    "nao ha dados especificos sobre",
    "nao tenho informacoes mais detalhadas sobre",
    "portanto, nao foi possivel detalhar",
    "a base nao trouxe informacoes sobre",
    "meu conhecimento atual nao contem",
    "nao ha informacoes suficientes na base sobre",
    "infelizmente nao encontrei dados sobre",
]

# Padrões de parágrafo-resumo final redundante (safety net para o consolidador e agentes).
# Só são removidos quando constituem o último parágrafo (lógica em strip_prohibited_phrases).
_PROHIBITED_SUMMARY_PARAGRAPHS: List[str] = [
    "resumo:",
    "resumindo:",
    "em resumo,",
    "em sintese,",
    "portanto, resumindo",
    "em sintese:",
    "resumo final:",
]

_PROHIBITED_INTERNAL_REFS: List[str] = [
    "base de conhecimento",
    "trechos recuperados",
    "agente 1",
    "agente 2",
    "agente 3",
    "agente 4",
    "agente 5",
    "agente 6",
    "agente 0",
    "meu conhecimento atual",
    "as informacoes que tenho hoje",
    "ferramenta de busca",
    "minha base de dados",
]

_PROHIBITED_FOLLOWUP_QUESTIONS: List[str] = [
    "voce gostaria de saber mais",
    "posso ajudar com algo mais",
    "ha algo especifico que deseja aprofundar",
    "gostaria que eu detalhasse",
    "deseja que eu explique",
    "quer saber mais sobre",
    "posso esclarecer algum ponto",
    "precisa de mais informacoes",
]


def strip_prohibited_phrases(text: str) -> str:
    """Remove frases proibidas da resposta final.

    Aplica remoção cirúrgica: preambles são removidos do início,
    disclaimers são removidos da cauda, referências internas são
    limpas em qualquer posição.

    Retorna o texto limpo sem alterar conteúdo técnico legítimo.
    """
    if not text:
        return text

    out = text.strip()
    normalized_full = _normalize_text(out)

    # 1. Remove preambles do início
    for phrase in _PROHIBITED_PREAMBLES:
        if normalized_full.startswith(phrase):
            # Encontra o ponto de corte no texto original
            phrase_len = len(phrase)
            # Procura o fim do preamble no texto normalizado e corta o original
            # na posição correspondente (preservando case e acentos)
            idx = _find_normalized_position(out, phrase)
            if idx >= 0:
                out = out[idx:].lstrip(" ,;:-–—")
                break

    # 2. Remove disclaimers e parágrafos-resumo redundantes da cauda (último parágrafo)
    paragraphs = out.rsplit("\n\n", 1)
    if len(paragraphs) == 2:
        last_para_norm = _normalize_text(paragraphs[1])
        removed = False
        for phrase in _PROHIBITED_DISCLAIMERS:
            if phrase in last_para_norm:
                out = paragraphs[0].rstrip()
                removed = True
                break
        if not removed:
            for phrase in _PROHIBITED_SUMMARY_PARAGRAPHS:
                if last_para_norm.startswith(phrase):
                    out = paragraphs[0].rstrip()
                    break

    # 3. Remove perguntas de followup do final
    lines = out.rstrip().rsplit("\n", 1)
    if len(lines) == 2:
        last_line_norm = _normalize_text(lines[1])
        for phrase in _PROHIBITED_FOLLOWUP_QUESTIONS:
            if phrase in last_line_norm:
                out = lines[0].rstrip()
                break
    elif lines:
        # Texto de uma linha só — verifica se termina com followup
        last_line_norm = _normalize_text(lines[0])
        for phrase in _PROHIBITED_FOLLOWUP_QUESTIONS:
            if last_line_norm.endswith(phrase + "?") or phrase in last_line_norm:
                # Tenta cortar só a frase final
                sentences = re.split(r"(?<=[.!])\s+", out)
                if len(sentences) > 1:
                    candidate = " ".join(sentences[:-1])
                    if len(candidate) > 20:
                        out = candidate
                break

    # 4. Remove referências internas em qualquer posição
    for phrase in _PROHIBITED_INTERNAL_REFS:
        # Substitui por espaço para evitar juntar palavras
        out = _remove_normalized_phrase(out, phrase)

    # Limpa espaços duplicados resultantes das remoções
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"^\s*[,;:]\s*", "", out)  # Pontuação órfã no início
    return out.strip()


def _find_normalized_position(original: str, normalized_phrase: str) -> int:
    """Encontra onde o preamble normalizado termina no texto original.

    Retorna o índice no texto original logo após o preamble.
    """
    norm = _normalize_text(original)
    idx = norm.find(normalized_phrase)
    if idx < 0:
        return -1

    # Mapeia posição normalizada → posição original
    # Itera caracteres do original consumindo a normalização
    end_norm = idx + len(normalized_phrase)
    orig_pos = 0
    norm_pos = 0
    text_lower = original.lower()

    for i, ch in enumerate(original):
        if norm_pos >= end_norm:
            return i
        # Skip combining characters e whitespace comprimido
        import unicodedata
        decomposed = unicodedata.normalize("NFKD", ch)
        for dch in decomposed:
            if unicodedata.combining(dch):
                continue
            norm_pos += 1

    return len(original)


def _remove_normalized_phrase(original: str, normalized_phrase: str) -> str:
    """Remove uma frase (normalizada) do texto original preservando formatação."""
    norm = _normalize_text(original)
    if normalized_phrase not in norm:
        return original

    # Approach simples: regex case-insensitive com acentos opcionais
    # Para frases curtas como "base de conhecimento", funciona bem
    escaped = re.escape(normalized_phrase)
    # Permite acentos opcionais entre caracteres
    flexible = re.sub(r"\\ ", r"\\s+", escaped)
    try:
        result = re.sub(flexible, " ", original, count=1, flags=re.IGNORECASE)
        return result
    except re.error:
        return original


# ============================================================
# 3. Classificador de qualidade de resposta
# ============================================================

class ResponseQuality:
    HIGH = "high"         # Factual com dados numéricos
    MEDIUM = "medium"     # Descritiva coerente sem números
    LOW = "low"           # Vaga/evasiva
    UNUSABLE = "unusable"  # Negativa de evidência


def classify_response_quality(text: str) -> str:
    """Classifica a qualidade da resposta para decisão de fallback.

    Retorna um dos valores de ResponseQuality.
    """
    if not text or not text.strip():
        return ResponseQuality.UNUSABLE

    normalized = _normalize_text(text)

    # Unusable: negativas explícitas como resposta principal
    unusable_markers = (
        "nao foi possivel consolidar",
        "nao foi possivel obter uma resposta",
        "nao foi possivel responder",
        "resposta confiavel no momento",
        "tente reformular sua pergunta",
        "nao encontrei informacoes",
        "nao foram encontradas informacoes",
        "nao foram encontrados dados",
        "nao ha dados especificos",
        "nao ha evidencias",
        "[fora_de_escopo]",
    )
    if any(marker in normalized for marker in unusable_markers):
        return ResponseQuality.UNUSABLE

    # HIGH: contém dados numéricos específicos (temperaturas, pHs, %, etc.)
    numeric_signals = len(re.findall(
        r"\d+(?:[.,]\d+)?\s*(?:°c|°d|%|ml|mg|g/l|ph|ufc|cel|ppm|kda|dalton|min|h\b|dias?|meses?|seg)",
        normalized,
    ))
    if numeric_signals >= 2:
        return ResponseQuality.HIGH

    # MEDIUM: resposta com conteúdo substancial
    words = normalized.split()
    if len(words) >= 15:
        # Verifica se há termos técnicos de laticínio
        dairy_terms = sum(1 for w in words if w in {
            "leite", "queijo", "mussarela", "iogurte", "fermentacao",
            "coagulacao", "maturacao", "pasteurizacao", "acidez",
            "temperatura", "proteina", "gordura", "caseina",
            "cultura", "fermento", "salga", "salmoura", "filagem",
        })
        if dairy_terms >= 2:
            return ResponseQuality.MEDIUM

    if len(words) >= 30:
        return ResponseQuality.MEDIUM

    # LOW: resposta curta ou vaga
    if len(words) >= 5:
        return ResponseQuality.LOW

    return ResponseQuality.UNUSABLE
