from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "tests" / "fixtures" / "rag" / "rag_queries.yaml"
OUT_DIR = ROOT / "docs" / "orchestrator" / "day1"
OUT_FILE = OUT_DIR / "ROUTING_SPECIALIST_HINTS.yaml"
DOCS_ROOT = ROOT / "docs"

SPECIALIST_IDS = {1, 2, 4, 5, 6}

# Tokens muito gerais para roteamento de especialista.
LOW_PRECISION_TERMS = {
    "leite",
    "qualidade",
    "queijo",
    "acidez",
    "ph",
    "cultura",
    "fermentado",
    "defeito",
    "problema",
    "sabor",
    "textura",
    "ingrediente",
    "receita",
    "validade",
    "analise",
    "norma",
    "produto",
    "lacteo",
    "laticinio",
    "limite",
    "partir",
    "tempo",
    "maximo",
    "maxima",
    "minimo",
    "minima",
    "total",
    "normal",
    "ideal",
    "permitido",
    "permitida",
    "permitidos",
    "permitidas",
    "regra",
    "exigido",
    "exigida",
    "depois",
    "antes",
    "deve",
    "podem",
    "pode",
    "qual",
    "quais",
    "quanto",
}

HINT_NOISE_TERMS = {
    "edicao", "edição", "dados", "serie", "série", "grafico", "gráfico",
    "imagem", "linha", "eixo", "bibliografia", "nacional", "oficiais",
    "metodos", "métodos", "comum",
}

STOPWORDS_PT = {
    "a", "o", "e", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "um", "uma", "uns", "umas", "para", "por", "com", "sem", "sob", "sobre",
    "ao", "aos", "as", "os", "ou", "que", "como", "qual", "quais", "quando",
    "quanto", "quanta", "quantas", "quantos", "se", "eu", "voce", "voces", "meu",
    "minha", "meus", "minhas", "seu", "sua", "seus", "suas", "isso", "essa",
    "esse", "essas", "esses", "esta", "este", "estas", "estes", "pode", "posso",
    "devo", "deve", "dever", "tem", "tenho", "ha", "havia", "ser", "sao", "é",
    "mais", "menos", "muito", "muita", "muitos", "muitas", "ja", "ainda", "entre",
    "ate", "apenas", "tambem", "também", "tipo", "dentro", "fora", "apos", "após",
    "pra", "pro", "pela", "pelas", "pelo", "pelos", "onde",
}


@dataclass
class TermStats:
    term: str
    support: int
    global_support: int
    groups: int
    precision: float
    coverage: float
    score: float


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> list[str]:
    # aceita termos alfanuméricos e removemos tokens muito curtos.
    return [
        t for t in re.findall(r"[a-z0-9]+", _normalize_text(text))
        if len(t) >= 3 and not t.isdigit() and t not in STOPWORDS_PT
    ]


def _regulatory_patterns(text_norm: str) -> set[str]:
    patterns = set()
    for m in re.finditer(r"\b(?:in|rdc|rtiq)\s*\d{1,4}\b", text_norm):
        patterns.add(re.sub(r"\s+", " ", m.group(0)).strip())
    if "riispoa" in text_norm:
        patterns.add("riispoa")
    return patterns


def _query_terms(pergunta: str) -> set[str]:
    text_norm = _normalize_text(pergunta)
    tokens = _tokenize(pergunta)
    terms: set[str] = set(_regulatory_patterns(text_norm))

    # unigrama: termos com pelo menos 4 chars e não genéricos.
    for tok in tokens:
        if len(tok) >= 4 and tok not in LOW_PRECISION_TERMS:
            terms.add(tok)

    # n-gramas 2..3 mais específicos.
    for n in (2, 3):
        if len(tokens) < n:
            continue
        for i in range(0, len(tokens) - n + 1):
            ng_tokens = tokens[i : i + n]
            if any(t in STOPWORDS_PT for t in ng_tokens):
                continue
            if all(t in LOW_PRECISION_TERMS for t in ng_tokens):
                continue
            phrase = " ".join(ng_tokens)
            # evita frases muito curtas/ruído
            if len(phrase) >= 8:
                terms.add(phrase)

    return terms


def _collect_query_texts(query: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    pergunta = str(query.get("pergunta", "") or "").strip()
    if pergunta:
        parts.append(pergunta)

    expected = query.get("expected")
    if isinstance(expected, str) and expected.strip():
        parts.append(expected.strip())
    elif isinstance(expected, list):
        parts.extend(str(x).strip() for x in expected if str(x).strip())

    expected_all = query.get("expected_all")
    if isinstance(expected_all, list):
        parts.extend(str(x).strip() for x in expected_all if str(x).strip())

    # Resposta ajuda a capturar termos técnicos canônicos por doc.
    answer = str(query.get("answer", "") or "").strip()
    if answer:
        parts.append(answer)
    return parts


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queries = data.get("queries") or []
    return [q for q in queries if isinstance(q, dict)]


def _iter_agent_markdown_docs(agent_id: int) -> list[Path]:
    out: list[Path] = []
    prefix = f"agente-{agent_id}-"
    if not DOCS_ROOT.exists():
        return out
    for d in DOCS_ROOT.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith(prefix):
            continue
        md_dir = d / "md"
        if not md_dir.exists():
            continue
        out.extend(sorted(md_dir.rglob("*.md")))
    return out


def _score_term(
    support: int,
    global_support: int,
    group_support: int,
    agent_queries: int,
) -> float:
    precision = support / max(1, global_support)
    coverage = support / max(1, agent_queries)
    idf = math.log((1 + agent_queries) / (1 + support)) + 1.0
    group_boost = min(group_support, 5) / 5
    return precision * (support ** 0.6) * (1 + group_boost) * (0.6 + coverage) * idf


def build_hints() -> dict[str, Any]:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    rows = _load_dataset(DATASET_FILE)

    agent_queries_count: Counter[int] = Counter()
    agent_groups: dict[int, set[str]] = defaultdict(set)
    term_global_queries: Counter[str] = Counter()
    term_agent_queries: dict[str, Counter[int]] = defaultdict(Counter)
    term_agent_groups: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    agent_docs_count: Counter[int] = Counter()
    term_global_docs: Counter[str] = Counter()
    term_agent_docs: dict[str, Counter[int]] = defaultdict(Counter)
    term_agent_doc_groups: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    for q in rows:
        try:
            aid = int(q.get("agent_id", -1))
        except (TypeError, ValueError):
            continue
        if aid not in SPECIALIST_IDS:
            continue

        group = str(q.get("group", "unknown") or "unknown")
        raw_texts = _collect_query_texts(q)
        if not raw_texts:
            continue
        terms: set[str] = set()
        for blob in raw_texts:
            terms.update(_query_terms(blob))
        if not terms:
            continue

        agent_queries_count[aid] += 1
        agent_groups[aid].add(group)
        for term in terms:
            term_global_queries[term] += 1
            term_agent_queries[term][aid] += 1
            term_agent_groups[term][aid].add(group)

    # Complementa com corpus de docs por agente para cobrir agentes/docs
    # não plenamente refletidos no rag_queries local.
    for aid in sorted(SPECIALIST_IDS):
        docs = _iter_agent_markdown_docs(aid)
        for doc_path in docs:
            terms: set[str] = set()
            # Título do doc costuma trazer sinais de domínio de alta precisão.
            title_blob = doc_path.stem.replace("_", " ").replace("-", " ")
            terms.update(_query_terms(title_blob))
            # Primeira heading markdown também tende a ser sinal útil.
            try:
                text = doc_path.read_text(encoding="utf-8", errors="ignore")
                for line in text.splitlines():
                    raw = line.strip()
                    if raw.startswith("#"):
                        heading = raw.lstrip("#").strip()
                        if heading:
                            terms.update(_query_terms(heading))
                            break
            except Exception:
                pass
            if not terms:
                continue
            agent_docs_count[aid] += 1
            doc_group = f"doc:{doc_path.stem}"
            for term in terms:
                term_global_docs[term] += 1
                term_agent_docs[term][aid] += 1
                term_agent_doc_groups[term][aid].add(doc_group)

    specialists: dict[int, dict[str, Any]] = {}
    for aid in sorted(SPECIALIST_IDS):
        query_total = int(agent_queries_count.get(aid, 0))
        docs_total = int(agent_docs_count.get(aid, 0))
        # Perguntas reais têm peso maior que corpus cru de docs.
        agent_total = (query_total * 2) + docs_total
        if agent_total <= 0:
            specialists[aid] = {
                "strong_hints_normalized": [],
                "medium_hints_normalized": [],
                "diagnostics": [],
            }
            continue

        strong: list[TermStats] = []
        medium: list[TermStats] = []

        total_groups = len(agent_groups.get(aid, set())) + docs_total
        min_group_spread = 2 if total_groups >= 2 else 1

        all_terms = set(term_agent_queries.keys()) | set(term_agent_docs.keys())
        for term in all_terms:
            query_support = int(term_agent_queries.get(term, Counter()).get(aid, 0))
            doc_support = int(term_agent_docs.get(term, Counter()).get(aid, 0))
            support = (query_support * 2) + doc_support
            if support <= 0:
                continue

            global_support = (int(term_global_queries.get(term, 0)) * 2) + int(term_global_docs.get(term, 0))
            groups = len(term_agent_groups[term][aid] | term_agent_doc_groups[term][aid])
            precision = support / max(1, global_support)
            coverage = support / max(1, agent_total)
            score = _score_term(
                support=support,
                global_support=global_support,
                group_support=groups,
                agent_queries=agent_total,
            )

            stats = TermStats(
                term=term,
                support=support,
                global_support=global_support,
                groups=groups,
                precision=precision,
                coverage=coverage,
                score=score,
            )

            if term in HINT_NOISE_TERMS:
                continue
            if any(ch.isdigit() for ch in term):
                continue

            # Critérios de alta confiança: aparece em múltiplas perguntas e grupos,
            # e é específico do agente vs demais.
            if query_total > 0:
                if support >= 4 and groups >= min_group_spread and precision >= 0.80:
                    strong.append(stats)
                elif support >= 2 and precision >= 0.65:
                    medium.append(stats)
            else:
                # Sem dados de queries para o agente: usa sinais ultra-específicos
                # do corpus documental para bootstrap.
                looks_specific = (" " in term) or (len(term) >= 8)
                if looks_specific and support >= 1 and precision >= 0.95:
                    strong.append(stats)
                elif looks_specific and support >= 1 and precision >= 0.80:
                    medium.append(stats)

        strong.sort(key=lambda s: (s.score, s.precision, s.support), reverse=True)
        medium.sort(key=lambda s: (s.score, s.precision, s.support), reverse=True)

        # evita duplicar medium que já está no strong
        strong_terms = {s.term for s in strong}
        medium = [m for m in medium if m.term not in strong_terms]

        # Limites para não inflar prompt/roteamento.
        strong = strong[:45]
        medium = medium[:60]

        diagnostics = [
            {
                "term": s.term,
                "support": s.support,
                "global_support": s.global_support,
                "groups": s.groups,
                "precision": round(s.precision, 4),
                "coverage": round(s.coverage, 4),
                "score": round(s.score, 4),
            }
            for s in (strong[:25] + medium[:25])
        ]

        specialists[aid] = {
            "strong_hints_normalized": [s.term for s in strong],
            "medium_hints_normalized": [m.term for m in medium],
            "diagnostics": diagnostics,
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_dataset": str(DATASET_FILE.relative_to(ROOT)).replace("\\", "/"),
        "methodology": {
            "notes": [
                "Hints derivados das perguntas reais do rag_queries.",
                "Quando necessário, o corpus markdown por agente complementa cobertura por documento.",
                "Score combina precisão por agente, cobertura e dispersão por grupos/docs.",
                "Termos genéricos foram excluídos para aumentar assertividade.",
            ],
            "strong_criteria": {
                "min_support": ">=4 no score combinado (query x2 + docs x1)",
                "min_group_spread": "2 (ou 1 quando agente tem apenas 1 grupo no dataset)",
                "min_precision": 0.80,
            },
            "medium_criteria": {
                "min_support": 2,
                "min_precision": 0.65,
            },
        },
        "specialists": {str(k): v for k, v in specialists.items()},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_hints()
    OUT_FILE.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=140),
        encoding="utf-8",
    )
    print(f"[ok] Wrote: {OUT_FILE}")


if __name__ == "__main__":
    main()
