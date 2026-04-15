"""Limpa markdown para ingestao RAG com foco em qualidade de texto.

Uso:
  python scripts/clean_agent_md_quality.py --agent-dir docs/agente-4-qualidade-leite/md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DATA_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:image\/[^\)]*\)", re.DOTALL | re.IGNORECASE)
DATA_IMAGE_ANY_RE = re.compile(
    r"data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+",
    re.DOTALL | re.IGNORECASE,
)
DATA_IMAGE_TOKEN_RE = re.compile(r"data:image\/[^\s\)]+", re.IGNORECASE)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
BLANKS_RE = re.compile(r"\n{3,}")


def looks_mojibake(text: str) -> bool:
    markers = ("Ã", "Â", "â€“", "â€”", "â€™", "â€œ", "â€", "�")
    hits = sum(text.count(m) for m in markers)
    return hits >= 3


def fix_mojibake(text: str) -> str:
    """Tenta reverter UTF-8 decodado como latin1/cp1252."""
    if not looks_mojibake(text):
        return text

    for enc in ("latin1", "cp1252"):
        try:
            candidate = text.encode(enc, errors="ignore").decode("utf-8", errors="ignore")
            if candidate.count("Ã") < text.count("Ã"):
                return candidate
        except Exception:
            continue
    return replace_common_mojibake(text)


def replace_common_mojibake(text: str) -> str:
    mapping = {
        "Ã§": "ç",
        "Ã£": "ã",
        "Ã¡": "á",
        "Ã ": "à",
        "Ã¢": "â",
        "Ãª": "ê",
        "Ã©": "é",
        "Ã­": "í",
        "Ã³": "ó",
        "Ã´": "ô",
        "Ãº": "ú",
        "Ã¼": "ü",
        "Ã‡": "Ç",
        "Ãƒ": "Ã",
        "Ã": "Á",
        "Ã€": "À",
        "Ã‚": "Â",
        "ÃŠ": "Ê",
        "Ã‰": "É",
        "Ã“": "Ó",
        "Ã”": "Ô",
        "Ãš": "Ú",
        "Ã–": "Ö",
        "Âº": "º",
        "Âª": "ª",
        "Â°": "°",
        "Â": "",
        "â€“": "-",
        "â€”": "-",
        "â€˜": "'",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€¦": "...",
    }
    out = text
    for bad, good in mapping.items():
        out = out.replace(bad, good)
    return out


def clean_markdown(text: str) -> tuple[str, dict[str, int]]:
    stats: dict[str, int] = {
        "base64_images_removed": 0,
        "comments_removed": 0,
        "blank_lines_compacted": 0,
        "mojibake_fixed": 0,
    }

    original = text
    fixed = fix_mojibake(text)
    if fixed != text:
        stats["mojibake_fixed"] = 1
    text = fixed

    text, n_img1 = DATA_IMAGE_RE.subn("", text)
    text, n_img2 = DATA_IMAGE_ANY_RE.subn("", text)
    text, n_img3 = DATA_IMAGE_TOKEN_RE.subn("", text)
    stats["base64_images_removed"] = n_img1 + n_img2 + n_img3

    text, n_comments = HTML_COMMENT_RE.subn("", text)
    stats["comments_removed"] = n_comments

    # Normalizacoes leves para RAG
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ")

    compacted = BLANKS_RE.sub("\n\n", text)
    if compacted != text:
        stats["blank_lines_compacted"] = 1
    text = compacted.strip() + "\n"

    if text == original:
        # evita gravacao desnecessaria
        return original, stats
    return text, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", required=True, help="Pasta com .md do agente")
    args = parser.parse_args()

    base = Path(args.agent_dir)
    files = sorted([p for p in base.glob("*.md") if p.is_file()])
    if not files:
        print(f"Nenhum .md encontrado em: {base}")
        return

    changed = 0
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned, stats = clean_markdown(raw)
        if cleaned != raw:
            path.write_text(cleaned, encoding="utf-8")
            changed += 1
        print(
            f"{path.name}: changed={cleaned != raw} "
            f"mojibake_fixed={stats['mojibake_fixed']} "
            f"base64_images_removed={stats['base64_images_removed']} "
            f"comments_removed={stats['comments_removed']}"
        )

    print(f"\nArquivos processados: {len(files)} | alterados: {changed}")


if __name__ == "__main__":
    main()
