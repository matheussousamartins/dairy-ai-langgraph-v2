"""
parsers.py — Conversão de PDF/DOCX/MD/TXT para Markdown limpo.

Responsabilidade única: dado bytes de um arquivo e seu nome,
retorna (markdown_text, page_count). Sem side-effects, sem I/O de rede.

Dependências opcionais (instaladas separadamente):
  - pymupdf4llm: para PDFs digitais
  - markitdown:  para DOCX
"""

import logging
import os
import re
import tempfile
from typing import Tuple

_log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset = frozenset({".pdf", ".docx", ".md", ".txt"})

_LEGISLACAO_FILENAME_RE = re.compile(
    r"(^|[_\s\-])in[\s_\-]?\d+|instrucao.normativa|\b(rdc|portaria|decreto|resolucao|normativa)\b"
)
_LEGISLACAO_CONTENT_RE = re.compile(
    r"art\.\s*\d+|instrução normativa|instrucao normativa"
    r"|portaria\s+n[°o]?\.?\s*\d+|§\s*\d+",
    re.IGNORECASE,
)
_GLOSSARIO_CONTENT_RE = re.compile(r"\btermo\s*:", re.IGNORECASE)
_GLOSSARIO_VALUE_RE = re.compile(
    r"(substitui[cç][aã]o|sin[oô]nimo|defini[cç][aã]o)\s*:", re.IGNORECASE
)
_FAQ_CONTENT_RE = re.compile(r"(pergunta|resposta)\s*:|\bfaq\b", re.IGNORECASE)


def detect_doc_type(filename: str, text_preview: str) -> str:
    """Infers doc_type from filename and first ~3000 chars of converted Markdown.

    Returns one of: legislacao | glossario | faq | formulacao |
                    tabela_nutricional | manual (fallback).

    Priority: filename signals > content signals > 'manual'.
    """
    name = (filename or "").lower()

    # --- Filename signals (high confidence, usually unambiguous) ---
    if _LEGISLACAO_FILENAME_RE.search(name):
        return "legislacao"
    if "glossar" in name:
        return "glossario"
    if "faq" in name or "perguntas_frequentes" in name:
        return "faq"
    if re.search(r"formula[cç]", name):
        return "formulacao"
    if re.search(r"tabela.nutri|composicao.nutri", name):
        return "tabela_nutricional"

    # --- Content signals (first ~3000 chars) ---
    preview = (text_preview or "").strip()
    if _LEGISLACAO_CONTENT_RE.search(preview):
        return "legislacao"
    if _GLOSSARIO_CONTENT_RE.search(preview) and _GLOSSARIO_VALUE_RE.search(preview):
        return "glossario"
    if _FAQ_CONTENT_RE.search(preview):
        return "faq"

    return "manual"


def convert_to_markdown(file_bytes: bytes, filename: str) -> Tuple[str, int]:
    """Converte PDF/DOCX/MD/TXT para Markdown.

    Retorna (texto_markdown, numero_de_paginas).
    Lança ValueError para formatos não suportados.
    Lança RuntimeError se a conversão falhar.
    """
    ext = _get_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Formato não suportado: '{ext}'. "
            f"Formatos aceitos: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        return _convert_pdf(file_bytes)
    if ext == ".docx":
        return _convert_docx(file_bytes)
    # .md e .txt: decodifica direto
    return _decode_text(file_bytes), 1


# ---------------------------------------------------------------------------
# Internos
# ---------------------------------------------------------------------------

def _get_extension(filename: str) -> str:
    parts = (filename or "").lower().rsplit(".", 1)
    return f".{parts[-1]}" if len(parts) == 2 else ""


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _with_tempfile(suffix: str, data: bytes):
    """Context manager: cria arquivo temporário, yield path, remove ao sair."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(data)
            tmp.flush()
            tmp.close()
            yield tmp.name
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return _ctx()


def _convert_pdf(file_bytes: bytes) -> Tuple[str, int]:
    """Converte PDF digital para Markdown usando pymupdf4llm.

    pymupdf4llm preserva estrutura de seções, tabelas e listas melhor que
    extratores simples. Para PDFs escaneados (sem camada de texto), retorna
    texto vazio — adicionar OCR em fase futura se necessário.
    """
    try:
        import pymupdf4llm
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "pymupdf4llm não instalado. Execute: pip install pymupdf4llm"
        ) from exc

    with _with_tempfile(".pdf", file_bytes) as tmp_path:
        try:
            with pymupdf.open(tmp_path) as doc:
                page_count = len(doc)
            md_text = pymupdf4llm.to_markdown(tmp_path)
        except Exception as exc:
            raise RuntimeError(f"Falha na conversão do PDF: {exc}") from exc

    if not md_text or not md_text.strip():
        raise RuntimeError(
            "O PDF não contém texto extraível. "
            "Pode ser um PDF escaneado — converta para texto antes de ingerir."
        )

    return md_text, page_count


def _convert_docx(file_bytes: bytes) -> Tuple[str, int]:
    """Converte DOCX para Markdown usando markitdown (Microsoft).

    markitdown preserva cabeçalhos, tabelas e listas do Word.
    DOCX não tem conceito de página — retorna page_count=1.
    """
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError(
            "markitdown não instalado. Execute: pip install markitdown"
        ) from exc

    with _with_tempfile(".docx", file_bytes) as tmp_path:
        try:
            result = MarkItDown().convert(tmp_path)
            md_text = result.text_content or ""
        except Exception as exc:
            raise RuntimeError(f"Falha na conversão do DOCX: {exc}") from exc

    if not md_text.strip():
        raise RuntimeError("O DOCX resultou em texto vazio após conversão.")

    return md_text, 1
