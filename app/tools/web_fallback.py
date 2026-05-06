"""Utilitários de fallback web com whitelist de domínios confiáveis."""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Tuple


_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def _normalize_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    d = d.lstrip(".")
    if d.startswith("www."):
        d = d[4:]
    return d


def _extract_domain(url: str) -> str:
    try:
        host = urllib.parse.urlparse(url).netloc or ""
    except Exception:
        return ""
    host = host.split("@")[-1].split(":")[0]
    return _normalize_domain(host)


def _is_allowed_domain(url: str, allowed_domains: List[str]) -> bool:
    domain = _extract_domain(url)
    if not domain:
        return False
    normalized = [_normalize_domain(d) for d in (allowed_domains or []) if d]
    if not normalized:
        return True
    for base in normalized:
        if domain == base or domain.endswith("." + base):
            return True
    return False


def _strip_html_to_text(raw_html: str, max_chars: int) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw_html or "")
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def _http_get(url: str, timeout_sec: float) -> str:
    req = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": _UA,
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # nosec B310
        charset = resp.headers.get_content_charset() or "utf-8"
        data = resp.read()
        return data.decode(charset, errors="ignore")


def search_web_duckduckgo(
    query: str,
    allowed_domains: List[str],
    max_results: int = 6,
    timeout_sec: float = 8,
    max_snippet_chars: int = 420,
) -> List[Dict[str, Any]]:
    """Busca web via DuckDuckGo HTML e aplica whitelist de domínios.

    Se allowed_domains vier vazio, a busca fica aberta. O orquestrador usa isso
    como segunda tentativa depois da whitelist configurada.

    Retorna resultados no formato:
      {title, url, domain, snippet}
    """
    q = (query or "").strip()
    if not q:
        return []

    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
    try:
        page = _http_get(url, timeout_sec=timeout_sec)
    except Exception:
        return []

    # Parsing leve baseado no HTML padrão do endpoint /html.
    # Captura pares (href, title, snippet) por bloco de resultado.
    pattern = re.compile(
        r'(?is)<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
        r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>'
    )
    matches = pattern.findall(page)

    results: List[Dict[str, Any]] = []
    seen_urls = set()
    for href, raw_title, raw_snippet in matches:
        href = html.unescape(href or "")
        # DDG usa redirect /l/?...&uddg=<encoded_url>
        parsed = urllib.parse.urlparse(href)
        if parsed.path.startswith("/l/"):
            qparams = urllib.parse.parse_qs(parsed.query)
            target = (qparams.get("uddg") or [""])[0]
            href = urllib.parse.unquote(target or "")

        if not href.startswith("http"):
            continue
        if href in seen_urls:
            continue
        if not _is_allowed_domain(href, allowed_domains):
            continue

        seen_urls.add(href)
        title = _strip_html_to_text(raw_title, max_chars=220)
        snippet = _strip_html_to_text(raw_snippet, max_chars=max_snippet_chars)
        domain = _extract_domain(href)
        results.append(
            {
                "title": title,
                "url": href,
                "domain": domain,
                "snippet": snippet,
            }
        )
        if len(results) >= max_results:
            break
    return results


def enrich_results_with_page_content(
    results: List[Dict[str, Any]],
    timeout_sec: float,
    max_page_chars: int,
) -> List[Dict[str, Any]]:
    """Opcional: enriquece resultados com trecho textual da página."""
    enriched: List[Dict[str, Any]] = []
    for item in results or []:
        url = str(item.get("url", "")).strip()
        page_text = ""
        if url:
            try:
                html_doc = _http_get(url, timeout_sec=timeout_sec)
                page_text = _strip_html_to_text(html_doc, max_chars=max_page_chars)
            except Exception:
                page_text = ""
        obj = dict(item)
        obj["page_text"] = page_text
        enriched.append(obj)
    return enriched


def build_web_fallback_evidence(results: List[Dict[str, Any]], max_sources: int = 3) -> Tuple[str, List[Dict[str, str]]]:
    """Renderiza evidência textual + lista de fontes para citação."""
    picked = (results or [])[: max(1, int(max_sources or 1))]
    if not picked:
        return "", []

    lines: List[str] = []
    sources: List[Dict[str, str]] = []
    for idx, item in enumerate(picked, start=1):
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        domain = str(item.get("domain", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        page_text = str(item.get("page_text", "")).strip()
        body = page_text or snippet
        if not url or not body:
            continue
        if len(body) > 600:
            body = body[:600].rstrip() + "..."
        lines.append(f"[Fonte {idx}] {title} ({domain}) - {body}")
        sources.append({"title": title, "domain": domain, "url": url})

    return "\n".join(lines).strip(), sources
