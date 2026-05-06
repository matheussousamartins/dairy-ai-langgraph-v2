from app.tools.web_fallback import _is_allowed_domain


def test_empty_allowed_domains_means_open_web_search():
    assert _is_allowed_domain("https://example.com/artigo", []) is True


def test_configured_allowed_domains_still_filter_results():
    assert _is_allowed_domain("https://sub.gov.br/pagina", ["gov.br"]) is True
    assert _is_allowed_domain("https://example.com/artigo", ["gov.br"]) is False
