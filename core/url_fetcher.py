"""Fetch a URL and return clean Markdown text."""

import logging
import re

import httpx
import trafilatura
from markdownify import markdownify

logger = logging.getLogger(__name__)

FETCH_TIMEOUT = 15
MAX_CONTENT_CHARS = 24_000

_CHROME_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_html(url: str) -> tuple[str | None, str | None]:
    """
    Return (html, error). Tries httpx first; falls back to curl_cffi with
    Chrome TLS fingerprinting when httpx gets a 4xx or suspiciously short body.
    """
    try:
        r = httpx.get(url, timeout=FETCH_TIMEOUT, headers=_CHROME_HEADERS, follow_redirects=True)
        r.raise_for_status()
        html = r.text
        content_type = r.headers.get("content-type", "")
        if any(t in content_type for t in ("text/plain", "text/xml", "application/xml", "application/json")):
            return html, None
        if "text/html" not in content_type:
            return None, f"Error: unsupported content type '{content_type}' at {url}"
        if len(html) > 500:
            return html, None
        # Short body — bot wall likely; fall through to curl_cffi
    except httpx.TimeoutException:
        return None, f"Error: request timed out fetching {url}"
    except httpx.HTTPStatusError as e:
        if e.response.status_code not in (403, 429):
            return None, f"Error: HTTP {e.response.status_code} fetching {url}"
        # Bot detection — fall through to curl_cffi
    except Exception as e:
        return None, f"Error: {e}"

    try:
        from curl_cffi import requests as cffi_requests
        r = cffi_requests.get(
            url,
            timeout=FETCH_TIMEOUT,
            headers=_CHROME_HEADERS,
            impersonate="chrome120",
            allow_redirects=True,
        )
        r.raise_for_status()
        return r.text, None
    except Exception as e:
        return None, f"Error: {e}"


def _extract(html: str) -> str:
    """Extract main content as Markdown. Falls back to markdownify on failure."""
    text = trafilatura.extract(
        html,
        include_formatting=True,
        include_links=True,
        include_tables=True,
        no_fallback=False,
    )
    if text and len(text.strip()) >= 200:
        return text.strip()

    # Fallback: structural HTML → Markdown (preserves headers, code, lists)
    md = markdownify(html, heading_style="ATX", strip=["script", "style", "nav", "footer", "header", "aside", "form"])
    # Collapse excessive blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _fetch_jina(url: str) -> str | None:
    """Last-resort fetch via Jina Reader — handles JS-rendered pages without a local browser."""
    try:
        r = httpx.get(
            f"https://r.jina.ai/{url}",
            timeout=20,
            headers={"Accept": "text/plain", "X-No-Cache": "true"},
            follow_redirects=True,
        )
        r.raise_for_status()
        text = r.text.strip()
        return text if len(text) >= 200 else None
    except Exception:
        return None


def fetch_url(url: str) -> str:
    """
    Fetch a URL and return clean Markdown text.

    Returns text on success, or an error string starting with "Error:".
    """
    if not url.startswith(("http://", "https://")):
        return (
            "I can only fetch http(s) URLs. "
            "To use a local file, attach it to the conversation instead."
        )

    html, error = _fetch_html(url)
    if error:
        return error
    if not html:
        return f"Error: no content returned from {url}"

    # Plain text, XML, JSON — return as-is without HTML extraction
    stripped = html.lstrip()
    if not stripped.startswith("<") or stripped.startswith("<?xml"):
        text = html.strip()
        raw_html_len = 0  # skip JS heuristic for non-HTML
    else:
        text = _extract(html)
        raw_html_len = len(html)

    # Jina fallback when: too little text, OR suspiciously low extraction ratio
    # (large HTML + tiny extracted text = JS-rendered page with client-side data loading)
    is_js_likely = raw_html_len > 50_000 and len(text) < 2_000
    if len(text) < 200 or is_js_likely:
        jina_text = _fetch_jina(url)
        if jina_text:
            text = jina_text
            logger.info("fetch_url (jina fallback): %s — %d chars", url, len(text))
        elif len(text) < 200:
            return (
                f"Error: page returned too little readable text ({len(text)} chars) — "
                f"likely JavaScript-rendered or login-gated. Try a different URL or use web_search."
            )
        # If is_js_likely but Jina failed, fall through with whatever we extracted

    if len(text) > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS] + f"\n\n[… content truncated at {MAX_CONTENT_CHARS} chars]"

    logger.info("fetch_url: %s — %d chars returned", url, len(text))
    return text
