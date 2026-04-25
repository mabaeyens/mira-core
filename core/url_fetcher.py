"""Fetch a URL and return its text content, stripped of HTML."""

import logging
import re

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

FETCH_TIMEOUT = 15        # seconds
MAX_CONTENT_CHARS = 12_000  # truncate very large pages


def fetch_url(url: str) -> str:
    """
    Fetch a URL and return clean plain text.

    Returns the text content on success, or an error string starting with
    "Error:" that the model can report to the user.
    """
    try:
        headers = {
            "User-Agent": "ollama-search-tool/1.0"
        }
        response = httpx.get(url, timeout=FETCH_TIMEOUT, headers=headers, follow_redirects=True)
        response.raise_for_status()
    except httpx.TimeoutException:
        return f"Error: request timed out fetching {url}"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} fetching {url}"
    except Exception as e:
        return f"Error: {e}"

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return f"Error: unsupported content type '{content_type}' at {url}"

    if "text/plain" in content_type:
        text = response.text
    else:
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script, style, nav, footer noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Prefer <article> or <main> if present
        body = soup.find("article") or soup.find("main") or soup.body or soup
        text = body.get_text(separator="\n")

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    if len(text) > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS] + f"\n\n[… content truncated at {MAX_CONTENT_CHARS} chars]"

    logger.info("fetch_url: %s — %d chars returned", url, len(text))
    return text
