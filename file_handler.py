"""File content extraction for PDFs, HTML, images, and plain text."""

import base64
import logging
import re
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
TEXT_EXTENSIONS = {
    '.txt', '.md', '.py', '.js', '.ts', '.jsx', '.tsx', '.css', '.json',
    '.yaml', '.yml', '.toml', '.xml', '.csv', '.sh', '.bash', '.zsh',
    '.c', '.cpp', '.h', '.java', '.go', '.rs', '.rb', '.php', '.swift',
    '.kt', '.r', '.sql',
}

# Truncate text attachments above this threshold to protect the context window
MAX_CONTENT_CHARS = 80_000


def load_file(path: str) -> Dict:
    """Load a file from disk. Returns an attachment dict for stream_chat()."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return load_file_bytes(p.name, p.read_bytes())


def load_file_bytes(name: str, data: bytes) -> Dict:
    """Load a file from raw bytes (used by the web upload endpoint)."""
    ext = Path(name).suffix.lower()

    if ext == '.pdf':
        return _extract_pdf(name, data)
    elif ext in ('.html', '.htm'):
        return _extract_html(name, data.decode('utf-8', errors='replace'))
    elif ext in IMAGE_EXTENSIONS:
        return {
            "type": "image",
            "name": name,
            "content": base64.b64encode(data).decode('utf-8'),
            "warning": None,
        }
    else:
        # Text, code, or unknown — try to decode as UTF-8
        return _guard({
            "type": "text",
            "name": name,
            "content": data.decode('utf-8', errors='replace'),
            "warning": None,
        })


# ── Extractors ────────────────────────────────────────────────────────────────

def _extract_pdf(name: str, data: bytes) -> Dict:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("pymupdf is required for PDF support: uv add pymupdf")

    doc = fitz.open(stream=data, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()

    text = '\n\n'.join(pages).strip()

    if not text:
        return {
            "type": "text",
            "name": name,
            "content": "",
            "warning": (
                f"'{name}' appears to be a scanned PDF with no extractable text. "
                "Try attaching individual pages as images instead."
            ),
        }

    return _guard({"type": "text", "name": name, "content": text, "warning": None})


def _extract_html(name: str, html: str) -> Dict:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 is required for HTML support: uv add beautifulsoup4")

    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'head', 'meta', 'link', 'noscript']):
        tag.decompose()

    text = soup.get_text(separator='\n', strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return _guard({"type": "text", "name": name, "content": text, "warning": None})


# ── Context guard ─────────────────────────────────────────────────────────────

def _guard(att: Dict) -> Dict:
    content = att["content"]
    if len(content) > MAX_CONTENT_CHARS:
        att["content"] = content[:MAX_CONTENT_CHARS]
        att["warning"] = (
            f"'{att['name']}' was truncated to {MAX_CONTENT_CHARS:,} characters "
            f"(original: {len(content):,} chars). For large documents consider RAG."
        )
    return att
