"""File content extraction for PDFs, HTML, images, and plain text."""

import base64
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# OCR fallback for scanned PDFs (optional — needs the `tesseract` binary on PATH).
_OCR_DPI = 200
_OCR_MAX_PAGES = 50            # cap OCR work on huge scans
_OCR_PER_PAGE_TIMEOUT = 30     # seconds per page

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
HEIC_EXTENSIONS = {'.heic', '.heif'}

# Maximum longest dimension for images sent to the model.
# Mobile camera photos (12–48 MP) are resized to stay under this limit.
_IMAGE_MAX_PX = 1024
_IMAGE_JPEG_QUALITY = 85
TEXT_EXTENSIONS = {
    '.txt', '.md', '.py', '.js', '.ts', '.jsx', '.tsx', '.css', '.json',
    '.yaml', '.yml', '.toml', '.xml', '.csv', '.sh', '.bash', '.zsh',
    '.c', '.cpp', '.h', '.java', '.go', '.rs', '.rb', '.php', '.swift',
    '.kt', '.r', '.sql',
}

# Magic-byte signatures for binary types we can auto-detect.
# Checked when the file extension is unrecognised.
_MAGIC_PDF   = b'%PDF'
_MAGIC_PNG   = b'\x89PNG\r\n\x1a\n'
_MAGIC_JPEG  = b'\xff\xd8'
_MAGIC_GIF87 = b'GIF87a'
_MAGIC_GIF89 = b'GIF89a'
_MAGIC_BMP   = b'BM'


def _sniff(data: bytes):
    """Return 'pdf', 'image', or None based on magic bytes."""
    if data[:4] == _MAGIC_PDF:
        return 'pdf'
    if data[:8] == _MAGIC_PNG:
        return 'image'
    if data[:2] == _MAGIC_JPEG:
        return 'image'
    if data[:6] in (_MAGIC_GIF87, _MAGIC_GIF89):
        return 'image'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image'
    if data[:2] == _MAGIC_BMP:
        return 'image'
    return None

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
    elif ext in HEIC_EXTENSIONS:
        return {
            "type": "text",
            "name": name,
            "content": "",
            "warning": (
                f"'{name}' is in HEIC/HEIF format, which is not supported. "
                "Please share the image as JPEG or PNG instead."
            ),
        }
    elif ext in IMAGE_EXTENSIONS:
        return _make_image(name, data)
    else:
        # Unknown extension — check magic bytes before falling back to text.
        # This handles files like "document.bump" that are really PDFs.
        detected = _sniff(data)
        if detected == 'pdf':
            result = _extract_pdf(name, data)
            display_ext = ext if ext else '(no extension)'
            result['warning'] = (
                f"'{name}' has a '{display_ext}' extension but is a PDF — "
                f"processed as PDF. {result['warning'] or ''}"
            ).strip()
            return result
        if detected == 'image':
            result = _make_image(name, data)
            display_ext = ext if ext else '(no extension)'
            result['warning'] = (
                f"'{name}' has a '{display_ext}' extension but is an image — "
                "processed as image."
            )
            return result
        # Genuine text / code / unknown binary — decode as UTF-8
        decoded = data.decode('utf-8', errors='replace')
        # Binary heuristic: if >5% of characters are UTF-8 replacement chars the
        # file is almost certainly binary (e.g. .qvf, .zip, .bin).  Return a
        # warning instead of indexing binary garbage into RAG.
        if decoded and decoded.count('\ufffd') / len(decoded) > 0.05:
            return {
                "type": "text",
                "name": name,
                "content": "",
                "warning": (
                    f"'{name}' appears to be a binary file — cannot extract text. "
                    "Attach a PDF, image, or plain-text file instead."
                ),
            }
        return _guard({
            "type": "text",
            "name": name,
            "content": decoded,
            "warning": None,
        })


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_image(name: str, data: bytes) -> Dict:
    warning = None
    try:
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if max(w, h) > _IMAGE_MAX_PX:
            scale = _IMAGE_MAX_PX / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            fmt = "JPEG" if img.mode in ("RGB", "L") else "PNG"
            if fmt == "JPEG" and img.mode != "RGB":
                img = img.convert("RGB")
            img.save(buf, format=fmt, quality=_IMAGE_JPEG_QUALITY)
            data = buf.getvalue()
            logger.debug("Resized image '%s' from %dx%d to %dx%d", name, w, h, img.width, img.height)
    except Exception as e:
        logger.warning("Could not resize image '%s': %s", name, e)
    return {
        "type": "image",
        "name": name,
        "content": base64.b64encode(data).decode('utf-8'),
        "warning": warning,
    }


# ── Extractors ────────────────────────────────────────────────────────────────

def _tesseract_available() -> bool:
    """True when the optional system `tesseract` binary is on PATH."""
    return shutil.which("tesseract") is not None


def _run_tesseract_on_bytes(image_bytes: bytes, suffix: str, timeout: float) -> str:
    """OCR raw image bytes via the system tesseract binary. Returns text or ''.

    Writes to a tempfile, then runs `tesseract <path> stdout` (explicit args list,
    never shell=True — per CLAUDE.md). Raises subprocess.TimeoutExpired on timeout.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["tesseract", tmp_path, "stdout", "-l", "eng"],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout if result.returncode == 0 else ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _ocr_page(page) -> str:
    """OCR a single fitz page via the system tesseract binary. Returns text or ''."""
    pix = page.get_pixmap(dpi=_OCR_DPI)
    return _run_tesseract_on_bytes(pix.tobytes("png"), suffix=".png", timeout=_OCR_PER_PAGE_TIMEOUT)


def ocr_image_from_base64(b64_content: str) -> Optional[str]:
    """OCR a standalone image attachment (screenshot of an error/menu, not a PDF page)
    via the system tesseract binary. Returns extracted text, or None if tesseract isn't
    installed or no readable text was found — callers should treat None as "OCR didn't
    help" and fall back to their own no-vision handling.
    """
    if not _tesseract_available():
        return None
    data = base64.b64decode(b64_content)
    suffix = ".png" if data[:8] == _MAGIC_PNG else ".jpg"
    try:
        text = _run_tesseract_on_bytes(data, suffix=suffix, timeout=_OCR_PER_PAGE_TIMEOUT).strip()
    except subprocess.TimeoutExpired:
        logger.warning("OCR timed out on attached image")
        return None
    except Exception as e:
        logger.warning("OCR failed on attached image: %s", e)
        return None
    return text or None


def _extract_pdf(name: str, data: bytes) -> Dict:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("pymupdf is required for PDF support: uv add pymupdf")

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        page_texts = [page.get_text() for page in doc]

        # OCR fallback: pages with no extractable text layer (scanned PDFs). Only triggers
        # on empty pages and only when tesseract is installed; text pages are left untouched
        # so the output for a normal PDF is identical to before. Mixed PDFs OCR only the
        # empty pages, merged in order.
        empty_idx = [i for i, t in enumerate(page_texts) if not t.strip()]
        ocr_warning = None
        if empty_idx and _tesseract_available():
            for i in empty_idx[:_OCR_MAX_PAGES]:
                try:
                    page_texts[i] = _ocr_page(doc[i])
                except subprocess.TimeoutExpired:
                    logger.warning("OCR timed out on page %d of '%s'; skipping", i + 1, name)
                    page_texts[i] = ""
                except Exception as e:
                    logger.warning("OCR failed on page %d of '%s': %s", i + 1, name, e)
                    page_texts[i] = ""
            if len(empty_idx) > _OCR_MAX_PAGES:
                ocr_warning = (
                    f"'{name}' has {len(empty_idx)} scanned pages; OCR was limited to the "
                    f"first {_OCR_MAX_PAGES}."
                )

        text = '\n\n'.join(page_texts).strip()
    finally:
        doc.close()

    if not text:
        hint = (
            "Install tesseract (brew install tesseract) to enable OCR of scanned PDFs."
            if not _tesseract_available()
            else "OCR produced no text — the scan may be too low quality."
        )
        return {
            "type": "text",
            "name": name,
            "content": "",
            "warning": f"'{name}' appears to be a scanned PDF with no extractable text. {hint}",
        }

    # PDFs always go through RAG regardless of size (consistent behaviour, better accuracy)
    return {"type": "rag", "name": name, "content": text, "warning": ocr_warning}


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
        # Upgrade to RAG instead of truncating — no token-cost concern with local models
        att["type"] = "rag"
    return att
