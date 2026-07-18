"""Tests for PDF extraction + scanned-PDF OCR fallback (core/file_handler)."""
import base64

import pytest

from core import file_handler

fitz = pytest.importorskip("fitz")  # PyMuPDF — required for PDF support


def _text_pdf() -> bytes:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Hello OCR world")
    data = doc.tobytes()
    doc.close()
    return data


def _blank_pdf() -> bytes:
    """A page with no text layer — stands in for a scanned page."""
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def test_text_pdf_extracts_without_ocr(monkeypatch):
    # OCR must never run on a PDF that already has a text layer.
    called = False
    def _boom(_page):
        nonlocal called
        called = True
        return "SHOULD NOT BE CALLED"
    monkeypatch.setattr(file_handler, "_ocr_page", _boom)
    monkeypatch.setattr(file_handler, "_tesseract_available", lambda: True)

    result = file_handler._extract_pdf("doc.pdf", _text_pdf())
    assert result["type"] == "rag"
    assert "Hello OCR world" in result["content"]
    assert result["warning"] is None
    assert called is False


def test_scanned_pdf_without_tesseract_warns(monkeypatch):
    monkeypatch.setattr(file_handler, "_tesseract_available", lambda: False)
    result = file_handler._extract_pdf("scan.pdf", _blank_pdf())
    assert result["type"] == "text"
    assert result["content"] == ""
    assert "scanned" in result["warning"].lower()
    assert "tesseract" in result["warning"].lower()


def test_scanned_pdf_with_ocr_recovers_text(monkeypatch):
    # Simulate tesseract present and returning text for the empty page.
    monkeypatch.setattr(file_handler, "_tesseract_available", lambda: True)
    monkeypatch.setattr(file_handler, "_ocr_page", lambda _page: "RECOVERED TEXT")
    result = file_handler._extract_pdf("scan.pdf", _blank_pdf())
    assert result["type"] == "rag"
    assert result["content"] == "RECOVERED TEXT"


# -- ocr_image_from_base64 -----------------------------------------------------

def _b64(data: bytes = b"\x89PNG\r\n\x1a\nfakepngbytes") -> str:
    return base64.b64encode(data).decode("utf-8")


def test_ocr_image_returns_none_without_tesseract(monkeypatch):
    monkeypatch.setattr(file_handler, "_tesseract_available", lambda: False)
    assert file_handler.ocr_image_from_base64(_b64()) is None


def test_ocr_image_returns_extracted_text(monkeypatch):
    monkeypatch.setattr(file_handler, "_tesseract_available", lambda: True)
    monkeypatch.setattr(
        file_handler, "_run_tesseract_on_bytes",
        lambda data, suffix, timeout: "Error: disk full\n",
    )
    assert file_handler.ocr_image_from_base64(_b64()) == "Error: disk full"


def test_ocr_image_returns_none_when_no_text_found(monkeypatch):
    monkeypatch.setattr(file_handler, "_tesseract_available", lambda: True)
    monkeypatch.setattr(
        file_handler, "_run_tesseract_on_bytes", lambda data, suffix, timeout: "   "
    )
    assert file_handler.ocr_image_from_base64(_b64()) is None


def test_ocr_image_returns_none_on_timeout(monkeypatch):
    import subprocess

    def _boom(data, suffix, timeout):
        raise subprocess.TimeoutExpired(cmd="tesseract", timeout=timeout)

    monkeypatch.setattr(file_handler, "_tesseract_available", lambda: True)
    monkeypatch.setattr(file_handler, "_run_tesseract_on_bytes", _boom)
    assert file_handler.ocr_image_from_base64(_b64()) is None
