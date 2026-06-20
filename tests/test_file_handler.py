"""Tests for PDF extraction + scanned-PDF OCR fallback (core/file_handler)."""
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
