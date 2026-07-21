"""Backend port identity verification (step 6).

A process answering /v1/models must be proven to serve the expected model before
Mira adopts it and routes prompts to it. A squatter that 200s with a different
(or no) model must be refused, not adopted.
"""
import json
import io
import pytest

from core import backend_manager as bm


# --- model-id matching -----------------------------------------------------

def test_matches_full_path_against_itself():
    assert bm._model_matches(
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        "mlx-community/Qwen3.6-35B-A3B-4bit")


def test_matches_short_name_against_full_path():
    # A backend that returns only the basename still matches the configured path.
    assert bm._model_matches("Qwen3.6-35B-A3B-4bit",
                             "mlx-community/Qwen3.6-35B-A3B-4bit")


def test_match_is_case_insensitive():
    assert bm._model_matches("qwen3.6-35b-a3b-4bit",
                             "mlx-community/Qwen3.6-35B-A3B-4bit")


def test_different_model_does_not_match():
    assert not bm._model_matches("evil-model",
                                 "mlx-community/Qwen3.6-35B-A3B-4bit")


def test_empty_served_never_matches():
    assert not bm._model_matches("", "mlx-community/Qwen3.6-35B-A3B-4bit")
    assert not bm._model_matches(None, "mlx-community/Qwen3.6-35B-A3B-4bit")


# --- verify_or_adopt behaviour --------------------------------------------

class _FakeResp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def _patch_served(monkeypatch, value):
    """Make _served_model return a fixed value regardless of URL."""
    monkeypatch.setattr(bm, "_served_model", lambda url, omlx=False: value)


def test_adopt_when_nobody_home(monkeypatch):
    # None => nothing listening => caller should start the backend.
    _patch_served(monkeypatch, None)
    assert bm._verify_or_adopt("http://x/v1/models", "the-model") is False


def test_adopt_when_model_matches(monkeypatch):
    _patch_served(monkeypatch, "mlx-community/the-model")
    assert bm._verify_or_adopt("http://x/v1/models", "the-model") is True


def test_refuse_when_model_differs(monkeypatch):
    _patch_served(monkeypatch, "evil-model")
    with pytest.raises(bm.BackendIdentityError):
        bm._verify_or_adopt("http://x/v1/models", "the-model")


def test_refuse_when_served_empty_but_listening(monkeypatch):
    # Answered but no model id in the body — a bare squatter. Must refuse.
    _patch_served(monkeypatch, "")
    with pytest.raises(bm.BackendIdentityError):
        bm._verify_or_adopt("http://x/v1/models", "the-model")


def test_no_expected_model_adopts_any_listener(monkeypatch):
    # When we have nothing to compare against, a live listener is adopted.
    _patch_served(monkeypatch, "whatever")
    assert bm._verify_or_adopt("http://x/v1/models", None) is True


def test_escape_hatch_adopts_mismatch(monkeypatch):
    _patch_served(monkeypatch, "evil-model")
    monkeypatch.setattr(bm, "_ADOPT_UNVERIFIED", True)
    assert bm._verify_or_adopt("http://x/v1/models", "the-model") is True


# --- _served_model parsing -------------------------------------------------

def test_served_model_parses_openai_shape(monkeypatch):
    body = json.dumps({"object": "list",
                       "data": [{"id": "mlx-community/the-model"}]}).encode()
    monkeypatch.setattr(bm.urllib.request, "urlopen",
                        lambda url, timeout=2: _FakeResp(body))
    assert bm._served_model("http://x/v1/models") == "mlx-community/the-model"


def test_served_model_none_on_error(monkeypatch):
    def boom(url, timeout=2):
        raise OSError("connection refused")
    monkeypatch.setattr(bm.urllib.request, "urlopen", boom)
    assert bm._served_model("http://x/v1/models") is None


def test_served_model_empty_on_no_data(monkeypatch):
    body = json.dumps({"object": "list", "data": []}).encode()
    monkeypatch.setattr(bm.urllib.request, "urlopen",
                        lambda url, timeout=2: _FakeResp(body))
    assert bm._served_model("http://x/v1/models") == ""
