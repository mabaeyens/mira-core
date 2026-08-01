"""The library view has to distinguish three states, not two.

Before this, `/models` reported flat `mlx_lm` and `ollama` lists only, and
every exception was swallowed into `[]`, so "not installed", "installed but not
responding" and "installed with an empty library" were the same answer.
`/backends` echoed mira.yaml with no check that any entry could be selected, and
returned active=False on every row whenever the running pair was not itself a
preset — which was the normal case.
"""
import pytest

from core import backend_manager as bm
from core import models_api
from core.models_api import BackendStatus, ModelEntry


def _entry(model_id, backend="mlx-lm"):
    return ModelEntry(model_id=model_id, display_name=model_id, size_gb=1.0, backend=backend)


# ── backend coverage ──────────────────────────────────────────────────────────

def test_every_known_backend_is_reported():
    """The whole point: six backends, not two."""
    reported = {b["backend"] for b in bm.list_models()["backends"]}
    assert reported == set(bm.KNOWN_BACKENDS)
    assert len(reported) > 2


def test_flat_keys_survive_for_existing_clients():
    d = bm.list_models()
    assert isinstance(d["mlx_lm"], list)
    # `ollama` is retired but the key stays, always empty, so an older app
    # build that still decodes it does not fail on a missing field.
    assert d["ollama"] == []


def test_missing_cli_makes_a_backend_unavailable(monkeypatch):
    monkeypatch.setattr(models_api.config, "VLLM_MLX_CLI", "/nope/vllm-mlx")
    status = models_api.backend_status("vllm-mlx")
    assert status.available is False
    assert status.models == []


def test_mira_mlx_needs_no_binary(monkeypatch):
    """It is an in-repo module launched with `python -m`, so nothing can be missing."""
    monkeypatch.setattr(models_api, "_cli_present", lambda _: False)
    assert models_api.backend_status("mira-mlx").available is True


def test_unknown_backend_is_refused_not_guessed():
    status = models_api.backend_status("llama-cpp")
    assert status.available is False
    assert "unknown backend" in status.detail


# ── /backends: selectability and the active row ───────────────────────────────

@pytest.fixture
def presets(monkeypatch):
    entries = [
        {"id": "a", "label": "A", "backend": "mira-mlx", "model": "org/present", "context_window": 1024},
        {"id": "b", "label": "B", "backend": "mlx-lm", "model": "absent-model", "context_window": 1024},
    ]
    import core.config
    monkeypatch.setattr(core.config, "BACKENDS", entries)

    def fake_status(backend):
        if backend == "mira-mlx":
            return BackendStatus("mira-mlx", True, "", [_entry("org/present")])
        return BackendStatus("mlx-lm", True, "mlx-lm is installed but has no models", [])

    monkeypatch.setattr(models_api, "backend_status", fake_status)
    return entries


def test_installed_preset_is_available(presets):
    rows = {r["id"]: r for r in bm.get_backends("mira-mlx", "org/present")}
    assert rows["a"]["available"] is True
    assert rows["a"]["detail"] == ""


def test_preset_whose_model_is_missing_is_flagged_with_a_reason(presets):
    rows = {r["id"]: r for r in bm.get_backends("mira-mlx", "org/present")}
    assert rows["b"]["available"] is False
    assert rows["b"]["detail"]


def test_active_preset_is_marked(presets):
    rows = {r["id"]: r for r in bm.get_backends("mira-mlx", "org/present")}
    assert rows["a"]["active"] is True
    assert rows["b"]["active"] is False


def test_running_pair_absent_from_presets_is_added_and_marked(presets):
    """The normal case on this machine: the default backend is in no preset."""
    rows = bm.get_backends("mira-mlx", "org/unlisted")
    active = [r for r in rows if r["active"]]
    assert len(active) == 1
    assert active[0]["model"] == "org/unlisted"
    assert active[0]["available"] is True
    # First, so a client rendering in order shows what is running at the top.
    assert rows[0] is active[0]


def test_synthesised_row_does_not_displace_the_configured_ones(presets):
    rows = bm.get_backends("mira-mlx", "org/unlisted")
    assert {"a", "b"}.issubset({r["id"] for r in rows})
    assert len(rows) == 3


def test_active_row_is_available_even_when_the_scan_disagrees(monkeypatch):
    """A model that is serving right now exists, whatever a disk scan concluded."""
    import core.config
    monkeypatch.setattr(core.config, "BACKENDS", [
        {"id": "a", "label": "A", "backend": "omlx", "model": "weird-name", "context_window": 1024},
    ])
    monkeypatch.setattr(models_api, "backend_status",
                        lambda b: BackendStatus("omlx", True, "", []))
    rows = bm.get_backends("omlx", "weird-name")
    assert rows[0]["active"] is True
    assert rows[0]["available"] is True


def test_backend_is_probed_once_per_backend_not_once_per_preset(monkeypatch):
    import core.config
    monkeypatch.setattr(core.config, "BACKENDS", [
        {"id": f"p{i}", "label": "P", "backend": "omlx", "model": f"m{i}", "context_window": 1}
        for i in range(5)
    ])
    calls = []

    def counting(backend):
        calls.append(backend)
        return BackendStatus(backend, True, "", [])

    monkeypatch.setattr(models_api, "backend_status", counting)
    bm.get_backends("omlx", "m0")
    assert calls == ["omlx"]
