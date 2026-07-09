"""Unit tests for core/hardware.py's RAM/disk-derived sizing (all pure functions —
no model download, no running server)."""
import subprocess

import pytest

from core import hardware

GB = hardware.BYTES_PER_GB

# Ministral 3 14B's text_config, used to validate the KV-bytes-per-token formula
# against a live measurement (2026-07-09): computed 163,840 vs. measured 163,820.
MINISTRAL_3_14B_CONFIG = {
    "text_config": {
        "num_hidden_layers": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_attention_heads": 32,
        "hidden_size": 5120,
    }
}


# -- estimate_kv_bytes_per_token -------------------------------------------

def test_estimate_kv_bytes_per_token_ministral_3_14b(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"text_config": {"num_hidden_layers": 40, "num_key_value_heads": 8, "head_dim": 128, "num_attention_heads": 32, "hidden_size": 5120}}')
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)

    assert hardware.estimate_kv_bytes_per_token("fake/model") == 2 * 40 * 8 * 128 * 2 == 163840


def test_estimate_kv_bytes_per_token_top_level_config(tmp_path, monkeypatch):
    # Text-only models have these keys at the top level, not nested under text_config.
    config_path = tmp_path / "config.json"
    config_path.write_text('{"num_hidden_layers": 10, "num_attention_heads": 4, "hidden_size": 256}')
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)

    # num_key_value_heads/head_dim absent -> falls back to num_attention_heads / hidden_size//heads
    assert hardware.estimate_kv_bytes_per_token("fake/model") == 2 * 10 * 4 * 64 * 2


def test_estimate_kv_bytes_per_token_no_cached_config(monkeypatch):
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: None)
    assert hardware.estimate_kv_bytes_per_token("fake/model") is None


def test_estimate_kv_bytes_per_token_malformed_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text("not json")
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)
    assert hardware.estimate_kv_bytes_per_token("fake/model") is None


# -- derive_context_window / derive_prompt_cache_max_bytes across RAM tiers -

@pytest.fixture
def fixed_model(monkeypatch):
    """8GB model weights, 163840 B/token KV cache (Ministral-3-14B's real figure)."""
    monkeypatch.setattr(hardware, "estimate_model_weight_bytes", lambda model_id: 8 * GB)
    monkeypatch.setattr(hardware, "estimate_kv_bytes_per_token", lambda model_id: 163840)


@pytest.mark.parametrize("ram_gb,expect_context", [
    (8, 1024),      # floored: available RAM can't even cover the model + margin
    (16, 32768),
    (24, 65536),
    (32, 65536),
    (64, 65536),
])
def test_derive_context_window_scales_with_ram(fixed_model, ram_gb, expect_context):
    assert hardware.derive_context_window("fake/model", ram_gb * GB, requested_context=65536) == expect_context


def test_derive_context_window_unknown_architecture_returns_requested(monkeypatch):
    monkeypatch.setattr(hardware, "estimate_model_weight_bytes", lambda model_id: 8 * GB)
    monkeypatch.setattr(hardware, "estimate_kv_bytes_per_token", lambda model_id: None)
    assert hardware.derive_context_window("fake/model", 32 * GB, requested_context=65536) == 65536


@pytest.mark.parametrize("ram_gb,expect_min_gb", [
    (8, 0.5),   # floored at 512MB even when nothing is really available
    (32, 10),
    (64, 26),
])
def test_derive_prompt_cache_max_bytes_scales_with_ram(fixed_model, ram_gb, expect_min_gb):
    budget = hardware.derive_prompt_cache_max_bytes("fake/model", ram_gb * GB)
    assert budget >= expect_min_gb * GB


# -- fits_in_memory boundary -------------------------------------------------

def test_fits_in_memory_unknown_model_allows(monkeypatch):
    monkeypatch.setattr(hardware, "estimate_model_weight_bytes", lambda model_id: None)
    ok, reason = hardware.fits_in_memory("fake/model", 8 * GB)
    assert ok is True


def test_fits_in_memory_clearly_too_big_rejected(fixed_model):
    ok, reason = hardware.fits_in_memory("fake/model", total_ram_bytes=8 * GB)
    assert ok is False
    assert "fake/model" in reason


def test_fits_in_memory_comfortable_ram_accepted(fixed_model):
    ok, reason = hardware.fits_in_memory("fake/model", total_ram_bytes=32 * GB)
    assert ok is True


# -- get_total_ram_bytes: sysctl PATH regression -----------------------------

def test_get_total_ram_bytes_uses_absolute_sysctl_path(monkeypatch):
    """Regression test for a real bug (2026-07-09): a bare 'sysctl' command name
    resolves fine in an interactive shell but FileNotFoundErrors under launchd's
    stripped PATH (no /usr/sbin), silently falling back to the 16GB guess. Assert
    the call site uses the absolute path so this can't regress."""
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="34359738368\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    total = hardware.get_total_ram_bytes()

    assert captured["args"][0] == "/usr/sbin/sysctl"
    assert total == 34359738368


def test_get_total_ram_bytes_falls_back_on_missing_sysctl(monkeypatch):
    def fake_run(args, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert hardware.get_total_ram_bytes() == 16 * GB


# -- derive_disk_cache_max_bytes ---------------------------------------------

def test_derive_disk_cache_max_bytes_caps_at_50gb(tmp_path, monkeypatch):
    class FakeUsage:
        free = 600 * GB  # 10% would be 60GB, above the 50GB cap

    monkeypatch.setattr(hardware.shutil, "disk_usage", lambda path: FakeUsage())
    assert hardware.derive_disk_cache_max_bytes(tmp_path) == 50 * GB


def test_derive_disk_cache_max_bytes_shrinks_on_low_free_space(tmp_path, monkeypatch):
    class FakeUsage:
        free = 20 * GB  # 10% is 2GB, below the 50GB cap -> the 10% term binds

    monkeypatch.setattr(hardware.shutil, "disk_usage", lambda path: FakeUsage())
    assert hardware.derive_disk_cache_max_bytes(tmp_path) == 2 * GB
