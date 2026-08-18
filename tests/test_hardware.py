"""Unit tests for core/hardware.py's RAM/disk-derived sizing (all pure functions —
no model download, no running server)."""
import json
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


def test_estimate_kv_bytes_per_token_kv_bits_none_unchanged(tmp_path, monkeypatch):
    """kv_bits=None (the default) must reproduce today's unquantized formula exactly."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(MINISTRAL_3_14B_CONFIG))
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)

    assert hardware.estimate_kv_bytes_per_token("fake/model", kv_bits=None) == 163840


def test_estimate_kv_bytes_per_token_quantized_smaller_than_unquantized(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(MINISTRAL_3_14B_CONFIG))
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)

    unquantized = hardware.estimate_kv_bytes_per_token("fake/model")
    quantized = hardware.estimate_kv_bytes_per_token("fake/model", kv_bits=8, kv_group_size=64)
    assert quantized < unquantized
    # bits/8 + (2*KV_DTYPE_BYTES)/group_size per element = 1 + 4/64 = 1.0625 B/element
    expected_bytes_per_element = 8 / 8 + (2 * hardware.KV_DTYPE_BYTES) / 64
    expected = int(2 * 40 * 8 * 128 * expected_bytes_per_element)
    assert quantized == expected


# A Qwen3.6-style hybrid: 40 layers, but only every 4th is full-attention (10
# grow a KV cache; the other 30 are GatedDeltaNet linear-attention with fixed
# recurrent state). num_key_value_heads=2, head_dim=256, 16 attention heads.
HYBRID_CONFIG = {
    "text_config": {
        "num_hidden_layers": 40,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_attention_heads": 16,
        "hidden_size": 2048,
        "layer_types": [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(40)
        ],
    }
}


def test_estimate_kv_bytes_per_token_hybrid_counts_only_full_attention(tmp_path, monkeypatch):
    """Hybrid models grow KV only on full_attention layers; counting all 40 would
    overcount 4x. layer_types is the source of truth."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(HYBRID_CONFIG))
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)
    # 10 full-attention layers, not 40
    assert hardware.estimate_kv_bytes_per_token("fake/hybrid") == 2 * 10 * 2 * 256 * 2


def test_estimate_prefill_transient_scales_with_heads_and_step(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(HYBRID_CONFIG))
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)
    t1024 = hardware.estimate_prefill_transient_bytes_per_token("fake/hybrid", 1024)
    t256 = hardware.estimate_prefill_transient_bytes_per_token("fake/hybrid", 256)
    assert t1024 == int(hardware.PREFILL_TRANSIENT_SCORE_FACTOR * 16 * hardware.KV_DTYPE_BYTES * 1024)
    assert t256 * 4 == t1024  # linear in prefill_step_size
    # Dominates KV: transient >> the growing-KV term at the same context
    assert t1024 > hardware.estimate_kv_bytes_per_token("fake/hybrid", kv_bits=4) * 5


def test_estimate_prefill_transient_unknown_config_returns_none(monkeypatch):
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: None)
    assert hardware.estimate_prefill_transient_bytes_per_token("fake/x", 1024) is None


def test_find_cached_config_accepts_local_dir(tmp_path, monkeypatch):
    """A locally-pathed model dir (the assembled/omlx MTP dir prod runs) must
    resolve, or every RAM-aware budget silently no-ops for it."""
    (tmp_path / "config.json").write_text("{}")
    # No HF hub match for this path; the local-dir branch must find it.
    assert hardware._find_cached_config(str(tmp_path)) == tmp_path / "config.json"


def test_derive_context_window_capped_by_prefill_transient(tmp_path, monkeypatch):
    """The transient term must pull a large requested window DOWN below the
    request on a 32GB Mac for the hybrid model — the whole point of the fix."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(HYBRID_CONFIG))
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)
    monkeypatch.setattr(hardware, "estimate_model_weight_bytes", lambda model_id: 20 * GB)
    capped = hardware.derive_context_window(
        "fake/hybrid", 32 * GB, requested_context=65536,
        kv_bits=4, kv_group_size=64, prefill_step_size=1024)
    # Lands well below the unfittable 65536, but still a usable window.
    assert 24000 < capped < 65536
    # A smaller prefill step shrinks the transient -> allows more context.
    capped_small_step = hardware.derive_context_window(
        "fake/hybrid", 32 * GB, requested_context=65536,
        kv_bits=4, kv_group_size=64, prefill_step_size=256)
    assert capped_small_step > capped


# -- derive_context_window / derive_prompt_cache_max_bytes across RAM tiers -

@pytest.fixture
def fixed_model(monkeypatch):
    """8GB model weights, 163840 B/token KV cache (Ministral-3-14B's real figure)."""
    monkeypatch.setattr(hardware, "estimate_model_weight_bytes", lambda model_id: 8 * GB)
    monkeypatch.setattr(hardware, "estimate_kv_bytes_per_token", lambda model_id, **kwargs: 163840)


def test_derive_context_window_scales_with_ram(fixed_model):
    """Derived window is non-decreasing in RAM, floored at 1024 when the model
    barely fits, and capped at the requested value once RAM is ample. Exact
    interior values depend on the Metal-ceiling + transient constants, so this
    asserts the shape rather than pinning brittle magic numbers."""
    windows = {
        gb: hardware.derive_context_window("fake/model", gb * GB, requested_context=65536)
        for gb in (8, 16, 24, 32, 64)
    }
    assert windows[8] == 1024          # model + margin can't fit in an 8GB Metal ceiling
    assert windows[64] == 65536        # ample RAM returns the full request
    ordered = [windows[gb] for gb in (8, 16, 24, 32, 64)]
    assert ordered == sorted(ordered)  # monotonic non-decreasing
    assert all(1024 <= w <= 65536 for w in windows.values())


def test_derive_context_window_kv_bits_none_matches_no_kv_bits_arg(fixed_model):
    """Passing kv_bits=None explicitly must be indistinguishable from omitting it."""
    with_none = hardware.derive_context_window("fake/model", 16 * GB, requested_context=65536, kv_bits=None)
    without_arg = hardware.derive_context_window("fake/model", 16 * GB, requested_context=65536)
    assert with_none == without_arg


def test_derive_context_window_kv_bits_set_forwards_to_estimate(monkeypatch):
    """kv_bits/kv_group_size must reach estimate_kv_bytes_per_token, not be dropped."""
    monkeypatch.setattr(hardware, "estimate_model_weight_bytes", lambda model_id: 8 * GB)
    captured = {}

    def fake_estimate(model_id, kv_bits=None, kv_group_size=64):
        captured["kv_bits"] = kv_bits
        captured["kv_group_size"] = kv_group_size
        return 163840

    monkeypatch.setattr(hardware, "estimate_kv_bytes_per_token", fake_estimate)
    hardware.derive_context_window("fake/model", 16 * GB, requested_context=65536, kv_bits=8, kv_group_size=32)
    assert captured == {"kv_bits": 8, "kv_group_size": 32}


def test_derive_context_window_unknown_architecture_returns_requested(monkeypatch):
    monkeypatch.setattr(hardware, "estimate_model_weight_bytes", lambda model_id: 8 * GB)
    monkeypatch.setattr(hardware, "estimate_kv_bytes_per_token", lambda model_id, **kwargs: None)
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


# -- estimate_active_weight_bytes / _classify_weight_bytes (MoE offloading) -

def _write_fake_safetensors(path, tensors):
    """Write a minimal, valid safetensors file: real header (name -> dtype/
    shape/data_offsets), zero-filled data section. Content is never read by
    _classify_weight_bytes (header-only), only shapes/offsets matter."""
    import struct as _struct

    header = {}
    offset = 0
    for name, (shape, nbytes) in tensors.items():
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(_struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        # Sparse data section: seek past it rather than materializing `offset`
        # bytes. Tests build multi-GB fake expert tables, and allocating those
        # for real blows up CI runners (MemoryError) while buying nothing —
        # only the header and st_size are ever read.
        if offset:
            f.seek(offset - 1, 1)
            f.write(b"\x00")


def test_estimate_active_weight_bytes_none_fraction_matches_full_size(monkeypatch):
    monkeypatch.setattr(hardware, "estimate_model_weight_bytes", lambda model_id: 20 * GB)
    assert hardware.estimate_active_weight_bytes("fake/model", None) == 20 * GB
    assert hardware.estimate_active_weight_bytes("fake/model", 1.0) == 20 * GB


def test_estimate_active_weight_bytes_dense_model_unaffected(tmp_path, monkeypatch):
    """A model with no 'num_experts' in its config (dense) must be returned
    at full size regardless of resident_expert_fraction — offloading only
    applies to MoE architectures."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"hidden_size": 4096}')
    monkeypatch.setattr(hardware, "estimate_model_weight_bytes", lambda model_id: 20 * GB)
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)
    assert hardware.estimate_active_weight_bytes("fake/dense-model", 0.2) == 20 * GB


def test_estimate_active_weight_bytes_moe_scales_expert_bytes_only(tmp_path, monkeypatch):
    """8 experts, 1MB each (8MB total expert bytes) + 2MB non-expert bytes.
    At resident_expert_fraction=0.25, only the expert portion shrinks."""
    num_experts = 8
    per_expert_bytes = 1 * 1024 * 1024
    other_bytes = 2 * 1024 * 1024
    shard = tmp_path / "model.safetensors"
    _write_fake_safetensors(shard, {
        "model.layers.0.mlp.switch_mlp.down_proj.weight": ((num_experts, 512, 512), num_experts * per_expert_bytes),
        "model.embed_tokens.weight": ((32000, 512), other_bytes),
    })
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"num_experts": num_experts}))

    # estimate_model_weight_bytes reports on-disk *file* size (includes the
    # safetensors header preamble), not just tensor data bytes — classify
    # only sums data_offsets, so use the same source of truth for "full".
    full_bytes = shard.stat().st_size
    monkeypatch.setattr(hardware, "estimate_model_weight_bytes", lambda model_id: full_bytes)
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)

    full = hardware.estimate_active_weight_bytes("fake/moe-model", None)
    reduced = hardware.estimate_active_weight_bytes("fake/moe-model", 0.25)
    assert full == full_bytes
    assert reduced == other_bytes + int(num_experts * per_expert_bytes * 0.25)
    assert reduced < full


def test_derive_context_window_resident_expert_fraction_forwards_to_estimate(monkeypatch):
    """resident_expert_fraction must reach estimate_active_weight_bytes, not
    be silently dropped along the derive_context_window call path."""
    captured = {}

    def fake_estimate(model_id, resident_expert_fraction=None):
        captured["resident_expert_fraction"] = resident_expert_fraction
        return 8 * GB

    monkeypatch.setattr(hardware, "estimate_active_weight_bytes", fake_estimate)
    monkeypatch.setattr(hardware, "estimate_kv_bytes_per_token", lambda model_id, **kwargs: 163840)
    hardware.derive_context_window("fake/model", 32 * GB, requested_context=65536, resident_expert_fraction=0.3)
    assert captured["resident_expert_fraction"] == 0.3


# -- derive_resident_expert_fraction (RAM-aware sizing for over-DRAM MoE) ----

def _moe_model(tmp_path, expert_gb, other_gb, num_experts=256):
    shard = tmp_path / "model.safetensors"
    _write_fake_safetensors(shard, {
        "model.layers.0.mlp.switch_mlp.down_proj.weight":
            ((num_experts, 4096, 4096), int(expert_gb * GB)),
        "model.embed_tokens.weight": ((32000, 4096), int(other_gb * GB)),
    })
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"num_experts": num_experts}))
    return config_path


def test_derive_resident_fraction_raises_above_floor_with_headroom(tmp_path, monkeypatch):
    """A big-but-offloadable expert table with RAM to spare must push the
    fraction above the configured floor, sized to the wired ceiling."""
    config_path = _moe_model(tmp_path, expert_gb=30, other_gb=3)
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)
    f = hardware.derive_resident_expert_fraction("fake/moe", floor_fraction=0.3, total_ram_bytes=32 * GB)
    # ceiling = min(32*0.55, 32*0.78 - 3, 32 - 3) = 17.6GB; budget = 17.6 - 3 = 14.6; f = 14.6/30
    ceiling = min(int(32 * GB * hardware.RAM_AWARE_PEAK_FRACTION),
                  int(32 * GB * hardware.METAL_WIRED_FRACTION) - hardware.WIRED_HEADROOM_BYTES,
                  32 * GB - hardware.SAFETY_MARGIN_BYTES)
    assert 0.3 < f < 0.85
    assert abs(f - round((ceiling - 3 * GB) / (30 * GB), 3)) < 1e-6


def test_derive_resident_fraction_never_below_floor(tmp_path, monkeypatch):
    """Tight RAM must not lower the fraction below the configured knob — the
    function can only ever RAISE residency, never reduce it."""
    config_path = _moe_model(tmp_path, expert_gb=40, other_gb=8)
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)
    f = hardware.derive_resident_expert_fraction("fake/moe", floor_fraction=0.3, total_ram_bytes=16 * GB)
    assert f == 0.3


def test_derive_resident_fraction_clamped_at_max(tmp_path, monkeypatch):
    """A small expert table with abundant RAM clamps at max_fraction, never 1.0
    (we're still an offloaded model; keep some slack)."""
    config_path = _moe_model(tmp_path, expert_gb=5, other_gb=2)
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)
    f = hardware.derive_resident_expert_fraction("fake/moe", floor_fraction=0.3,
                                                 total_ram_bytes=64 * GB, max_fraction=0.85)
    assert f == 0.85


def test_derive_resident_fraction_dense_model_returns_floor(tmp_path, monkeypatch):
    """A dense model (no num_experts) isn't offloaded — return the floor."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"hidden_size": 4096}')
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: config_path)
    assert hardware.derive_resident_expert_fraction("fake/dense", floor_fraction=0.3) == 0.3


def test_derive_resident_fraction_uncached_model_returns_floor(monkeypatch):
    monkeypatch.setattr(hardware, "_find_cached_config", lambda model_id: None)
    assert hardware.derive_resident_expert_fraction("fake/unknown", floor_fraction=0.3) == 0.3
