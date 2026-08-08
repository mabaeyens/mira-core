"""Gating for the idle-branch decompression.

The touch itself costs ~17s of real memory traffic on a machine that is by
definition short of memory, so almost all the value here is in the conditions
that stop it running. Every test below is a case where it must NOT act.

No model is loaded: `_maybe_decompress_model` is driven off `_system_state`, so
the state can be written directly and the touch stubbed.
"""
import pytest

pytest.importorskip("mlx.core")  # mlx is macOS-only (Apple Silicon), absent on Linux CI

from core import hardware
from core.inference.mira_mlx_server import (
    DECOMPRESS_MIN_AVAILABLE_BYTES,
    TOUCH_ID_STRIDE,
    TOUCH_TOKENS,
    GenerationEngine,
)

GB = 1024**3
EVICTED = 18_800_000_000


def make_engine(monkeypatch, enabled=True, **state):
    """An engine with a known advisory state and a touch that only records itself."""
    engine = GenerationEngine(model_path="fake/model", proactive_decompress=enabled)
    engine._system_state = {
        "advisory": "evicted",
        "self_compressed_bytes": EVICTED,
        "available_bytes": 8 * GB,
        "pressure_level": hardware.PRESSURE_NORMAL,
        **state,
    }
    touches = []
    monkeypatch.setattr(engine, "_touch_model_weights",
                        lambda: (touches.append(1), (2.5, EVICTED))[1])
    monkeypatch.setattr(hardware, "on_battery", lambda: False)
    return engine, touches


def test_acts_once_on_a_real_eviction(monkeypatch):
    engine, touches = make_engine(monkeypatch)
    engine._maybe_decompress_model()
    assert len(touches) == 1
    assert engine._decompress_events == 1
    assert engine._decompress_last_reclaimed_bytes == EVICTED
    assert engine._decompress_last_seconds == 2.5


def test_does_not_loop_while_the_same_eviction_persists(monkeypatch):
    """The machine may keep re-compressing the model. That is it saying it has no
    room, and the answer is to stop, not to keep winning the page back."""
    engine, touches = make_engine(monkeypatch)
    for _ in range(50):
        engine._maybe_decompress_model()
    assert len(touches) == 1


def test_rearms_only_after_the_advisory_clears(monkeypatch):
    engine, touches = make_engine(monkeypatch)
    engine._maybe_decompress_model()
    engine._system_state["advisory"] = "ok"
    engine._maybe_decompress_model()          # clears the latch, does not act
    assert len(touches) == 1
    engine._system_state["advisory"] = "evicted"
    engine._maybe_decompress_model()          # a genuinely new event
    assert len(touches) == 2


def test_disabled_by_default_config(monkeypatch):
    engine, touches = make_engine(monkeypatch, enabled=False)
    engine._maybe_decompress_model()
    assert touches == []


def test_never_acts_on_the_system_wide_signal(monkeypatch):
    """Without a per-process reading there is no evidence the compressed pages
    are Mira's, and 17s of memory traffic because Xcode is busy is a bug."""
    engine, touches = make_engine(monkeypatch, self_compressed_bytes=None)
    engine._maybe_decompress_model()
    assert touches == []
    assert engine._decompress_events == 0


def test_skips_when_there_is_no_headroom(monkeypatch):
    engine, touches = make_engine(
        monkeypatch, available_bytes=DECOMPRESS_MIN_AVAILABLE_BYTES - 1)
    engine._maybe_decompress_model()
    assert touches == []
    assert engine._decompress_skipped_no_headroom == 1


def test_headroom_floor_does_not_require_room_for_the_whole_expansion(monkeypatch):
    """The real 18.44GB event ran with 6.49GB available and left 5.12GB, because
    emptying the compressor pays for most of the expansion. A precondition sized
    against the compressed total would have blocked the case worth acting on."""
    engine, touches = make_engine(monkeypatch, available_bytes=6.49 * GB)
    engine._maybe_decompress_model()
    assert len(touches) == 1


def test_skips_on_battery(monkeypatch):
    engine, touches = make_engine(monkeypatch)
    monkeypatch.setattr(hardware, "on_battery", lambda: True)
    engine._maybe_decompress_model()
    assert touches == []


def test_skips_at_critical_pressure(monkeypatch):
    """At critical the machine is already in trouble; adding memory traffic to
    it makes Mira the problem rather than the victim."""
    engine, touches = make_engine(monkeypatch,
                                  pressure_level=hardware.PRESSURE_CRITICAL)
    engine._maybe_decompress_model()
    assert touches == []


def test_forces_a_re_probe_so_the_advisory_stops_lying(monkeypatch):
    """The advisory drives a user-visible notification. Leaving it to expire on
    the 30s gate would keep reporting an eviction that has just been fixed."""
    engine, touches = make_engine(monkeypatch)
    engine._system_state_checked_at = 12345.0
    engine._maybe_decompress_model()
    assert engine._system_state_checked_at == 0.0


def test_a_touch_that_reclaims_nothing_is_recorded_as_nothing(monkeypatch):
    """Measured, not assumed. If the forward pass stops faulting the weights in
    (a different model, an MLX change), the counters must show it rather than
    reporting a successful decompression of zero bytes."""
    engine, _ = make_engine(monkeypatch)
    monkeypatch.setattr(engine, "_touch_model_weights", lambda: (0.01, 0))
    engine._maybe_decompress_model()
    assert engine._decompress_events == 1
    assert engine._decompress_last_reclaimed_bytes == 0


def test_a_failing_touch_does_not_take_the_engine_down(monkeypatch):
    """This runs on the model thread, the only thread that can serve requests.
    An unhandled exception would stop the server entirely, which is a
    catastrophic price for an optimization nobody asked for."""
    engine, _ = make_engine(monkeypatch)

    def boom():
        raise RuntimeError("wrong forward signature for this model")

    monkeypatch.setattr(engine, "_touch_model_weights", boom)
    engine._maybe_decompress_model()  # must not raise
    assert engine._decompress_failures == 1
    assert engine._decompress_events == 0
    # Disabled for the process: the failure is a property of this model, so
    # retrying every 30s would only reprint the same traceback forever.
    assert engine.proactive_decompress is False


def test_a_failing_touch_is_not_retried(monkeypatch):
    engine, _ = make_engine(monkeypatch)
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("nope")

    monkeypatch.setattr(engine, "_touch_model_weights", boom)
    for _ in range(10):
        engine._system_state["advisory"] = "ok"
        engine._maybe_decompress_model()
        engine._system_state["advisory"] = "evicted"
        engine._maybe_decompress_model()
    assert len(calls) == 1


def test_stats_expose_the_counters(monkeypatch):
    engine, _ = make_engine(monkeypatch)
    engine._maybe_decompress_model()
    block = engine.stats_snapshot()["decompress"]
    assert block["events"] == 1
    assert block["last_reclaimed_bytes"] == EVICTED
    assert block["skipped_no_headroom"] == 0


class _FakeTokenizer:
    vocab_size = 151_936


def _touch_ids(engine, monkeypatch):
    """Run the real _touch_model_weights against a recording stand-in model."""
    seen = {}

    def fake_model(arr):
        seen["shape"] = arr.shape
        seen["ids"] = arr.tolist()[0]
        return arr

    monkeypatch.setattr(engine, "tokenizer", _FakeTokenizer())
    monkeypatch.setattr(engine, "model", fake_model)
    engine._touch_model_weights()
    return seen


def test_touch_prompt_is_long_enough_to_cover_the_expert_table(monkeypatch):
    """A single token reaches ~3% of a top-8-of-256 table. That version was
    measured on the real model: 1.19s, zero bytes reclaimed, model still 13.39GB
    compressed. Covering the table needs about (E/k)*ln(E) ~ 177 tokens."""
    engine = GenerationEngine(model_path="fake/model", proactive_decompress=True)
    seen = _touch_ids(engine, monkeypatch)
    assert seen["shape"] == (1, TOUCH_TOKENS)
    assert TOUCH_TOKENS >= 256


def test_touch_ids_are_spread_and_avoid_special_tokens(monkeypatch):
    """Consecutive low ids are mostly special tokens and route through the same
    few experts, which would defeat the length above."""
    engine = GenerationEngine(model_path="fake/model", proactive_decompress=True)
    ids = _touch_ids(engine, monkeypatch)["ids"]
    assert all(i >= 1024 for i in ids)
    assert all(i < _FakeTokenizer.vocab_size for i in ids)
    # Distinct ids, not a short repeating cycle.
    assert len(set(ids)) == TOUCH_TOKENS
    assert TOUCH_ID_STRIDE > 1


def test_touch_survives_a_tokenizer_without_vocab_size(monkeypatch):
    """Falls back rather than raising, since raising here disables the feature."""
    engine = GenerationEngine(model_path="fake/model", proactive_decompress=True)

    class Bare:
        pass

    seen = {}
    monkeypatch.setattr(engine, "tokenizer", Bare())
    monkeypatch.setattr(engine, "model",
                        lambda arr: seen.setdefault("ids", arr.tolist()[0]) or arr)
    engine._touch_model_weights()
    assert len(seen["ids"]) == TOUCH_TOKENS


def test_stats_include_the_per_process_eviction_fields(monkeypatch):
    engine, _ = make_engine(monkeypatch)
    engine._system_state["eviction_signal"] = "self"
    block = engine.stats_snapshot()["system_memory"]
    assert block["self_compressed_bytes"] == EVICTED
    assert block["eviction_signal"] == "self"
