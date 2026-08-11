"""What _check_memory_pressure gives back, and in what order.

There was no test here at all, which is how the trim came to free 7x to 53x more
than the overshoot for three days without anyone noticing. The counters said
"1 trim event"; nothing said what it cost.

The distinction these tests protect: MLX's reuse cache is free to discard (it
costs a reallocation) while the prompt cache is not (every entry is a prefill
somebody pays again). So the reuse cache goes first, and the prompt cache is
touched only if that was not enough.
"""
import pytest

pytest.importorskip("mlx.core")

from types import SimpleNamespace  # noqa: E402

import core.inference.mira_mlx_server as srv  # noqa: E402
from core.inference.mira_mlx_server import GenerationEngine  # noqa: E402

GB = 1024 ** 3


class _FakeMX:
    """Stands in for the mx module inside _check_memory_pressure.

    active is fixed; the reuse cache is what clear_cache() releases. Modelling
    them separately is the whole point -- a fake where clear_cache() does
    nothing would pass either ordering and prove nothing.
    """

    def __init__(self, active, reuse):
        self.active = active
        self.reuse = reuse
        self.clear_calls = 0

    def get_active_memory(self):
        return self.active

    def get_cache_memory(self):
        return self.reuse

    def get_peak_memory(self):
        return self.active + self.reuse

    def clear_cache(self):
        self.clear_calls += 1
        self.reuse = 0


class _Pool:
    """trim_to() pops whole entries, so the pool cannot shrink to an arbitrary
    byte count. Modelling that matters: it is why the margin in the production
    code does not need to be finely tuned."""

    ENTRY = 88 * 1024 * 1024

    def __init__(self, entries):
        self.entries = entries
        self.trim_calls = []

    @property
    def nbytes(self):
        return self.entries * self.ENTRY

    def trim_to(self, *, n_bytes=None, n_sequences=None):
        self.trim_calls.append(n_bytes)
        while self.entries > 0 and self.nbytes > n_bytes:
            self.entries -= 1


@pytest.fixture
def engine(monkeypatch):
    e = GenerationEngine(model_path="fake/model")
    e._memory_ceiling_bytes = 20 * GB
    return e


def _run(engine, monkeypatch, *, active, reuse, entries):
    fake = _FakeMX(active, reuse)
    pool = _Pool(entries)
    monkeypatch.setattr(srv, "mx", fake)
    engine.prompt_cache = pool
    engine._check_memory_pressure()
    return fake, pool


def test_under_the_ceiling_nothing_happens(engine, monkeypatch):
    fake, pool = _run(engine, monkeypatch, active=18 * GB, reuse=1 * GB, entries=10)
    assert fake.clear_calls == 0
    assert pool.entries == 10
    assert engine._memory_pressure_trim_events == 0


def test_the_reuse_cache_goes_first_and_the_prompt_cache_survives(engine, monkeypatch):
    """The case every recorded event was: a tiny overshoot against a reuse cache
    far larger than it. Before this ordering, all 15 entries' worth of prefill
    was thrown away to reclaim 30MB."""
    fake, pool = _run(engine, monkeypatch,
                      active=19 * GB, reuse=1 * GB + 30 * 1024 * 1024, entries=15)

    assert fake.clear_calls == 1
    assert pool.entries == 15, "prompt cache was trimmed despite the reuse cache covering it"
    assert pool.trim_calls == []
    assert engine._memory_pressure_trim_events == 0
    assert engine._buffer_cache_sufficed == 1
    assert engine._last_trim["freed_by_buffer_cache_alone"] is True


def test_when_the_reuse_cache_is_not_enough_the_trim_is_proportional(engine, monkeypatch):
    """Still over by ~1 entry's worth after the clear, so about one entry should
    go -- not half the pool. Ten entries halved would be five."""
    over = 80 * 1024 * 1024
    fake, pool = _run(engine, monkeypatch,
                      active=20 * GB + over, reuse=100 * 1024 * 1024, entries=10)

    assert fake.clear_calls == 1
    assert engine._memory_pressure_trim_events == 1
    assert engine._buffer_cache_sufficed == 0
    assert engine._last_trim["freed_by_buffer_cache_alone"] is False
    dropped = 10 - pool.entries
    assert 1 <= dropped <= 2, f"dropped {dropped} entries for a {over/1024**2:.0f}MB overshoot"
    assert pool.entries >= 5, "this is the old halve-the-pool behaviour returning"


def test_a_huge_overshoot_empties_the_pool_rather_than_underreacting(engine, monkeypatch):
    """Clearing the condition still outranks keeping the cache. Freeing too
    little would be a worse bug than freeing too much, because what is on the
    other side is the machine swapping."""
    fake, pool = _run(engine, monkeypatch,
                      active=25 * GB, reuse=1 * GB, entries=10)

    assert pool.entries == 0
    assert engine._memory_pressure_trim_events == 1


def test_the_two_outcomes_are_counted_separately(engine, monkeypatch):
    """Same trigger, very different costs. One counter for both would hide which
    one is actually happening, which is what the old stats did."""
    _run(engine, monkeypatch, active=19 * GB, reuse=2 * GB, entries=10)
    _run(engine, monkeypatch, active=25 * GB, reuse=1 * GB, entries=10)

    assert engine._buffer_cache_sufficed == 1
    assert engine._memory_pressure_trim_events == 1


def test_stats_report_both_counters(engine, monkeypatch):
    _run(engine, monkeypatch, active=19 * GB, reuse=2 * GB, entries=10)
    engine.batch_generator = SimpleNamespace(_prompt_batch=None)
    stats = engine.stats_snapshot()

    assert stats["memory_pressure_buffer_cache_sufficed"] == 1
    assert stats["memory_pressure_trim_events"] == 0
    assert stats["last_trim"]["freed_by_buffer_cache_alone"] is True
