"""The cache says WHY it missed, not just that it did.

fetch_nearest_cache can reuse an entry two ways, and they fail for different
reasons. `shorter` is an entry that is a *whole* prefix of the prompt, so one
differing token anywhere inside it drops reuse from thousands of tokens to zero.
`longer` is an entry extending past the divergence, trimmed back — and that one
is gated on can_trim_prompt_cache().

On 2026-08-08 bench Q10's second turn took 48.7s re-prefilling 27,614 tokens
when a 27,551-token entry covered all but the last handful, and the hit/miss
counters could not say which of the two paths had failed or where. It was both:
the entry was not a whole prefix (Qwen3's generation prompt ends with
`<think>\\n`, which the template does not re-emit when replaying that assistant
turn), and the trim fallback was unavailable because Qwen3.6's cache contains an
ArraysCache, whose is_trimmable() is False.

These tests pin the diagnostic against sequences whose divergence index is known
by construction, so the number in the log can be trusted on a real prompt.
"""
import logging
import re

import pytest

from core.inference.disk_prompt_cache import DiskBackedPromptCache

MODEL = "test-model"


class _FakeCacheArray:
    """Stands in for an mlx KV cache array. nbytes is read by the LRU accounting
    and is_trimmable() by can_trim_prompt_cache(); non-trimmable keeps the
    `longer` branch of fetch_nearest_cache out of the way so these tests pin the
    whole-prefix path that Q10 actually took."""
    nbytes = 1024

    def is_trimmable(self):
        # False on purpose, and not an arbitrary stub: Qwen3.6's real prompt
        # cache contains an ArraysCache whose is_trimmable() is False, which
        # makes can_trim_prompt_cache() False for the whole entry and takes the
        # trim-back branch out of play. These tests therefore exercise the same
        # whole-prefix-only path the live model is restricted to.
        return False


def _entry():
    return [_FakeCacheArray()]


@pytest.fixture
def cache():
    # No disk store: this exercises the in-memory path, which is where the
    # divergence happens. A disk store would only add an exact-match lookup.
    return DiskBackedPromptCache(max_size=10, max_bytes=1 << 30, disk_store=None)


def _detail(caplog):
    for rec in caplog.records:
        if "cache miss detail" in rec.getMessage():
            return rec.getMessage()
    return None


def _field(msg, name):
    m = re.search(rf"{name}=(\d+)", msg)
    return int(m.group(1)) if m else None


def test_it_reports_the_divergence_index(cache, caplog):
    """The Q10 shape: a stored entry shares a long prefix and then differs."""
    stored = list(range(1000))
    cache.insert_cache(MODEL, stored, _entry())

    query = list(range(900)) + [99999] + list(range(901, 1100))
    with caplog.at_level(logging.INFO, logger="core.inference.disk_prompt_cache"):
        got, rest = cache.fetch_nearest_cache(MODEL, query)

    assert got is None, "sanity: this query must actually miss"
    msg = _detail(caplog)
    assert msg is not None, "a miss produced no explanation"
    assert re.search(r"diverged at index 900\b", msg), msg


def test_a_whole_prefix_entry_is_a_hit_and_needs_no_explanation(cache, caplog):
    """The control. If this ever misses, the diagnostic above is measuring a
    broken cache rather than a diverging prompt."""
    stored = list(range(1000))
    cache.insert_cache(MODEL, stored, _entry())

    with caplog.at_level(logging.INFO, logger="core.inference.disk_prompt_cache"):
        got, rest = cache.fetch_nearest_cache(MODEL, stored + [1000, 1001])

    assert got is not None, "a stored entry that IS a whole prefix did not hit"
    assert len(rest) == 2
    assert _detail(caplog) is None


def test_one_differing_token_at_the_end_costs_the_whole_entry(cache, caplog):
    """This is the behaviour that makes Q10 expensive: divergence in the last
    token of a 1000-token entry reuses nothing, not 999 tokens."""
    stored = list(range(1000))
    cache.insert_cache(MODEL, stored, _entry())

    query = list(range(999)) + [77777] + [1000]
    with caplog.at_level(logging.INFO, logger="core.inference.disk_prompt_cache"):
        got, rest = cache.fetch_nearest_cache(MODEL, query)

    assert got is None
    assert len(rest) == len(query), "expected zero reuse"
    assert re.search(r"diverged at index 999\b", _detail(caplog))


def test_it_reports_zero_when_nothing_is_stored(cache, caplog):
    with caplog.at_level(logging.INFO, logger="core.inference.disk_prompt_cache"):
        cache.fetch_nearest_cache(MODEL, [1, 2, 3])
    msg = _detail(caplog)
    assert msg is not None
    assert re.search(r"diverged at index 0\b", msg), msg


def test_it_reports_the_prompt_length(cache, caplog):
    cache.insert_cache(MODEL, list(range(50)), _entry())
    with caplog.at_level(logging.INFO, logger="core.inference.disk_prompt_cache"):
        cache.fetch_nearest_cache(MODEL, [999] * 123)
    assert "123 prompt tokens" in _detail(caplog)


def test_a_broken_trie_does_not_break_the_request(cache, caplog):
    """Diagnostics run on the request path. A failure here must cost the log
    line, not the answer."""
    class Boom:
        def search(self, *a, **k):
            raise RuntimeError("trie exploded")

    cache._trie = Boom()
    with caplog.at_level(logging.INFO, logger="core.inference.disk_prompt_cache"):
        cache._explain_miss(MODEL, [1, 2, 3])
    assert _detail(caplog) is None
