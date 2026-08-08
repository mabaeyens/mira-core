"""Pin the two facts that make Qwen3.6 multi-turn chat uncacheable.

Both were established on 2026-08-08 while diagnosing why bench Q10's second turn
re-prefilled 27,614 tokens. Neither is Mira's own code, which is exactly why
they need a test: an mlx-lm upgrade that changes either one silently changes
Mira's cache behaviour, and the failure mode is a slow reply rather than a wrong
one, so nothing else would catch it.
"""
import pytest

pytest.importorskip("mlx.core")

from mlx_lm.models.cache import ArraysCache, can_trim_prompt_cache  # noqa: E402


def test_arrayscache_is_not_trimmable():
    """Qwen3.6's prompt cache contains ArraysCache entries alongside the KV
    caches. can_trim_prompt_cache() is all(), so this single False disables
    fetch_nearest_cache's trim-back branch for the whole model: the only reuse
    left is an entry that is a *whole* prefix of the new prompt.

    If this ever starts returning True, the trim path comes back to life and the
    Q10 class of miss should disappear. That would be good news worth noticing.
    """
    assert ArraysCache(1).is_trimmable() is False
    assert ArraysCache.is_trimmable.__qualname__ == "_BaseCache.is_trimmable", (
        "ArraysCache gained its own is_trimmable; re-check whether trimming is "
        "now available and whether it is actually correct for this cache type"
    )


def test_one_untrimmable_entry_disables_the_whole_cache():
    """The all() semantics, pinned. This is why a single ArraysCache among many
    trimmable KV caches is enough to cost every token of reuse."""
    class Trimmable:
        def is_trimmable(self):
            return True

    assert can_trim_prompt_cache([Trimmable(), Trimmable()]) is True
    assert can_trim_prompt_cache([Trimmable(), ArraysCache(1)]) is False
