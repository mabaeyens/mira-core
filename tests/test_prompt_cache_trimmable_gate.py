"""DiskBackedPromptCache.insert_cache must drop entries that can never be reused
on a non-trimmable model (Qwen3.6 hybrid), while always keeping "system" entries
(reused via whole-prefix match) and never changing behaviour on a trimmable model.

See specs/prompt-cache-skip-nonreusable.md.
"""
import pytest

pytest.importorskip("mlx.core")

from unittest.mock import patch  # noqa: E402

import core.inference.disk_prompt_cache as dpc  # noqa: E402


@pytest.mark.parametrize(
    "trimmable, cache_type, expect_inserted",
    [
        # non-trimmable model: only "system" survives, the rest are deadweight
        (False, "assistant", False),
        (False, "user", True),       # history: whole prefix of the next tool step; newest only
        (False, "system", True),
        # trimmable model: every class inserts exactly as before
        (True, "assistant", True),
        (True, "user", True),
        (True, "system", True),
    ],
)
def test_insert_gated_by_trimmability(trimmable, cache_type, expect_inserted):
    cache = dpc.DiskBackedPromptCache(max_size=10, max_bytes=10 ** 12)
    with patch.object(dpc, "can_trim_prompt_cache", return_value=trimmable), \
         patch.object(dpc.LRUPromptCache, "insert_cache") as base_insert:
        cache.insert_cache("model-id", [1, 2, 3], [object()], cache_type=cache_type)
    assert base_insert.called is expect_inserted


def test_predicate_error_falls_toward_inserting():
    """A can_trim_prompt_cache error must never drop a working cache."""
    cache = dpc.DiskBackedPromptCache(max_size=10, max_bytes=10 ** 12)
    with patch.object(dpc, "can_trim_prompt_cache", side_effect=RuntimeError("boom")), \
         patch.object(dpc.LRUPromptCache, "insert_cache") as base_insert:
        cache.insert_cache("model-id", [1, 2, 3], [object()], cache_type="assistant")
    assert base_insert.called is True


def test_non_trimmable_keeps_only_the_newest_history_entry():
    """Each tool step supersedes the last one, so older history entries are deadweight."""
    class C:
        nbytes = 100

        def is_trimmable(self):
            return False

    cache = dpc.DiskBackedPromptCache(max_size=10, max_bytes=10 ** 12)
    with patch.object(dpc, "can_trim_prompt_cache", return_value=False):
        cache.insert_cache("m", [1, 2, 3], [C()], cache_type="system")
        cache.insert_cache("m", [1, 2, 3, 4], [C()], cache_type="user")
        cache.insert_cache("m", [1, 2, 3, 4, 5], [C()], cache_type="user")
    assert len(cache) == 2          # system + newest user
    assert cache.nbytes == 200
    got, rest = cache.fetch_nearest_cache("m", [1, 2, 3, 4, 5, 6])
    assert got is not None and rest == [6]
