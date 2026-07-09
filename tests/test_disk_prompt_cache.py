"""Regression + unit tests for core/inference/disk_prompt_cache.py.

Covers two real bugs from the mira-mlx work (2026-07-09): the prompt-cache
byte-cap silently evicting an entry right after inserting it (too-small
max_bytes), and the disk-overflow feature meant to soften that eviction
instead of discarding the entry outright. All model-free: uses mlx-lm's real
KVCache with small random tensors, no downloaded model needed.
"""
import pytest

mx = pytest.importorskip("mlx.core")  # mlx is macOS-only (Apple Silicon), absent on Linux CI
from mlx_lm.models.cache import KVCache

from core.inference.disk_prompt_cache import DiskBackedPromptCache, DiskPromptCacheStore


def make_kv_cache(n_tokens=8):
    c = KVCache()
    k = mx.random.normal((1, 4, n_tokens, 16))
    v = mx.random.normal((1, 4, n_tokens, 16))
    c.update_and_fetch(k, v)
    return [c]


# -- byte-cap eviction (regression: too-small max_bytes evicts on insert) ----

def test_undersized_max_bytes_evicts_immediately_without_disk_store():
    """The original bug shape: max_bytes smaller than a single entry causes
    insert_cache's own eviction loop to pop the entry right back out."""
    cache = DiskBackedPromptCache(max_size=10, max_bytes=1, disk_store=None)
    cache.insert_cache("model", list(range(100)), make_kv_cache())

    hit, rest = cache.fetch_nearest_cache("model", list(range(100)))
    assert hit is None
    assert len(rest) == 100


def test_adequately_sized_max_bytes_retains_entry():
    entry_bytes = sum(c.nbytes for c in make_kv_cache())
    cache = DiskBackedPromptCache(max_size=10, max_bytes=entry_bytes * 2, disk_store=None)
    cache.insert_cache("model", list(range(100)), make_kv_cache())

    hit, rest = cache.fetch_nearest_cache("model", list(range(100)))
    assert hit is not None
    assert rest == []


# -- disk overflow ------------------------------------------------------------

def test_evicted_entry_persists_to_disk_and_is_recoverable(tmp_path):
    store = DiskPromptCacheStore(cache_dir=tmp_path, max_bytes=10 * 1024 * 1024)
    one_entry = sum(c.nbytes for c in make_kv_cache())
    cache = DiskBackedPromptCache(max_size=10, max_bytes=int(one_entry * 1.5), disk_store=store)

    cache.insert_cache("model", list(range(200, 210)), make_kv_cache())
    cache.insert_cache("model", list(range(100)), make_kv_cache())  # evicts the first, over budget

    hit, rest = cache.fetch_nearest_cache("model", list(range(200, 210)))
    assert hit is not None, "evicted entry should be recoverable from disk instead of gone"
    assert rest == []


def test_disk_index_survives_simulated_restart(tmp_path):
    store = DiskPromptCacheStore(cache_dir=tmp_path, max_bytes=10 * 1024 * 1024)
    one_entry = sum(c.nbytes for c in make_kv_cache())
    cache = DiskBackedPromptCache(max_size=10, max_bytes=int(one_entry * 1.5), disk_store=store)
    cache.insert_cache("model", list(range(200, 210)), make_kv_cache())
    cache.insert_cache("model", list(range(100)), make_kv_cache())

    # A fresh process would construct a brand-new store pointed at the same
    # directory; its index must be rebuilt purely by scanning the directory.
    restarted_store = DiskPromptCacheStore(cache_dir=tmp_path, max_bytes=10 * 1024 * 1024)
    restarted_cache = DiskBackedPromptCache(max_size=10, max_bytes=1, disk_store=restarted_store)

    hit, rest = restarted_cache.fetch_nearest_cache("model", list(range(200, 210)))
    assert hit is not None
    assert rest == []


def test_cross_model_entries_never_collide(tmp_path):
    store = DiskPromptCacheStore(cache_dir=tmp_path, max_bytes=10 * 1024 * 1024)
    one_entry = sum(c.nbytes for c in make_kv_cache())
    cache = DiskBackedPromptCache(max_size=10, max_bytes=int(one_entry * 1.5), disk_store=store)
    cache.insert_cache("model-a", list(range(200, 210)), make_kv_cache())
    cache.insert_cache("model-a", list(range(100)), make_kv_cache())

    hit, rest = cache.fetch_nearest_cache("model-b", list(range(200, 210)))
    assert hit is None


def test_persist_skipped_when_disk_budget_is_zero(tmp_path):
    store = DiskPromptCacheStore(cache_dir=tmp_path, max_bytes=0)
    one_entry = sum(c.nbytes for c in make_kv_cache())
    cache = DiskBackedPromptCache(max_size=10, max_bytes=int(one_entry * 1.5), disk_store=store)
    cache.insert_cache("model", list(range(200, 210)), make_kv_cache())
    cache.insert_cache("model", list(range(100)), make_kv_cache())

    hit, rest = cache.fetch_nearest_cache("model", list(range(200, 210)))
    assert hit is None
    assert list(store.cache_dir.glob("*.safetensors")) == []
