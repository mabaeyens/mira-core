"""Disk-backed overflow for mlx-lm's in-memory LRUPromptCache.

An entry LRUPromptCache would otherwise discard (its own byte-budget eviction
in insert_cache(), or GenerationEngine's proactive trim_to() under memory
pressure) is written to disk instead, using mlx-lm's existing
save_prompt_cache/load_prompt_cache (safetensors). A later request that misses
the in-memory cache checks the disk store before falling back to a full
reprocess.

Entries are content-addressed by sha256(model_id + token ids) — the hash IS
the filename, so no separate sidecar index is needed: on startup the store
just re-derives its index by listing the cache directory, and a lookup
recomputes the same hash from the incoming (model, tokens) and does a
dictionary lookup. This also makes cross-model collisions a non-issue (model
id is part of the hashed input).
"""
import hashlib
import logging
import shutil
import time
from pathlib import Path
from typing import Any, List, Optional

from mlx_lm.models.cache import LRUPromptCache, PromptTrie, load_prompt_cache, save_prompt_cache

logger = logging.getLogger(__name__)


def _key(model: Any, tokens: List[int]) -> str:
    h = hashlib.sha256()
    h.update(str(model).encode())
    h.update(b"\x00")
    h.update(",".join(map(str, tokens)).encode())
    return h.hexdigest()


class DiskPromptCacheStore:
    def __init__(self, cache_dir: Path, max_bytes: int):
        self.cache_dir = Path(cache_dir)
        self.max_bytes = max_bytes
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = {}
        self.hits = 0  # diagnostics only (GET /v1/stats)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        for f in self.cache_dir.glob("*.safetensors"):
            try:
                st = f.stat()
            except OSError:
                continue
            self._index[f.stem] = {"path": f, "nbytes": st.st_size, "mtime": st.st_mtime}

    def _total_bytes(self) -> int:
        return sum(v["nbytes"] for v in self._index.values())

    def _evict_to_fit(self, incoming_bytes: int) -> None:
        if self._total_bytes() + incoming_bytes <= self.max_bytes:
            return
        for key, meta in sorted(self._index.items(), key=lambda kv: kv[1]["mtime"]):
            if self._total_bytes() + incoming_bytes <= self.max_bytes:
                break
            try:
                meta["path"].unlink(missing_ok=True)
            except OSError:
                pass
            del self._index[key]

    def persist(self, model: Any, tokens: List[int], prompt_cache: List[Any], nbytes: int) -> None:
        """Best-effort: never raises, a failed persist just means a future
        request reprocesses from scratch instead of hitting disk — no worse
        than today's discard-on-evict behavior."""
        if self.max_bytes <= 0 or nbytes <= 0:
            return
        try:
            free = shutil.disk_usage(self.cache_dir).free
        except OSError:
            return
        if nbytes > free:
            logger.warning("disk prompt cache: skipping persist, insufficient free space")
            return

        self._evict_to_fit(nbytes)
        key = _key(model, tokens)
        path = self.cache_dir / f"{key}.safetensors"
        try:
            save_prompt_cache(str(path), prompt_cache, metadata={"model": str(model)})
        except Exception as exc:  # noqa: BLE001 - best-effort persistence
            logger.warning("disk prompt cache: failed to persist entry: %s", exc)
            return
        try:
            actual_bytes = path.stat().st_size
        except OSError:
            actual_bytes = nbytes
        self._index[key] = {"path": path, "nbytes": actual_bytes, "mtime": time.time()}

    def load(self, model: Any, tokens: List[int]) -> Optional[List[Any]]:
        key = _key(model, tokens)
        meta = self._index.get(key)
        if meta is None:
            return None
        try:
            cache = load_prompt_cache(str(meta["path"]))
        except Exception as exc:  # noqa: BLE001 - a corrupt/missing file is just a miss
            logger.warning("disk prompt cache: failed to load entry (%s); dropping from index", exc)
            self._index.pop(key, None)
            return None
        meta["mtime"] = time.time()
        self.hits += 1
        return cache

    @property
    def nbytes(self) -> int:
        return self._total_bytes()


class _PersistingPromptTrie(PromptTrie):
    """Same as PromptTrie, but every entry popped out of the trie (i.e. every
    in-memory eviction) gets a chance to land on disk first."""

    def __init__(self, store: DiskPromptCacheStore):
        super().__init__()
        self._store = store

    def pop(self, model: Any, tokens: List[int]):
        entry = super().pop(model, tokens)
        self._store.persist(model, tokens, entry.prompt_cache, entry.nbytes)
        return entry


class DiskBackedPromptCache(LRUPromptCache):
    """LRUPromptCache whose evictions overflow to a DiskPromptCacheStore
    instead of being discarded, and whose fetch_nearest_cache() falls back to
    a disk hit (exact match only) on a full in-memory miss."""

    def __init__(self, max_size: int = 10, max_bytes: int = 1 << 63, disk_store: Optional[DiskPromptCacheStore] = None):
        super().__init__(max_size=max_size, max_bytes=max_bytes)
        self.disk_store = disk_store
        if disk_store is not None:
            self._trie = _PersistingPromptTrie(disk_store)

    def fetch_nearest_cache(self, model: Any, tokens: List[int]):
        cache, rest = super().fetch_nearest_cache(model, tokens)
        if cache is not None or self.disk_store is None:
            return cache, rest
        # Full in-memory miss — try an exact disk hit before giving up.
        disk_cache = self.disk_store.load(model, tokens)
        if disk_cache is None:
            return cache, rest
        logger.info("disk prompt cache: exact-match hit for %d tokens", len(tokens))
        return disk_cache, []
