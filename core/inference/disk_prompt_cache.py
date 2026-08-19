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

from mlx_lm.models.cache import (
    LRUPromptCache,
    PromptTrie,
    can_trim_prompt_cache,
    load_prompt_cache,
    save_prompt_cache,
)

logger = logging.getLogger(__name__)


def _key(model: Any, tokens: List[int], kv_bits: Optional[int] = None, kv_group_size: int = 64) -> str:
    h = hashlib.sha256()
    h.update(str(model).encode())
    h.update(b"\x00")
    h.update(",".join(map(str, tokens)).encode())
    # A quantized and unquantized entry for the same model+tokens are NOT
    # interchangeable (different cache class, different array shapes/dtypes) —
    # fold kv_bits/kv_group_size in so a config change (or a restart that
    # flips --kv-bits) can never load the wrong entry for a hash collision.
    h.update(b"\x00")
    h.update(f"{kv_bits}:{kv_group_size}".encode())
    return h.hexdigest()


class DiskPromptCacheStore:
    def __init__(self, cache_dir: Path, max_bytes: int, kv_bits: Optional[int] = None, kv_group_size: int = 64):
        self.cache_dir = Path(cache_dir)
        self.max_bytes = max_bytes
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
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
        key = _key(model, tokens, self.kv_bits, self.kv_group_size)
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
        key = _key(model, tokens, self.kv_bits, self.kv_group_size)
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
        if cache is None:
            self._explain_miss(model, tokens)
        if cache is not None or self.disk_store is None:
            return cache, rest
        # Full in-memory miss — try an exact disk hit before giving up.
        disk_cache = self.disk_store.load(model, tokens)
        if disk_cache is None:
            return cache, rest
        logger.info("disk prompt cache: exact-match hit for %d tokens", len(tokens))
        return disk_cache, []

    def insert_cache(self, model: Any, tokens: List[int], prompt_cache: List[Any],
                     *, cache_type: str = "assistant"):
        """Drop entries that can never be reused on a non-trimmable model.

        On a hybrid model (Qwen3.6: GatedDeltaNet layers make the whole cache
        non-trimmable) a non-"system" entry is reusable ONLY via fetch's
        `result.longer` trim path, which is gated on can_trim_prompt_cache() and so
        never fires — it becomes pure hold-and-evict deadweight (up to the derived
        pool budget, ~4.5GB on a 32GB Mac) that starves the prefill score transient.
        "system" entries are reused via a whole-prefix match (no trim needed — the
        system-checkpoint openers), so they are always kept. On a trimmable dense
        model the predicate is True and every class inserts exactly as before.
        """
        try:
            reusable = cache_type == "system" or can_trim_prompt_cache(prompt_cache)
        except Exception:  # noqa: BLE001 — never let a predicate error drop a working cache
            reusable = True
        if not reusable:
            logger.info(
                "insert_cache SKIPPED (non-trimmable, cache_type=%s): %d tokens "
                "would never be reused", cache_type, len(tokens),
            )
            return False
        super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)
        return True

    def _explain_miss(self, model: Any, tokens: List[int]) -> None:
        """Say WHY a lookup missed, which the hit/miss counters cannot.

        fetch_nearest_cache has two ways to reuse an entry and they fail for
        different reasons. `result.shorter` is an entry that is a *whole* prefix
        of this prompt, and with Qwen3's chat template that essentially never
        happens across turns: turn N's prompt ends with the generation prompt
        `<|im_start|>assistant\\n<think>\\n`, which the template does not
        reproduce when it replays that assistant turn from history, so the entry
        diverges a handful of tokens before its own end. Real reuse therefore
        comes from `result.longer`, an entry extending past the divergence that
        gets trimmed back to `common_prefix` — and that path is gated on
        can_trim_prompt_cache().

        So the two numbers worth having are the divergence index (already
        computed by search() and then discarded) and whether the extending entry
        was trimmable. "Not a prefix" is expected design; "not trimmable" is a
        bug, and without this line the two are indistinguishable."""
        try:
            result = self._trie.search(model, tokens)
            stored = len(self._lru) if hasattr(self._lru, "__len__") else "?"
            # An entry that merely *extends past* the divergence point is still
            # reusable: fetch_nearest_cache trims it back to common_prefix. That
            # path is gated on can_trim_prompt_cache(), so when it is False the
            # request loses every one of those tokens with no other explanation
            # in the log. Report it: "not a prefix" is design, "not trimmable"
            # is a bug.
            trimmable = None
            if result.longer is not None:
                try:
                    entry = self._trie.get(result.model, result.longer)
                    trimmable = can_trim_prompt_cache(entry.prompt_cache)
                    kinds = sorted({type(c).__name__ for c in entry.prompt_cache})
                except Exception:  # noqa: BLE001
                    kinds = ["?"]
            else:
                kinds = []
            logger.info(
                "cache miss detail: %d prompt tokens, diverged at index %s, "
                "longest whole-prefix entry=%s, extending entry=%s, trimmable=%s, "
                "cache_types=%s, entries_held=%s",
                len(tokens),
                result.common_prefix,
                len(result.shorter) if result.shorter else 0,
                len(result.longer) if result.longer else 0,
                trimmable,
                ",".join(kinds),
                stored,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must never break a request
            logger.debug("cache miss detail unavailable: %s", exc)
