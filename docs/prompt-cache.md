# The prompt cache

How mira-mlx reuses prefill work, why it currently cannot do so for plain
multi-turn chat on Qwen3.6, and what the 39.82GB of files on disk actually are.

Everything below was measured on 2026-08-08 (M5, 32GB, `mlx-community/Qwen3.6-35B-A3B-4bit`).
The engine now logs its own cache decisions to `~/.local/share/mira/mira-mlx.log`; before
`099d381` its stdout went to `DEVNULL` and none of this was observable.

## 1. Two layers

**In memory** — `LRUPromptCache` (mlx-lm), a trie of token sequences, budgeted by
`--prompt-cache-max-bytes` (currently **5.00GB**) and by entry count (`max_size`,
default 10). One entry is registered per completed request
(`mira_mlx_server.py:1112`), keyed on `all_tokens` = the prompt plus everything
generated.

**On disk** — `DiskPromptCacheStore` (`core/inference/disk_prompt_cache.py`).
Entries evicted from the trie are written here rather than discarded, via
`_PersistingPromptTrie.pop()`.

## 2. How a lookup can succeed, and why it usually doesn't

`fetch_nearest_cache` has exactly two ways to reuse an entry:

1. **Whole prefix** (`result.shorter`) — a stored entry that is entirely a prefix
   of the new prompt. Reuse is all-or-nothing: one differing token *anywhere*
   inside an entry drops reuse to zero, not to a partial match.
2. **Trim back** (`result.longer`) — an entry extending past the divergence,
   trimmed to the common prefix. Gated on `can_trim_prompt_cache()`.

**Path 2 is permanently unavailable on Qwen3.6.** It is `qwen3_5_moe`, a hybrid:
`make_cache()` returns `ArraysCache(size=2)` for linear-attention layers and
`KVCache()` for full-attention ones (`mlx_lm/models/qwen3_5.py:305`). Slot 0
holds the last `conv_kernel_size - 1` positions of the conv input, slot 1 the
Gated DeltaNet recurrent state. Both are running summaries with **no per-token
slots**, so removing the last N tokens' contribution is undefined without
replaying from a checkpoint. `ArraysCache.is_trimmable()` returning `False` is
correct, and since `can_trim_prompt_cache()` is an `all()`, one such layer
disables trimming for the whole entry. This applies to **any** hybrid
linear-attention model, not just this one. There is nothing for upstream to fix.

**So only path 1 is left, and the chat template breaks it.** A prompt ends with
the generation prompt `<|im_start|>assistant\n<think>\n`. When the template later
replays that assistant turn from history it emits the header and the content and
**never re-emits the scaffold**, so turn N's sequence is not a prefix of turn
N+1's.

| shape | is step N a prefix of step N+1? |
|---|---|
| agentic tool step (assistant carries `tool_calls`) | **yes** |
| plain chat turn | **no** |
| plain chat, `enable_thinking=False` | **no** (scaffold becomes `<think>\n\n</think>\n\n`) |
| plain chat, reasoning re-wrapped into `content` | **no** |

Pinned in `tests/test_template_prefix.py` and `tests/test_cache_trimmability.py`,
because neither fact is Mira's code and an mlx-lm upgrade could change either
silently, with a slow reply as the only symptom.

**Consequence:** agentic tool loops cache well; **plain multi-turn chat
re-prefills the entire conversation on every turn.** In the 2026-08-08 bench a
27,614-token second turn reused 0 tokens and took 48.7s.

## 3. Reading the log

Every request logs a hit/miss line, and every miss now also logs why:

```
cache MISS: 0/27621 prompt tokens reused (fetch_nearest_cache took 0.01s)
cache miss detail: 27621 prompt tokens, diverged at index 27503,
  longest whole-prefix entry=0, extending entry=27558, trimmable=False,
  cache_types=ArraysCache,RotatingQuantizedKVCache, entries_held=2
```

- `diverged at index` — where the prompt stops matching any stored path.
- `longest whole-prefix entry` — path 1's candidate; 0 means none.
- `extending entry` + `trimmable` — path 2's candidate and whether it was usable.
  **`trimmable=False` with a non-zero extending entry is the signature of the
  problem above**, not of a corrupt cache.

`/v1/stats` carries `cache_hits`, `cache_misses` and `disk_cache_hits`, but
cannot distinguish these cases; the log is the only place that can.

## 4. The files on disk

Location `~/.local/share/mira/mira_mlx_cache`, one `.safetensors` per entry.

| | |
|---|---|
| entries | 302 |
| total | 39.82 GB, against a 39.86 GB cap (`hardware.derive_disk_cache_max_bytes`) |
| entry size | min 61.7 MB, median 102.1 MB, max **2,103.5 MB** |
| accumulated since | 2026-07-18 |
| reads served, ever | **0** |

**Filename is the key**: `sha256(model_id + token_ids + "kv_bits:kv_group_size")`,
so the store needs no sidecar index — it rebuilds by listing the directory
(`_rebuild_index`). Folding `kv_bits`/`kv_group_size` in is deliberate: a
quantized and an unquantized entry for the same tokens are not interchangeable.

**Contents of one entry**: one serialized cache per layer. For Qwen3.6 that is 40
layer caches — **30 `ArraysCache` + 10 `RotatingQuantizedKVCache`** — 120 tensors
in a median file. A 102.1 MB file holds 61.4 MB of arrays once loaded; loading it
takes ~0.02s.

**Provenance is thin.** The safetensors metadata carries the model id
(`{'model': 'mlx-community/Qwen3.6-35B-A3B-4bit'}`) and nothing else — no token
count, no conversation, no creation time beyond the file's mtime. You cannot tell
what an entry was for without loading it and counting tensors.

**They are never read.** `DiskBackedPromptCache.fetch_nearest_cache` falls back to
`disk_store.load()`, which is an **exact match on the full token list**
(`disk_prompt_cache.py:106`). A hit needs a byte-identical repeat of an entire
prompt, which outside a regenerate does not happen. The store is at its cap and
evicting old entries to make room for new ones that will also never be read.

**The pending decision** is therefore: make the disk layer prefix-capable, or
delete it and reclaim 39.82GB. Leaving it as-is is the one option that is
definitely wrong. A prefix-capable disk layer must be costed first — a load is
~0.02s for a median entry, but an index over 302 entries that compares candidates
by loading them is a different proposition from one hash lookup.

## 5. Copies are cheap, which matters for any fix

`fetch_nearest_cache` already `deepcopy`s an entry on every hit before handing it
to the generator. Measured on the largest entry on disk (2,103.5 MB):

| | |
|---|---|
| `copy.deepcopy` wall | **0.8 ms** |
| MLX active memory delta | **+0.0 MB** |

`mx.array` implements `__deepcopy__`, and mutating a copy does not disturb the
original — verified for both `KVCache` (which appends) and `ArraysCache` (whose
slots are reassigned wholesale). So **duplicating a cache entry costs a
reference, not bytes or time**; what costs is *retaining* it, since a held entry
keeps its buffers alive and occupies one of the LRU's 10 slots and part of its
5.00GB budget.

That is what makes the fix in `specs/assistant-boundary-snapshot.md` plausible.
