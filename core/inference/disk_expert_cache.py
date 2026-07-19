"""Read-through disk store for MoE expert weights, sourced directly from the
model's existing safetensors shards — no repacking, no separate on-disk
format.

Unlike disk_prompt_cache.py (which persists cold KV *to* disk from an
always-authoritative in-memory source), this is a read-through cache *from*
a disk source of truth: the safetensors shards mlx_lm.utils.load() already
read at startup. A per-expert slice is fetched by computing its byte offset
within the stacked (num_experts, out, in) tensor and doing a plain seek+read
against the shard file — a spike against the actual Qwen3.6-35B-A3B-4bit
checkpoint measured 0.3-0.6ms per expert slice this way, well under the
tens-of-ms Eliseev & Mazur report for their (network-attached-storage)
setup, so no repacked layout is needed on this hardware.

fetch_fn() (from reader_for()) returns raw (numpy array, safetensors dtype
string) data, NOT mx.array — deliberately. The fork's offload code
(mlx_lm/models/switch_layers.py) parallelizes calls to fetch_fn across
threads to overlap disk I/O latency (a large/diverse prefill can mean tens
of thousands of misses in one forward pass; sequential reads there dominate
wall time), but MLX arrays are thread-affinity-sensitive — mira-mlx's engine
pins model execution to one dedicated thread, and an mx.array constructed on
a different thread crashes the first time it's used there
("RuntimeError: There is no Stream(gpu, N) in current thread", confirmed via
a real crash during Phase C validation). The disk I/O + numpy parsing here
is thread-safe (each call opens its own file handle, touches no shared
mutable state); mx.array construction is deferred to the fork's own
`_offload_to_mx()`, called back on the correct thread after the parallel
fetch completes — this file has no mlx-lm import, so that conversion can't
live here without creating a mira-core -> fork dependency in the wrong
direction.

See specs/moe-expert-offload-02-runtime-cache.md for the design this
implements (Phase A).
"""
import json
import logging
import struct
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# safetensors dtype string -> numpy dtype usable with np.frombuffer.
# BF16 has no native numpy dtype: read as raw uint16; the caller bit-casts to
# bfloat16 after mx.array construction (this string is what tells it to).
_NUMPY_DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # placeholder; caller bit-casts to bfloat16
    "U64": np.uint64,
    "I64": np.int64,
    "U32": np.uint32,
    "I32": np.int32,
    "U16": np.uint16,
    "I16": np.int16,
    "U8": np.uint8,
    "I8": np.int8,
    "BOOL": np.bool_,
}


class DiskExpertCacheStore:
    """Resolves (module_path, attr) -> safetensors shard + tensor metadata
    once, then serves fast per-expert byte-range reads against that shard.

    `model_path` is the local snapshot directory mlx_lm already loaded from
    (same directory that holds model.safetensors.index.json or a single
    *.safetensors file).
    """

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._weight_map: dict = {}
        self._single_shard: Optional[str] = None
        self._shard_headers: dict = {}  # shard filename -> (header dict, data_start)
        self._resolved_cache: dict = {}  # (module_path, attr) -> (shard, meta) or None
        # Hit/miss tracking lives per-module (mlx_lm's switch_layers.py owns
        # the cache dict — a hit never calls back into this store at all), so
        # this only ever counts real disk reads (cache misses across every
        # SwitchLinear/QuantizedSwitchLinear this store backs). Phase B's
        # /v1/stats wiring sums module._offload_hits across patched modules
        # for the hit side.
        self.misses = 0
        self._load_index()

    def _load_index(self) -> None:
        index_file = self.model_path / "model.safetensors.index.json"
        if index_file.exists():
            try:
                data = json.loads(index_file.read_text())
                self._weight_map = data.get("weight_map", {})
                return
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("expert cache: failed to read %s: %s", index_file, exc)
        shards = sorted(self.model_path.glob("*.safetensors"))
        if shards:
            self._single_shard = shards[0].name
        else:
            logger.warning("expert cache: no safetensors shards found under %s", self.model_path)

    def _header(self, shard_name: str):
        cached = self._shard_headers.get(shard_name)
        if cached is not None:
            return cached
        path = self.model_path / shard_name
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        result = (header, data_start)
        self._shard_headers[shard_name] = result
        return result

    def _resolve(self, module_path: str, attr: str):
        cache_key = (module_path, attr)
        if cache_key in self._resolved_cache:
            return self._resolved_cache[cache_key]

        suffix = f"{module_path}.{attr}"
        result = None
        if self._weight_map:
            for key, shard in self._weight_map.items():
                if key.endswith(suffix):
                    header, data_start = self._header(shard)
                    result = (shard, data_start, header[key])
                    break
        elif self._single_shard is not None:
            header, data_start = self._header(self._single_shard)
            for key in header:
                if key.endswith(suffix):
                    result = (self._single_shard, data_start, header[key])
                    break

        self._resolved_cache[cache_key] = result
        return result

    def _read_expert_slice_raw(self, shard: str, data_start: int, meta: dict, expert_id: int) -> Tuple[np.ndarray, str]:
        """Pure disk I/O + numpy — no mx calls, safe to run on any thread."""
        shape = meta["shape"]
        dtype = meta["dtype"]
        start, end = meta["data_offsets"]
        n_experts = shape[0]
        per_expert_bytes = (end - start) // n_experts
        offset = data_start + start + per_expert_bytes * expert_id
        path = self.model_path / shard
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read(per_expert_bytes)
        np_array = np.frombuffer(chunk, dtype=_NUMPY_DTYPES[dtype]).reshape(shape[1:])
        return np_array, dtype

    def reader_for(self, module_path: str, attrs: List[str]) -> Callable[[int], object]:
        """Build a fetch_fn(expert_id) closure for one SwitchLinear /
        QuantizedSwitchLinear instance, resolving each attr's shard/offset
        metadata once up front (cheap: one linear scan of the weight map per
        attr, only ever done at offload-enable time, not per token).

        Returns raw (np.ndarray, dtype_str) per attr — see module docstring
        for why mx.array construction is deliberately NOT done here."""
        resolved = []
        for attr in attrs:
            found = self._resolve(module_path, attr)
            if found is None and attr != "biases":
                raise KeyError(f"expert cache: no safetensors key found for suffix '{module_path}.{attr}'")
            resolved.append(found)

        def fetch(expert_id: int):
            out = []
            for found in resolved:
                if found is None:
                    out.append(None)
                    continue
                shard, data_start, meta = found
                out.append(self._read_expert_slice_raw(shard, data_start, meta, expert_id))
            self.misses += 1
            return out[0] if len(out) == 1 else tuple(out)

        return fetch
