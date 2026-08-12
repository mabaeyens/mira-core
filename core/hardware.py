"""RAM-aware sizing for local inference: total memory detection and per-model
KV-cache/context budgeting, so the same code behaves sensibly on an 8GB Mac
and a 128GB Mac Studio instead of using one fixed set of constants everywhere.

A prompt whose own token count is >= the derived max_kv_size ceiling is
rejected outright by GenerationEngine._admit_job (mira_mlx_server.py) with a
clear ValueError, rather than left to RotatingKVCache's undefined behavior for
a single over-budget submission.
"""
import ctypes
import ctypes.util
import json
import logging
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3
KV_DTYPE_BYTES = 2  # mlx_lm's KV cache is fp16 by default regardless of model quantization

# Held back from every RAM-based budget for the OS, other apps, and Mira's own
# non-model processes (embeddings, RAG, orchestrator). Deliberately conservative.
SAFETY_MARGIN_BYTES = 3 * BYTES_PER_GB

# Apple-silicon Metal caps a process's wired working set at roughly 78% of unified
# RAM (recommendedMaxWorkingSetSize; measured 24.96GB on this 32GB Mac). Resident
# weights past it force a Metal OOM regardless of free RAM, so RAM-aware expert
# sizing bounds peak against this as a hard upper limit.
METAL_WIRED_FRACTION = 0.78
# Extra headroom kept below the wired ceiling for byte-model estimation error.
WIRED_HEADROOM_BYTES = 3 * BYTES_PER_GB
# RAM-aware sizing's PRIMARY (lower) target: keep peak footprint at/below this
# fraction of unified RAM, leaving the rest free for the prefill transient gather.
# Pushing residency to the wired ceiling (f~0.59 on the 8bit) gains more decode
# but starves that transient and costs ~-36% prefill/TTFT from memory pressure;
# capping peak near 55% of RAM (f~0.45) keeps prefill healthy while still lifting
# decode (measured 2026-07-19). Deliberately conservative; raise to trade TTFT
# for decode throughput.
RAM_AWARE_PEAK_FRACTION = 0.55


def get_total_ram_bytes() -> int:
    """Total physical RAM on this Mac, via `sysctl hw.memsize` (no new dependency)."""
    try:
        # Absolute path: launchd-managed processes run with a minimal PATH that
        # doesn't include /usr/sbin, so bare "sysctl" silently FileNotFoundErrors
        # and this would fall through to the 16GB guess below on the real server
        # (confirmed 2026-07-09 — found via a live/interactive-shell discrepancy).
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5, check=True
        )
        return int(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        logger.warning("Could not read hw.memsize (%s); assuming 16GB", exc)
        return 16 * BYTES_PER_GB


# Lowest the dynamic ceiling is ever allowed to fall to. Below this the model
# cannot serve a useful turn anyway, so reporting a smaller budget would only
# make Mira thrash its own caches while the real problem is elsewhere.
MIN_DYNAMIC_CEILING_BYTES = 6 * BYTES_PER_GB

# macOS memory pressure levels from kern.memorystatus_vm_pressure_level.
PRESSURE_NORMAL, PRESSURE_WARN, PRESSURE_CRITICAL = 1, 2, 4


def read_memory_pressure_level() -> Optional[int]:
    """macOS's own verdict: 1 normal, 2 warn, 4 critical. None if unreadable.

    None means unknown, never healthy — a probe that fails must not read as an
    all-clear (this is the whole reason it is Optional rather than defaulting).
    """
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return int(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        logger.debug("Could not read memory pressure level (%s)", exc)
        return None


def read_swap_used_bytes() -> Optional[int]:
    """Swap currently in use, via `sysctl vm.swapusage`. None if unreadable.

    Only useful as a delta. This machine sits at ~800MB of swap in use while
    completely idle, so an absolute value says nothing about pressure.
    """
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        # "total = 2048.00M  used = 796.12M  free = 1251.88M  (encrypted)"
        parts = out.stdout.replace("=", " ").split()
        i = parts.index("used")
        raw = parts[i + 1]
        mult = {"K": 1024, "M": 1024**2, "G": 1024**3}.get(raw[-1].upper(), 1)
        return int(float(raw.rstrip("KMGkmg")) * mult)
    except (subprocess.SubprocessError, ValueError, OSError, IndexError) as exc:
        logger.debug("Could not read swap usage (%s)", exc)
        return None


def read_vm_state() -> Optional[dict]:
    """One `vm_stat` call, returning both availability and compressor occupancy.

    `available` is free + inactive + speculative. `inactive` is the load-bearing
    term and omitting it is the classic mistake: on this machine free alone reads
    0.06GB where several GB is genuinely available, which would make any budget
    derived from it conclude the machine is permanently starving. `purgeable` is
    deliberately NOT added, because it is already counted inside active/inactive.

    `compressor_bytes` is the one that actually matters and is why this returns a
    dict rather than a single number. Measured 2026-08-08: under external memory
    pressure macOS compressed 21.4GB of pages into 17.05GB of compressor, and
    most of it was Mira's own model weights. The next request then took **15.37s
    against a warm 0.47s, a 33x penalty**, and decompressing the model emptied
    the compressor back to 0.20GB in that one turn. Availability is the leading
    indicator; compressor occupancy is the damage already done.
    """
    try:
        out = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=5, check=True
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("Could not run vm_stat (%s)", exc)
        return None

    page_size = 16384
    counts = {}
    for line in out.stdout.splitlines():
        if "page size of" in line:
            try:
                page_size = int(line.split("page size of")[1].split()[0])
            except (IndexError, ValueError):
                pass
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if value.isdigit():
            counts[key.strip().lower()] = int(value)

    try:
        available = (counts["pages free"] + counts["pages inactive"]
                     + counts.get("pages speculative", 0)) * page_size
    except KeyError as exc:
        logger.debug("vm_stat missing expected field (%s)", exc)
        return None
    return {
        "available_bytes": available,
        "compressor_bytes": counts.get("pages occupied by compressor", 0) * page_size,
        "wired_bytes": counts.get("pages wired down", 0) * page_size,
    }


def on_battery() -> bool:
    """True only when macOS reports the Mac is running on battery.

    Returns False on any doubt, including an unreadable pmset: this gates
    optional background work, so "could not tell" must not silently disable a
    feature the way a None-means-maybe would.
    """
    try:
        out = subprocess.run(
            ["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, timeout=5, check=True
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("could not read pmset (%s); assuming wall power", exc)
        return False
    return "Battery Power" in out.stdout


def read_available_ram_bytes() -> Optional[int]:
    """Availability alone, for callers that do not need the full state."""
    state = read_vm_state()
    return None if state is None else state["available_bytes"]


# --- Per-process eviction, via task_info(TASK_VM_INFO) -----------------------
#
# The system-wide compressor figure cannot say WHOSE pages are compressed, so the
# fraction heuristic below is a guess with a wide margin around it. task_info on
# mach_task_self() answers for this process exactly, needs no entitlement for
# your own task, and costs ~1.4us against ~1s for a `vmmap -summary` subprocess.
#
# Validated 2026-08-08 against both instruments at once: a 6GB victim process
# self-reported 1.75GB compressed while an external `vmmap` read SWAPPED as
# 1.6GiB (= 1.72GB) and `footprint -p` agreed on the 6.46GB footprint.
_TASK_VM_INFO = 22
_KERN_SUCCESS = 0

_mach_vm_size_t = ctypes.c_uint64
_integer_t = ctypes.c_int32


class _TaskVMInfo(ctypes.Structure):
    """task_vm_info through rev3.

    The rev0 prefix is not enough. Its `compressed` and `resident_size` are pmap
    statistics (task.c:5303 fills them from `map->pmap->stats`), and the pmap
    does not account IOKit/Metal mappings, which is where essentially all of
    Mira's memory lives. Both therefore misreport for the engine: `compressed`
    read 16.19GB on a fully resident model, and `resident_size` reads near zero
    against a 19GB footprint. It is the same blind spot that makes `ps` RSS
    report 8.91GB for a process MLX says is holding 19.66GB.

    The rev3 tail carries LEDGER entries instead, including the graphics tags,
    and ledgers are what actually track this memory.
    """

    _fields_ = [
        ("virtual_size", _mach_vm_size_t),
        ("region_count", _integer_t),
        ("page_size", _integer_t),
        ("resident_size", _mach_vm_size_t),
        ("resident_size_peak", _mach_vm_size_t),
        ("device", _mach_vm_size_t),
        ("device_peak", _mach_vm_size_t),
        ("internal", _mach_vm_size_t),
        ("internal_peak", _mach_vm_size_t),
        ("external", _mach_vm_size_t),
        ("external_peak", _mach_vm_size_t),
        ("reusable", _mach_vm_size_t),
        ("reusable_peak", _mach_vm_size_t),
        ("purgeable_volatile_pmap", _mach_vm_size_t),
        ("purgeable_volatile_resident", _mach_vm_size_t),
        ("purgeable_volatile_virtual", _mach_vm_size_t),
        ("compressed", _mach_vm_size_t),
        ("compressed_peak", _mach_vm_size_t),
        ("compressed_lifetime", _mach_vm_size_t),
        # --- rev1 ---
        ("phys_footprint", _mach_vm_size_t),
        # --- rev2 ---
        ("min_address", ctypes.c_uint64),
        ("max_address", ctypes.c_uint64),
        # --- rev3: ledger entries, which unlike the pmap stats above DO account
        # IOKit/Metal memory. The graphics pair is the one that matters here.
        ("ledger_phys_footprint_peak", ctypes.c_int64),
        ("ledger_purgeable_nonvolatile", ctypes.c_int64),
        ("ledger_purgeable_nonvolatile_compressed", ctypes.c_int64),
        ("ledger_purgeable_volatile", ctypes.c_int64),
        ("ledger_purgeable_volatile_compressed", ctypes.c_int64),
        ("ledger_tag_network_nonvolatile", ctypes.c_int64),
        ("ledger_tag_network_nonvolatile_compressed", ctypes.c_int64),
        ("ledger_tag_network_volatile", ctypes.c_int64),
        ("ledger_tag_network_volatile_compressed", ctypes.c_int64),
        ("ledger_tag_media_footprint", ctypes.c_int64),
        ("ledger_tag_media_footprint_compressed", ctypes.c_int64),
        ("ledger_tag_media_nofootprint", ctypes.c_int64),
        ("ledger_tag_media_nofootprint_compressed", ctypes.c_int64),
        ("ledger_tag_graphics_footprint", ctypes.c_int64),
        ("ledger_tag_graphics_footprint_compressed", ctypes.c_int64),
        ("ledger_tag_graphics_nofootprint", ctypes.c_int64),
        ("ledger_tag_graphics_nofootprint_compressed", ctypes.c_int64),
        ("ledger_tag_neural_footprint", ctypes.c_int64),
        ("ledger_tag_neural_footprint_compressed", ctypes.c_int64),
        ("ledger_tag_neural_nofootprint", ctypes.c_int64),
        ("ledger_tag_neural_nofootprint_compressed", ctypes.c_int64),
    ]


_TASK_VM_INFO_COUNT = ctypes.sizeof(_TaskVMInfo) // ctypes.sizeof(_integer_t)
_libsystem_handle = None
_libsystem_tried = False


def _libsystem():
    """libSystem with task_info bound, or None off-darwin / if it cannot load.

    Loaded lazily and remembered, including the failure: this is called from the
    engine's idle branch and must never pay for a retry it already lost.
    """
    global _libsystem_handle, _libsystem_tried
    if _libsystem_tried:
        return _libsystem_handle
    _libsystem_tried = True
    if sys.platform != "darwin":
        return None
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        lib.mach_task_self.restype = ctypes.c_uint32
        lib.task_info.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(_integer_t),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.task_info.restype = ctypes.c_int
        _libsystem_handle = lib
    except (OSError, AttributeError) as exc:
        logger.debug("task_info unavailable (%s); falling back to system-wide signals", exc)
    return _libsystem_handle


def read_self_memory_state() -> Optional[dict]:
    """This process's own resident and paged-out bytes. None if unreadable.

    `compressed_bytes` comes from the GRAPHICS LEDGER entries, not from the
    struct's own `compressed` field and not from `phys_footprint - resident_size`.
    Both of those were tried against the live engine and both are wrong; the raw
    values are returned alongside so the difference stays visible.

    XNU fills this struct from two different accounting systems. The rev0 fields
    are PMAP STATISTICS, set by a macro over `map->pmap->stats` (task.c:5303):

        #define _VM_INFO(_name) \\
            vm_info->_name = ((mach_vm_size_t) map->pmap->stats._name) * PAGE_SIZE

    The pmap does not account IOKit/Metal mappings, which is where essentially
    all of Mira's memory lives, so for this process both of those fields lie.
    Measured on the live engine while the model was fully resident and replying
    in 0.40s, against `vmmap` reporting 292MB swapped and 18.9G resident:

        compressed      (pmap)   15.46 GB   <- would be a permanent false alarm
        resident_size   (pmap)    4.65 GB   <- undercounts by ~14GB
        phys_footprint  (ledger) 20.28 GB   <- correct total, but counts a page
                                               whether resident or compressed,
                                               so footprint-minus-resident just
                                               inherits the resident_size error

    The rev3 tail carries LEDGER entries, which do account this memory:

        graphics_footprint             19.76 GB  <- matches MLX's own 19.66 GB
        graphics_footprint_compressed   0.00 GB  <- correct: nothing paged out

    and under real eviction the compressed entry rises and falls with the model,
    verified by a touch that reclaimed 8.34GB of it in 1.82s.

    Returns None rather than zeros off-darwin or on any failure, so a caller
    cannot mistake "could not measure" for "nothing is compressed" - the same
    rule the system-wide probes follow.

    NOTE this answers for the CALLING process. It belongs to whichever process
    actually holds the weights, which is the mira-mlx engine, not the API server.
    """
    lib = _libsystem()
    if lib is None:
        return None
    info = _TaskVMInfo()
    count = ctypes.c_uint32(_TASK_VM_INFO_COUNT)
    try:
        rc = lib.task_info(
            lib.mach_task_self(),
            _TASK_VM_INFO,
            ctypes.cast(ctypes.byref(info), ctypes.POINTER(_integer_t)),
            ctypes.byref(count),
        )
    except OSError as exc:
        logger.debug("task_info raised (%s)", exc)
        return None
    if rc != _KERN_SUCCESS:
        logger.debug("task_info(TASK_VM_INFO) returned %d", rc)
        return None
    graphics_compressed = (
        int(info.ledger_tag_graphics_footprint_compressed)
        + int(info.ledger_tag_graphics_nofootprint_compressed)
    )
    return {
        "compressed_bytes": max(graphics_compressed, 0),
        "resident_bytes": int(info.resident_size),
        "footprint_bytes": int(info.phys_footprint),
        # Diagnostics. Every one of these was a candidate for the verdict above
        # and each is kept so the next person can see why it is not used.
        "pmap_compressed_bytes": int(info.compressed),
        "graphics_footprint_bytes": int(info.ledger_tag_graphics_footprint),
        "graphics_footprint_compressed_bytes": int(
            info.ledger_tag_graphics_footprint_compressed),
        "graphics_nofootprint_bytes": int(info.ledger_tag_graphics_nofootprint),
        "graphics_nofootprint_compressed_bytes": int(
            info.ledger_tag_graphics_nofootprint_compressed),
    }


# Mira's OWN compressed graphics bytes above this fraction of its MLX footprint
# means the model has been paged out. Measured on this 32GB Mac 2026-08-08
# against a ~19.7GB graphics footprint: the ledger reads exactly 0 whenever the
# model is warm, and rose past 8.34GB under a hog before the idle touch cleared
# it. There is no grey zone to tune around, so this only has to sit somewhere
# between "nothing" and "gigabytes"; it is not a threshold the way the
# system-wide fallback below is.
EVICTED_SELF_FRACTION = 0.25

# Fallback for when the per-process reading is unavailable. Compressor occupancy
# above this fraction of Mira's own MLX footprint means a large share of the
# model is very likely paged out. Chosen with a wide margin around the two
# measured states: 0.91 when evicted (17.05GB against 18.81GB) and 0.01 when
# resident (0.20GB). Nothing observed has landed between them. This is a guess
# about attribution in a way EVICTED_SELF_FRACTION is not: the compressor is
# system-wide, so another app's compressed pages read here as Mira's.
EVICTED_COMPRESSOR_FRACTION = 0.25


def derive_dynamic_ceiling_bytes(
    mira_used_bytes: int,
    total_ram_bytes: Optional[int] = None,
    available_bytes: Optional[int] = None,
    self_compressed_bytes: Optional[int] = None,
) -> tuple[int, dict]:
    """How much MLX memory Mira may hold, given what the rest of the Mac is doing.

    The static ceiling (`total - SAFETY_MARGIN`) assumes the machine is Mira's
    alone. It is not; this is a computer someone also uses. This returns the same
    value when nothing else is running and a smaller one when something is.

    `mira_used_bytes` is Mira's own MLX footprint (active + cache) and is ADDED
    back, because it already shows up as unavailable in the system's numbers.
    Leaving it out makes the budget shrink in response to Mira's own size, which
    shrinks nothing and then shrinks again — a feedback loop, not a measurement.

    Never exceeds the static ceiling: a transient "lots free" reading must not
    let the model claim memory it cannot keep. Never falls below
    MIN_DYNAMIC_CEILING_BYTES.

    `self_compressed_bytes` is the CALLER's own compressed bytes from
    read_self_memory_state(). Pass it and the eviction verdict becomes a fact
    about this process; omit it and the verdict falls back to a system-wide
    compressor heuristic that cannot tell Mira's compressed pages from Xcode's.
    It is a parameter rather than a probe made in here because this function is
    also reachable from the API server process, which holds no weights and whose
    self-reading would be meaningless.

    Returns (ceiling_bytes, diagnostics) so callers can report WHY it moved
    rather than just that it did.
    """
    total = total_ram_bytes if total_ram_bytes is not None else get_total_ram_bytes()
    static_ceiling = total - SAFETY_MARGIN_BYTES
    state = read_vm_state()
    if state is None and available_bytes is not None:
        state = {"available_bytes": available_bytes, "compressor_bytes": 0}
    elif state is not None and available_bytes is not None:
        state = dict(state, available_bytes=available_bytes)

    pressure = read_memory_pressure_level()
    diag = {
        "static_ceiling_bytes": static_ceiling,
        "available_bytes": None if state is None else state["available_bytes"],
        "compressor_bytes": None if state is None else state["compressor_bytes"],
        "mira_used_bytes": mira_used_bytes,
        "pressure_level": pressure,
        # Overwritten below once an eviction verdict is actually reached. Present
        # here so every return path has the same shape, including the early one
        # where the probe failed and no verdict exists.
        "self_compressed_bytes": self_compressed_bytes,
        "eviction_signal": None,
    }

    if state is None:
        # Unknown, not healthy. Fall back to the static ceiling and say so, so a
        # broken probe is visible instead of silently reading as an all-clear.
        diag["source"] = "static (probe unavailable)"
        diag["advisory"] = "unknown"
        return static_ceiling, diag

    available = state["available_bytes"]
    headroom = mira_used_bytes + available - SAFETY_MARGIN_BYTES
    ceiling = max(min(headroom, static_ceiling), MIN_DYNAMIC_CEILING_BYTES)
    diag["source"] = "dynamic"
    diag["other_processes_bytes"] = max(total - available - mira_used_bytes, 0)
    diag["ceiling_bytes"] = ceiling

    # Advisory, in increasing order of how much the user will actually feel it.
    # `evicted` is the one worth surfacing: it means the next reply pays to fault
    # the whole model back in. Measured 2026-08-08 on an unforced eviction, all
    # 18.80GB compressed: 17.60s against a warm 0.45s. A half-evicted model cost
    # 3.38s, so the penalty scales with how much went out and this is not a
    # binary the user experiences as one fixed cost.
    #
    # Prefer the per-process reading. The system-wide compressor cannot attribute
    # anything: a machine where something ELSE is compressed reads identically.
    if self_compressed_bytes is not None and mira_used_bytes > 0:
        diag["self_compressed_bytes"] = self_compressed_bytes
        diag["eviction_signal"] = "self"
        evicted = self_compressed_bytes > EVICTED_SELF_FRACTION * mira_used_bytes
    else:
        diag["self_compressed_bytes"] = None
        diag["eviction_signal"] = "system-wide (per-process unavailable)"
        evicted = (mira_used_bytes > 0
                   and state["compressor_bytes"] > EVICTED_COMPRESSOR_FRACTION * mira_used_bytes)
    if evicted:
        diag["advisory"] = "evicted"
    elif pressure is not None and pressure >= PRESSURE_CRITICAL:
        diag["advisory"] = "critical"
    elif pressure is not None and pressure >= PRESSURE_WARN:
        diag["advisory"] = "busy"
    elif pressure is None:
        diag["advisory"] = "unknown"
    else:
        diag["advisory"] = "ok"
    return ceiling, diag


def _find_cached_config(model_id: str) -> Optional[Path]:
    """Locate a HF-cached model's config.json without invoking transformers/mlx_lm."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    if not repo_dir.exists():
        return None
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return None
    for snapshot in snapshots.iterdir():
        candidate = snapshot / "config.json"
        if candidate.exists():
            return candidate
    return None


def estimate_kv_bytes_per_token(
    model_id: str, kv_bits: Optional[int] = None, kv_group_size: int = 64
) -> Optional[int]:
    """KV cache bytes for one token of context, derived from the model's own config.

    Formula (validated 2026-07-09 against a live measurement of Ministral 3 14B —
    computed 163,840 B/token vs. a measured 163,820 B/token): for each of K and V,
    one value per layer per KV head per head-dim element, at KV_DTYPE_BYTES each
    (KV cache stays fp16 by default regardless of model quantization).

    kv_bits: when set (mira-mlx's --kv-bits), each element instead costs
    bits/8 packed bytes plus a per-group fp16 scale+bias amortized across
    kv_group_size elements — mirrors mlx_lm.models.cache.QuantizedKVCache's
    own packing (mx.quantize), not an independent guess.
    """
    config_path = _find_cached_config(model_id)
    if config_path is None:
        return None
    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    # Multimodal architectures (e.g. mistral3) nest the actual LM config under
    # text_config; text-only models have these keys at the top level.
    cfg = raw.get("text_config", raw)
    try:
        num_layers = cfg["num_hidden_layers"]
        num_kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    except KeyError:
        return None

    if kv_bits is not None:
        bytes_per_element = kv_bits / 8 + (2 * KV_DTYPE_BYTES) / kv_group_size
    else:
        bytes_per_element = KV_DTYPE_BYTES

    return int(2 * num_layers * num_kv_heads * head_dim * bytes_per_element)


def estimate_model_weight_bytes(model_id: str) -> Optional[int]:
    """On-disk size of a HF-cached model's weight shards (already downloaded)."""
    config_path = _find_cached_config(model_id)
    if config_path is None:
        return None
    snapshot_dir = config_path.parent
    total = sum(
        f.stat().st_size for f in snapshot_dir.glob("*.safetensors") if f.is_file()
    )
    return total or None


def _classify_weight_bytes(snapshot_dir: Path, num_experts: int):
    """Split on-disk shard bytes into (per-expert-stacked, everything else)
    by reading safetensors headers only (no tensor data read) — a tensor
    whose first dimension equals num_experts and has >= 3 dims is a stacked
    per-expert tensor, matching the shape mlx_lm's SwitchLinear/
    QuantizedSwitchLinear always store (num_experts, out, in). Returns
    (None, None) if bytes can't be classified (caller falls back to full
    on-disk size rather than guessing)."""
    shards = sorted(snapshot_dir.glob("*.safetensors"))
    if not shards:
        return None, None
    expert_bytes = 0
    other_bytes = 0
    try:
        for shard in shards:
            with open(shard, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                shape = meta.get("shape")
                offsets = meta.get("data_offsets")
                if not shape or not offsets:
                    continue
                nbytes = offsets[1] - offsets[0]
                if len(shape) >= 3 and shape[0] == num_experts:
                    expert_bytes += nbytes
                else:
                    other_bytes += nbytes
    except (OSError, json.JSONDecodeError, struct.error) as exc:
        logger.warning("hardware: failed to classify weight bytes at %s: %s", snapshot_dir, exc)
        return None, None
    return expert_bytes, other_bytes


def estimate_active_weight_bytes(model_id: str, resident_expert_fraction: Optional[float] = None) -> Optional[int]:
    """Resident weight footprint with MoE expert offloading active: every
    non-expert byte stays resident as before, but only
    resident_expert_fraction of each per-expert stacked tensor's bytes do.

    Returns the same value as estimate_model_weight_bytes() (offloading has
    no effect on the budget) when resident_expert_fraction is None or >= 1.0,
    or when the model isn't MoE / bytes can't be classified — this is what
    keeps every derive_* caller's default behavior unchanged unless a caller
    explicitly opts in.
    """
    total_bytes = estimate_model_weight_bytes(model_id)
    if total_bytes is None:
        return None
    if resident_expert_fraction is None or resident_expert_fraction >= 1.0:
        return total_bytes

    config_path = _find_cached_config(model_id)
    if config_path is None:
        return total_bytes
    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return total_bytes
    cfg = raw.get("text_config", raw)
    num_experts = cfg.get("num_experts")
    if not num_experts:
        return total_bytes  # dense model — offloading doesn't apply

    expert_bytes, other_bytes = _classify_weight_bytes(config_path.parent, num_experts)
    if expert_bytes is None:
        return total_bytes

    return int(other_bytes + expert_bytes * resident_expert_fraction)


def derive_resident_expert_fraction(
    model_id: str,
    floor_fraction: float,
    total_ram_bytes: Optional[int] = None,
    max_fraction: float = 0.85,
) -> float:
    """Largest resident-expert fraction whose peak footprint stays under a safe
    ceiling, for an over-DRAM MoE model that is being offloaded anyway.

    Rationale: offload defaults to a flat 0.3, but an over-DRAM model leaves RAM
    idle — the 8bit Qwen3.6 peaks ~12.7GB on a 32GB Mac at 0.3, ~19GB unused.
    Raising the resident fraction cashes that headroom for decode throughput
    (measured +26% at 0.5, no prediction, no quality change). This returns the
    highest fraction whose resident-weight bytes stay under the Metal wired
    ceiling with headroom. It NEVER returns below floor_fraction (the configured
    knob), so it can only raise residency, never lower it.

    The on-disk resident-weight bytes track measured PEAK closely (byte-model
    19.2GB vs measured 19.07GB at f=0.5), so peak is bounded directly on them.
    Falls back to floor_fraction whenever the model can't be classified as MoE.
    """
    total_ram_bytes = total_ram_bytes if total_ram_bytes is not None else get_total_ram_bytes()
    config_path = _find_cached_config(model_id)
    if config_path is None:
        return floor_fraction
    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return floor_fraction
    cfg = raw.get("text_config", raw)
    num_experts = cfg.get("num_experts")
    if not num_experts:
        return floor_fraction  # dense model — offloading doesn't apply
    expert_bytes, other_bytes = _classify_weight_bytes(config_path.parent, num_experts)
    if not expert_bytes:
        return floor_fraction

    ceiling = min(
        int(total_ram_bytes * RAM_AWARE_PEAK_FRACTION),          # primary target: leave prefill headroom
        int(total_ram_bytes * METAL_WIRED_FRACTION) - WIRED_HEADROOM_BYTES,  # hard wired-limit cap
        total_ram_bytes - SAFETY_MARGIN_BYTES,
    )
    budget_for_experts = ceiling - other_bytes
    if budget_for_experts <= 0:
        return floor_fraction
    f = budget_for_experts / expert_bytes
    return round(max(floor_fraction, min(f, max_fraction)), 3)


def derive_prompt_cache_max_bytes(
    model_id: str,
    total_ram_bytes: Optional[int] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    resident_expert_fraction: Optional[float] = None,
) -> int:
    """Safe prompt-cache pool size: whatever's left after the model, KV working
    set, and a fixed safety margin, floored so it's never negative/degenerate."""
    total_ram_bytes = total_ram_bytes if total_ram_bytes is not None else get_total_ram_bytes()
    model_bytes = estimate_active_weight_bytes(model_id, resident_expert_fraction) or 8 * BYTES_PER_GB  # conservative guess
    available = total_ram_bytes - model_bytes - SAFETY_MARGIN_BYTES
    # Leave room for at least the active generation's own KV cache on top of the
    # cache *pool* — cap the pool at half of whatever's left.
    budget = max(available // 2, 512 * 1024 * 1024)
    return int(budget)


def derive_context_window(model_id: str, total_ram_bytes: Optional[int] = None,
                           requested_context: int = 65536,
                           kv_bits: Optional[int] = None, kv_group_size: int = 64,
                           resident_expert_fraction: Optional[float] = None) -> int:
    """Cap a requested context window to what this machine can actually hold in
    KV cache for a single active generation, without touching the cache pool."""
    total_ram_bytes = total_ram_bytes if total_ram_bytes is not None else get_total_ram_bytes()
    model_bytes = estimate_active_weight_bytes(model_id, resident_expert_fraction) or 8 * BYTES_PER_GB
    kv_bytes_per_token = estimate_kv_bytes_per_token(model_id, kv_bits=kv_bits, kv_group_size=kv_group_size)
    if kv_bytes_per_token is None:
        return requested_context  # unknown architecture — don't guess, use the caller's value

    available = total_ram_bytes - model_bytes - SAFETY_MARGIN_BYTES
    max_tokens_by_ram = available // kv_bytes_per_token
    return int(max(min(requested_context, max_tokens_by_ram), 1024))


def derive_disk_cache_max_bytes(cache_dir: Path, cap_bytes: int = 50 * BYTES_PER_GB) -> int:
    """Disk-backed prompt-cache budget: min(cap, 10% of free space) at cache_dir's
    volume — disk is comfortable on most Macs (measured 513GB free/926GB total,
    3% used, on the 2026-07-09 dev machine) but must still be bounded and re-
    checked live, not assumed fixed, since free space can drop between calls."""
    try:
        probe = cache_dir
        while not probe.exists():
            probe = probe.parent
        free = shutil.disk_usage(probe).free
    except OSError as exc:
        logger.warning("Could not stat disk usage at %s (%s); assuming 0 budget", cache_dir, exc)
        return 0
    return int(min(cap_bytes, free // 10))


def format_bytes(n: int) -> str:
    """Human size that stays informative at both ends of the range.

    A fixed GB unit renders a few megabytes as "0.00 GB", which reads as nothing
    at all and quietly undermines the message it appears in — seen 2026-08-08 in
    the orphaned-cache warning, where the real case is tens of GB but the test
    case was 4.5MB."""
    if n >= BYTES_PER_GB:
        return f"{n / BYTES_PER_GB:.2f} GB"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} bytes"


def orphaned_prompt_cache(cache_dir: Path) -> tuple[int, int]:
    """Files and total bytes left in the disk prompt-cache directory.

    Meaningful only when `DISK_PROMPT_CACHE` is off, and then it is a leak rather
    than a cache: the only code that ever deleted one of these files was the
    store's own LRU eviction, and that no longer runs. An install upgrading past
    the flag keeps everything it had accumulated (39.75GB on the machine this was
    found on) with nothing referencing it and nothing to make it visible. Hence
    the warning at server startup and in `mira doctor` — deleting it is the user's
    call, but not knowing about it should not be.

    Never raises: a missing directory, a permission error or a file vanishing
    mid-scan all read as "nothing to report", because this exists to add a line
    to a health check and must never be able to take one down.
    """
    total = 0
    count = 0
    try:
        entries = list(cache_dir.glob("*.safetensors"))
    except OSError:
        return 0, 0
    for p in entries:
        try:
            total += p.stat().st_size
            count += 1
        except OSError:
            continue
    return count, total


def derive_cache_limit_bytes(total_ram_bytes: Optional[int] = None) -> int:
    """Metal allocator reuse-cache cap: a small slice of the same ceiling
    `_check_memory_pressure` already trims against, so the *reactive* clear
    there stays a rare fallback instead of doing the routine work of keeping
    the reuse cache in check."""
    total_ram_bytes = total_ram_bytes if total_ram_bytes is not None else get_total_ram_bytes()
    ceiling = total_ram_bytes - SAFETY_MARGIN_BYTES
    return int(max(ceiling // 8, 256 * 1024 * 1024))


def fits_in_memory(model_id: str, total_ram_bytes: Optional[int] = None,
                    min_context: int = 4096,
                    kv_bits: Optional[int] = None, kv_group_size: int = 64,
                    resident_expert_fraction: Optional[float] = None) -> tuple[bool, str]:
    """Preflight check: would this model + a minimum usable context even fit?"""
    total_ram_bytes = total_ram_bytes if total_ram_bytes is not None else get_total_ram_bytes()
    model_bytes = estimate_active_weight_bytes(model_id, resident_expert_fraction)
    if model_bytes is None:
        return True, "model not yet downloaded — cannot preflight, allowing"

    kv_bytes_per_token = estimate_kv_bytes_per_token(model_id, kv_bits=kv_bits, kv_group_size=kv_group_size) or 0
    required = model_bytes + kv_bytes_per_token * min_context + SAFETY_MARGIN_BYTES
    if required > total_ram_bytes:
        return False, (
            f"{model_id} needs ~{required / BYTES_PER_GB:.1f}GB "
            f"(model {model_bytes / BYTES_PER_GB:.1f}GB + minimum context + safety margin), "
            f"this Mac has {total_ram_bytes / BYTES_PER_GB:.1f}GB total"
        )
    return True, "ok"
