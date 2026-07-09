"""RAM-aware sizing for local inference: total memory detection and per-model
KV-cache/context budgeting, so the same code behaves sensibly on an 8GB Mac
and a 128GB Mac Studio instead of using one fixed set of constants everywhere.

A prompt whose own token count is >= the derived max_kv_size ceiling is
rejected outright by GenerationEngine._start_job (mira_mlx_server.py) with a
clear ValueError, rather than left to RotatingKVCache's undefined behavior for
a single over-budget submission.
"""
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3
KV_DTYPE_BYTES = 2  # mlx_lm's KV cache is fp16 by default regardless of model quantization

# Held back from every RAM-based budget for the OS, other apps, and Mira's own
# non-model processes (embeddings, RAG, orchestrator). Deliberately conservative.
SAFETY_MARGIN_BYTES = 3 * BYTES_PER_GB


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


def estimate_kv_bytes_per_token(model_id: str) -> Optional[int]:
    """KV cache bytes for one token of context, derived from the model's own config.

    Formula (validated 2026-07-09 against a live measurement of Ministral 3 14B —
    computed 163,840 B/token vs. a measured 163,820 B/token): for each of K and V,
    one value per layer per KV head per head-dim element, at KV_DTYPE_BYTES each
    (KV cache stays fp16 even for a 4-bit-quantized model — quantization only
    applies to weights unless mlx-lm's separate kv_bits option is explicitly used,
    which mira-mlx does not yet support — see the low-RAM plan's non-goals).
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

    return 2 * num_layers * num_kv_heads * head_dim * KV_DTYPE_BYTES


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


def derive_prompt_cache_max_bytes(model_id: str, total_ram_bytes: Optional[int] = None) -> int:
    """Safe prompt-cache pool size: whatever's left after the model, KV working
    set, and a fixed safety margin, floored so it's never negative/degenerate."""
    total_ram_bytes = total_ram_bytes if total_ram_bytes is not None else get_total_ram_bytes()
    model_bytes = estimate_model_weight_bytes(model_id) or 8 * BYTES_PER_GB  # conservative guess
    available = total_ram_bytes - model_bytes - SAFETY_MARGIN_BYTES
    # Leave room for at least the active generation's own KV cache on top of the
    # cache *pool* — cap the pool at half of whatever's left.
    budget = max(available // 2, 512 * 1024 * 1024)
    return int(budget)


def derive_context_window(model_id: str, total_ram_bytes: Optional[int] = None,
                           requested_context: int = 65536) -> int:
    """Cap a requested context window to what this machine can actually hold in
    KV cache for a single active generation, without touching the cache pool."""
    total_ram_bytes = total_ram_bytes if total_ram_bytes is not None else get_total_ram_bytes()
    model_bytes = estimate_model_weight_bytes(model_id) or 8 * BYTES_PER_GB
    kv_bytes_per_token = estimate_kv_bytes_per_token(model_id)
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


def fits_in_memory(model_id: str, total_ram_bytes: Optional[int] = None,
                    min_context: int = 4096) -> tuple[bool, str]:
    """Preflight check: would this model + a minimum usable context even fit?"""
    total_ram_bytes = total_ram_bytes if total_ram_bytes is not None else get_total_ram_bytes()
    model_bytes = estimate_model_weight_bytes(model_id)
    if model_bytes is None:
        return True, "model not yet downloaded — cannot preflight, allowing"

    kv_bytes_per_token = estimate_kv_bytes_per_token(model_id) or 0
    required = model_bytes + kv_bytes_per_token * min_context + SAFETY_MARGIN_BYTES
    if required > total_ram_bytes:
        return False, (
            f"{model_id} needs ~{required / BYTES_PER_GB:.1f}GB "
            f"(model {model_bytes / BYTES_PER_GB:.1f}GB + minimum context + safety margin), "
            f"this Mac has {total_ram_bytes / BYTES_PER_GB:.1f}GB total"
        )
    return True, "ok"
