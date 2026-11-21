"""Locate / assemble the MTP head sidecar so native MTP needs no external app.

The public 4-bit quant (``mlx-community/Qwen3.6-35B-A3B-4bit``) strips the MTP
head, so an MTP run needs a bf16 ``model-mtp.safetensors`` sidecar (~1.7 GB, 19
tensors) alongside the base shards, plus a ``config.json`` carrying
``mtp_num_hidden_layers``. omlx requires the user to hand-build
``~/.omlx/models/<name>/`` for this; mira assembles its own dir under the mira
data dir instead — the zero-setup win.

An assembled MTP model dir is: symlinks to every base-model file + the real
``model-mtp.safetensors`` + a merged ``config.json``. ``mlx_lm.load`` globs all
``*.safetensors`` in the dir, so the head loads with the base.

The bf16 ``mtp.*`` tensors are sourced from the full-precision upstream repo
(``Qwen/Qwen3.6-35B-A3B``, apache-2.0) by :func:`fetch_mtp_sidecar`, which pulls only
those 19 tensors with HTTP range requests (~1.7 GB) instead of the 6 GB of shards they
live in. ``assemble_mtp_model_dir`` takes the resulting file as its ``sidecar_src``.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import time
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SIDECAR_FILENAME = "model-mtp.safetensors"
_MTP_KEY_PREFIX = "mtp."


def has_mtp_head(model_dir: os.PathLike | str) -> bool:
    """True if ``model_dir`` already carries the MTP sidecar."""
    return (Path(model_dir) / SIDECAR_FILENAME).exists()


def mtp_num_hidden_layers(model_dir: os.PathLike | str) -> int:
    """Read ``mtp_num_hidden_layers`` from a model dir's config.json (0 if absent).
    Handles both flat and ``text_config``-nested layouts."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        return 0
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        return 0
    if "mtp_num_hidden_layers" in cfg:
        return int(cfg.get("mtp_num_hidden_layers") or 0)
    text_cfg = cfg.get("text_config") or {}
    return int(text_cfg.get("mtp_num_hidden_layers") or 0)


def sidecar_has_mtp_tensors(sidecar_path: os.PathLike | str) -> bool:
    """Validate a sidecar file actually holds ``mtp.*`` tensors (reads only the
    safetensors header, not the payload)."""
    path = Path(sidecar_path)
    if not path.exists():
        return False
    try:
        with path.open("rb") as fh:
            (header_len,) = _read_u64(fh)
            header = json.loads(fh.read(header_len))
    except (OSError, ValueError):
        return False
    return any(k.startswith(_MTP_KEY_PREFIX) for k in header if k != "__metadata__")


def _read_u64(fh) -> tuple[int]:
    import struct

    return struct.unpack("<Q", fh.read(8))


def assemble_mtp_model_dir(
    base_model_dir: os.PathLike | str,
    sidecar_src: os.PathLike | str,
    dest_dir: os.PathLike | str,
    *,
    num_mtp_layers: int = 1,
    force: bool = False,
) -> Path:
    """Build an MTP-enabled model dir at ``dest_dir``: symlinks to every file in
    ``base_model_dir``, the ``model-mtp.safetensors`` sidecar, and a merged
    ``config.json`` with ``mtp_num_hidden_layers`` set. Idempotent — returns the
    dir untouched if it already looks assembled (unless ``force``).

    Raises if the sidecar is missing or carries no ``mtp.*`` tensors, so a broken
    sidecar fails loudly here rather than as ~0% draft acceptance later.
    """
    base = Path(base_model_dir)
    sidecar = Path(sidecar_src)
    dest = Path(dest_dir)

    if not base.is_dir():
        raise FileNotFoundError(f"base model dir not found: {base}")
    if not sidecar_has_mtp_tensors(sidecar):
        raise ValueError(
            f"sidecar {sidecar} is missing or carries no mtp.* tensors; "
            "cannot assemble an MTP model dir"
        )

    if dest.is_dir() and has_mtp_head(dest) and not force:
        logger.debug("MTP model dir already assembled at %s", dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    # Symlink every base file except an existing config.json (merged below) and any
    # stale sidecar. Relative-resolve so the links survive a data-dir move.
    for entry in base.iterdir():
        if entry.name in (SIDECAR_FILENAME, "config.json"):
            continue
        link = dest / entry.name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(entry.resolve(), link)

    # Real sidecar (symlink so we don't duplicate ~1.7 GB).
    sidecar_link = dest / SIDECAR_FILENAME
    if sidecar_link.exists() or sidecar_link.is_symlink():
        sidecar_link.unlink()
    os.symlink(sidecar.resolve(), sidecar_link)

    # Merge config.json with mtp_num_hidden_layers (respect nested text_config).
    cfg = json.loads((base / "config.json").read_text())
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        cfg["text_config"]["mtp_num_hidden_layers"] = int(num_mtp_layers)
    else:
        cfg["mtp_num_hidden_layers"] = int(num_mtp_layers)
    (dest / "config.json").write_text(json.dumps(cfg, indent=2))

    logger.info("assembled MTP model dir at %s (base=%s, sidecar=%s)", dest, base, sidecar)
    return dest


# ── sourcing the sidecar from the upstream repo ───────────────────────────────

UPSTREAM_MTP_REPO = "Qwen/Qwen3.6-35B-A3B"
_CHUNK = 64 * 1024 * 1024


def _hf_resolve(repo: str, filename: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def _http_range(url: str, start: int, end: int, *, retries: int = 4) -> bytes:
    """Bytes [start, end] inclusive via a Range request. Retries on transient errors.
    Public repo, so no credentials are ever sent."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise OSError(f"short read: wanted {end - start + 1} bytes, got {len(data)}")
            return data
        except (OSError, ValueError) as exc:  # URLError and short reads are OSError
            last = exc
            time.sleep(2 * (attempt + 1))
    raise OSError(f"range request failed after {retries} tries: {last}")


def _http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def fetch_mtp_sidecar(
    dest: os.PathLike | str,
    *,
    repo: str = UPSTREAM_MTP_REPO,
    revision: str = "main",
    progress=None,
) -> Path:
    """Build ``model-mtp.safetensors`` at ``dest`` from the upstream repo's ``mtp.*``
    tensors, downloading only those bytes (~1.7 GB, not the 6 GB of shards that hold
    them). Atomic (writes ``.part`` then renames) and idempotent: an existing valid
    sidecar is returned untouched. ``progress(done_bytes, total_bytes)`` is optional.
    """
    dest = Path(dest)
    if sidecar_has_mtp_tensors(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    index = _http_json(_hf_resolve(repo, "model.safetensors.index.json", revision))
    names = sorted(k for k in index["weight_map"] if k.startswith(_MTP_KEY_PREFIX))
    if not names:
        raise ValueError(f"{repo} has no {_MTP_KEY_PREFIX}* tensors in its index")

    # Per shard: read the safetensors header once, then plan the byte span of each tensor.
    shard_hdr: dict[str, tuple[int, dict]] = {}
    plan = []  # (name, shard, abs_start, nbytes, meta)
    for name in names:
        shard = index["weight_map"][name]
        if shard not in shard_hdr:
            url = _hf_resolve(repo, shard, revision)
            (hlen,) = struct.unpack("<Q", _http_range(url, 0, 7))
            header = json.loads(_http_range(url, 8, 8 + hlen - 1))
            shard_hdr[shard] = (8 + hlen, header)
        base, header = shard_hdr[shard]
        meta = header[name]
        s0, e0 = meta["data_offsets"]
        plan.append((name, shard, base + s0, e0 - s0, meta))

    total = sum(p[3] for p in plan)
    out_header, offset = {}, 0
    for name, _shard, _start, nbytes, meta in plan:
        out_header[name] = {"dtype": meta["dtype"], "shape": meta["shape"],
                            "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(out_header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # pad so the data section is 8-byte aligned

    part = dest.with_name(dest.name + ".part")
    done = 0
    with part.open("wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        for name, shard, start, nbytes, _meta in plan:
            url = _hf_resolve(repo, shard, revision)
            pos = 0
            while pos < nbytes:
                n = min(_CHUNK, nbytes - pos)
                fh.write(_http_range(url, start + pos, start + pos + n - 1))
                pos += n
                done += n
                if progress:
                    progress(done, total)
    part.replace(dest)
    if not sidecar_has_mtp_tensors(dest):
        dest.unlink(missing_ok=True)
        raise ValueError("fetched sidecar failed validation (no mtp.* tensors)")
    logger.info("fetched MTP sidecar %s (%d tensors, %.2f GB)", dest, len(plan), total / 1e9)
    return dest
