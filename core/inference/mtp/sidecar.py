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

OPEN (spec §8): where the bf16 ``mtp.*`` tensors are *sourced* from — a published
HF sidecar vs. extraction from the full-precision source repo — is not yet
settled. This module takes the sidecar as an input path (``sidecar_src``); wiring
an automatic download is the follow-up. Until then, an existing sidecar (e.g. the
one under ``~/.omlx/models/Qwen3.6-35B-A3B-MTP/``) can be passed as the source.
"""

from __future__ import annotations

import json
import logging
import os
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
