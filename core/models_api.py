"""Utilities for listing locally available models and downloading mlx-lm models."""

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# Imported as a module, not `from .config import OMLX_CLI`: a by-value import
# copies the string into this namespace and a test that patches `core.config`
# would leave this module pointing at the real path. Same reasoning as
# core/workspace.py's WORKSPACE_ROOT.
from core import config

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    model_id: str
    display_name: str
    size_gb: float
    backend: str  # "mira-mlx" | "mlx-lm" | "vllm-mlx" | "omlx"


@dataclass
class BackendStatus:
    """What a client needs to decide whether a backend is worth offering.

    `available` answers "could this be started at all", which is a different
    question from "does it have models". A backend whose CLI is present but
    reports available=True, an empty model list, and a detail saying why the
    whose library is empty needs a different fix from one that is not installed.
    """
    backend: str
    available: bool
    detail: str
    models: list = field(default_factory=list)


def _hf_cache_dir() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def _humanize(model_id: str) -> str:
    """Turn 'mlx-community/gemma-4-26b-a4b-it-4bit' into 'Gemma 4 26B (4-bit)'."""
    name = model_id.split("/")[-1]
    # Replace hyphens/underscores with spaces, title-case
    name = re.sub(r"[-_]", " ", name).strip()
    # Normalize common suffixes
    name = re.sub(r"\b4bit\b", "(4-bit)", name, flags=re.IGNORECASE)
    name = re.sub(r"\b8bit\b", "(8-bit)", name, flags=re.IGNORECASE)
    name = re.sub(r"\bq4 0\b", "(Q4_0)", name, flags=re.IGNORECASE)
    return name.title()


def _dir_size_gb(path: Path) -> float:
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / (1024 ** 3), 1)
    except Exception:
        return 0.0


# Keywords that identify non-chat utility models (rerankers, embedders).
# These stay on disk but are hidden from the model selector.
_NON_CHAT = {"reranker", "embed", "nomic"}


def list_mlx_models(backend: str = "mlx-lm") -> list[ModelEntry]:
    """Scan the HuggingFace cache for mlx-community chat models.

    `backend` only labels the returned entries. Every MLX-serving backend
    (mira-mlx, mlx-lm, vllm-mlx) loads from this same cache, so they all
    see the same set and differ only in the name they are offered under.

    Deliberately restricted to `mlx-community`. Widening it to every `models--*`
    repo would pull in the cross-encoder and nomic entries that are also cached
    here, and, worse, any torch-format repo, which would put un-runnable models
    in front of the user. Every non-mlx-community repo currently cached is a
    reranker or an embedder.
    """
    cache = _hf_cache_dir()
    if not cache.exists():
        return []

    entries = []
    for d in sorted(cache.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        # HF cache dirs are named like: models--mlx-community--gemma-4-26b-a4b-it-4bit
        if not name.startswith("models--mlx-community--"):
            continue
        # Reconstruct the repo id
        model_id = name[len("models--"):].replace("--", "/", 1)
        if any(kw in model_id.lower() for kw in _NON_CHAT):
            continue
        size_gb = _dir_size_gb(d / "blobs") if (d / "blobs").exists() else _dir_size_gb(d)
        entries.append(ModelEntry(
            model_id=model_id,
            display_name=_humanize(model_id),
            size_gb=size_gb,
            backend=backend,
        ))
    return entries


def list_omlx_models() -> list[ModelEntry]:
    """Scan oMLX's own model directory.

    Entries here are usually symlinks into the HuggingFace cache (both currently
    are), so their bytes are already counted by `list_mlx_models`. Sizes are
    resolved through the link deliberately: a user choosing a model wants to know
    what it costs to run, not how much new disk it would claim.
    """
    root = Path.home() / ".omlx" / "models"
    if not root.exists():
        return []
    entries = []
    for d in sorted(root.iterdir()):
        if not (d.is_dir() or d.is_symlink()):
            continue
        if d.name.startswith("."):
            continue
        entries.append(ModelEntry(
            model_id=d.name,
            display_name=_humanize(d.name),
            size_gb=_dir_size_gb(d.resolve()),
            backend="omlx",
        ))
    return entries


# Backends that serve straight out of the HuggingFace cache. They see an
# identical model set; only the label differs.
_HF_CACHE_BACKENDS = ("mira-mlx", "mlx-lm", "vllm-mlx")


def _cli_present(path: str) -> bool:
    return bool(path) and (os.path.exists(path) or bool(shutil.which(path)))


def backend_status(backend: str) -> BackendStatus:
    """Whether `backend` can be started here, and what it has to offer."""
    if backend == "mira-mlx":
        # In-repo module launched with `python -m`, so there is no binary that
        # could be missing. It is available wherever mira-core itself runs.
        return BackendStatus(backend, True, "", list_mlx_models("mira-mlx"))

    if backend in _HF_CACHE_BACKENDS:
        cli = {
            "mlx-lm": config.MLX_LM_CLI,
            "vllm-mlx": config.VLLM_MLX_CLI,
        }[backend]
        if not _cli_present(cli):
            return BackendStatus(backend, False, f"{backend} is not installed ({cli} not found)", [])
        return BackendStatus(backend, True, "", list_mlx_models(backend))

    if backend == "omlx":
        if not _cli_present(config.OMLX_CLI):
            return BackendStatus(backend, False, f"oMLX is not installed ({config.OMLX_CLI} not found)", [])
        models = list_omlx_models()
        detail = "" if models else "oMLX is installed but has no models in ~/.omlx/models"
        return BackendStatus(backend, True, detail, models)

    return BackendStatus(backend, False, f"unknown backend '{backend}'", [])


def list_backend_status(backends) -> list[BackendStatus]:
    """`backend_status` for each name, in the order given."""
    return [backend_status(b) for b in backends]
