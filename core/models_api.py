"""Utilities for listing locally available models and downloading mlx-lm models."""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelEntry:
    model_id: str
    display_name: str
    size_gb: float
    backend: str  # "mlx-lm" | "ollama"


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


def list_mlx_models() -> list[ModelEntry]:
    """Scan the HuggingFace cache for mlx-community chat models."""
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
            backend="mlx-lm",
        ))
    return entries


def list_ollama_models() -> list[ModelEntry]:
    """Return all models installed in Ollama."""
    try:
        import ollama as _ollama
        result = _ollama.list()
        entries = []
        for m in result.get("models", []):
            model_id = m.get("name", "")
            size_bytes = m.get("size", 0)
            entries.append(ModelEntry(
                model_id=model_id,
                display_name=_humanize(model_id),
                size_gb=round(size_bytes / (1024 ** 3), 1),
                backend="ollama",
            ))
        return entries
    except Exception:
        return []
