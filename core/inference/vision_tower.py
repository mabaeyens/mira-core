"""Optional vision for mira-mlx: turn images into language-model embeddings.

The checkpoint Mira runs (`mlx-community/Qwen3.6-35B-A3B-4bit`) ships a complete
vision tower, 333 tensors under `vision_tower.*`, unquantized while the language
model around it is 4-bit. `mlx_lm` throws all of it away: `qwen3_5_moe.sanitize()`
drops every `vision_tower*` key at load, so `mlx_lm.utils.load()` hands back a
text-only model. This module loads those weights separately, on demand, and only
when vision is switched on.

What it produces is deliberately narrow: one `(num_image_tokens, hidden)` array
per image, in the language model's hidden size, ready to be spliced into the text
embeddings wherever `image_token_id` appears. Everything downstream of that is
the ordinary text path, which is why the server change on top of this is small.

Off by default. Nothing here is imported or loaded unless `--vision` is set, so
the text-only path keeps its startup time and its RAM.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from .qwen3_vl_vision import VisionConfig, VisionModel

logger = logging.getLogger(__name__)

# Qwen's own preprocessing constants. These come from the checkpoint's
# preprocessor_config.json; the defaults here are only a fallback.
_DEFAULT_IMAGE_MEAN = (0.5, 0.5, 0.5)
_DEFAULT_IMAGE_STD = (0.5, 0.5, 0.5)


def _smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Tuple[int, int]:
    """Qwen's resize rule, ported from `transformers`' `Qwen2VLImageProcessor`.

    Both sides end up divisible by `factor` (patch_size * merge_size), the total
    pixel count lands inside [min_pixels, max_pixels], and the aspect ratio moves
    as little as possible. Getting this wrong does not raise, it just produces a
    grid the tower and the token count disagree about, so it is ported exactly
    rather than approximated.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class VisionTower:
    """Loads the checkpoint's vision tower and embeds images with it."""

    def __init__(self, model_path: str | Path, max_pixels: Optional[int] = None):
        self.model_path = self._resolve(model_path)
        config = json.loads((self.model_path / "config.json").read_text())
        vision_config = config.get("vision_config")
        if not vision_config:
            raise ValueError(
                f"{self.model_path.name} has no vision_config, so it carries no "
                "vision tower and cannot answer about images."
            )

        self.config = VisionConfig.from_dict(vision_config)

        # Qwen3-VL's deepstack feeds intermediate tower features back into the
        # first few language layers. mlx-lm's text-only qwen3_5 has no such hook,
        # so a checkpoint that wants deepstack would silently lose part of its
        # visual signal. This one lists no deepstack layers, so refuse loudly
        # rather than degrade quietly if that ever changes.
        if self.config.deepstack_visual_indexes:
            raise ValueError(
                "This checkpoint uses deepstack visual layers "
                f"({self.config.deepstack_visual_indexes}), which the text-only "
                "language model cannot consume. Vision would be silently degraded."
            )

        self.image_token_id = config.get("image_token_id")
        if self.image_token_id is None:
            raise ValueError(f"{self.model_path.name} declares no image_token_id")

        text_config = config.get("text_config", config)
        self.hidden_size = text_config["hidden_size"]
        if self.config.out_hidden_size != self.hidden_size:
            raise ValueError(
                f"vision tower emits {self.config.out_hidden_size}-wide embeddings "
                f"but the language model expects {self.hidden_size}"
            )

        preprocessor = self._load_preprocessor_config()
        self.patch_size = preprocessor.get("patch_size", self.config.patch_size)
        self.merge_size = preprocessor.get("merge_size", self.config.spatial_merge_size)
        self.temporal_patch_size = preprocessor.get(
            "temporal_patch_size", self.config.temporal_patch_size
        )
        self.image_mean = np.array(
            preprocessor.get("image_mean", _DEFAULT_IMAGE_MEAN), dtype=np.float32
        )
        self.image_std = np.array(
            preprocessor.get("image_std", _DEFAULT_IMAGE_STD), dtype=np.float32
        )
        size = preprocessor.get("size", {})
        self.min_pixels = size.get("shortest_edge", 56 * 56)
        checkpoint_max = size.get("longest_edge", 16777216)

        # The checkpoint ships longest_edge = 16,777,216 (16.7 MP), which is no
        # practical cap: a 5712x4284 phone photo survives at 16,170 image tokens,
        # costing 243s in the tower, 126 MB of embeddings and 12% of a 128k
        # context window. Measured on real photos, capping to ~2 MP costs about
        # 4.4s and 2k tokens instead, and to 1 MP about 1.6s and 1k tokens -
        # a 157x speedup at the top end with no code path of its own.
        #
        # Only ever lowers the ceiling: a checkpoint that already asks for less
        # than the cap keeps its own value.
        self.max_pixels = min(checkpoint_max, max_pixels) if max_pixels else checkpoint_max
        if self.max_pixels < checkpoint_max:
            logger.info(
                "vision: capping max_pixels to %d (checkpoint asks for %d)",
                self.max_pixels,
                checkpoint_max,
            )

        self.model: Optional[VisionModel] = None
        self._weight_bytes = 0

    @staticmethod
    def _resolve(model_path: str | Path) -> Path:
        """Accept either a local snapshot directory or a HuggingFace repo id.

        The engine carries the configured model as a repo id
        (`mlx-community/Qwen3.6-35B-A3B-4bit`), which is what `mlx_lm.load`
        takes, so the tower has to resolve it the same way rather than treating
        it as a directory that does not exist.
        """
        path = Path(model_path)
        if (path / "config.json").exists():
            return path
        from mlx_lm.utils import hf_repo_to_path

        return hf_repo_to_path(str(model_path))

    def _load_preprocessor_config(self) -> dict:
        path = self.model_path / "preprocessor_config.json"
        if not path.exists():
            logger.warning(
                "no preprocessor_config.json in %s, falling back to vision_config "
                "defaults for image preprocessing",
                self.model_path.name,
            )
            return {}
        return json.loads(path.read_text())

    def load(self) -> None:
        """Read the `vision_tower.*` tensors and build the tower.

        Only the vision keys are pulled out of the shards, so this costs the
        tower's own weights (about 0.89 GB in bf16 for this checkpoint) and not a
        second copy of the language model. The tower is left in its checkpoint
        dtype: it ships unquantized and must not go through the language model's
        quantization predicate.
        """
        if self.model is not None:
            return

        prefix = "vision_tower."
        weights = {}
        for shard in sorted(self.model_path.glob("*.safetensors")):
            for key, value in mx.load(str(shard)).items():
                if key.startswith(prefix):
                    weights[key[len(prefix) :]] = value

        if not weights:
            raise ValueError(
                f"no vision_tower.* tensors found in {self.model_path.name}; "
                "this checkpoint declares a vision_config but ships no tower"
            )

        model = VisionModel(self.config)
        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))
        model.eval()

        self.model = model
        self._weight_bytes = sum(v.nbytes for v in weights.values())
        logger.info(
            "vision tower loaded: %d tensors, %.2f GB, %d layers at hidden %d",
            len(weights),
            self._weight_bytes / 1e9,
            self.config.depth,
            self.config.hidden_size,
        )

    @property
    def weight_bytes(self) -> int:
        return self._weight_bytes

    def preprocess(self, image) -> Tuple[mx.array, mx.array]:
        """PIL image to (pixel_values, grid_thw).

        Ported from `transformers`' `Qwen2VLImageProcessorFast._preprocess` so the
        patch ordering matches what the tower was trained on. The temporal axis is
        the subtle part: a still image is repeated `temporal_patch_size` times
        inside each patch vector while `grid_t` stays 1. Get that backwards and the
        grid maths is off by a factor of two with no error, just wrong answers.
        """
        from PIL import Image

        if image.mode != "RGB":
            image = image.convert("RGB")

        factor = self.patch_size * self.merge_size
        height, width = image.height, image.width
        resized_h, resized_w = _smart_resize(
            height, width, factor, self.min_pixels, self.max_pixels
        )
        if (resized_h, resized_w) != (height, width):
            image = image.resize((resized_w, resized_h), Image.BICUBIC)

        # (H, W, C) to normalized (C, H, W)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - self.image_mean) / self.image_std
        arr = arr.transpose(2, 0, 1)

        channel = arr.shape[0]
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        m = self.merge_size
        p = self.patch_size

        patches = arr.reshape(channel, grid_h // m, m, p, grid_w // m, m, p)
        # (grid_h/m, grid_w/m, m, m, channel, p, p) - grid outer, patch inner
        patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
        # Insert the temporal axis after channel and repeat the still image into it.
        patches = np.repeat(patches[..., None, :, :], self.temporal_patch_size, axis=-3)
        flat = patches.reshape(
            grid_h * grid_w, channel * self.temporal_patch_size * p * p
        )

        return mx.array(flat), mx.array([[1, grid_h, grid_w]])

    def num_image_tokens(self, image) -> int:
        """How many `image_token_id` placeholders this image expands to.

        Cheap: resizes nothing, just does the grid arithmetic, so callers can
        budget context before paying for preprocessing.
        """
        factor = self.patch_size * self.merge_size
        resized_h, resized_w = _smart_resize(
            image.height, image.width, factor, self.min_pixels, self.max_pixels
        )
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        return (grid_h * grid_w) // (self.merge_size**2)

    def embed(self, images: Sequence) -> List[mx.array]:
        """Embed PIL images, one `(num_image_tokens, hidden)` array each.

        One tower forward per image. That is once per turn, not once per token,
        so it does not touch decode throughput.
        """
        if self.model is None:
            self.load()

        out = []
        for image in images:
            pixel_values, grid_thw = self.preprocess(image)
            embeds, deepstack = self.model(pixel_values, grid_thw)
            if deepstack:
                # Guarded at construction; belt and braces in case a future
                # checkpoint slips through with a non-empty index list.
                raise ValueError(
                    "vision tower returned deepstack features, which the text-only "
                    "language model cannot consume"
                )
            mx.eval(embeds)
            expected = self.num_image_tokens(image)
            if embeds.shape[0] != expected:
                raise ValueError(
                    f"vision tower produced {embeds.shape[0]} embeddings but the "
                    f"token budget expected {expected}; grid maths is out of step"
                )
            out.append(embeds)
        return out


def splice_image_embeddings(
    text_embeddings: mx.array,
    token_ids: Sequence[int],
    image_embeddings: Sequence[mx.array],
    image_token_id: int,
) -> mx.array:
    """Replace the rows at `image_token_id` positions with the image embeddings.

    `text_embeddings` is what the language model's own table produced for the
    whole prompt, including one placeholder row per image token. The images'
    embeddings are laid down in order across those positions, so the result is
    the sequence the model would have built for itself if it could see.
    """
    positions = [i for i, t in enumerate(token_ids) if t == image_token_id]
    total = sum(e.shape[0] for e in image_embeddings)
    if len(positions) != total:
        raise ValueError(
            f"prompt has {len(positions)} image token placeholders but the images "
            f"embed to {total} positions"
        )
    if not positions:
        return text_embeddings

    stacked = mx.concatenate(list(image_embeddings), axis=0).astype(
        text_embeddings.dtype
    )
    index = mx.array(positions, dtype=mx.uint32)
    out = mx.array(text_embeddings)
    out[index] = stacked
    return out
