"""Tests for the optional vision path.

Everything here runs without the 0.89GB tower weights. The parts that need real
weights (a tower forward, an end-to-end image answer) are live checks, not unit
tests, and are recorded in docs/architecture.md.
"""

import json

import numpy as np
import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core")  # mlx is macOS-only (Apple Silicon), absent on Linux CI

# Imported after the guard: vision_tower pulls in mlx at module level, so on Linux
# this line would raise during collection rather than skip.
from core.inference.vision_tower import (  # noqa: E402
    VisionTower,
    _smart_resize,
    splice_image_embeddings,
)


def _write_checkpoint(tmp_path, **vision_overrides):
    """A checkpoint directory with just the two json files the tower reads."""
    vision_config = {
        "depth": 27,
        "hidden_size": 1152,
        "out_hidden_size": 2048,
        "num_heads": 16,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "num_position_embeddings": 2304,
        "deepstack_visual_indexes": [],
    }
    vision_config.update(vision_overrides)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "vision_config": vision_config,
                "image_token_id": 248056,
                "text_config": {"hidden_size": 2048},
            }
        )
    )
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "patch_size": 16,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "size": {"shortest_edge": 65536, "longest_edge": 16777216},
            }
        )
    )
    return tmp_path


# -- smart_resize -------------------------------------------------------------


@pytest.mark.parametrize(
    "h,w",
    [(768, 1024), (480, 640), (200, 320), (313, 97), (1, 200), (4000, 3000)],
)
def test_smart_resize_output_is_divisible_by_the_factor(h, w):
    rh, rw = _smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=16777216)
    assert rh % 32 == 0 and rw % 32 == 0
    assert rh > 0 and rw > 0


def test_smart_resize_respects_the_pixel_floor():
    rh, rw = _smart_resize(50, 50, factor=32, min_pixels=65536, max_pixels=16777216)
    assert rh * rw >= 65536


def test_smart_resize_respects_the_pixel_ceiling():
    rh, rw = _smart_resize(8000, 8000, factor=32, min_pixels=65536, max_pixels=262144)
    assert rh * rw <= 262144


def test_smart_resize_rejects_absurd_aspect_ratios():
    with pytest.raises(ValueError, match="aspect ratio"):
        _smart_resize(1, 500, factor=32, min_pixels=1, max_pixels=10**9)


# -- token accounting ---------------------------------------------------------


@pytest.mark.parametrize(
    "size,expected",
    [((1024, 768), 768), ((640, 480), 300), ((1024, 1024), 1024)],
)
def test_num_image_tokens_matches_the_documented_budget(tmp_path, size, expected):
    """These are the numbers the context budget in the docs is built on, so a
    change in the grid maths should break a test rather than quietly cost
    context."""
    tower = VisionTower(_write_checkpoint(tmp_path))
    assert tower.num_image_tokens(Image.new("RGB", size)) == expected


def test_preprocess_shape_agrees_with_the_token_count(tmp_path):
    tower = VisionTower(_write_checkpoint(tmp_path))
    img = Image.new("RGB", (1024, 768))
    pixel_values, grid = tower.preprocess(img)
    grid_t, grid_h, grid_w = [int(x) for x in np.array(grid)[0]]
    assert grid_t == 1
    assert pixel_values.shape[0] == grid_h * grid_w
    # channel * temporal * patch * patch = 3 * 2 * 16 * 16
    assert pixel_values.shape[1] == 3 * 2 * 16 * 16
    assert (grid_h * grid_w) // 4 == tower.num_image_tokens(img)


def test_preprocess_converts_non_rgb(tmp_path):
    tower = VisionTower(_write_checkpoint(tmp_path))
    pixel_values, _ = tower.preprocess(Image.new("L", (640, 480)))
    assert pixel_values.shape[1] == 3 * 2 * 16 * 16


# -- guards -------------------------------------------------------------------


def test_deepstack_checkpoint_is_refused(tmp_path):
    """The text-only language model has no hook to consume deepstack features,
    so a checkpoint that wants them must fail loudly rather than answer with
    part of its visual signal missing."""
    path = _write_checkpoint(tmp_path, deepstack_visual_indexes=[0, 1, 2])
    with pytest.raises(ValueError, match="deepstack"):
        VisionTower(path)


def test_hidden_size_mismatch_is_refused(tmp_path):
    path = _write_checkpoint(tmp_path, out_hidden_size=1536)
    with pytest.raises(ValueError, match="language model expects"):
        VisionTower(path)


def test_checkpoint_without_vision_config_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 2048}))
    with pytest.raises(ValueError, match="no vision_config"):
        VisionTower(tmp_path)


def test_load_without_tower_weights_is_refused(tmp_path):
    tower = VisionTower(_write_checkpoint(tmp_path))
    with pytest.raises(ValueError, match="no vision_tower"):
        tower.load()


# -- splicing -----------------------------------------------------------------


def test_splice_replaces_only_the_image_positions():
    hidden = 8
    tokens = [5, 5, 999, 999, 999, 7]
    text = mx.arange(len(tokens) * hidden).reshape(len(tokens), hidden).astype(
        mx.float32
    )
    out = splice_image_embeddings(text, tokens, [mx.full((3, hidden), -1.0)], 999)
    assert np.allclose(np.array(out)[2:5], -1.0)
    assert np.allclose(np.array(out)[[0, 1, 5]], np.array(text)[[0, 1, 5]])


def test_splice_lays_multiple_images_down_in_order():
    hidden = 4
    tokens = [1, 999, 999, 2, 999, 3]
    text = mx.zeros((len(tokens), hidden))
    out = splice_image_embeddings(
        text, tokens, [mx.full((2, hidden), 1.0), mx.full((1, hidden), 2.0)], 999
    )
    got = np.array(out)
    assert np.allclose(got[[1, 2]], 1.0)
    assert np.allclose(got[4], 2.0)


def test_splice_rejects_a_placeholder_count_mismatch():
    """This is the check that turns an off-by-one in the grid maths into an
    error instead of a silently misaligned prompt."""
    hidden = 4
    tokens = [999, 999, 999]
    with pytest.raises(ValueError, match="placeholders"):
        splice_image_embeddings(
            mx.zeros((3, hidden)), tokens, [mx.zeros((2, hidden))], 999
        )


def test_splice_with_no_images_is_a_passthrough():
    text = mx.ones((3, 4))
    out = splice_image_embeddings(text, [1, 2, 3], [], 999)
    assert np.allclose(np.array(out), np.array(text))
