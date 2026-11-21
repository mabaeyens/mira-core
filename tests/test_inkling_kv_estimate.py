"""The KV estimate for a looped model counts growing cache visits, not layers."""

import json

from core import hardware

# Inkling 2 Mini's text config, trimmed to what the estimator reads.
INKLING_TEXT = {
    "hidden_size": 2304,
    "num_hidden_layers": 24,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
    "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3},
}


def _model_dir(tmp_path, text_config):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "inkling2_loop", "text_config": text_config})
    )
    return str(tmp_path)


def test_counts_full_attention_visits_across_loop(tmp_path):
    # Full-attention layers 5, 11, 17, 23. 11 and 17 are inside the loop (x3).
    # visits = 1 + 3 + 3 + 1 = 8 -> 2 (K,V) * 8 * 8 heads * 128 * 2 bytes.
    got = hardware.estimate_kv_bytes_per_token(_model_dir(tmp_path, INKLING_TEXT))
    assert got == 2 * 8 * 8 * 128 * hardware.KV_DTYPE_BYTES == 32768


def test_kv_bits_is_ignored_because_the_cache_cannot_be_quantized(tmp_path):
    model = _model_dir(tmp_path, INKLING_TEXT)
    assert hardware.estimate_kv_bytes_per_token(model, kv_bits=4) == \
        hardware.estimate_kv_bytes_per_token(model)


def test_plain_model_is_unchanged(tmp_path):
    plain = {k: v for k, v in INKLING_TEXT.items() if k not in ("local_layer_ids", "loop_config")}
    got = hardware.estimate_kv_bytes_per_token(_model_dir(tmp_path, plain))
    assert got == 2 * 24 * 8 * 128 * hardware.KV_DTYPE_BYTES
