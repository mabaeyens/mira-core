"""Unit tests for native MTP model-side patch + sidecar (spec §5.1/§5.2).

Tiny synthetic tensors only — no model download, no full load. The full-model
live smoke test (load the 4-bit base + bf16 sidecar, confirm head logits are
non-flat and output is lossless vs. non-MTP greedy) is a manual bench-discipline
step, not part of this suite.
"""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402

from core.inference import mtp  # noqa: E402
from core.inference.mtp import qwen3_mtp, sidecar  # noqa: E402

pytest.importorskip("mlx_lm.models.qwen3_5")


def _tiny_args():
    """A small but complete Qwen3.5 dense (num_experts=0) config so the MTP head,
    including its full-attention block, actually constructs."""
    from mlx_lm.models.qwen3_5 import TextModelArgs

    qwen3_mtp.apply()  # installs the from_dict wrapper that keeps mtp_num_hidden_layers
    return TextModelArgs.from_dict(
        {
            "model_type": "qwen3_5",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 256,
            "rms_norm_eps": 1e-6,
            "num_experts": 0,
            "mtp_num_hidden_layers": 1,
            "rope_parameters": {"type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 1.0},
        }
    )


def test_apply_is_idempotent():
    assert qwen3_mtp.apply() is True
    assert qwen3_mtp.apply() is True  # second call must not chain wraps or raise


def test_args_surfaces_mtp_layers():
    args = _tiny_args()
    assert args.mtp_num_hidden_layers == 1


def test_head_forward_runs_and_is_non_constant():
    from mlx_lm.models import qwen3_5 as q35

    args = _tiny_args()
    head = q35._MiraMTPHead(args)
    mx.eval(head.parameters())

    S, H = 3, args.hidden_size
    hidden = mx.random.normal((1, S, H))
    next_ids = mx.array([[1, 2, 3]])
    embed = nn.Embedding(args.vocab_size, H)
    mx.eval(embed.parameters())

    out = head(hidden, next_ids, embed)
    mx.eval(out)
    assert out.shape == (1, S, H)
    assert bool(mx.all(mx.isfinite(out)))
    # A collapsed head (the norm-convention bug) yields ~constant output.
    assert float(mx.std(out).item()) > 1e-4


def test_sanitize_shifts_all_head_norms_on_raw_hf_sidecar():
    # A head sidecar carries ONE convention. The reliable head norms (pre_fc_* + the
    # head layer norms) center near 0 when raw-HF → shift ALL head norms uniformly,
    # INCLUDING q_norm/k_norm/mtp.norm whose raw gammas sit ABOVE 0.5. Regression: a
    # per-key mean<0.5 test misclassified those three and left them -1 off (~14pp of
    # draft acceptance, worst on the deep drafts). Base is already-MLX (no conv1d
    # transpose), so backbone_shift stays False and the head signal must carry it.
    stub = SimpleNamespace(mtp=object(), args=SimpleNamespace(tie_word_embeddings=False))
    weights = {
        # reliable discriminators — raw-HF, center near 0
        "language_model.mtp.pre_fc_norm_hidden.weight": mx.full((8,), -0.16),
        "language_model.mtp.pre_fc_norm_embedding.weight": mx.full((8,), -0.46),
        "language_model.mtp.layers.0.input_layernorm.weight": mx.full((8,), 0.04),
        "language_model.mtp.layers.0.post_attention_layernorm.weight": mx.full((8,), 0.21),
        # the three whose raw-HF gamma sits above 0.5 — must STILL shift
        "language_model.mtp.layers.0.self_attn.q_norm.weight": mx.full((8,), 0.79),
        "language_model.mtp.layers.0.self_attn.k_norm.weight": mx.full((8,), 0.78),
        "language_model.mtp.norm.weight": mx.full((8,), 1.25),
        # non-norm + backbone
        "language_model.mtp.fc.weight": mx.ones((8, 16)),
        "language_model.model.norm.weight": mx.full((8,), 0.7),  # MLX base, no shift
    }
    out = qwen3_mtp._sanitize_text_model(stub, weights)

    def val(k):
        return float(out[k][0].item())

    assert "language_model.mtp.norm.weight" in out                       # mtp.* kept
    assert val("language_model.mtp.pre_fc_norm_hidden.weight") == pytest.approx(0.84, abs=1e-4)
    assert val("language_model.mtp.layers.0.self_attn.q_norm.weight") == pytest.approx(1.79, abs=1e-4)
    assert val("language_model.mtp.layers.0.self_attn.k_norm.weight") == pytest.approx(1.78, abs=1e-4)
    assert val("language_model.mtp.norm.weight") == pytest.approx(2.25, abs=1e-4)  # the fix
    assert out["language_model.mtp.fc.weight"].shape == (8, 16)          # 2D untouched
    assert val("language_model.model.norm.weight") == pytest.approx(0.7, abs=1e-4)  # backbone unshifted


def test_sanitize_leaves_head_norms_on_already_mlx_sidecar():
    # Reliable head norms near 1 → sidecar already in MLX convention → shift NOTHING
    # (guards against a double +1 on a pre-converted head).
    stub = SimpleNamespace(mtp=object(), args=SimpleNamespace(tie_word_embeddings=False))
    weights = {
        "language_model.mtp.pre_fc_norm_hidden.weight": mx.full((8,), 0.84),
        "language_model.mtp.pre_fc_norm_embedding.weight": mx.full((8,), 0.54),
        "language_model.mtp.norm.weight": mx.full((8,), 2.25),
        "language_model.mtp.fc.weight": mx.ones((8, 16)),
    }
    out = qwen3_mtp._sanitize_text_model(stub, weights)
    assert float(out["language_model.mtp.norm.weight"][0].item()) == pytest.approx(2.25, abs=1e-4)
    assert float(out["language_model.mtp.pre_fc_norm_hidden.weight"][0].item()) == pytest.approx(0.84, abs=1e-4)


def test_sanitize_drops_mtp_when_no_head():
    stub = SimpleNamespace(args=SimpleNamespace(tie_word_embeddings=False))  # no .mtp
    out = qwen3_mtp._sanitize_text_model(stub, {"language_model.mtp.norm.weight": mx.ones((4,))})
    assert out == {}


def test_sanitize_raises_when_head_but_no_weights():
    stub = SimpleNamespace(mtp=object(), args=SimpleNamespace(tie_word_embeddings=False))
    with pytest.raises(ValueError, match="mtp.* tensors"):
        qwen3_mtp._sanitize_text_model(stub, {"language_model.model.norm.weight": mx.ones((4,))})


def test_moe_sanitize_unfuses_backbone_and_mtp_experts():
    moe = pytest.importorskip("mlx_lm.models.qwen3_5_moe")
    qwen3_mtp.apply()

    E, M, D = 4, 6, 8  # experts, 2*inter (gate_up split at M//2), hidden
    stub = SimpleNamespace(
        language_model=SimpleNamespace(
            args=SimpleNamespace(num_hidden_layers=1, mtp_num_hidden_layers=1),
            sanitize=lambda w: w,  # identity so we can inspect the unfused keys
        )
    )
    weights = {
        "model.layers.0.mlp.experts.gate_up_proj": mx.ones((E, M, D)),
        "model.layers.0.mlp.experts.down_proj": mx.ones((E, D, M // 2)),
        "mtp.layers.0.mlp.experts.gate_up_proj": mx.ones((E, M, D)),
        "mtp.layers.0.mlp.experts.down_proj": mx.ones((E, D, M // 2)),
    }
    out = moe.Model.sanitize(stub, weights)

    for prefix in ("language_model.model.layers.0.mlp", "language_model.mtp.layers.0.mlp"):
        assert f"{prefix}.switch_mlp.gate_proj.weight" in out
        assert f"{prefix}.switch_mlp.up_proj.weight" in out
        assert f"{prefix}.switch_mlp.down_proj.weight" in out
        assert out[f"{prefix}.switch_mlp.gate_proj.weight"].shape == (E, M // 2, D)
        assert f"{prefix}.experts.gate_up_proj" not in out


# --- sidecar helpers -------------------------------------------------------- #

def _write_safetensors(path: Path, keys):
    """Minimal valid safetensors: header names `keys` as tiny f32 tensors."""
    offset = 0
    header = {}
    blobs = []
    for k in keys:
        data = struct.pack("<f", 1.0)
        header[k] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        blobs.append(data)
        offset += 4
    hdr = json.dumps(header).encode()
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr)))
        fh.write(hdr)
        for b in blobs:
            fh.write(b)


def test_sidecar_tensor_detection(tmp_path):
    good = tmp_path / "good.safetensors"
    bad = tmp_path / "bad.safetensors"
    _write_safetensors(good, ["mtp.fc.weight", "mtp.norm.weight"])
    _write_safetensors(bad, ["model.norm.weight"])
    assert sidecar.sidecar_has_mtp_tensors(good) is True
    assert sidecar.sidecar_has_mtp_tensors(bad) is False
    assert sidecar.sidecar_has_mtp_tensors(tmp_path / "missing.safetensors") is False


def test_assemble_mtp_model_dir(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (base / "tokenizer.json").write_bytes(b"{}")
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe", "hidden_size": 64}))

    side = tmp_path / "src" / sidecar.SIDECAR_FILENAME
    side.parent.mkdir()
    _write_safetensors(side, ["mtp.fc.weight"])

    dest = tmp_path / "assembled"
    out = sidecar.assemble_mtp_model_dir(base, side, dest, num_mtp_layers=1)

    assert sidecar.has_mtp_head(out)
    assert (out / "model-00001-of-00001.safetensors").is_symlink()
    assert (out / "tokenizer.json").is_symlink()
    assert sidecar.mtp_num_hidden_layers(out) == 1
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["mtp_num_hidden_layers"] == 1 and cfg["hidden_size"] == 64
    # idempotent second call
    assert sidecar.assemble_mtp_model_dir(base, side, dest) == dest


def test_assemble_rejects_bad_sidecar(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    bad = tmp_path / "bad.safetensors"
    _write_safetensors(bad, ["model.norm.weight"])
    with pytest.raises(ValueError, match="mtp.* tensors"):
        sidecar.assemble_mtp_model_dir(base, bad, tmp_path / "dest")
