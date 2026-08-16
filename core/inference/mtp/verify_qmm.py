"""Skinny-M 4-bit quantized matmul for the MTP verify window (self-authored).

MEASURED VERDICT (2026-08-16, M5 + MLX 0.32.0): NOT the lever at Mira's MTP depth, kept
as a correct scaffold. Direct timing shows stock `mx.quantized_matmul` is already
amortized for M=2-4 (MLP-up 1.0-1.1x, MLP-down 1.0-1.46x, lm_head 1.0-1.20x vs M=1); the
skinny-M penalty only bites at M>=6 (deeper than MTP runs). `qmv_wide` (PR #3764, in
0.32.0) closed the gap that omlx's kernel — built against the older mlx_lm 0.31.3 — fixed.
The v1 scalar kernel below is CORRECT (as accurate as stock) but SLOWER at M<=4 (it does
M scalar FMAs per weight element and skips the matrix units). The 1.88x->2.36x native-vs-
omlx gap is accept-rate + Python-loop overhead, not this matmul. `skinny_qmm`'s N-floor/M
gate means it never routes in production. Do NOT wire this in without re-measuring on a
newer MLX or at higher depth. See specs/native-mtp-verify-qmm-kernel.md, [[project_mtp_verify_kernel]].

The MTP verify forward multiplies a SKINNY activation (M = draft_depth+1 = 2-4 rows)
by large 4-bit projection weights. On MLX 0.32.0 the stock `qmm` amortizes the weight
FETCH across small M but not the per-row 4-bit dequant/compute, so it runs ~1.6x off
roofline at M=4 (mlx issue #4265). This kernel does the weight K-sweep for each output
column ONCE and reuses it across all M activation rows — the skinny-M win.

v1 = scalar-FMA, one simdgroup (32 lanes) per output column. This is the morphology
that provably wins on large N (lm_head, big MLP); it is a NET LOSS on small N (kernel
dispatch + scalar loop overhead), so `skinny_qmm` gates on an N-floor and falls back to
`mx.quantized_matmul` otherwise. A roofline-closing simdgroup_matrix (MMA) variant is
future work (see specs/native-mtp-verify-qmm-kernel.md).

Understanding (not code) drawn from omlx's Apache-2.0 `qwen35_verify_qmm.py` (MTPLX
lineage) and MLX's own `quantized.h`; the Metal below is original.

Correctness bar is trunk-verified equivalence, not bit-identity: fp32 accumulation makes
this AT LEAST as close to a dequantize@matmul reference as stock bf16 qmm is.
"""
from __future__ import annotations

import functools

import mlx.core as mx

# Only route projections at or above this N to the custom kernel; below it, the
# scalar kernel loses to stock qmm (mlx #4265). Tunable per machine/model.
DEFAULT_N_FLOOR = 4096
_MAX_M = 6  # acc[] is statically sized; larger M falls back.

_SOURCE = r"""
    uint lane = thread_position_in_grid.x;   // 0..31, one simdgroup per column
    uint n    = thread_position_in_grid.y;   // output column (weight row)

    const uint M  = x_shape[0];
    const uint K  = x_shape[1];
    const uint N  = wq_shape[0];
    const uint Kw = wq_shape[1];              // K / 8  (8 nibbles per uint32)
    const uint Kg = scales_shape[1];          // K / GROUP_SIZE

    if (n >= N) return;

    float acc[6];
    for (uint r = 0; r < M; ++r) acc[r] = 0.0f;

    // Each lane strides over the packed weight words of row n. The dequantized
    // weight for this k is reused across all M activation rows.
    for (uint w = lane; w < Kw; w += 32) {
        uint word = wq[n * Kw + w];
        uint k0 = w * 8;
        for (uint i = 0; i < 8; ++i) {
            uint k  = k0 + i;
            uint gi = k / GROUP_SIZE;
            float scale = (float)scales[n * Kg + gi];
            float bias  = (float)biases[n * Kg + gi];
            float wv = (float)((word >> (4u * i)) & 0xFu) * scale + bias;
            for (uint r = 0; r < M; ++r) {
                acc[r] += (float)x[r * K + k] * wv;
            }
        }
    }

    // Reduce each row's partial across the 32 lanes of the simdgroup.
    for (uint r = 0; r < M; ++r) {
        float s = simd_sum(acc[r]);
        if (lane == 0) {
            out[r * N + n] = (T)s;
        }
    }
"""


@functools.lru_cache(maxsize=None)
def _build_kernel(group_size: int):
    return mx.fast.metal_kernel(
        name=f"skinny_qmm_4bit_gs{group_size}",
        input_names=["x", "wq", "scales", "biases"],
        output_names=["out"],
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def skinny_qmm_kernel(x, wq, scales, biases, *, group_size: int):
    """Run the custom skinny-M 4-bit qmm unconditionally (no gating).

    Computes ``x @ dequantize(wq).T`` for a 4-bit affine-quantized weight, i.e. the
    equivalent of ``mx.quantized_matmul(x, wq, scales, biases, transpose=True,
    group_size=group_size, bits=4)``. Shapes: x (M, K), wq (N, K/8) uint32,
    scales/biases (N, K/group_size). Returns (M, N) in x's dtype.
    """
    M, K = x.shape
    N = wq.shape[0]
    if M > _MAX_M:
        raise ValueError(f"skinny_qmm_kernel supports M<= {_MAX_M}, got {M}")
    kernel = _build_kernel(group_size)
    (out,) = kernel(
        inputs=[x, wq, scales, biases],
        template=[("T", x.dtype), ("GROUP_SIZE", group_size)],
        grid=(32, N, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )
    return out


def _eligible(x, wq, scales, biases, group_size, bits, transpose, n_floor) -> bool:
    if bits != 4 or not transpose:
        return False
    if x.ndim != 2 or wq.ndim != 2:
        return False
    M, K = x.shape
    N = wq.shape[0]
    if not (2 <= M <= _MAX_M):
        return False
    if N < n_floor or (K % group_size) != 0 or (K % 8) != 0:
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    return True


def skinny_qmm(
    x, wq, scales, biases, *, group_size: int, bits: int = 4,
    transpose: bool = True, n_floor: int = DEFAULT_N_FLOOR,
):
    """Dispatch: custom kernel when it wins (skinny M, large N, 4-bit), else stock qmm.

    Drop-in for ``mx.quantized_matmul(..., transpose=True)`` on the verify path.
    """
    if _eligible(x, wq, scales, biases, group_size, bits, transpose, n_floor):
        return skinny_qmm_kernel(x, wq, scales, biases, group_size=group_size)
    return mx.quantized_matmul(
        x, wq, scales=scales, biases=biases,
        transpose=transpose, group_size=group_size, bits=bits,
    )
