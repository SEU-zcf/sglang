"""Triton implementation of the DeepSeek-V4 mHC pre operator.

The implementation keeps the public contract of :func:`mhc_pre`:

1. project the flattened multi-stream residual into pre/post/comb logits while
   accumulating its RMS square sum;
2. turn the logits into pre/post gates and a Sinkhorn-normalized combination
   matrix;
3. combine the input streams and optionally fuse the following RMSNorm.

The final kernel never materializes the unnormalized ``(tokens, hidden_size)``
layer input in HBM when RMSNorm is requested.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mhc_router_projection_kernel(
    residual_ptr,
    fn_ptr,
    mixes_ptr,
    sqrsum_ptr,
    num_tokens: tl.constexpr,
    k_total: tl.constexpr,
    num_mix_outputs: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute ``residual @ fn.T`` and residual square sums in one K sweep."""

    pid_m = tl.program_id(0)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in tl.range(0, k_total, BLOCK_K):
        k = k_start + offs_k
        a = tl.load(
            residual_ptr + offs_m[:, None] * k_total + k[None, :],
            mask=(offs_m[:, None] < num_tokens) & (k[None, :] < k_total),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            fn_ptr + offs_n[:, None] * k_total + k[None, :],
            mask=(offs_n[:, None] < num_mix_outputs) & (k[None, :] < k_total),
            other=0.0,
        ).to(tl.float32)

        # DSV4's existing DeepGEMM path also uses TF32 for this FP32 router
        # projection. Keeping TF32 here makes the numerical comparison useful.
        acc += tl.dot(a, tl.trans(b), input_precision="tf32")
        sumsq += tl.sum(a * a, axis=1)

    tl.store(
        mixes_ptr + offs_m[:, None] * num_mix_outputs + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < num_tokens)
        & (offs_n[None, :] < num_mix_outputs),
    )
    tl.store(sqrsum_ptr + offs_m, sumsq, mask=offs_m < num_tokens)


@triton.jit
def _mhc_finalize_combine_rmsnorm_kernel(
    mixes_ptr,
    sqrsum_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    norm_weight_ptr,
    output_ptr,
    k_total: tl.constexpr,
    hidden_size: tl.constexpr,
    HC: tl.constexpr,
    HC2: tl.constexpr,
    NUM_MIX_OUTPUTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RMS_EPS: tl.constexpr,
    HC_PRE_EPS: tl.constexpr,
    HC_SINKHORN_EPS: tl.constexpr,
    HC_POST_MULT: tl.constexpr,
    SINKHORN_REPEAT: tl.constexpr,
    APPLY_NORM: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    """Finalize routing, combine HC streams, and optionally apply RMSNorm."""

    token = tl.program_id(0).to(tl.int64)
    hc_offs = tl.arange(0, HC)
    comb_offs = tl.arange(0, HC2)

    rms = tl.rsqrt(tl.load(sqrsum_ptr + token) / k_total + RMS_EPS)
    scale_pre = tl.load(hc_scale_ptr).to(tl.float32)
    scale_post = tl.load(hc_scale_ptr + 1).to(tl.float32)
    scale_comb = tl.load(hc_scale_ptr + 2).to(tl.float32)

    pre_logits = tl.load(mixes_ptr + token * NUM_MIX_OUTPUTS + hc_offs)
    post_logits = tl.load(mixes_ptr + token * NUM_MIX_OUTPUTS + HC + hc_offs)
    pre_base = tl.load(hc_base_ptr + hc_offs)
    post_base = tl.load(hc_base_ptr + HC + hc_offs)
    pre = tl.sigmoid(pre_logits * rms * scale_pre + pre_base) + HC_PRE_EPS
    post = tl.sigmoid(post_logits * rms * scale_post + post_base) * HC_POST_MULT

    comb_logits = tl.load(
        mixes_ptr + token * NUM_MIX_OUTPUTS + 2 * HC + comb_offs
    )
    comb_base = tl.load(hc_base_ptr + 2 * HC + comb_offs)
    comb = (comb_logits * rms * scale_comb + comb_base).reshape((HC, HC))

    row_max = tl.max(comb, axis=1)
    comb = tl.exp(comb - row_max[:, None])
    row_sum = tl.sum(comb, axis=1)
    comb = comb / row_sum[:, None] + HC_SINKHORN_EPS
    col_sum = tl.sum(comb, axis=0)
    comb = comb / (col_sum[None, :] + HC_SINKHORN_EPS)
    for _ in tl.static_range(0, SINKHORN_REPEAT - 1):
        row_sum = tl.sum(comb, axis=1)
        comb = comb / (row_sum[:, None] + HC_SINKHORN_EPS)
        col_sum = tl.sum(comb, axis=0)
        comb = comb / (col_sum[None, :] + HC_SINKHORN_EPS)

    tl.store(post_ptr + token * HC + hc_offs, post)
    tl.store(comb_ptr + token * HC2 + comb_offs, comb.reshape((HC2,)))

    d = tl.arange(0, BLOCK_D)
    d_mask = d < hidden_size
    residual_row = residual_ptr + token * HC * hidden_size

    layer_input = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for route in tl.static_range(0, HC):
        route_input = tl.load(
            residual_row + route * hidden_size + d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        route_weight = tl.sum(tl.where(hc_offs == route, pre, 0.0), axis=0)
        layer_input += route_weight * route_input

    # Match the existing TileLang implementation: its weighted sum is rounded
    # to BF16 before the following RMSNorm reads it back from shared memory.
    layer_input = layer_input.to(tl.bfloat16).to(tl.float32)

    if APPLY_NORM:
        sumsq = tl.sum(layer_input * layer_input, axis=0)
        inv_rms = tl.rsqrt(sumsq / hidden_size + NORM_EPS)
        norm_weight = tl.load(norm_weight_ptr + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        layer_input *= inv_rms * norm_weight

    tl.store(
        output_ptr + token * hidden_size + d,
        layer_input,
        mask=d_mask,
    )


def mhc_pre_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    n_splits_pre: int = 32,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton implementation matching ``sglang.srt.layers.mhc.mhc_pre``.

    ``n_splits`` and ``n_splits_pre`` are accepted for API compatibility. The
    Triton projection uses a token-tiled GEMM and therefore does not expose the
    split-K workspace used by the TileLang/DeepGEMM implementation.
    """

    del n_splits, n_splits_pre
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32
    assert residual.is_cuda, "Triton mHC requires a CUDA tensor"
    assert residual.is_contiguous(), "residual must be contiguous"
    assert fn.is_contiguous(), "fn must be contiguous"
    assert hc_scale.is_contiguous(), "hc_scale must be contiguous"
    assert hc_base.is_contiguous(), "hc_base must be contiguous"

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    num_mix_outputs = hc_mult * 2 + hc_mult2
    k_total = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    # The DSV4 checkpoint and the specialized Sinkhorn kernel both use HC=4.
    # Keeping this explicit prevents silently compiling a wrong padded matrix.
    assert hc_mult == 4, f"Triton mHC currently supports hc_mult=4, got {hc_mult}"
    assert fn.shape == (num_mix_outputs, k_total)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (num_mix_outputs,)
    assert sinkhorn_repeat >= 1
    if norm_weight is not None:
        assert norm_eps is not None, "norm_eps is required with norm_weight"
        assert norm_weight.shape == (hidden_size,)
        norm_weight_bf16 = norm_weight.to(torch.bfloat16).contiguous()
    else:
        norm_weight_bf16 = residual.new_empty((1,))

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    post_mix = torch.empty(
        (num_tokens, hc_mult), dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        (num_tokens, hc_mult2), dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=residual.device
    )
    if num_tokens == 0:
        return (
            post_mix.view(*outer_shape, hc_mult, 1),
            comb_mix.view(*outer_shape, hc_mult, hc_mult),
            layer_input.view(*outer_shape, hidden_size),
        )

    mixes = torch.empty(
        (num_tokens, num_mix_outputs),
        dtype=torch.float32,
        device=residual.device,
    )
    sqrsum = torch.empty((num_tokens,), dtype=torch.float32, device=residual.device)
    # tl.dot requires a tensor-core-compatible M tile even for single-token
    # decode. Masking the unused rows is cheaper than maintaining a scalar path.
    block_m = 16
    _mhc_router_projection_kernel[(triton.cdiv(num_tokens, block_m),)](
        residual_flat,
        fn,
        mixes,
        sqrsum,
        num_tokens=num_tokens,
        k_total=k_total,
        num_mix_outputs=num_mix_outputs,
        BLOCK_M=block_m,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    block_d = triton.next_power_of_2(hidden_size)
    assert block_d <= 8192, f"hidden_size={hidden_size} is too large for this kernel"
    _mhc_finalize_combine_rmsnorm_kernel[(num_tokens,)](
        mixes,
        sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        norm_weight_bf16,
        layer_input,
        k_total=k_total,
        hidden_size=hidden_size,
        HC=hc_mult,
        HC2=hc_mult2,
        NUM_MIX_OUTPUTS=num_mix_outputs,
        BLOCK_D=block_d,
        RMS_EPS=rms_eps,
        HC_PRE_EPS=hc_pre_eps,
        HC_SINKHORN_EPS=hc_sinkhorn_eps,
        HC_POST_MULT=hc_post_mult_value,
        SINKHORN_REPEAT=sinkhorn_repeat,
        APPLY_NORM=norm_weight is not None,
        NORM_EPS=0.0 if norm_eps is None else norm_eps,
        num_warps=8 if block_d >= 4096 else 4,
    )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )
