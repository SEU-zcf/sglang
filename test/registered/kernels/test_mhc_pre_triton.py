import pytest
import torch

from sglang.kernels.ops.layernorm.mhc_triton import mhc_pre_triton
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


def _reference(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    norm_weight,
    norm_eps,
):
    num_tokens, hc_mult, hidden_size = residual.shape
    x = residual.flatten(1).float()
    inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = torch.nn.functional.linear(x, fn) * inv_rms

    pre = (
        torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
        + hc_pre_eps
    )
    post = (
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        * hc_post_mult_value
    )
    comb = (
        mixes[:, 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
    ).view(num_tokens, hc_mult, hc_mult)
    comb = torch.exp(comb - comb.max(dim=2, keepdim=True).values)
    comb = comb / comb.sum(dim=2, keepdim=True) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=2, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).bfloat16()
    if norm_weight is not None:
        inv_layer_rms = torch.rsqrt(
            layer_input.float().square().mean(dim=-1, keepdim=True) + norm_eps
        )
        layer_input = (
            layer_input.float() * inv_layer_rms * norm_weight.bfloat16().float()
        ).bfloat16()
    # ``mhc_pre`` returns post with a trailing singleton dimension so it can be
    # broadcast directly by ``hc_post``: [T, H, 1].
    return post.unsqueeze(-1), comb, layer_input


@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("num_tokens", [0, 1, 8, 33])
@pytest.mark.parametrize("use_norm", [False, True])
def test_mhc_pre_triton_matches_torch(hidden_size, num_tokens, use_norm):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton mHC kernel")

    torch.manual_seed(0)
    device = torch.device("cuda")
    hc_mult = 4
    num_mix_outputs = hc_mult * (2 + hc_mult)
    residual = (
        torch.randn(
            num_tokens,
            hc_mult,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    ).contiguous()
    fn = (
        torch.randn(
            num_mix_outputs,
            hc_mult * hidden_size,
            device=device,
            dtype=torch.float32,
        )
        * 0.01
    ).contiguous()
    hc_scale = torch.tensor([0.5, 0.25, 0.25], device=device)
    hc_base = torch.zeros(num_mix_outputs, device=device)
    norm_weight = (
        torch.randn(hidden_size, device=device, dtype=torch.bfloat16) * 0.05 + 1.0
        if use_norm
        else None
    )
    rms_eps = hc_pre_eps = hc_sinkhorn_eps = 1e-6
    norm_eps = 1e-6 if use_norm else None
    sinkhorn_repeat = 20

    if num_tokens == 0:
        post, comb, layer_input = mhc_pre_triton(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            2.0,
            sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        assert post.shape == (0, hc_mult, 1)
        assert comb.shape == (0, hc_mult, hc_mult)
        assert layer_input.shape == (0, hidden_size)
        return

    expected = _reference(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        2.0,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
    )
    actual = mhc_pre_triton(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        2.0,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-2, rtol=2e-2)
