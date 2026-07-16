"""Compare the existing TileLang and experimental Triton DSV4 mHC pre paths.

Example on an H200:

    python3 benchmark/kernels/deepseek/benchmark_mhc_pre_triton.py \
        --hidden-sizes 4096 7168 --token-counts 1 8 32 128 512 2048
"""

from __future__ import annotations

import argparse
import socket

import torch
import triton

from sglang.kernels.ops.layernorm.mhc import mhc_pre_tilelang
from sglang.kernels.ops.layernorm.mhc_triton import mhc_pre_triton


def make_inputs(num_tokens: int, hidden_size: int):
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
    scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
    base = torch.zeros(num_mix_outputs, device=device, dtype=torch.float32)
    norm_weight = torch.ones(hidden_size, device=device, dtype=torch.bfloat16)
    return residual, fn, scale, base, norm_weight


def call_impl(impl, args):
    residual, fn, scale, base, norm_weight = args
    return impl(
        residual,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )


def check_correctness(args):
    tilelang_out = call_impl(mhc_pre_tilelang, args)
    triton_out = call_impl(mhc_pre_triton, args)
    torch.cuda.synchronize()
    tolerances = ((2e-3, 2e-3), (2e-3, 2e-3), (2e-2, 2e-2))
    for name, tile, tri, (atol, rtol) in zip(
        ("post", "comb", "layer_input"),
        tilelang_out,
        triton_out,
        tolerances,
    ):
        try:
            torch.testing.assert_close(tri, tile, atol=atol, rtol=rtol)
        except AssertionError as exc:
            raise AssertionError(f"{name} mismatch:\n{exc}") from exc


def bench(fn, warmup: int, rep: int):
    median, p20, p80 = triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        quantiles=[0.5, 0.2, 0.8],
    )
    return median * 1000.0, p20 * 1000.0, p80 * 1000.0


def init_single_rank_model_parallel():
    """Initialize the TP group required by the current TileLang baseline.

    ``mhc_pre_tilelang`` allocates its output through the symmetric-memory
    helper, which expects a tensor-parallel group even at TP=1. The benchmark
    measures a single GPU, so this creates a one-rank NCCL group only; no
    collective communication is performed by either implementation.
    """
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    torch.cuda.set_device(0)
    # Avoid assuming that a fixed TCP port is free on a shared benchmark host.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[4096, 7168])
    parser.add_argument(
        "--token-counts", type=int, nargs="+", default=[1, 8, 32, 128, 512, 2048]
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    init_single_rank_model_parallel()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("Times are median [p20, p80] in microseconds; speedup = TileLang/Triton.")
    print(
        f"{'tokens':>8} {'hidden':>8} {'TileLang us':>28} "
        f"{'Triton us':>28} {'speedup':>10}"
    )

    for hidden_size in args.hidden_sizes:
        for num_tokens in args.token_counts:
            inputs = make_inputs(num_tokens, hidden_size)
            if not args.skip_correctness:
                check_correctness(inputs)

            # Compile both implementations before timing.
            call_impl(mhc_pre_tilelang, inputs)
            call_impl(mhc_pre_triton, inputs)
            torch.cuda.synchronize()

            tile = bench(
                lambda: call_impl(mhc_pre_tilelang, inputs), args.warmup, args.rep
            )
            tri = bench(lambda: call_impl(mhc_pre_triton, inputs), args.warmup, args.rep)
            speedup = tile[0] / tri[0]
            print(
                f"{num_tokens:8d} {hidden_size:8d} "
                f"{tile[0]:9.2f} [{tile[1]:7.2f}, {tile[2]:7.2f}] "
                f"{tri[0]:9.2f} [{tri[1]:7.2f}, {tri[2]:7.2f}] "
                f"{speedup:9.3f}x"
            )


if __name__ == "__main__":
    main()
