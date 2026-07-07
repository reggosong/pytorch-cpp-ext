"""Benchmark the SmoothReLU C++ extension against PyTorch eager and torch.compile.

Measures forward and backward latency at several tensor sizes with proper
warmup, reporting the median of repeated timed runs. Run from the repo root:

    python benchmarks/bench.py
"""

import platform
import sys
from pathlib import Path

import torch
import torch.utils.benchmark as benchmark

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoothrelu import smooth_relu  # noqa: E402


def smooth_relu_eager(x, alpha=1.0):
    """Naive eager PyTorch implementation."""
    zero = torch.zeros_like(x)
    mid = x * x / (2.0 * alpha)
    return torch.where(x <= 0, zero, torch.where((x > 0) & (x < alpha), mid, x - alpha / 2))


SIZES = [10_000, 1_000_000, 100_000_000]
ALPHA = 1.0
MIN_RUN_TIME = 2.0  # seconds per measurement; Timer picks the run count
WARMUP_ITERS = 5


def bench_forward(fn, x):
    for _ in range(WARMUP_ITERS):
        fn(x, ALPHA)
    t = benchmark.Timer(
        stmt="fn(x, alpha)",
        globals={"fn": fn, "x": x, "alpha": ALPHA},
    ).blocked_autorange(min_run_time=MIN_RUN_TIME)
    return t.median


def bench_backward(fn, x):
    x = x.detach().clone().requires_grad_(True)
    y = fn(x, ALPHA)
    grad_out = torch.ones_like(y)
    for _ in range(WARMUP_ITERS):
        torch.autograd.grad(y, x, grad_out, retain_graph=True)
    t = benchmark.Timer(
        stmt="torch.autograd.grad(y, x, g, retain_graph=True)",
        globals={"torch": torch, "y": y, "x": x, "g": grad_out},
    ).blocked_autorange(min_run_time=MIN_RUN_TIME)
    return t.median


def fmt(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} us"
    return f"{seconds * 1e3:8.2f} ms"


def main():
    torch.manual_seed(0)
    print(f"torch {torch.__version__} | {platform.platform()} | "
          f"{torch.get_num_threads()} threads")

    impls = {"eager": smooth_relu_eager, "cpp ext": smooth_relu}
    try:
        compiled = torch.compile(smooth_relu_eager)
        compiled(torch.randn(16), ALPHA)  # trigger compilation outside timing
        impls["torch.compile"] = compiled
    except Exception as e:
        print(f"torch.compile unavailable, skipping: {e}")

    header = f"{'size':>12} {'pass':>9}" + "".join(f"{name:>16}" for name in impls)
    print(header)
    print("-" * len(header))

    for n in SIZES:
        x = torch.randn(n)
        fwd = [bench_forward(fn, x) for fn in impls.values()]
        bwd = [bench_backward(fn, x) for fn in impls.values()]
        print(f"{n:>12,} {'forward':>9}" + "".join(f"{fmt(t):>16}" for t in fwd))
        print(f"{n:>12,} {'backward':>9}" + "".join(f"{fmt(t):>16}" for t in bwd))


if __name__ == "__main__":
    main()
