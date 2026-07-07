# SmoothReLU: a custom C++ PyTorch operator

A fused SmoothReLU activation implemented as a PyTorch C++ extension with
explicitly vectorized CPU kernels, dispatcher registration, and a manual
autograd function validated by `gradcheck`.

**Interactive visualizer**: open [`docs/visualizer.html`](docs/visualizer.html)
in a browser (from a local clone) — the piecewise math with a live α slider, a
lane-by-lane walkthrough of the SIMD `blendv` selection, the benchmark numbers,
and a clickable map of the dispatch/autograd pipeline.

## Benchmarks

Median latency, float32, measured with `torch.utils.benchmark` (warmup + median
of repeated runs). Hardware: Apple M3 Pro (11-core, 5 performance cores),
18 GB, macOS; torch 2.8.0, 5 intra-op threads; Apple clang 17, `-O3`.

| Elements | Pass     | Eager PyTorch | C++ extension | torch.compile |
|---------:|----------|--------------:|--------------:|--------------:|
| 1e4      | forward  |       28.9 us |    **5.2 us** |       18.8 us |
| 1e4      | backward |       34.1 us |   **12.8 us** |       29.6 us |
| 1e6      | forward  |       1.84 ms |  **181.7 us** |      729.7 us |
| 1e6      | backward |       1.55 ms |  **184.7 us** |      771.3 us |
| 1e8      | forward  |     184.89 ms |  **17.52 ms** |      70.95 ms |
| 1e8      | backward |     252.87 ms |  **17.70 ms** |      74.86 ms |

The extension is 5.5-14x faster than eager and 2.5-4x faster than
`torch.compile` on this machine — single-threaded, against eager using all 5
intra-op threads. Reproduce with `python benchmarks/bench.py`.

Why the gap: the eager implementation
(`torch.where(x <= 0, 0, torch.where(x < alpha, x*x/(2*alpha), x - alpha/2))`)
launches a chain of elementwise kernels and materializes every intermediate
(masks, products, both branches), so it is memory-bandwidth-bound on tensors it
touches many times. The C++ kernel makes exactly one pass: one load, a few
registers of SIMD arithmetic, one store.

## The operator

SmoothReLU is a piecewise activation that smooths ReLU's kink at zero with a
quadratic ramp of width `alpha`:

```
f(x) = 0              x <= 0
       x^2 / (2a)     0 < x < a
       x - a/2        x >= a
```

Its derivative is continuous (0, then x/a, then 1), which is the point: no
sudden gradient jump at the origin.

## Implementation notes

**Vectorized kernels** (`csrc/smooth_relu.cpp`): the forward and backward
kernels process the bulk of the tensor with `at::vec::Vectorized<scalar_t>`
(NEON on ARM, AVX on x86) — both piecewise branches are computed for a whole
SIMD lane and combined with `blendv` masks, avoiding per-element branching. A
scalar tail loop handles the remainder in `at::opmath_type<scalar_t>` so
half/bfloat16 tails accumulate in float. `AT_DISPATCH_FLOATING_TYPES_AND2`
instantiates the templates for float32/float64/float16/bfloat16.

**Dispatcher registration**: the ops are declared with `TORCH_LIBRARY` under
the `smoothrelu` namespace with proper schemas, and the CPU kernels are
registered via `TORCH_LIBRARY_IMPL(smoothrelu, CPU, ...)`. Compared to raw
pybind functions, going through the dispatcher makes the op a first-class
citizen: it is callable as `torch.ops.smoothrelu.smooth_relu` from Python or
TorchScript, shows up in the profiler by name, participates in schema
checking, and leaves a clean seam for registering CUDA or Autograd kernels
under the same op name later.

**Contiguity**: the kernels iterate raw data pointers linearly, which is only
correct for contiguous memory. Rather than silently handling strides (slower)
or crashing (wrong), the kernel `TORCH_CHECK`s contiguity, and the Python
wrapper calls `.contiguous()` before dispatching — non-contiguous callers get
correct results at the cost of one copy, and anyone hitting the raw op gets a
clear error instead of garbage.

**Autograd**: `smoothrelu/__init__.py` wires the op into autograd with a
manual `torch.autograd.Function` that saves the input and calls the C++
backward kernel. The analytic gradient is validated against finite differences
with `torch.autograd.gradcheck` in float64 (see `tests/`), alongside
elementwise comparisons to a pure-PyTorch reference in all four dtypes.

## Install and run

Requires Python with PyTorch and Ninja (the extension JIT-compiles on first
import, with `-O3`):

```bash
pip install torch ninja pytest
python -c "from smoothrelu import smooth_relu; import torch; print(smooth_relu(torch.randn(5), 1.0))"
```

Run the tests:

```bash
python -m pytest tests/
```

Run the benchmark (takes a few minutes; sizes up to 1e8 elements):

```bash
python benchmarks/bench.py
```

Profiling walkthroughs (torch.profiler, Chrome traces, TensorBoard) live in
`benchmarks/profile_smoothrelu.py` and `benchmarks/demo_profiling.py`.

## Layout

```
csrc/            C++ kernels, schema + dispatcher registration
smoothrelu/      Python package: JIT build, autograd Function, public API
tests/           pytest suite: reference comparison, gradcheck, error paths
benchmarks/      bench.py (eager vs extension vs torch.compile) + profiling
notebooks/       reference implementation notebook
```
