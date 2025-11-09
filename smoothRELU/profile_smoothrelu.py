"""
Advanced profiling examples for SmoothReLU C++ extension
"""

import torch
from smooth_relu import smooth_relu
from torch.profiler import profile, record_function, ProfilerActivity
import torch.utils.benchmark as benchmark


def profile_with_tensorboard():
    """Profile and export to TensorBoard for visualization"""
    print("=== TENSORBOARD PROFILING ===")

    from torch.profiler import tensorboard_trace_handler

    # Small 1-D tensor is enough to exercise the kernel while keeping logs tiny.
    x = torch.randn(10000, requires_grad=True)

    # Schedule skips the first iteration (wait), records warmup, then exports the active window via TensorBoard handler.
    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1
        ),
        on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

        for i in range(5):
            # Custom record_function blocks show up as named ranges in TensorBoard.
            with record_function(f"iteration_{i}"):
                y = smooth_relu(x, 1.0 + i * 0.1)
                loss = y.sum()
                loss.backward()

    print("TensorBoard traces saved to ./profiler_logs/")
    print("Run: tensorboard --logdir=./profiler_logs/")


def profile_gpu_if_available():
    """Profile GPU operations if CUDA is available"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU profiling")
        return

    print("=== GPU PROFILING ===")

    device = torch.device("cuda")
    # Larger tensor keeps the GPU busy long enough for meaningful timings.
    x = torch.randn(50000, requires_grad=True, device=device)

    # Collect both CPU + CUDA timelines so we can attribute host vs device cost.
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True
    ) as prof:

        with record_function("gpu_forward"):
            y = smooth_relu(x, 1.0)

        with record_function("gpu_backward"):
            loss = y.sum()
            loss.backward()

    print("GPU profiling results:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def compare_implementations():
    """Profile and compare different SmoothReLU implementations"""
    print("=== COMPARATIVE PROFILING ===")

    def smooth_relu_numpy(x, alpha=1.0):
        """Pure NumPy implementation for comparison"""
        import numpy as np
        x_np = x.detach().numpy()
        result = np.where(x_np <= 0, 0,
                 np.where((x_np > 0) & (x_np < alpha),
                         x_np * x_np / (2.0 * alpha),
                         x_np - alpha / 2.0))
        return torch.from_numpy(result)

    def smooth_relu_pytorch(x, alpha=1.0):
        """Pure PyTorch implementation"""
        zero = torch.zeros_like(x)
        mid = x * x / (2.0 * alpha)
        return torch.where(x <= 0, zero,
                          torch.where((x > 0) & (x < alpha), mid, x - alpha / 2.0))

    x = torch.randn(100000, requires_grad=True)

    implementations = {
        "C++ Extension": lambda: smooth_relu(x, 1.0),
        "PyTorch": lambda: smooth_relu_pytorch(x, 1.0),
        "NumPy": lambda: smooth_relu_numpy(x, 1.0)
    }

    for name, func in implementations.items():
        print(f"\n--- Profiling {name} ---")

        with profile(activities=[ProfilerActivity.CPU],
                     record_shapes=True) as prof:

            # Wrap each variant so its stats are grouped under a unique label.
            with record_function(f"{name}_forward"):
                y = func()
                loss = y.sum()
                loss.backward()

        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))


def profile_memory_leaks():
    """Check for memory leaks during repeated operations"""
    print("=== MEMORY LEAK DETECTION ===")

    import gc
    import psutil
    import os

    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    x = torch.randn(10000, requires_grad=True)
    initial_memory = get_memory_usage()

    # Memory stats are critical here, so profile_memory=True stays on.
    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True
    ) as prof:

        for i in range(100):
            # Reuse the same tensor to detect cumulative allocator drift.
            y = smooth_relu(x, 1.0)
            loss = y.sum()
            loss.backward()

            if i % 20 == 0:
                gc.collect()  # Force garbage collection
                current_memory = get_memory_usage()
                print(".1f")

    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    print(f"\nTotal memory increase: {memory_increase:.1f} MB")

    if memory_increase > 50:  # Arbitrary threshold
        print("⚠️  Potential memory leak detected!")
    else:
        print("✅ Memory usage looks stable")

    print("\nMemory profiling details:")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


def profile_async_execution():
    """Profile asynchronous execution patterns"""
    print("=== ASYNC EXECUTION PROFILING ===")

    import asyncio

    async def async_smoothrelu_task(task_id, size):
        x = torch.randn(size, requires_grad=True)

        # Each async task gets its own profiler context so results stay isolated.
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function(f"task_{task_id}"):
                y = smooth_relu(x, 1.0)
                loss = y.sum()
                loss.backward()

        return prof

    async def run_async_tasks():
        tasks = []
        for i in range(3):
            task = asyncio.create_task(async_smoothrelu_task(i, 5000 + i * 1000))
            tasks.append(task)

        profilers = await asyncio.gather(*tasks)

        # Combine results
        all_events = []
        for prof in profilers:
            all_events.extend(prof.events())

        print(f"Total events from async execution: {len(all_events)}")
        return profilers

    # Run async profiling
    profilers = asyncio.run(run_async_tasks())

    # Show combined results
    for i, prof in enumerate(profilers):
        print(f"\nTask {i} profiling:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=3))


if __name__ == "__main__":
    print("Advanced SmoothReLU Profiling Suite")
    print("=" * 50)

    # Run all profiling examples
    try:
        profile_with_tensorboard()
        profile_gpu_if_available()
        compare_implementations()
        profile_memory_leaks()
        profile_async_execution()
    except Exception as e:
        print(f"Profiling failed: {e}")
        import traceback
        traceback.print_exc()
