import torch
from smooth_relu import smooth_relu
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity

# Pure PyTorch reference used for gradcheck + microbenchmarks so we have
# a correctness oracle for the C++/CUDA implementation.
def smooth_relu_ref(x, alpha=1.0):
    zero = torch.zeros_like(x)
    mid = x * x / (2.0*alpha)
    y = torch.where(x <= 0, zero, torch.where((x>0)&(x<alpha), mid, x - alpha/2))
    return y

# =============================================================================
# PROFILING EXAMPLES
# =============================================================================

def profile_smoothrelu_operations():
    """Profile forward and backward passes of SmoothReLU"""
    print("\n=== PROFILING SMOOTHRELU OPERATIONS ===")

    # Create test data
    x = torch.randn(10000, requires_grad=True)
    alpha = 1.5

    with profile(activities=[ProfilerActivity.CPU],
                 record_shapes=True,
                 profile_memory=True) as prof:

        with record_function("forward_pass"):
            y = smooth_relu(x, alpha)

        with record_function("backward_pass"):
            loss = y.sum()
            loss.backward()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

def profile_memory_usage():
    """Profile memory usage during SmoothReLU operations"""
    print("\n=== MEMORY PROFILING ===")

    # Large tensor magnifies allocation patterns so cpu_memory stats stand out.
    x = torch.randn(50000, requires_grad=True)

    with profile(activities=[ProfilerActivity.CPU],
                 profile_memory=True,
                 record_shapes=True) as prof:

        y = smooth_relu(x, 1.0)
        loss = y.sum()
        loss.backward()

    print("Memory usage summary:")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

def profile_compilation():
    """Profile the C++ extension compilation time"""
    print("\n=== PROFILING COMPILATION ===")

    import time
    from torch.utils.cpp_extension import load_inline

    src = r"""
    #include <torch/extension.h>
    using torch::Tensor;

    Tensor test_forward(Tensor x, float alpha) {
        return torch::relu(x) * alpha;
    }
    """

    start_time = time.time()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        ext = load_inline(
            name="test_compile",
            cpp_sources=[src],
            functions=["test_forward"],
            verbose=False  # Reduce compilation output
        )

    compile_time = time.time() - start_time
    print(f"Compilation time: {compile_time:.2f}s")

    # Show compilation profiling
    print("Compilation profiling:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

def profile_detailed_execution():
    """Detailed profiling with stack traces and export"""
    print("\n=== DETAILED PROFILING WITH EXPORT ===")

    x = torch.randn(1000, requires_grad=True)

    # Stack traces + memory/shape info allow deep dive in Chrome tracing.
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True  # Include Python stack traces
    ) as prof:

        y = smooth_relu(x, 1.0)
        loss = y.sum()
        loss.backward()

    # Export for visualization (can be viewed in Chrome://tracing)
    prof.export_chrome_trace("smoothrelu_trace.json")
    print("Trace exported to smoothrelu_trace.json")

    # Print detailed summary
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cpu_time_total", row_limit=20))

def profile_kernel_analysis():
    """Analyze kernel launches and operator fusion"""
    print("\n=== KERNEL ANALYSIS ===")

    x = torch.randn(10000, requires_grad=True)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_modules=True  # Include module hierarchy
    ) as prof:

        # Multiple operations to see kernel patterns
        for i in range(10):
            y = smooth_relu(x, 1.0 + i * 0.1)
            y.sum().backward()

    # Group by operator to see which operations dominate
    print("Operations by total time:")
    print(prof.key_averages(group_by_stack_n=5).table(
        sort_by="cpu_time_total", row_limit=15))

    # Group by input shape
    print("\nOperations by input shape:")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cpu_time_total", row_limit=10))

if __name__ == "__main__":
    # sanity check 
    x = torch.tensor([-1., 0.2, 2.0], requires_grad=True)
    y = smooth_relu(x, 1.0).sum()
    y.backward()
    print("grad:", x.grad)

    # gradient check (finite-diff)
    x = torch.randn(5, dtype=torch.double, requires_grad=True)
    alpha = 1.3
    result = torch.autograd.gradcheck(lambda z: smooth_relu(z, alpha).to(torch.double), (x,))
    print(result)

    # compare runtime between extension and Python reference on large tensors.
    x_bench = torch.randn(1000000, dtype=torch.double, requires_grad=True)
    t_cpp = benchmark.Timer(
        stmt="smooth_relu(x_bench, 1.0)",
        globals={"smooth_relu": smooth_relu, "x_bench": x_bench},
    ).timeit(50)

    t_py = benchmark.Timer(
        stmt="smooth_relu_ref(x_bench, 1.0)",
        globals={"smooth_relu_ref": smooth_relu_ref, "x_bench": x_bench},
    ).timeit(50)

    print(f"t_cpp: {t_cpp}")
    print(f"t_py: {t_py}")


    # Run profiling examples
    # profile_smoothrelu_operations()
    # profile_memory_usage()
    # profile_compilation()
    # profile_detailed_execution()
    # profile_kernel_analysis()
