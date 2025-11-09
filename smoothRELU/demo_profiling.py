"""
Simple profiling demo for SmoothReLU - runs after compilation
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Import the compiled extension (assumes smooth_relu.py was run first)
try:
    from smooth_relu import smooth_relu
except ImportError:
    print("Please run smooth_relu.py first to compile the extension")
    exit(1)

def demo_basic_profiling():
    """Demonstrate basic profiler usage"""
    print("=== BASIC PROFILING DEMO ===")

    # Use a moderately sized tensor so the trace contains enough signal.
    x = torch.randn(1000, requires_grad=True)
    alpha = 1.5

    # Profile the forward and backward passes
    with profile(activities=[ProfilerActivity.CPU],
                 record_shapes=True,
                 profile_memory=True) as prof:

        # record_function blocks show up as named ranges in the trace viewer.
        with record_function("smoothrelu_forward"):
            y = smooth_relu(x, alpha)

        with record_function("smoothrelu_backward"):
            loss = y.sum()
            loss.backward()

    # Display results
    print("Top 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print("\nTop 10 operations by memory usage:")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

def demo_memory_profiling():
    """Show memory usage patterns"""
    print("\n=== MEMORY PROFILING DEMO ===")

    # Bigger tensor magnifies allocation differences when alpha changes.
    x = torch.randn(50000, requires_grad=True)

    with profile(activities=[ProfilerActivity.CPU],
                 profile_memory=True,
                 record_shapes=True) as prof:

        # Multiple operations to see memory patterns
        for i in range(5):
            y = smooth_relu(x, 1.0 + i * 0.1)
            y.sum().backward()

    print("Memory usage summary:")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=15))

def demo_export_trace():
    """Export trace for external visualization"""
    print("\n=== TRACE EXPORT DEMO ===")

    # Smaller tensor keeps exported trace manageable for Chrome tracing.
    x = torch.randn(1000, requires_grad=True)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

        y = smooth_relu(x, 1.0)
        loss = y.sum()
        loss.backward()

    # Export to Chrome trace format
    prof.export_chrome_trace("smoothrelu_demo_trace.json")
    print("Trace exported to smoothrelu_demo_trace.json")
    print("View it by opening chrome://tracing and loading the file")

if __name__ == "__main__":
    print("SmoothReLU Profiler Demo")
    print("=" * 30)

    # Run demos sequentially so readers can correlate console output with trace files.
    demo_basic_profiling()
    demo_memory_profiling()
    demo_export_trace()

    print("\nDemo complete! Check smoothrelu_demo_trace.json for detailed visualization.")
