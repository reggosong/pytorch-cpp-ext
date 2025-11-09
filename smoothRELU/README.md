# SmoothReLU PyTorch C++ Extension

Implements a custom operator with manual autograd and gradient check.

## Build

```bash
python test.py
```

## Profiling

The codebase includes comprehensive profiling tools to analyze performance:

### Basic Profiling

Run the test suite to see basic profiling results:

```bash
python test.py
```

This includes:

- CPU time profiling for forward/backward passes
- Memory usage analysis
- Compilation time measurement
- Kernel operation analysis

### Advanced Profiling

For detailed performance analysis:

```bash
python profile_smoothrelu.py
```

This provides:

- TensorBoard trace export for visualization
- GPU profiling (if CUDA available)
- Comparative analysis vs PyTorch/NumPy implementations
- Memory leak detection
- Asynchronous execution profiling

### Profiler Features

**Key Profiling Options:**

- `record_shapes=True`: Track tensor shapes through operations
- `profile_memory=True`: Monitor memory allocation/deallocation
- `with_stack=True`: Include Python call stacks
- `with_modules=True`: Group by module hierarchy

**Visualization:**

- Chrome traces: Open `chrome://tracing` and load `smoothrelu_trace.json`
- TensorBoard: `tensorboard --logdir=./profiler_logs/`

**Common Profiling Commands:**

```python
from torch.profiler import profile, record_function, ProfilerActivity

# Basic profiling
with profile(activities=[ProfilerActivity.CPU]) as prof:
    # Your code here
    pass

print(prof.key_averages().table(sort_by="cpu_time_total"))

# With memory profiling
with profile(profile_memory=True, record_shapes=True) as prof:
    # Your code here
    pass
```
