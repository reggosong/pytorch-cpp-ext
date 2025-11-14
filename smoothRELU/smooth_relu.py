import os
import torch
from pathlib import Path
from torch.utils.cpp_extension import load

os.environ.setdefault("PYTORCH_JIT_USE_NINJA", "0")

_THIS_DIR = Path(__file__).resolve().parent
_BUILD_DIR = _THIS_DIR / ".torch_extensions"
_BUILD_DIR.mkdir(parents=True, exist_ok=True)

# Build/load extension so dispatcher kernels register when this module imports.
_ext = load(
    name="smooth_relu_ext",
    sources=[str(_THIS_DIR / "smooth_relu.cpp")],
    build_directory=str(_BUILD_DIR),
    verbose=False,
)


class _SmoothReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha: float = 1.0):
        x_contig = x.contiguous()
        ctx.save_for_backward(x_contig)
        ctx.alpha = float(alpha)
        return torch.ops.smoothrelu.smooth_relu(x_contig, ctx.alpha)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_output_contig = grad_output.contiguous()
        grad_x = torch.ops.smoothrelu.smooth_relu_backward(x, grad_output_contig, ctx.alpha)
        return grad_x, None, None
    
def smooth_relu(x, alpha: float = 1.0):
    """
    SmoothReLU autograd entry point.

    Args:
        x: Input tensor.
        alpha: Smoothness parameter.
    """
    return _SmoothReLU.apply(x, float(alpha))
