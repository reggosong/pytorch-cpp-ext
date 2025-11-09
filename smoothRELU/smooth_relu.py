import torch
from pathlib import Path
from torch.utils.cpp_extension import load

_THIS_DIR = Path(__file__).resolve().parent
_BUILD_DIR = _THIS_DIR / ".torch_extensions"
_BUILD_DIR.mkdir(parents=True, exist_ok=True)

ext = load(
    name="smooth_relu_ext",
    sources=[str(_THIS_DIR / "smooth_relu.cpp")],
    build_directory=str(_BUILD_DIR),
    verbose=False,
)


class _SmoothReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha: float = 1.0):
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return ext.smooth_relu_forward(x, ctx.alpha)

    @staticmethod
    def backward(ctx, grad_output):
        (x, ) = ctx.saved_tensors
        grad_x = ext.smooth_relu_backward(x, grad_output, ctx.alpha)
        return grad_x, None
    
def smooth_relu(x, alpha: float = 1.0):
    return _SmoothReLU.apply(x, alpha)