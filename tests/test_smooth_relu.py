import pytest
import torch

from smoothrelu import smooth_relu


def smooth_relu_ref(x, alpha=1.0):
    """Pure-PyTorch reference implementation (correctness oracle)."""
    zero = torch.zeros_like(x)
    mid = x * x / (2.0 * alpha)
    return torch.where(x <= 0, zero, torch.where((x > 0) & (x < alpha), mid, x - alpha / 2))


FLOAT_DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
# Loose tolerances for reduced-precision dtypes; the vectorized main loop
# computes in native precision.
TOLERANCES = {
    torch.float32: 1e-6,
    torch.float64: 1e-12,
    torch.float16: 1e-3,
    torch.bfloat16: 1e-2,
}


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(3,), (33,), (1024,), (64, 64), (7, 5, 3)])
def test_forward_matches_reference(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    alpha = 1.3
    got = smooth_relu(x, alpha)
    want = smooth_relu_ref(x.to(torch.float64), alpha).to(dtype)
    tol = TOLERANCES[dtype]
    torch.testing.assert_close(got, want, rtol=tol, atol=tol)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.7])
def test_backward_matches_reference(alpha):
    torch.manual_seed(1)
    x = torch.randn(257, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    smooth_relu(x, alpha).sum().backward()
    smooth_relu_ref(x_ref, alpha).sum().backward()

    torch.testing.assert_close(x.grad, x_ref.grad)


def test_gradcheck():
    torch.manual_seed(2)
    x = torch.randn(64, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: smooth_relu(z, 1.3), (x,))


def test_piecewise_regions():
    # x <= 0 -> 0; 0 < x < alpha -> x^2/(2a); x >= alpha -> x - a/2
    alpha = 1.0
    x = torch.tensor([-2.0, 0.0, 0.5, 1.0, 3.0])
    expected = torch.tensor([0.0, 0.0, 0.125, 0.5, 2.5])
    torch.testing.assert_close(smooth_relu(x, alpha), expected)


def test_non_contiguous_input_handled():
    # The Python wrapper makes inputs contiguous before hitting the kernel.
    torch.manual_seed(3)
    x = torch.randn(32, 16).t()
    assert not x.is_contiguous()
    torch.testing.assert_close(smooth_relu(x, 1.0), smooth_relu_ref(x, 1.0))


def test_raw_op_rejects_non_contiguous():
    x = torch.randn(32, 16).t()
    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops.smoothrelu.smooth_relu(x, 1.0)


def test_rejects_non_positive_alpha():
    x = torch.randn(8)
    with pytest.raises(RuntimeError, match="alpha must be positive"):
        smooth_relu(x, 0.0)


def test_rejects_integer_input():
    x = torch.arange(8)
    with pytest.raises(RuntimeError, match="floating point"):
        smooth_relu(x, 1.0)
