#include <torch/extension.h>

using torch::Tensor;

namespace {

inline void check_inputs(const Tensor& x, float alpha) {
    TORCH_CHECK(x.is_floating_point(), "x must be a floating point tensor");
    TORCH_CHECK(alpha > 0.0f, "alpha must be positive");
    TORCH_CHECK(!x.is_cuda(), "smooth_relu extension only implements CPU kernels");
}

inline void check_grad_output(const Tensor& grad_output, const Tensor& input) {
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be a floating point tensor");
    TORCH_CHECK(grad_output.sizes() == input.sizes(),
                "grad_output must have the same shape as the input");
    TORCH_CHECK(grad_output.device() == input.device(),
                "grad_output must be on the same device as the input");
}

}  // namespace

Tensor smooth_relu_forward(const Tensor& x, float alpha) {
    check_inputs(x, alpha);

    auto zero = torch::zeros_like(x);
    auto mid = x * x / (2.0 * alpha);
    auto hi = x - (alpha / 2.0);

    auto is_le0 = x <= 0;
    auto is_lt_alpha = (x > 0) & (x < alpha);

    return torch::where(is_le0, zero, torch::where(is_lt_alpha, mid, hi));
}

Tensor smooth_relu_backward(const Tensor& x, const Tensor& grad_output, float alpha) {
    check_inputs(x, alpha);
    check_grad_output(grad_output, x);

    auto grad = torch::zeros_like(x);
    auto slope_region = (x > 0) & (x < alpha);
    auto high_region = x >= alpha;

    grad = torch::where(slope_region, x / alpha, grad);
    grad = torch::where(high_region, torch::ones_like(x), grad);

    return grad * grad_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smooth_relu_forward", &smooth_relu_forward, "SmoothReLU forward (CPU)");
    m.def("smooth_relu_backward", &smooth_relu_backward, "SmoothReLU backward (CPU)");
}
