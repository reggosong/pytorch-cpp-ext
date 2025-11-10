#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/vec.h>

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

template <typename scalar_t>
Tensor smooth_relu_forward_vec_impl(const Tensor& x, double alpha) {
    using Vec = at::vec::Vectorized<scalar_t>;
    auto x_contig = x.contiguous();
    auto out = torch::empty_like(x_contig);

    const scalar_t* src = x_contig.data_ptr<scalar_t>();
    scalar_t* dst = out.data_ptr<scalar_t>();
    const int64_t n = x_contig.numel();
    constexpr int64_t kWidth = Vec::size();

    const scalar_t zero_val = scalar_t(0);
    const scalar_t alpha_val = static_cast<scalar_t>(alpha);
    const scalar_t half_alpha_val = static_cast<scalar_t>(alpha * 0.5);
    const scalar_t two_alpha_val = static_cast<scalar_t>(alpha * 2.0);

    Vec zero_vec(zero_val);
    Vec alpha_vec(alpha_val);
    Vec half_alpha_vec(half_alpha_val);
    Vec two_alpha_vec(two_alpha_val);

    int64_t i = 0;
    for (; i + kWidth <= n; i += kWidth) {
        Vec x_vec = Vec::loadu(src + i);
        Vec mid = (x_vec * x_vec) / two_alpha_vec;
        Vec hi = x_vec - half_alpha_vec;

        Vec mask_lo = x_vec <= zero_vec;
        Vec mask_mid = (x_vec > zero_vec) & (x_vec < alpha_vec);

        Vec blended = Vec::blendv(
            Vec::blendv(hi, mid, mask_mid),
            zero_vec,
            mask_lo
        );

        blended.store(dst + i);
    }

    using acc_t = at::opmath_type<scalar_t>;
    const acc_t zero_acc = acc_t(0);
    const acc_t alpha_acc = static_cast<acc_t>(alpha);
    const acc_t half_alpha_acc = alpha_acc * acc_t(0.5);
    const acc_t denom_acc = alpha_acc * acc_t(2.0);

    for (; i < n; ++i) {
        acc_t xi = static_cast<acc_t>(src[i]);
        acc_t out_acc = zero_acc;

        if (xi > zero_acc) {
            if (xi < alpha_acc) {
                out_acc = (xi * xi) / denom_acc;
            } else {
                out_acc = xi - half_alpha_acc;
            }
        }

        dst[i] = static_cast<scalar_t>(out_acc);
    }

    return out;
}

template <typename scalar_t>
Tensor smooth_relu_backward_vec_impl(const Tensor& x, const Tensor& grad_output, double alpha) {
    using Vec = at::vec::Vectorized<scalar_t>;
    auto x_contig = x.contiguous();
    auto grad_out_contig = grad_output.contiguous();
    auto grad_input = torch::empty_like(x_contig);

    const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
    const scalar_t* grad_ptr = grad_out_contig.data_ptr<scalar_t>();
    scalar_t* grad_in_ptr = grad_input.data_ptr<scalar_t>();

    const int64_t n = x_contig.numel();
    constexpr int64_t kWidth = Vec::size();

    const scalar_t zero_val = scalar_t(0);
    const scalar_t one_val = scalar_t(1);
    const scalar_t alpha_val = static_cast<scalar_t>(alpha);

    Vec zero_vec(zero_val);
    Vec one_vec(one_val);
    Vec alpha_vec(alpha_val);

    int64_t i = 0;
    for (; i + kWidth <= n; i += kWidth) {
        Vec x_vec = Vec::loadu(x_ptr + i);
        Vec go_vec = Vec::loadu(grad_ptr + i);
        Vec mid = x_vec / alpha_vec;

        Vec rising = (x_vec > zero_vec) & (x_vec < alpha_vec);
        Vec high = x_vec >= alpha_vec;

        Vec base = Vec::blendv(zero_vec, mid, rising);
        Vec grad_vec = Vec::blendv(base, one_vec, high);

        grad_vec = grad_vec * go_vec;
        grad_vec.store(grad_in_ptr + i);
    }

    using acc_t = at::opmath_type<scalar_t>;
    const acc_t zero_acc = acc_t(0);
    const acc_t one_acc = acc_t(1);
    const acc_t alpha_acc = static_cast<acc_t>(alpha);

    for (; i < n; ++i) {
        acc_t xi = static_cast<acc_t>(x_ptr[i]);
        acc_t go = static_cast<acc_t>(grad_ptr[i]);
        acc_t grad_acc = zero_acc;

        if (xi > zero_acc && xi < alpha_acc) {
            grad_acc = xi / alpha_acc;
        } else if (xi >= alpha_acc) {
            grad_acc = one_acc;
        }

        grad_in_ptr[i] = static_cast<scalar_t>(grad_acc * go);
    }

    return grad_input;
}

}  // namespace

Tensor smooth_relu_forward(const Tensor& x, float alpha) {
    check_inputs(x, alpha);
    Tensor result;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "smooth_relu_forward",
        [&] {
            result = smooth_relu_forward_vec_impl<scalar_t>(x, alpha);
        });
    return result;
}

Tensor smooth_relu_backward(const Tensor& x, const Tensor& grad_output, float alpha) {
    check_inputs(x, alpha);
    check_grad_output(grad_output, x);
    Tensor grad;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "smooth_relu_backward",
        [&] {
            grad = smooth_relu_backward_vec_impl<scalar_t>(x, grad_output, alpha);
        });
    return grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smooth_relu_forward", &smooth_relu_forward, "SmoothReLU forward (CPU)");
    m.def("smooth_relu_backward", &smooth_relu_backward, "SmoothReLU backward (CPU)");
}
