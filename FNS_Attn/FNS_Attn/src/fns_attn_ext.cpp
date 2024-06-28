#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int fns_attn_forward_cuda(at::Tensor features, at::Tensor masks,
                              int kernel_size, 
                              at::Tensor output);

int fns_attn_backward_cuda(at::Tensor top_grad, at::Tensor features,
                               at::Tensor masks, int kernel_size,
                               at::Tensor bottom_grad, at::Tensor mask_grad);
#endif

int fns_attn_forward(at::Tensor features, at::Tensor masks,
               int kernel_size, at::Tensor output) {
  if (features.device().is_cuda()) {
#ifdef WITH_CUDA
    return fns_attn_forward_cuda(features, masks, kernel_size, output);
#else
    AT_ERROR("fns_attn is not compiled with GPU support");
#endif
  }
  AT_ERROR("fns_attn is not implemented on CPU");
}

int fns_attn_backward(at::Tensor top_grad, at::Tensor features,
                at::Tensor masks, int kernel_size,
                at::Tensor bottom_grad, at::Tensor mask_grad) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
    return fns_attn_backward_cuda(top_grad, features, masks, kernel_size,
                            bottom_grad, mask_grad);
#else
    AT_ERROR("fns_attn is not compiled with GPU support");
#endif
  }
  AT_ERROR("fns_attn is not implemented on CPU");

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fns_attn_forward, "fns_attn forward");
  m.def("backward", &fns_attn_backward, "fns_attn backward");
}