#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

int FNSAttnForwardLauncher(const at::Tensor features, const at::Tensor attn,
                              const int kernel_size,
                              const int batch_size,
                              const int channels, const int height,
                              const int width, const int in_height,
                              const int in_width, at::Tensor output);

int FNSAttnBackwardLauncher(const at::Tensor top_grad,
                               const at::Tensor features,
                               const at::Tensor attn, const int kernel_size,
                               const int batch_size, const int channels,
                               const int height, const int width, 
                               const int in_height, const int in_width,
                               at::Tensor bottom_grad, at::Tensor mask_grad);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int fns_attn_forward_cuda(at::Tensor features, at::Tensor attn,
                    int kernel_size, at::Tensor output)
{
  CHECK_INPUT(features);
  CHECK_INPUT(attn);
  CHECK_INPUT(output);
  at::DeviceGuard guard(features.device());

  int batch_size = output.size(0);
  int num_channels = output.size(1);
  int data_height = output.size(2);
  int data_width = output.size(3);
  int in_height = features.size(2);
  int in_width = features.size(3);

  FNSAttnForwardLauncher(features, attn, kernel_size,
                    batch_size, num_channels, data_height,
                    data_width, in_height, in_width, output);
  return 1;
}

int fns_attn_backward_cuda(at::Tensor top_grad, at::Tensor features,
                     at::Tensor attn, int kernel_size,
                     at::Tensor bottom_grad, at::Tensor mask_grad)
{
  CHECK_INPUT(top_grad);
  CHECK_INPUT(features);
  CHECK_INPUT(attn);
  CHECK_INPUT(bottom_grad);
  CHECK_INPUT(mask_grad);
  at::DeviceGuard guard(top_grad.device());

  int batch_size = top_grad.size(0);
  int num_channels = top_grad.size(1);
  int data_height = top_grad.size(2);
  int data_width = top_grad.size(3);
  int in_height = features.size(2);
  int in_width = features.size(3);

  FNSAttnBackwardLauncher(top_grad, features, attn, kernel_size,
                     batch_size, num_channels, data_height, data_width,
                     in_height, in_width, bottom_grad, mask_grad);
  return 1;
}

