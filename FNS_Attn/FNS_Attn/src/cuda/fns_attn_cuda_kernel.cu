/* 
   CUDA extension for SAPA
   by https://github.com/Teleppo
   modified from https://github.com/myownskyW7/CARAFE
*/
#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <cmath>

using namespace at;  // temporal fix for pytorch<=0.4.1 (see #9848)

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65536;
  return min(optimal_block_num, max_block_num);
}

__device__ inline int Loc2Index(const int n, const int c, const int h,
                                const int w, const int channel_num,
                                const int height, const int width) {
  int index = w + (h + (c + n * channel_num) * height) * width;
  return index;
}


template <typename scalar_t>
__global__ void FNSAttnForward(const int nthreads,
                          const scalar_t *bottom_data,
                          const scalar_t *bottom_masks,
                          const int kernel_size,
                          const int channels,
                          const int height, const int width,
                          const int in_height, const int in_width,
                          scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int mask_channels = kernel_size * kernel_size;

    int down_pw =  int(float(pw)*float(in_width)/ float(width)); 
    int down_ph = int(float(ph)*float(in_height)/ float(height));

    float rh = float(height)/float(in_height);
    float rw = float(width)/float(in_width);

    float shift_h = (float(ph)/rh - floor(float(ph)/rh)) -(rh-1.0)/(2.0*rh);
    float shift_w = (float(pw)/rw - floor(float(pw)/rw)) -(rw-1.0)/(2.0*rw);

    scalar_t output_val = 0;
    for (int iy = 0; iy < kernel_size; iy++) {
      for (int ix = 0; ix < kernel_size; ix++) {
        int mask_c = iy*kernel_size+ix;
        int mask_index = Loc2Index(n, mask_c, ph, pw, mask_channels, height, width);
        float shifted_h = min(max(float(down_ph+iy-(kernel_size - 1) / 2) + shift_h, 0.), float(in_height-1));
        float shifted_w = min(max(float(down_pw+ix-(kernel_size - 1) / 2) + shift_w, 0.), float(in_width-1));

        int h1 = int(shifted_h);
        int h2 = min(h1+1, in_height-1);
        int w1 = int(shifted_w);
        int w2 = min(w1+1, in_width-1);

        int feat_index = Loc2Index(n, c, h1, w1, channels, in_height, in_width);
        int feat_index_h = Loc2Index(n, c, h2, w1, channels, in_height, in_width);
        int feat_index_w = Loc2Index(n, c, h1, w2, channels, in_height, in_width);
        int feat_index_hw = Loc2Index(n, c, h2, w2, channels, in_height, in_width);

        auto bilinear = bottom_data[feat_index] *(h1+1-shifted_h)*(w1+1-shifted_w) +\
                        bottom_data[feat_index_h]*(shifted_h-h1)*(w1+1-shifted_w) +\
                        bottom_data[feat_index_w]*(h1+1-shifted_h)*(shifted_w-w1) +\
                        bottom_data[feat_index_hw]*(shifted_h-h1)*(shifted_w-w1);

        output_val += bilinear * bottom_masks[mask_index];
     
      }
    }
    top_data[index] = output_val;
  }
}

int FNSAttnForwardLauncher(const at::Tensor features, const at::Tensor masks,
                      const int kernel_size, const int batch_size,
                      const int channels, const int height, const int width,
                      const int in_height, const int in_width,at::Tensor output) {
  const int output_size = batch_size * channels * height * width;

  // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "FNSAttnLauncherForward", ([&] {
        const scalar_t *bottom_data = features.data_ptr<scalar_t>();
        const scalar_t *bottom_masks = masks.data_ptr<scalar_t>();
        scalar_t *top_data = output.data_ptr<scalar_t>();

        FNSAttnForward<scalar_t>
            // <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK,0, stream>>>(
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK,0>>>(
                output_size, bottom_data, bottom_masks, kernel_size,
                channels, height, width, in_height, in_width, top_data);
      }));
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}

template <typename scalar_t>
__global__ void FNSAttnBackward(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_data,
    const scalar_t *bottom_masks, const int kernel_size,
    const int channels, const int height, const int width,
    const int in_height, const int in_width, scalar_t *bottom_diff, scalar_t *mask_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int mask_channels = kernel_size * kernel_size;

    int down_pw =  int(float(pw)*float(in_width)/ float(width)); 
    int down_ph = int(float(ph)*float(in_height)/ float(height));

    float rh = float(height)/float(in_height);
    float rw = float(width)/float(in_width);


    float shift_h = (float(ph)/rh - floor(float(ph)/rh)) -(rh-1.0)/(2.0*rh);
    float shift_w = (float(pw)/rw - floor(float(pw)/rw)) -(rw-1.0)/(2.0*rw);

    // scalar_t output_val = 0;
    for (int iy = 0; iy < kernel_size; iy++) {
      for (int ix = 0; ix < kernel_size; ix++) {
        int mask_c = iy*kernel_size+ix;
        int mask_index = Loc2Index(n, mask_c, ph, pw, mask_channels, height, width);
        // if (bottom_masks[mask_index]>0.0001){
          float shifted_h = min(max(float(down_ph+iy-(kernel_size - 1) / 2) + shift_h, 0.), float(in_height-1));
          float shifted_w = min(max(float(down_pw+ix-(kernel_size - 1) / 2) + shift_w, 0.), float(in_width-1));

          int h1 = int(shifted_h);
          int h2 = min(h1+1, in_height-1);
          int w1 = int(shifted_w);
          int w2 = min(w1+1, in_width-1);

          int feat_index = Loc2Index(n, c, h1, w1, channels, in_height, in_width);
          int feat_index_h = Loc2Index(n, c, h2, w1, channels, in_height, in_width);
          int feat_index_w = Loc2Index(n, c, h1, w2, channels, in_height, in_width);
          int feat_index_hw = Loc2Index(n, c, h2, w2, channels, in_height, in_width);

          auto bilinear = bottom_data[feat_index] *(h1+1-shifted_h)*(w1+1-shifted_w) +\
                          bottom_data[feat_index_h]*(shifted_h-h1)*(w1+1-shifted_w) +\
                          bottom_data[feat_index_w]*(h1+1-shifted_h)*(shifted_w-w1) +\
                          bottom_data[feat_index_hw]*(shifted_h-h1)*(shifted_w-w1);

          atomicAdd(mask_diff+mask_index, bilinear*top_diff[index]);
          atomicAdd(bottom_diff+feat_index, bottom_masks[mask_index]*top_diff[index]*(h2-shifted_h)*(w2-shifted_w));
          atomicAdd(bottom_diff+feat_index_h, bottom_masks[mask_index]*top_diff[index]*(shifted_h-h1)*(w2-shifted_w));
          atomicAdd(bottom_diff+feat_index_w, bottom_masks[mask_index]*top_diff[index]*(h2-shifted_h)*(shifted_w-w1));
          atomicAdd(bottom_diff+feat_index_hw, bottom_masks[mask_index]*top_diff[index]*(shifted_h-h1)*(shifted_w-w1));
        // }
      }
    }
  }
}

int FNSAttnBackwardLauncher(const at::Tensor top_grad,
                      const at::Tensor features,
                      const at::Tensor masks, const int kernel_size,
                      const int batch_size, const int channels,
                      const int height, const int width,
                      const int in_height, const int in_width,
                      at::Tensor bottom_grad, at::Tensor mask_grad) {
  const int output_size = batch_size * channels * height * width;
  // cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "FNSAttnLauncherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data_ptr<scalar_t>();
        const scalar_t *bottom_data = features.data_ptr<scalar_t>();
        const scalar_t *bottom_masks = masks.data_ptr<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data_ptr<scalar_t>();
        scalar_t *mask_diff = mask_grad.data_ptr<scalar_t>();

        FNSAttnBackward<scalar_t>
         //<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0>>>(
                output_size, top_diff, bottom_data, bottom_masks, kernel_size,
                channels, height, width, in_height, in_width, bottom_diff,
                mask_diff);
      }));

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}