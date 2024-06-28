from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NVCC_ARGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
]

setup(
    name='FNS_Attn',
    version='0.0.1',
    license='L1',
    ext_modules=[
        CUDAExtension(
            'fns_attn_ext', [
                'FNS_Attn/src/cuda/fns_attn_cuda.cpp',
                'FNS_Attn/src/cuda/fns_attn_cuda_kernel.cu',
                'FNS_Attn/src/fns_attn_ext.cpp'
            ],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            })
    ],
    packages=find_packages(exclude=('test', )),
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
