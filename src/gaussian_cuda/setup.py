import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    '-O3', '-std=c++17',
    # '-Xptxas', '-v',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
]

c_flags = ['-O3', '-std=c++17']

include_dirs = [os.path.join(_src_path, "src", "include")]
setup(
    name='gaussian_cuda',  # package name, import this to use python API
    ext_modules=[
        CUDAExtension(
            name='gaussian_cuda',  # extension name, import this to use CUDA API
            sources=[os.path.join(_src_path, 'src', f) for f in [
                # 'matmul.cu',
                # 'include/common.hpp',
                'gaussian_cuda_kernel.cu',
                'gaussian_cuda.cpp',
            ]],
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)
