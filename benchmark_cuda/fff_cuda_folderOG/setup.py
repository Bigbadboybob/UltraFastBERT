from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fff_cudaOG',
    ext_modules=[
        CUDAExtension('fff_cudaOG', [
            'fff_cuda.cpp',
            'fff_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
