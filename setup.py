from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="marlin",
    version="0.2.1",
    author="Roberto Lopez Castro",
    author_email="roberto.lopez.castro@udc.es",
    description="Highly optimized FP16x(INT4+2:4 sparsity) CUDA matmul kernel.",
    install_requires=["numpy", "torch"],
    packages=["marlin"],
    ext_modules=[
        cpp_extension.CUDAExtension(
            "marlin_cuda",
            [
                "marlin/marlin_cuda.cpp",
                "marlin/marlin_cuda_kernel.cu",
                "marlin/marlin_cuda_kernel_nm.cu",
            ],
            extra_compile_args={
                "nvcc": ["-arch=sm_86", "--ptxas-options=-v", "-lineinfo"]
            },
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
