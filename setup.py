from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="gsplat_raytracing",
    packages=['gsplat_raytracing'],
    ext_modules=[
        CUDAExtension(
            name="gsplat_raytracing.cu",
            sources=[
            "src/main.cu",
            "src/ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
