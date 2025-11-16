from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cluster_attention',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='cu_weighted_sum',
            sources=['weighted_sum.cu'],
        ),
        CUDAExtension(
            name='cu_broadcast',
            sources=['broadcast.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=["torch"]
)
