import torch.cuda

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

gencode = [
        '-gencode=arch=compute_30,code=sm_30',
        '-gencode=arch=compute_35,code=sm_35',
        '-gencode=arch=compute_50,code=sm_50',
        '-gencode=arch=compute_52,code=sm_52',
        '-gencode=arch=compute_60,code=sm_60',
        '-gencode=arch=compute_61,code=sm_61',
        '-gencode=arch=compute_61,code=compute_61',
]

ext_modules = [
    CppExtension(
        name = 'satnet._cpp',
        include_dirs = ['./src'],
        sources = [
            'src/satnet.cpp',
            'src/satnet_cpu.cpp',
        ],
        extra_compile_args = ['-fopenmp', '-msse4.1', '-Wall', '-g']
    )
]

if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        name = 'satnet._cuda',
        include_dirs = ['./src'],
        sources = [
            'src/satnet.cpp',
            'src/satnet_cuda.cu',
        ],
        extra_compile_args = {
            'cxx': ['-DMIX_USE_GPU', '-g'],
            'nvcc': ['-g', '-restrict', '-maxrregcount', '32', '-lineinfo', '-Xptxas=-v']
        }
    )
    ext_modules.append(extension)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Python interface
setup(
    name='satnet',
    version='0.1.3',
    install_requires=['torch>=1.3'],
    packages=['satnet'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    author='Po-Wei Wang',
    author_email='poweiw@cs.cmu.edu',
    url='https://github.com/locuslab/SATNet',
    zip_safe=False,
    description='Bridging deep learning and logical reasoning using a differentiable satisfiability solver',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
)
