from setuptools import setup
from torch.utils import cpp_extension
import pybind11

setup(name='custom_dtw',
      ext_modules=[
                   cpp_extension.CppExtension('cpp_dtw', ['timewarp_lib/utils/cpp_dtw.cpp']),
                   cpp_extension.CppExtension('param_cpp_dtw', ['timewarp_lib/utils/param_cpp_dtw.cpp']),
                   cpp_extension.CppExtension('pvect_cpp_dtw', ['timewarp_lib/utils/pvect_cpp_dtw.cpp']),
                   cpp_extension.CppExtension('cpp_linear_dtw', ['timewarp_lib/utils/cpp_linear_dtw.cpp']),
                   cpp_extension.CUDAExtension('cpp_dtw_cuda', ['timewarp_lib/utils/cpp_dtw_cuda_kernel.cpp','timewarp_lib/utils/basic_dtw.cu']),
                   cpp_extension.CUDAExtension('cpp_dtw_cuda_split', ['timewarp_lib/utils/cpp_dtw_cuda_kernel_split.cpp','timewarp_lib/utils/split_basic_dtw.cu']),
                   ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=['timewarp_lib','timewarp_lib/utils'],
      include_dirs = [pybind11.get_include()])
