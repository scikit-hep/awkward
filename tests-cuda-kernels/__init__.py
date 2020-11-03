# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import ctypes
import platform

import pkg_resources

# awkward1-cuda-kernels is only supported on Linux, but let's leave the placeholder.
if platform.system() == "Windows":
    shared_library_name = "awkward-cuda-kernels.dll"
elif platform.system() == "Darwin":
    shared_library_name = "libawkward-cuda-kernels.dylib"
else:
    shared_library_name = "libawkward-cuda-kernels.so"

CUDA_KERNEL_SO = pkg_resources.resource_filename(
    "awkward1_cuda_kernels", shared_library_name
)

lib = ctypes.CDLL(CUDA_KERNEL_SO)


class Error(ctypes.Structure):
    _fields_ = [
        ("str", ctypes.POINTER(ctypes.c_char)),
        ("filename", ctypes.POINTER(ctypes.c_char)),
        ("identity", ctypes.c_int64),
        ("attempt", ctypes.c_int64),
        ("pass_through", ctypes.c_bool),
    ]


del platform
del pkg_resources
