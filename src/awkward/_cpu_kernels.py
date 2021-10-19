# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: no change; keep this file.

from __future__ import absolute_import

import ctypes
import platform
import pkg_resources

import awkward._kernel_signatures

if platform.system() == "Windows":
    name = "awkward-cpu-kernels.dll"
elif platform.system() == "Darwin":
    name = "libawkward-cpu-kernels.dylib"
else:
    name = "libawkward-cpu-kernels.so"
libpath = pkg_resources.resource_filename("awkward", name)

lib = ctypes.cdll.LoadLibrary(libpath)
kernel = awkward._kernel_signatures.by_signature(lib)
