# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import ctypes
import platform
import pkg_resources

if platform.system() == "Windows":
    name = "awkward-cpu-kernels.dll"
elif platform.system() == "Darwin":
    name = "libawkward-cpu-kernels.dylib"
else:
    name = "libawkward-cpu-kernels.so"
libpath = pkg_resources.resource_filename("awkward1", name)

lib = ctypes.cdll.LoadLibrary(libpath)
