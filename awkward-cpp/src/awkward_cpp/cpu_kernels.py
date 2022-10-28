# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: no change; keep this file.

import ctypes
import platform
import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

import awkward_cpp._kernel_signatures

if platform.system() == "Windows":
    name = "awkward-cpu-kernels.dll"
elif platform.system() == "Darwin":
    name = "libawkward-cpu-kernels.dylib"
else:
    name = "libawkward-cpu-kernels.so"

libpath_ref = importlib_resources.files(awkward_cpp) / "lib" / name
with importlib_resources.as_file(libpath_ref) as libpath:
    lib = ctypes.cdll.LoadLibrary(str(libpath))

kernel = awkward_cpp._kernel_signatures.by_signature(lib)
