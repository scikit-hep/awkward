# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import ctypes
import importlib.resources as importlib_resources
import platform

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
