# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import os
import ctypes
import platform

import awkward1.layout

__version__ = awkward1.layout.__version__

if platform.system() == "Darwin":
    lib1 = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libawkward-cpu-kernels.dylib"))
elif platform.system() == "Windows":
    lib1 = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "awkward-cpu-kernels.dll"))
else:
    lib1 = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libawkward-cpu-kernels.so"))

dummy1 = lib1.dummy1
dummy3 = layout.dummy3
