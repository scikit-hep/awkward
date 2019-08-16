import os
import ctypes
import platform

import awkward1.layout

__version__ = awkward1.layout.__version__

print("platform.system()", platform.system())
print("__file__", __file__)
print("listdir", os.listdir(os.path.dirname(os.path.abspath(__file__))))

if platform.system() == "Windows":
    lib1 = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "awkward-cpu-kernels.dll"))
else:
    lib1 = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libawkward-cpu-kernels.so"))

dummy1 = lib1.dummy1
dummy3 = layout.dummy3
