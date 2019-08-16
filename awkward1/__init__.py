import os
import ctypes
import platform

import awkward1.layout

__version__ = awkward1.layout.__version__

print("RUNNING1", __file__)
print("RUNNING2", os.listdir(os.path.dirname(os.path.abspath(__file__))))

if platform.system() == "Windows":
    lib1 = ctypes.windll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "awkward-cpu-kernels.dll"))
else:
    lib1 = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libawkward-cpu-kernels.so"))

print("RUNNING3", lib1)
print("RUNNING4", dir(lib1))

dummy1 = lib1.dummy1
dummy3 = layout.dummy3
