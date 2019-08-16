import os
import ctypes

import awkward1.layout

__version__ = awkward1.layout.__version__

print("RUNNING", __file__)
print("RUNNING", os.listdir(os.path.dirname(os.path.abspath(__file__))))

lib1 = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libawkward-cpu-kernels.so"))

dummy1 = lib1.dummy1
dummy3 = layout.dummy3
