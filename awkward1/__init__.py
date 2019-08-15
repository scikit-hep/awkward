import os
import ctypes

import awkward1.layout

lib1 = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libawkward-cpu-kernels.so"))
lib2 = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libawkward.so"))

dummy = lib1.dummy
dummy2 = lib2._Z6dummy2i
dummy3 = layout.dummy3
