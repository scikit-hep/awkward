import os
import ctypes

import awkward1.layout

__version__ = awkward1.layout.__version__

lib1 = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libawkward-cpu-kernels.so"))
lib2 = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libawkward.so"))

dummy1 = lib1.dummy1
dummy2 = lib2._Z6dummy2i
dummy3 = layout.dummy3
