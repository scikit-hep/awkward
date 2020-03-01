# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import os
import ctypes
import ctypes.util

libpath = ctypes.util.find_library("awkward-cpu-kernels")
lib = ctypes.cdll.LoadLibrary(libpath)
