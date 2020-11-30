# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import ctypes
import os
import platform

import pkg_resources

if platform.system() == "Windows":
    name = "awkward-cpu-kernels.dll"
elif platform.system() == "Darwin":
    name = "libawkward-cpu-kernels.dylib"
else:
    name = "libawkward-cpu-kernels.so"

CPU_KERNEL_SO = None
try:
    CPU_KERNEL_SO = pkg_resources.resource_filename("awkward1", name)
except ModuleNotFoundError:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.join(CURRENT_DIR, "..")
    for root, _, files in os.walk(TOP_DIR):
        for filename in files:
            if filename == name:
                CPU_KERNEL_SO = os.path.join(root, filename)
                break

if CPU_KERNEL_SO is None:
    raise Exception("Unable to find {0}.".format(name))
lib = ctypes.CDLL(CPU_KERNEL_SO)


class Error(ctypes.Structure):
    _fields_ = [
        ("str", ctypes.POINTER(ctypes.c_char)),
        ("filename", ctypes.POINTER(ctypes.c_char)),
        ("identity", ctypes.c_int64),
        ("attempt", ctypes.c_int64),
        ("pass_through", ctypes.c_bool),
    ]
