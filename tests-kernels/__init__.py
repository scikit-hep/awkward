import os
import ctypes

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(CURRENT_DIR, "..")

for root, _, files in os.walk(TOP_DIR):
    for filename in files:
        if filename.endswith("libawkward-cpu-kernels.so"):
            CPU_KERNEL_SO = os.path.join(root, filename)
            break

lib = ctypes.CDLL(CPU_KERNEL_SO)

class Error(ctypes.Structure):
    _fields_ = [
        ("str", ctypes.POINTER(ctypes.c_char)),
        ("identity", ctypes.c_int64),
        ("attempt", ctypes.c_int64),
        ("pass_through", ctypes.c_bool),
    ]
