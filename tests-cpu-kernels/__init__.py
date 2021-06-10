# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import ctypes
import warnings

import awkward as ak


warnings.warn(
    "HERE {0} {1}".format(repr(ak._cpu_kernels.libpath), ak._cpu_kernels.lib),
    UserWarning,
)

lib = ak._cpu_kernels.lib


class Error(ctypes.Structure):
    _fields_ = [
        ("str", ctypes.POINTER(ctypes.c_char)),
        ("filename", ctypes.POINTER(ctypes.c_char)),
        ("identity", ctypes.c_int64),
        ("attempt", ctypes.c_int64),
        ("pass_through", ctypes.c_bool),
    ]
