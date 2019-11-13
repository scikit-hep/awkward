# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import os
import ctypes
import platform

import numba.typing.ctypes_utils

if platform.system() == "Windows":
    libname = "awkward.dll"
elif platform.system() == "Darwin":
    libname = "libawkward.dylib"
else:
    libname = "libawkward.so"

libpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), libname)

lib = ctypes.cdll.LoadLibrary(libpath)

# bool awkward_FillableArray_length(void* fillablearray, int64_t* result);
FillableArray_length = lib.awkward_FillableArray_length
FillableArray_length.argtypes = [ctypes.c_voidp, ctypes.POINTER(ctypes.c_int64)]
FillableArray_length.restype  = ctypes.c_uint8

# bool awkward_FillableArray_clear(void* fillablearray);
FillableArray_clear = lib.awkward_FillableArray_clear
FillableArray_clear.argtypes = [ctypes.c_voidp]
FillableArray_clear.restype  = ctypes.c_uint8

# bool awkward_FillableArray_null(void* fillablearray);
FillableArray_null = lib.awkward_FillableArray_null
FillableArray_null.argtypes = [ctypes.c_voidp]
FillableArray_null.restype  = ctypes.c_uint8

# bool awkward_FillableArray_boolean(void* fillablearray, bool x);
FillableArray_boolean = lib.awkward_FillableArray_boolean
FillableArray_boolean.argtypes = [ctypes.c_voidp, ctypes.c_uint8]
FillableArray_boolean.restype  = ctypes.c_uint8

# bool awkward_FillableArray_integer(void* fillablearray, int64_t x);
FillableArray_integer = lib.awkward_FillableArray_integer
FillableArray_integer.argtypes = [ctypes.c_voidp, ctypes.c_int64]
FillableArray_integer.restype  = ctypes.c_uint8

# bool awkward_FillableArray_real(void* fillablearray, double x);
FillableArray_real = lib.awkward_FillableArray_real
FillableArray_real.argtypes = [ctypes.c_voidp, ctypes.c_double]
FillableArray_real.restype  = ctypes.c_uint8

# bool awkward_FillableArray_beginlist(void* fillablearray);
FillableArray_beginlist = lib.awkward_FillableArray_beginlist
FillableArray_beginlist.argtypes = [ctypes.c_voidp]
FillableArray_beginlist.restype  = ctypes.c_uint8

# bool awkward_FillableArray_endlist(void* fillablearray);
FillableArray_endlist = lib.awkward_FillableArray_endlist
FillableArray_endlist.argtypes = [ctypes.c_voidp]
FillableArray_endlist.restype  = ctypes.c_uint8
