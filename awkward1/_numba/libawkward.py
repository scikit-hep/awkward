# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import os
import ctypes
import platform

import numba
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
FillableArray_length.name = "FillableArray.length"
FillableArray_length.argtypes = [ctypes.c_voidp, ctypes.POINTER(ctypes.c_int64)]
FillableArray_length.restype  = ctypes.c_uint8
FillableArray_length.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_length)

# bool awkward_FillableArray_clear(void* fillablearray);
FillableArray_clear = lib.awkward_FillableArray_clear
FillableArray_clear.name = "FillableArray.clear"
FillableArray_clear.argtypes = [ctypes.c_voidp]
FillableArray_clear.restype  = ctypes.c_uint8
FillableArray_clear.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_clear)

# bool awkward_FillableArray_null(void* fillablearray);
FillableArray_null = lib.awkward_FillableArray_null
FillableArray_null.name = "FillableArray.null"
FillableArray_null.argtypes = [ctypes.c_voidp]
FillableArray_null.restype  = ctypes.c_uint8
FillableArray_null.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_null)

# bool awkward_FillableArray_boolean(void* fillablearray, bool x);
FillableArray_boolean = lib.awkward_FillableArray_boolean
FillableArray_boolean.name = "FillableArray.boolean"
FillableArray_boolean.argtypes = [ctypes.c_voidp, ctypes.c_uint8]
FillableArray_boolean.restype  = ctypes.c_uint8
FillableArray_boolean.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_boolean)

# bool awkward_FillableArray_integer(void* fillablearray, int64_t x);
FillableArray_integer = lib.awkward_FillableArray_integer
FillableArray_integer.name = "FillableArray.integer"
FillableArray_integer.argtypes = [ctypes.c_voidp, ctypes.c_int64]
FillableArray_integer.restype  = ctypes.c_uint8
FillableArray_integer.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_integer)

# bool awkward_FillableArray_real(void* fillablearray, double x);
FillableArray_real = lib.awkward_FillableArray_real
FillableArray_real.name = "FillableArray.real"
FillableArray_real.argtypes = [ctypes.c_voidp, ctypes.c_double]
FillableArray_real.restype  = ctypes.c_uint8
FillableArray_real.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_real)

# bool awkward_FillableArray_beginlist(void* fillablearray);
FillableArray_beginlist = lib.awkward_FillableArray_beginlist
FillableArray_beginlist.name = "FillableArray.beginlist"
FillableArray_beginlist.argtypes = [ctypes.c_voidp]
FillableArray_beginlist.restype  = ctypes.c_uint8
FillableArray_beginlist.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_beginlist)

# bool awkward_FillableArray_endlist(void* fillablearray);
FillableArray_endlist = lib.awkward_FillableArray_endlist
FillableArray_endlist.name = "FillableArray.endlist"
FillableArray_endlist.argtypes = [ctypes.c_voidp]
FillableArray_endlist.restype  = ctypes.c_uint8
FillableArray_endlist.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_endlist)
