# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

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

# uint8_t awkward_FillableArray_begintuple(void* fillablearray, int64_t numfields);
FillableArray_begintuple = lib.awkward_FillableArray_begintuple
FillableArray_begintuple.name = "FillableArray.begintuple"
FillableArray_begintuple.argtypes = [ctypes.c_voidp, ctypes.c_int64]
FillableArray_begintuple.restype  = ctypes.c_uint8
FillableArray_begintuple.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_begintuple)

# uint8_t awkward_FillableArray_index(void* fillablearray, int64_t index);
FillableArray_index = lib.awkward_FillableArray_index
FillableArray_index.name = "FillableArray.index"
FillableArray_index.argtypes = [ctypes.c_voidp, ctypes.c_int64]
FillableArray_index.restype  = ctypes.c_uint8
FillableArray_index.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_index)

# uint8_t awkward_FillableArray_endtuple(void* fillablearray);
FillableArray_endtuple = lib.awkward_FillableArray_endtuple
FillableArray_endtuple.name = "FillableArray.endtuple"
FillableArray_endtuple.argtypes = [ctypes.c_voidp]
FillableArray_endtuple.restype  = ctypes.c_uint8
FillableArray_endtuple.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_endtuple)

# uint8_t awkward_FillableArray_beginrecord(void* fillablearray);
FillableArray_beginrecord = lib.awkward_FillableArray_beginrecord
FillableArray_beginrecord.name = "FillableArray.beginrecord"
FillableArray_beginrecord.argtypes = [ctypes.c_voidp]
FillableArray_beginrecord.restype  = ctypes.c_uint8
FillableArray_beginrecord.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_beginrecord)

# uint8_t awkward_FillableArray_beginrecord_fast(void* fillablearray, const char* name);
FillableArray_beginrecord_fast = lib.awkward_FillableArray_beginrecord_fast
FillableArray_beginrecord_fast.name = "FillableArray.beginrecord_fast"
FillableArray_beginrecord_fast.argtypes = [ctypes.c_voidp, ctypes.c_voidp]
FillableArray_beginrecord_fast.restype  = ctypes.c_uint8
FillableArray_beginrecord_fast.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_beginrecord_fast)

# uint8_t awkward_FillableArray_beginrecord_check(void* fillablearray, const char* name);
FillableArray_beginrecord_check = lib.awkward_FillableArray_beginrecord_check
FillableArray_beginrecord_check.name = "FillableArray.beginrecord_check"
FillableArray_beginrecord_check.argtypes = [ctypes.c_voidp, ctypes.c_voidp]
FillableArray_beginrecord_check.restype  = ctypes.c_uint8
FillableArray_beginrecord_check.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_beginrecord_check)

# uint8_t awkward_FillableArray_field_fast(void* fillablearray, const char* key);
FillableArray_field_fast = lib.awkward_FillableArray_field_fast
FillableArray_field_fast.name = "FillableArray.field_fast"
FillableArray_field_fast.argtypes = [ctypes.c_voidp, ctypes.c_voidp]
FillableArray_field_fast.restype  = ctypes.c_uint8
FillableArray_field_fast.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_field_fast)

# uint8_t awkward_FillableArray_field_check(void* fillablearray, const char* key);
FillableArray_field_check = lib.awkward_FillableArray_field_check
FillableArray_field_check.name = "FillableArray.field_check"
FillableArray_field_check.argtypes = [ctypes.c_voidp, ctypes.c_voidp]
FillableArray_field_check.restype  = ctypes.c_uint8
FillableArray_field_check.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_field_check)

# uint8_t awkward_FillableArray_endrecord(void* fillablearray);
FillableArray_endrecord = lib.awkward_FillableArray_endrecord
FillableArray_endrecord.name = "FillableArray.endrecord"
FillableArray_endrecord.argtypes = [ctypes.c_voidp]
FillableArray_endrecord.restype  = ctypes.c_uint8
FillableArray_endrecord.numbatpe = numba.typing.ctypes_utils.make_function_type(FillableArray_endrecord)
