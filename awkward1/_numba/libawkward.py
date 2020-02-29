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

# bool awkward_ArrayBuilder_length(void* arraybuilder, int64_t* result);
ArrayBuilder_length = lib.awkward_ArrayBuilder_length
ArrayBuilder_length.name = "ArrayBuilder.length"
ArrayBuilder_length.argtypes  = [ctypes.c_voidp, ctypes.POINTER(ctypes.c_int64)]
ArrayBuilder_length.restype   = ctypes.c_uint8
ArrayBuilder_length.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_length)

# bool awkward_ArrayBuilder_clear(void* arraybuilder);
ArrayBuilder_clear = lib.awkward_ArrayBuilder_clear
ArrayBuilder_clear.name = "ArrayBuilder.clear"
ArrayBuilder_clear.argtypes  = [ctypes.c_voidp]
ArrayBuilder_clear.restype   = ctypes.c_uint8
ArrayBuilder_clear.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_clear)

# bool awkward_ArrayBuilder_null(void* arraybuilder);
ArrayBuilder_null = lib.awkward_ArrayBuilder_null
ArrayBuilder_null.name = "ArrayBuilder.null"
ArrayBuilder_null.argtypes  = [ctypes.c_voidp]
ArrayBuilder_null.restype   = ctypes.c_uint8
ArrayBuilder_null.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_null)

# bool awkward_ArrayBuilder_boolean(void* arraybuilder, bool x);
ArrayBuilder_boolean = lib.awkward_ArrayBuilder_boolean
ArrayBuilder_boolean.name = "ArrayBuilder.boolean"
ArrayBuilder_boolean.argtypes  = [ctypes.c_voidp, ctypes.c_uint8]
ArrayBuilder_boolean.restype   = ctypes.c_uint8
ArrayBuilder_boolean.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_boolean)

# bool awkward_ArrayBuilder_integer(void* arraybuilder, int64_t x);
ArrayBuilder_integer = lib.awkward_ArrayBuilder_integer
ArrayBuilder_integer.name = "ArrayBuilder.integer"
ArrayBuilder_integer.argtypes  = [ctypes.c_voidp, ctypes.c_int64]
ArrayBuilder_integer.restype   = ctypes.c_uint8
ArrayBuilder_integer.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_integer)

# bool awkward_ArrayBuilder_real(void* arraybuilder, double x);
ArrayBuilder_real = lib.awkward_ArrayBuilder_real
ArrayBuilder_real.name = "ArrayBuilder.real"
ArrayBuilder_real.argtypes  = [ctypes.c_voidp, ctypes.c_double]
ArrayBuilder_real.restype   = ctypes.c_uint8
ArrayBuilder_real.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_real)

# bool awkward_ArrayBuilder_beginlist(void* arraybuilder);
ArrayBuilder_beginlist = lib.awkward_ArrayBuilder_beginlist
ArrayBuilder_beginlist.name = "ArrayBuilder.beginlist"
ArrayBuilder_beginlist.argtypes  = [ctypes.c_voidp]
ArrayBuilder_beginlist.restype   = ctypes.c_uint8
ArrayBuilder_beginlist.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_beginlist)

# bool awkward_ArrayBuilder_endlist(void* arraybuilder);
ArrayBuilder_endlist = lib.awkward_ArrayBuilder_endlist
ArrayBuilder_endlist.name = "ArrayBuilder.endlist"
ArrayBuilder_endlist.argtypes  = [ctypes.c_voidp]
ArrayBuilder_endlist.restype   = ctypes.c_uint8
ArrayBuilder_endlist.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_endlist)

# uint8_t awkward_ArrayBuilder_begintuple(void* arraybuilder, int64_t numfields);
ArrayBuilder_begintuple = lib.awkward_ArrayBuilder_begintuple
ArrayBuilder_begintuple.name = "ArrayBuilder.begintuple"
ArrayBuilder_begintuple.argtypes  = [ctypes.c_voidp, ctypes.c_int64]
ArrayBuilder_begintuple.restype   = ctypes.c_uint8
ArrayBuilder_begintuple.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_begintuple)

# uint8_t awkward_ArrayBuilder_index(void* arraybuilder, int64_t index);
ArrayBuilder_index = lib.awkward_ArrayBuilder_index
ArrayBuilder_index.name = "ArrayBuilder.index"
ArrayBuilder_index.argtypes  = [ctypes.c_voidp, ctypes.c_int64]
ArrayBuilder_index.restype   = ctypes.c_uint8
ArrayBuilder_index.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_index)

# uint8_t awkward_ArrayBuilder_endtuple(void* arraybuilder);
ArrayBuilder_endtuple = lib.awkward_ArrayBuilder_endtuple
ArrayBuilder_endtuple.name = "ArrayBuilder.endtuple"
ArrayBuilder_endtuple.argtypes  = [ctypes.c_voidp]
ArrayBuilder_endtuple.restype   = ctypes.c_uint8
ArrayBuilder_endtuple.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_endtuple)

# uint8_t awkward_ArrayBuilder_beginrecord(void* arraybuilder);
ArrayBuilder_beginrecord = lib.awkward_ArrayBuilder_beginrecord
ArrayBuilder_beginrecord.name = "ArrayBuilder.beginrecord"
ArrayBuilder_beginrecord.argtypes  = [ctypes.c_voidp]
ArrayBuilder_beginrecord.restype   = ctypes.c_uint8
ArrayBuilder_beginrecord.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_beginrecord)

# uint8_t awkward_ArrayBuilder_beginrecord_fast(void* arraybuilder, const char* name);
ArrayBuilder_beginrecord_fast = lib.awkward_ArrayBuilder_beginrecord_fast
ArrayBuilder_beginrecord_fast.name = "ArrayBuilder.beginrecord_fast"
ArrayBuilder_beginrecord_fast.argtypes  = [ctypes.c_voidp, ctypes.c_voidp]
ArrayBuilder_beginrecord_fast.restype   = ctypes.c_uint8
ArrayBuilder_beginrecord_fast.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_beginrecord_fast)

# uint8_t awkward_ArrayBuilder_beginrecord_check(void* arraybuilder, const char* name);
ArrayBuilder_beginrecord_check = lib.awkward_ArrayBuilder_beginrecord_check
ArrayBuilder_beginrecord_check.name = "ArrayBuilder.beginrecord_check"
ArrayBuilder_beginrecord_check.argtypes  = [ctypes.c_voidp, ctypes.c_voidp]
ArrayBuilder_beginrecord_check.restype   = ctypes.c_uint8
ArrayBuilder_beginrecord_check.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_beginrecord_check)

# uint8_t awkward_ArrayBuilder_field_fast(void* arraybuilder, const char* key);
ArrayBuilder_field_fast = lib.awkward_ArrayBuilder_field_fast
ArrayBuilder_field_fast.name = "ArrayBuilder.field_fast"
ArrayBuilder_field_fast.argtypes  = [ctypes.c_voidp, ctypes.c_voidp]
ArrayBuilder_field_fast.restype   = ctypes.c_uint8
ArrayBuilder_field_fast.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_field_fast)

# uint8_t awkward_ArrayBuilder_field_check(void* arraybuilder, const char* key);
ArrayBuilder_field_check = lib.awkward_ArrayBuilder_field_check
ArrayBuilder_field_check.name = "ArrayBuilder.field_check"
ArrayBuilder_field_check.argtypes  = [ctypes.c_voidp, ctypes.c_voidp]
ArrayBuilder_field_check.restype   = ctypes.c_uint8
ArrayBuilder_field_check.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_field_check)

# uint8_t awkward_ArrayBuilder_endrecord(void* arraybuilder);
ArrayBuilder_endrecord = lib.awkward_ArrayBuilder_endrecord
ArrayBuilder_endrecord.name = "ArrayBuilder.endrecord"
ArrayBuilder_endrecord.argtypes  = [ctypes.c_voidp]
ArrayBuilder_endrecord.restype   = ctypes.c_uint8
ArrayBuilder_endrecord.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_endrecord)

# uint8_t awkward_ArrayBuilder_append_nowrap(void* arraybuilder, const void* shared_ptr_ptr, int64_t at);
ArrayBuilder_append_nowrap = lib.awkward_ArrayBuilder_append_nowrap
ArrayBuilder_append_nowrap.name = "ArrayBuilder.append_nowrap"
ArrayBuilder_append_nowrap.argtypes  = [ctypes.c_voidp, ctypes.c_voidp, ctypes.c_int64]
ArrayBuilder_append_nowrap.restype   = ctypes.c_uint8
ArrayBuilder_append_nowrap.numbatype = numba.typing.ctypes_utils.make_function_type(ArrayBuilder_append_nowrap)
