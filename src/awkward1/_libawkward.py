# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import ctypes
import platform
import pkg_resources

if platform.system() == "Windows":
    name = "awkward.dll"
elif platform.system() == "Darwin":
    name = "libawkward.dylib"
else:
    name = "libawkward.so"
libpath = pkg_resources.resource_filename("awkward1", name)

lib = ctypes.cdll.LoadLibrary(libpath)

# bool awkward_ArrayBuilder_length(void* fillablearray,
#                                  int64_t* result);
ArrayBuilder_length = lib.awkward_ArrayBuilder_length
ArrayBuilder_length.name = "ArrayBuilder.length"
ArrayBuilder_length.argtypes = [ctypes.c_voidp, ctypes.POINTER(ctypes.c_int64)]
ArrayBuilder_length.restype = ctypes.c_uint8

# bool awkward_ArrayBuilder_clear(void* fillablearray);
ArrayBuilder_clear = lib.awkward_ArrayBuilder_clear
ArrayBuilder_clear.name = "ArrayBuilder.clear"
ArrayBuilder_clear.argtypes = [ctypes.c_voidp]
ArrayBuilder_clear.restype = ctypes.c_uint8

# bool awkward_ArrayBuilder_null(void* fillablearray);
ArrayBuilder_null = lib.awkward_ArrayBuilder_null
ArrayBuilder_null.name = "ArrayBuilder.null"
ArrayBuilder_null.argtypes = [ctypes.c_voidp]
ArrayBuilder_null.restype = ctypes.c_uint8

# bool awkward_ArrayBuilder_boolean(void* fillablearray,
#                                   bool x);
ArrayBuilder_boolean = lib.awkward_ArrayBuilder_boolean
ArrayBuilder_boolean.name = "ArrayBuilder.boolean"
ArrayBuilder_boolean.argtypes = [ctypes.c_voidp, ctypes.c_uint8]
ArrayBuilder_boolean.restype = ctypes.c_uint8

# bool awkward_ArrayBuilder_integer(void* fillablearray,
#                                   int64_t x);
ArrayBuilder_integer = lib.awkward_ArrayBuilder_integer
ArrayBuilder_integer.name = "ArrayBuilder.integer"
ArrayBuilder_integer.argtypes = [ctypes.c_voidp, ctypes.c_int64]
ArrayBuilder_integer.restype = ctypes.c_uint8

# bool awkward_ArrayBuilder_real(void* fillablearray,
#                                double x);
ArrayBuilder_real = lib.awkward_ArrayBuilder_real
ArrayBuilder_real.name = "ArrayBuilder.real"
ArrayBuilder_real.argtypes = [ctypes.c_voidp, ctypes.c_double]
ArrayBuilder_real.restype = ctypes.c_uint8

# bool awkward_ArrayBuilder_beginlist(void* fillablearray);
ArrayBuilder_beginlist = lib.awkward_ArrayBuilder_beginlist
ArrayBuilder_beginlist.name = "ArrayBuilder.beginlist"
ArrayBuilder_beginlist.argtypes = [ctypes.c_voidp]
ArrayBuilder_beginlist.restype = ctypes.c_uint8

# bool awkward_ArrayBuilder_endlist(void* fillablearray);
ArrayBuilder_endlist = lib.awkward_ArrayBuilder_endlist
ArrayBuilder_endlist.name = "ArrayBuilder.endlist"
ArrayBuilder_endlist.argtypes = [ctypes.c_voidp]
ArrayBuilder_endlist.restype = ctypes.c_uint8

# uint8_t awkward_ArrayBuilder_begintuple(void* fillablearray,
#                                         int64_t numfields);
ArrayBuilder_begintuple = lib.awkward_ArrayBuilder_begintuple
ArrayBuilder_begintuple.name = "ArrayBuilder.begintuple"
ArrayBuilder_begintuple.argtypes = [ctypes.c_voidp, ctypes.c_int64]
ArrayBuilder_begintuple.restype = ctypes.c_uint8

# uint8_t awkward_ArrayBuilder_index(void* fillablearray,
#                                    int64_t index);
ArrayBuilder_index = lib.awkward_ArrayBuilder_index
ArrayBuilder_index.name = "ArrayBuilder.index"
ArrayBuilder_index.argtypes = [ctypes.c_voidp, ctypes.c_int64]
ArrayBuilder_index.restype = ctypes.c_uint8

# uint8_t awkward_ArrayBuilder_endtuple(void* fillablearray);
ArrayBuilder_endtuple = lib.awkward_ArrayBuilder_endtuple
ArrayBuilder_endtuple.name = "ArrayBuilder.endtuple"
ArrayBuilder_endtuple.argtypes = [ctypes.c_voidp]
ArrayBuilder_endtuple.restype = ctypes.c_uint8

# uint8_t awkward_ArrayBuilder_beginrecord(void* fillablearray);
ArrayBuilder_beginrecord = lib.awkward_ArrayBuilder_beginrecord
ArrayBuilder_beginrecord.name = "ArrayBuilder.beginrecord"
ArrayBuilder_beginrecord.argtypes = [ctypes.c_voidp]
ArrayBuilder_beginrecord.restype = ctypes.c_uint8

# uint8_t awkward_ArrayBuilder_beginrecord_fast(void* fillablearray,
#                                               const char* name);
ArrayBuilder_beginrecord_fast = lib.awkward_ArrayBuilder_beginrecord_fast
ArrayBuilder_beginrecord_fast.name = "ArrayBuilder.beginrecord_fast"
ArrayBuilder_beginrecord_fast.argtypes = [ctypes.c_voidp, ctypes.c_voidp]
ArrayBuilder_beginrecord_fast.restype = ctypes.c_uint8

# uint8_t awkward_ArrayBuilder_beginrecord_check(void* fillablearray,
#                                                const char* name);
ArrayBuilder_beginrecord_check = lib.awkward_ArrayBuilder_beginrecord_check
ArrayBuilder_beginrecord_check.name = "ArrayBuilder.beginrecord_check"
ArrayBuilder_beginrecord_check.argtypes = [ctypes.c_voidp, ctypes.c_voidp]
ArrayBuilder_beginrecord_check.restype = ctypes.c_uint8

# uint8_t awkward_ArrayBuilder_field_fast(void* fillablearray,
#                                         const char* key);
ArrayBuilder_field_fast = lib.awkward_ArrayBuilder_field_fast
ArrayBuilder_field_fast.name = "ArrayBuilder.field_fast"
ArrayBuilder_field_fast.argtypes = [ctypes.c_voidp, ctypes.c_voidp]
ArrayBuilder_field_fast.restype = ctypes.c_uint8

# uint8_t awkward_ArrayBuilder_field_check(void* fillablearray,
#                                          const char* key);
ArrayBuilder_field_check = lib.awkward_ArrayBuilder_field_check
ArrayBuilder_field_check.name = "ArrayBuilder.field_check"
ArrayBuilder_field_check.argtypes = [ctypes.c_voidp, ctypes.c_voidp]
ArrayBuilder_field_check.restype = ctypes.c_uint8

# uint8_t awkward_ArrayBuilder_endrecord(void* fillablearray);
ArrayBuilder_endrecord = lib.awkward_ArrayBuilder_endrecord
ArrayBuilder_endrecord.name = "ArrayBuilder.endrecord"
ArrayBuilder_endrecord.argtypes = [ctypes.c_voidp]
ArrayBuilder_endrecord.restype = ctypes.c_uint8

# uint8_t awkward_ArrayBuilder_append_nowrap(void* fillablearray,
#                                            const void* shared_ptr_ptr,
#                                            int64_t at);
ArrayBuilder_append_nowrap = lib.awkward_ArrayBuilder_append_nowrap
ArrayBuilder_append_nowrap.name = "ArrayBuilder.append_nowrap"
ArrayBuilder_append_nowrap.argtypes = [ctypes.c_voidp, ctypes.c_voidp, ctypes.c_int64]
ArrayBuilder_append_nowrap.restype = ctypes.c_uint8
