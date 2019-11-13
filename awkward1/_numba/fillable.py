# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numba

import awkward1.layout
from .._numba import libawkward, util

@numba.extending.typeof_impl.register(awkward1.layout.FillableArray)
def typeof(val, c):
    return FillableArrayType()

class FillableArrayType(numba.types.Type):
    def __init__(self):
        super(FillableArrayType, self).__init__("FillableArrayType")

@numba.datamodel.registry.register_default(FillableArrayType)
class FillableArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("rawptr", numba.types.voidptr),
                   ("pyptr", numba.types.pyobject)]
        super(FillableArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(FillableArrayType)
def unbox(tpe, obj, c):
    rawptr_obj = c.pyapi.object_getattr_string(obj, "_ptr")
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.rawptr = c.pyapi.long_as_voidptr(rawptr_obj)
    proxyout.pyptr = obj
    c.pyapi.decref(rawptr_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(FillableArrayType)
def box(tpe, val, c):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    c.pyapi.incref(proxyin.pyptr)
    return proxyin.pyptr
