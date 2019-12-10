# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numba

import awkward1.layout
from .._numba import util

class TypeType(numba.types.Type):
    def __init__(self, type):
        super(TypeType, self).__init__(name="ak::Type({0})".format(repr(str(type))))
        self.type = type

@numba.extending.typeof_impl.register(awkward1.layout.Type)
def typeof(val, c):
    return TypeType(val)

@numba.extending.register_model(TypeType)
class TypeModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        super(TypeModel, self).__init__(dmm, fe_type, [])

@numba.extending.unbox(TypeType)
def unbox(tpe, obj, c):
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(TypeType)
def box(tpe, val, c):
    return c.pyapi.unserialize(c.pyapi.serialize_object(tpe.type))
