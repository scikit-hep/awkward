# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numba

import awkward1.layout
from .._numba import util

class AwkwardType(numba.types.Type):
    def __init__(self, awkwardtype):
        super(AwkwardType, self).__init__(name="ak::Type({})".format(repr(str(awkwardtype))))
        self.awkwardtype = awkwardtype

@numba.extending.typeof_impl.register(awkward1.layout.Type)
def typeof(val, c):
    return AwkwardType(val)

@numba.extending.register_model(AwkwardType)
class AwkwardTypeModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        super(AwkwardTypeModel, self).__init__(dmm, fe_type, [])

@numba.extending.unbox(AwkwardType)
def unbox(tpe, obj, c):
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(AwkwardType)
def box(tpe, val, c):
    return c.pyapi.unserialize(c.pyapi.serialize_object(tpe.awkwardtype))
