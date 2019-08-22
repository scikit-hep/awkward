# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numpy
import numba
import numba.typing.arraydecl

import awkward1.layout
from .common import ContentType

@numba.extending.typeof_impl.register(awkward1.layout.NumpyArray)
def NumpyArray_typeof(val, c):
    return NumpyArrayType(numba.typeof(numpy.asarray(val)))

class NumpyArrayType(ContentType):
    def __init__(self, arraytype):
        super(NumpyArrayType, self).__init__(name="NumpyArrayType({0})".format(arraytype.name))
        self.arraytype = arraytype

@numba.extending.register_model(NumpyArrayType)
class NumpyArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("array", fe_type.arraytype)]
        super(NumpyArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(NumpyArrayType)
def NumpyArray_unbox(typ, obj, c):
    asarray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numpy.asarray))
    array_obj = c.pyapi.call_function_objargs(asarray_obj, (obj,))
    array_val = c.pyapi.to_native_value(typ.arraytype, array_obj).value
    proxy = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    proxy.array = array_val
    c.pyapi.decref(asarray_obj)
    c.pyapi.decref(array_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxy._getvalue(), is_error)
