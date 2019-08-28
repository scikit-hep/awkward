# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl
import numba.typing.ctypes_utils

import awkward1.layout
from .._numba import cpu, common

@numba.extending.typeof_impl.register(awkward1.layout.Identity)
def typeof(val, c):
    return IdentityType(numba.typeof(numpy.asarray(val)))

class IdentityType(common.ContentType):
    def __init__(self, arraytpe):
        super(IdentityType, self).__init__(name="IdentityType({0})".format(arraytpe.name))
        self.arraytpe = arraytpe

FieldLocation = numba.types.List(numba.types.Tuple((common.IndexType, numba.types.string)))

@numba.extending.register_model(IdentityType)
class IdentityModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("ref", common.RefType),
                   ("fieldloc", FieldLocation),
                   ("chunkdepth", common.IndexType),
                   ("indexdepth", common.IndexType),
                   ("array", fe_type.arraytpe)]
        super(IdentityModel, self).__init__(dmm, fe_type, members)

numba.extending.make_attribute_wrapper(IdentityType, "ref", "ref")
numba.extending.make_attribute_wrapper(IdentityType, "fieldloc", "fieldloc")
numba.extending.make_attribute_wrapper(IdentityType, "chunkdepth", "chunkdepth")
numba.extending.make_attribute_wrapper(IdentityType, "indexdepth", "indexdepth")
numba.extending.make_attribute_wrapper(IdentityType, "array", "array")

@numba.extending.unbox(IdentityType)
def unbox(tpe, obj, c):
    ref_obj = c.pyapi.object_getattr_string(obj, "ref")
    fieldloc_obj = c.pyapi.object_getattr_string(obj, "fieldloc")
    chunkdepth_obj = c.pyapi.object_getattr_string(obj, "chunkdepth")
    indexdepth_obj = c.pyapi.object_getattr_string(obj, "indexdepth")
    array_obj = c.pyapi.object_getattr_string(obj, "array")
    ref_val = c.pyapi.to_native_value(common.RefType, ref_obj).value
    fieldloc_val = c.pyapi.to_native_value(FieldLocation, fieldloc_obj).value
    chunkdepth_val = c.pyapi.to_native_value(common.IndexType, chunkdepth_obj)
    indexdepth_val = c.pyapi.to_native_value(common.IndexType, indexdepth_obj)
    array_val = c.pyapi.to_native_value(tpe.arraytpe, array_obj)
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    HERE
