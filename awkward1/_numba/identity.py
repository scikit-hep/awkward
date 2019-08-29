# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl
import numba.typing.ctypes_utils

import awkward1.layout
from .._numba import cpu, util

@numba.extending.typeof_impl.register(awkward1.layout.Identity)
def typeof(val, c):
    return IdentityType()

class IdentityType(numba.types.Type):
    fieldloctpe = numba.types.List(numba.types.Tuple((util.IndexType, numba.types.string)))
    arraytpe = util.IndexType[:,:]

    def __init__(self):
        super(IdentityType, self).__init__(name="IdentityType")

@numba.extending.register_model(IdentityType)
class IdentityModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("ref", util.RefType),
                   ("fieldloc", fe_type.fieldloctpe),
                   ("chunkdepth", util.IndexType),
                   ("indexdepth", util.IndexType),
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
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.ref = c.pyapi.to_native_value(util.RefType, ref_obj).value
    proxyout.fieldloc = c.pyapi.to_native_value(tpe.fieldloctpe, fieldloc_obj).value
    proxyout.chunkdepth = c.pyapi.to_native_value(util.IndexType, chunkdepth_obj).value
    proxyout.indexdepth = c.pyapi.to_native_value(util.IndexType, indexdepth_obj).value
    proxyout.array = c.pyapi.to_native_value(tpe.arraytpe, array_obj).value
    if c.context.enable_nrt:
        c.context.nrt.incref(c.builder, tpe.fieldloctpe, proxyout.fieldloc)
    c.pyapi.decref(ref_obj)
    c.pyapi.decref(fieldloc_obj)
    c.pyapi.decref(chunkdepth_obj)
    c.pyapi.decref(indexdepth_obj)
    c.pyapi.decref(array_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(IdentityType)
def box(tpe, val, c):
    Identity_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Identity))
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    ref_obj = c.pyapi.from_native_value(util.RefType, proxyin.ref, c.env_manager)
    fieldloc_obj = c.pyapi.from_native_value(tpe.fieldloctpe, proxyin.fieldloc, c.env_manager)
    chunkdepth_obj = c.pyapi.from_native_value(util.IndexType, proxyin.chunkdepth, c.env_manager)
    indexdepth_obj = c.pyapi.from_native_value(util.IndexType, proxyin.indexdepth, c.env_manager)
    array_obj = c.pyapi.from_native_value(tpe.arraytpe, proxyin.array, c.env_manager)
    out = c.pyapi.call_function_objargs(Identity_obj, (ref_obj, fieldloc_obj, chunkdepth_obj, indexdepth_obj, array_obj))
    c.pyapi.decref(Identity_obj)
    c.pyapi.decref(ref_obj)
    c.pyapi.decref(fieldloc_obj)
    c.pyapi.decref(chunkdepth_obj)
    c.pyapi.decref(indexdepth_obj)
    c.pyapi.decref(array_obj)
    return out

@numba.extending.lower_builtin(len, IdentityType)
def lower_len(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    return numba.targets.arrayobj.array_len(context, builder, numba.types.intp(tpe.arraytpe), (proxyin.array,))

@numba.typing.templates.infer_getattr
class type_methods(numba.typing.templates.AttributeTemplate):
    key = IdentityType

    def generic_resolve(self, tpe, attr):
        if attr == "keydepth":
            return util.IndexType

@numba.extending.lower_getattr(IdentityType, "keydepth")
def lower_keydepth(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    multiplier = context.get_constant(util.IndexType, int(numba.int64.bitwidth / numba.int32.bitwidth))
    return builder.add(builder.mul(multiplier, proxyin.chunkdepth), proxyin.indexdepth)
