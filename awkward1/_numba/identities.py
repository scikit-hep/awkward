# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import operator

import numpy
import numba
import numba.typing.arraydecl
import numba.typing.ctypes_utils

import awkward1.layout
from .._numba import util

@numba.extending.typeof_impl.register(awkward1.layout.Identities32)
@numba.extending.typeof_impl.register(awkward1.layout.Identities64)
def typeof(val, c):
    return IdentitiesType(numba.typeof(val.array))

class IdentitiesType(numba.types.Type):
    fieldloctpe = numba.types.List(numba.types.Tuple((numba.int64, numba.types.string)))

    def bitwidth(self):
        return self.arraytpe.dtype.bitwidth

    def __init__(self, arraytpe):
        super(IdentitiesType, self).__init__(name="ak::Identities{0}Type".format(arraytpe.dtype.bitwidth))
        self.arraytpe = arraytpe

@numba.extending.register_model(IdentitiesType)
class IdentitiesModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("ref", util.RefType),
                   ("fieldloc", fe_type.fieldloctpe),
                   ("array", fe_type.arraytpe)]
        super(IdentitiesModel, self).__init__(dmm, fe_type, members)

numba.extending.make_attribute_wrapper(IdentitiesType, "ref", "ref")
numba.extending.make_attribute_wrapper(IdentitiesType, "fieldloc", "fieldloc")
numba.extending.make_attribute_wrapper(IdentitiesType, "array", "array")

@numba.extending.unbox(IdentitiesType)
def unbox(tpe, obj, c):
    ref_obj = c.pyapi.object_getattr_string(obj, "ref")
    fieldloc_obj = c.pyapi.object_getattr_string(obj, "fieldloc")
    array_obj = c.pyapi.object_getattr_string(obj, "array")
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.ref = c.pyapi.to_native_value(util.RefType, ref_obj).value
    proxyout.fieldloc = c.pyapi.to_native_value(tpe.fieldloctpe, fieldloc_obj).value
    proxyout.array = c.pyapi.to_native_value(tpe.arraytpe, array_obj).value
    if c.context.enable_nrt:
        c.context.nrt.incref(c.builder, tpe.fieldloctpe, proxyout.fieldloc)
    c.pyapi.decref(ref_obj)
    c.pyapi.decref(fieldloc_obj)
    c.pyapi.decref(array_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(IdentitiesType)
def box(tpe, val, c):
    if tpe.bitwidth() == 32:
        Identities_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Identities32))
    elif tpe.bitwidth() == 64:
        Identities_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Identities64))
    else:
        assert False, "unrecognized bitwidth"
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    ref_obj = c.pyapi.from_native_value(util.RefType, proxyin.ref, c.env_manager)
    fieldloc_obj = c.pyapi.from_native_value(tpe.fieldloctpe, proxyin.fieldloc, c.env_manager)
    array_obj = c.pyapi.from_native_value(tpe.arraytpe, proxyin.array, c.env_manager)
    out = c.pyapi.call_function_objargs(Identities_obj, (ref_obj, fieldloc_obj, array_obj))
    c.pyapi.decref(Identities_obj)
    c.pyapi.decref(ref_obj)
    c.pyapi.decref(fieldloc_obj)
    c.pyapi.decref(array_obj)
    return out

@numba.extending.lower_builtin(len, IdentitiesType)
def lower_len(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    return numba.targets.arrayobj.array_len(context, builder, numba.types.intp(tpe.arraytpe), (proxyin.array,))

def lower_getitem_any(context, builder, identitiestpe, wheretpe, idval, whereval):
    proxyin = numba.cgutils.create_struct_proxy(identitiestpe)(context, builder, value=idval)

    if isinstance(wheretpe, numba.types.Integer):
        proxyslice = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
        proxyslice.start = util.cast(context, builder, wheretpe, numba.intp, whereval)
        proxyslice.stop = builder.add(proxyslice.start, context.get_constant(numba.intp, 1))
        proxyslice.step = context.get_constant(numba.intp, 1)
        wheretpe = numba.types.slice2_type
        whereval = proxyslice._getvalue()

    proxyout = numba.cgutils.create_struct_proxy(identitiestpe)(context, builder)
    proxyout.ref = proxyin.ref
    proxyout.fieldloc = proxyin.fieldloc
    if isinstance(wheretpe, numba.types.BaseTuple):
        proxyout.array = numba.targets.arrayobj.getitem_array_tuple(context, builder, identitiestpe.arraytpe(identitiestpe.arraytpe, wheretpe), (proxyin.array, whereval))
    elif isinstance(wheretpe, numba.types.Array):
        proxyout.array = numba.targets.arrayobj.fancy_getitem_array(context, builder, identitiestpe.arraytpe(identitiestpe.arraytpe, wheretpe), (proxyin.array, whereval))
    else:
        proxyout.array = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, identitiestpe.arraytpe(identitiestpe.arraytpe, wheretpe), (proxyin.array, whereval))

    return proxyout._getvalue()

@numba.typing.templates.infer_getattr
class type_methods(numba.typing.templates.AttributeTemplate):
    key = IdentitiesType

    def generic_resolve(self, tpe, attr):
        if attr == "width":
            return numba.int64

@numba.extending.lower_getattr(IdentitiesType, "width")
def lower_content(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    array_proxy = numba.cgutils.create_struct_proxy(tpe.arraytpe)(context, builder, value=proxyin.array)
    out = builder.extract_value(array_proxy.shape, 1)
    return util.cast(context, builder, numba.intp, numba.int64, out)
