# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl
import numba.typing.ctypes_utils

import awkward1.layout
from .._numba import cpu, util

@numba.extending.typeof_impl.register(awkward1.layout.Identity32)
@numba.extending.typeof_impl.register(awkward1.layout.Identity64)
def typeof(val, c):
    return IdentityType(numba.typeof(val.array))

class IdentityType(numba.types.Type):
    fieldloctpe = numba.types.List(numba.types.Tuple((numba.int64, numba.types.string)))

    def bitwidth(self):
        return self.arraytpe.dtype.bitwidth

    def __init__(self, arraytpe):
        super(IdentityType, self).__init__(name="Identity{0}Type".format(arraytpe.dtype.bitwidth))
        self.arraytpe = arraytpe

@numba.extending.register_model(IdentityType)
class IdentityModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("ref", util.RefType),
                   ("fieldloc", fe_type.fieldloctpe),
                   ("array", fe_type.arraytpe)]
        super(IdentityModel, self).__init__(dmm, fe_type, members)

numba.extending.make_attribute_wrapper(IdentityType, "ref", "ref")
numba.extending.make_attribute_wrapper(IdentityType, "fieldloc", "fieldloc")
numba.extending.make_attribute_wrapper(IdentityType, "array", "array")

@numba.extending.unbox(IdentityType)
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

@numba.extending.box(IdentityType)
def box(tpe, val, c):
    if tpe.bitwidth() == 32:
        Identity_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Identity32))
    elif tpe.bitwidth() == 64:
        Identity_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Identity64))
    else:
        assert False, "unrecognized bitwidth"
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    ref_obj = c.pyapi.from_native_value(util.RefType, proxyin.ref, c.env_manager)
    fieldloc_obj = c.pyapi.from_native_value(tpe.fieldloctpe, proxyin.fieldloc, c.env_manager)
    array_obj = c.pyapi.from_native_value(tpe.arraytpe, proxyin.array, c.env_manager)
    out = c.pyapi.call_function_objargs(Identity_obj, (ref_obj, fieldloc_obj, array_obj))
    c.pyapi.decref(Identity_obj)
    c.pyapi.decref(ref_obj)
    c.pyapi.decref(fieldloc_obj)
    c.pyapi.decref(array_obj)
    return out

@numba.extending.lower_builtin(len, IdentityType)
def lower_len(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    return numba.targets.arrayobj.array_len(context, builder, numba.types.intp(tpe.arraytpe), (proxyin.array,))

def lower_getitem_any(context, builder, idtpe, wheretpe, idval, whereval):
    proxyin = numba.cgutils.create_struct_proxy(idtpe)(context, builder, value=idval)

    proxyout = numba.cgutils.create_struct_proxy(idtpe)(context, builder)
    proxyout.ref = proxyin.ref
    proxyout.fieldloc = proxyin.fieldloc
    if isinstance(wheretpe, numba.types.BaseTuple):
        proxyout.array = numba.targets.arrayobj.getitem_array_tuple(context, builder, idtpe.arraytpe(idtpe.arraytpe, wheretpe), (proxyin.array, whereval))
    elif isinstance(wheretpe, numba.types.Array):
        proxyout.array = numba.targets.arrayobj.fancy_getitem_array(context, builder, idtpe.arraytpe(idtpe.arraytpe, wheretpe), (proxyin.array, whereval))
    else:
        proxyout.array = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, idtpe.arraytpe(idtpe.arraytpe, wheretpe), (proxyin.array, whereval))

    return proxyout._getvalue()

@numba.typing.templates.infer_getattr
class type_methods(numba.typing.templates.AttributeTemplate):
    key = IdentityType

    def generic_resolve(self, tpe, attr):
        if attr == "width":
            return numba.int64

@numba.extending.lower_getattr(IdentityType, "width")
def lower_content(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    array_proxy = numba.cgutils.create_struct_proxy(tpe.arraytpe)(context, builder, value=proxyin.array)
    out = builder.extract_value(array_proxy.shape, 1)
    return util.cast(context, builder, numba.intp, numba.int64, out)
