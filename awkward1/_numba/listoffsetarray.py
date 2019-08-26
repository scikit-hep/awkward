# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl

import awkward1.layout
from .._numba import cpu, common

@numba.extending.typeof_impl.register(awkward1.layout.ListOffsetArray)
def typeof(val, c):
    return ListOffsetArrayType(numba.typeof(numpy.asarray(val.offsets)), numba.typeof(val.content))

class ListOffsetArrayType(common.ContentType):
    def __init__(self, offsetstpe, contenttpe):
        super(ListOffsetArrayType, self).__init__(name="ListOffsetArrayType({0}, {1})".format(offsetstpe.name, contenttpe.name))
        self.offsetstpe = offsetstpe
        self.contenttpe = contenttpe

    def getitem(self, wheretpe):
        if len(wheretpe.types) == 0:
            return self
        else:
            headtpe = wheretpe.types[0]
            tailtpe = numba.types.Tuple(wheretpe.types[1:])
            if isinstance(headtpe, numba.types.Integer):
                return self.contenttpe.getitem(tailtpe)
            else:
                return self.getitem(tailtpe)

@numba.extending.register_model(ListOffsetArrayType)
class ListOffsetArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("offsets", fe_type.offsetstpe),
                   ("content", fe_type.contenttpe)]
        super(ListOffsetArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(ListOffsetArrayType)
def unbox(tpe, obj, c):
    asarray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numpy.asarray))
    offsets_obj = c.pyapi.object_getattr_string(obj, "offsets")
    content_obj = c.pyapi.object_getattr_string(obj, "content")
    offsetsarray_obj = c.pyapi.call_function_objargs(asarray_obj, (offsets_obj,))
    offsetsarray_val = c.pyapi.to_native_value(tpe.offsetstpe, offsetsarray_obj).value
    content_val = c.pyapi.to_native_value(tpe.contenttpe, content_obj).value
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.offsets = offsetsarray_val
    proxyout.content = content_val
    c.pyapi.decref(asarray_obj)
    c.pyapi.decref(offsets_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(offsetsarray_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(ListOffsetArrayType)
def box(tpe, val, c):
    Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Index))
    ListOffsetArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.ListOffsetArray))
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    offsetsarray_obj = c.pyapi.from_native_value(tpe.offsetstpe, proxyin.offsets, c.env_manager)
    content_obj = c.pyapi.from_native_value(tpe.contenttpe, proxyin.content, c.env_manager)
    offsets_obj = c.pyapi.call_function_objargs(Index_obj, (offsetsarray_obj,))
    out = c.pyapi.call_function_objargs(ListOffsetArray_obj, (offsets_obj, content_obj))
    c.pyapi.decref(Index_obj)
    c.pyapi.decref(ListOffsetArray_obj)
    c.pyapi.decref(offsetsarray_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(offsets_obj)
    return out

@numba.extending.lower_builtin(len, ListOffsetArrayType)
def lower_len(context, builder, sig, args):
    rettpe, (tpe,) = sig.return_type, sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    offsetlen = numba.targets.arrayobj.array_len(context, builder, numba.types.intp(tpe.offsetstpe), (proxyin.offsets,))
    return builder.sub(offsetlen, context.get_constant(rettpe, 1))

@numba.extending.lower_builtin(operator.getitem, ListOffsetArrayType, numba.types.Integer)
def lower_getitem(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    wherevalp1 = builder.add(whereval, context.get_constant(wheretpe, 1))
    if isinstance(wheretpe, numba.types.Literal):
        wherevalp1_tpe = wheretpe.literal_type
    else:
        wherevalp1_tpe = wheretpe

    start = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, common.IndexType(tpe.offsetstpe, wheretpe), (proxyin.offsets, whereval))
    stop = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, common.IndexType(tpe.offsetstpe, wherevalp1_tpe), (proxyin.offsets, wherevalp1))
    proxyslice = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxyslice.start = builder.zext(start, context.get_value_type(numba.types.intp))
    proxyslice.stop = builder.zext(stop, context.get_value_type(numba.types.intp))
    proxyslice.step = context.get_constant(numba.types.intp, 1)

    fcn = context.get_function(operator.getitem, rettpe(tpe.contenttpe, numba.types.slice2_type))
    return fcn(builder, (proxyin.content, proxyslice._getvalue()))

@numba.extending.lower_builtin(operator.getitem, ListOffsetArrayType, numba.types.slice2_type)
def lower_getitem(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    proxyout = numba.cgutils.create_struct_proxy(tpe)(context, builder)
    proxyout.offsets = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.offsetstpe(tpe.offsetstpe, wheretpe), (proxyin.offsets, whereval))
    proxyout.content = proxyin.content
    out = proxyout._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out
