# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl

import awkward1.layout
from .._numba import cpu, util, content

@numba.extending.typeof_impl.register(awkward1.layout.ListOffsetArray32)
@numba.extending.typeof_impl.register(awkward1.layout.ListOffsetArray64)
def typeof(val, c):
    return ListOffsetArrayType(numba.typeof(numpy.asarray(val.offsets)), numba.typeof(val.content), numba.typeof(val.id))

class ListOffsetArrayType(content.ContentType):
    def __init__(self, offsetstpe, contenttpe, idtpe):
        super(ListOffsetArrayType, self).__init__(name="ListOffsetArray{}Type({}, id={})".format(offsetstpe.dtype.bitwidth, contenttpe.name, idtpe.name))
        self.offsetstpe = offsetstpe
        self.contenttpe = contenttpe
        self.idtpe = idtpe

    @property
    def bitwidth(self):
        return self.offsetstpe.dtype.bitwidth

    @property
    def ndim(self):
        return 1 + self.contenttpe.ndim

    def getitem_int(self):
        return self.getitem_tuple(numba.types.Tuple((numba.int64,)))

    def getitem_range(self):
        return self.getitem_tuple(numba.types.Tuple((numba.types.slice2_type,)))

    def getitem_tuple(self, wheretpe):
        return self.getitem_next(wheretpe, False)

    def getitem_next(self, wheretpe, isadvanced):
        if len(wheretpe.types) == 0:
            return self
        else:
            headtpe = wheretpe.types[0]
            tailtpe = numba.types.Tuple(wheretpe.types[1:])
            if isinstance(headtpe, numba.types.Integer):
                return self.contenttpe.getitem_next(tailtpe, isadvanced)
            else:
                return self.getitem_next(tailtpe, isadvanced)

    def carry(self):
        import awkward1._numba.listarray
        return awkward1._numba.listarray.ListArrayType(self.offsettpe, self.offsettpe, self.contenttpe, self.idtpe)

    @property
    def lower_len(self):
        return lower_len

    @property
    def lower_getitem_int(self):
        return lower_getitem_int

    @property
    def lower_getitem_range(self):
        return lower_getitem_range

    @property
    def lower_getitem_next(self):
        return lower_getitem_next

    @property
    def lower_carry(self):
        return lower_carry

@numba.extending.register_model(ListOffsetArrayType)
class ListOffsetArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("offsets", fe_type.offsetstpe),
                   ("content", fe_type.contenttpe)]
        if fe_type.idtpe != numba.none:
            members.append(("id", fe_type.idtpe))
        super(ListOffsetArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(ListOffsetArrayType)
def unbox(tpe, obj, c):
    asarray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numpy.asarray))
    offsets_obj = c.pyapi.object_getattr_string(obj, "offsets")
    content_obj = c.pyapi.object_getattr_string(obj, "content")
    offsetsarray_obj = c.pyapi.call_function_objargs(asarray_obj, (offsets_obj,))
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.offsets = c.pyapi.to_native_value(tpe.offsetstpe, offsetsarray_obj).value
    proxyout.content = c.pyapi.to_native_value(tpe.contenttpe, content_obj).value
    c.pyapi.decref(asarray_obj)
    c.pyapi.decref(offsets_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(offsetsarray_obj)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.object_getattr_string(obj, "id")
        proxyout.id = c.pyapi.to_native_value(tpe.idtpe, id_obj).value
        c.pyapi.decref(id_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(ListOffsetArrayType)
def box(tpe, val, c):
    if tpe.bitwidth == 32:
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Index32))
        ListOffsetArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.ListOffsetArray32))
    elif tpe.bitwidth == 64:
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Index64))
        ListOffsetArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.ListOffsetArray64))
    else:
        raise AssertionError("unrecognized bitwidth")
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    offsetsarray_obj = c.pyapi.from_native_value(tpe.offsetstpe, proxyin.offsets, c.env_manager)
    content_obj = c.pyapi.from_native_value(tpe.contenttpe, proxyin.content, c.env_manager)
    offsets_obj = c.pyapi.call_function_objargs(Index_obj, (offsetsarray_obj,))
    c.pyapi.decref(Index_obj)
    c.pyapi.decref(offsetsarray_obj)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.from_native_value(tpe.idtpe, proxyin.id, c.env_manager)
        out = c.pyapi.call_function_objargs(ListOffsetArray_obj, (offsets_obj, content_obj, id_obj))
        c.pyapi.decref(id_obj)
    else:
        out = c.pyapi.call_function_objargs(ListOffsetArray_obj, (offsets_obj, content_obj))
    c.pyapi.decref(ListOffsetArray_obj)
    c.pyapi.decref(offsets_obj)
    c.pyapi.decref(content_obj)
    return out

@numba.extending.lower_builtin(len, ListOffsetArrayType)
def lower_len(context, builder, sig, args):
    rettpe, (tpe,) = sig.return_type, sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    offsetlen = numba.targets.arrayobj.array_len(context, builder, numba.intp(tpe.offsetstpe), (proxyin.offsets,))
    return builder.sub(offsetlen, context.get_constant(rettpe, 1))

@numba.extending.lower_builtin(operator.getitem, ListOffsetArrayType, numba.types.Integer)
def lower_getitem_int(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    wherevalp1 = builder.add(whereval, context.get_constant(wheretpe, 1))
    if isinstance(wheretpe, numba.types.Literal):
        wherevalp1_tpe = wheretpe.literal_type
    else:
        wherevalp1_tpe = wheretpe

    start = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, numba.int64(tpe.offsetstpe, wheretpe), (proxyin.offsets, whereval))
    stop = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, numba.int64(tpe.offsetstpe, wherevalp1_tpe), (proxyin.offsets, wherevalp1))
    proxyslice = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxyslice.start = builder.zext(start, context.get_value_type(numba.intp))
    proxyslice.stop = builder.zext(stop, context.get_value_type(numba.intp))
    proxyslice.step = context.get_constant(numba.intp, 1)

    fcn = context.get_function(operator.getitem, rettpe(tpe.contenttpe, numba.types.slice2_type))
    return fcn(builder, (proxyin.content, proxyslice._getvalue()))

@numba.extending.lower_builtin(operator.getitem, ListOffsetArrayType, numba.types.slice2_type)
def lower_getitem_slice(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    proxyslicein = numba.cgutils.create_struct_proxy(wheretpe)(context, builder, value=whereval)
    proxysliceout = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxysliceout.start = proxyslicein.start
    proxysliceout.stop = builder.add(proxyslicein.stop, context.get_constant(numba.intp, 1))
    proxysliceout.step = context.get_constant(numba.intp, 1)

    proxyout = numba.cgutils.create_struct_proxy(tpe)(context, builder)
    proxyout.offsets = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.offsetstpe(tpe.offsetstpe, numba.types.slice2_type), (proxyin.offsets, proxysliceout._getvalue()))
    proxyout.content = proxyin.content
    out = proxyout._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out

# def lower_carry(context, builder, arraytpe, carrytpe, arrayval, carryval):
#     proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)
#     proxyout = numba.cgutils.create_struct_proxy(arraytpe)(context, builder)
#     proxyout.starts = numba.targets.arrayobj.fancy_getitem_array(context, builder, arraytpe.startstpe(arraytpe.startstpe, carrytpe), (proxyin.starts, carryval))
#     proxyout.stops = numba.targets.arrayobj.fancy_getitem_array(context, builder, arraytpe.stopstpe(arraytpe.stopstpe, carrytpe), (proxyin.stops, carryval))
#     proxyout.content = proxyin.content
#     if not isinstance(arraytpe.idtpe, numba.types.NoneType):
#         raise NotImplementedError("array.id is not None")
#     outval = proxyout._getvalue()
#     if context.enable_nrt:
#         context.nrt.incref(builder, arraytpe, outval)
#     return outval

@numba.typing.templates.infer_getattr
class type_methods(numba.typing.templates.AttributeTemplate):
    key = ListOffsetArrayType

    def generic_resolve(self, tpe, attr):
        if attr == "offsets":
            return tpe.offsetstpe

        elif attr == "content":
            return tpe.contenttpe

        elif attr == "id":
            if tpe.idtpe == numba.none:
                return numba.optional(identity.IdentityType(numba.int32[:, :]))
            else:
                return tpe.idtpe

@numba.extending.lower_getattr(ListOffsetArrayType, "offsets")
def lower_offsets(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if context.enable_nrt:
        context.nrt.incref(builder, tpe.offsetstpe, proxyin.offsets)
    return proxyin.offsets

@numba.extending.lower_getattr(ListOffsetArrayType, "content")
def lower_content(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if context.enable_nrt:
        context.nrt.incref(builder, tpe.contenttpe, proxyin.content)
    return proxyin.content

@numba.extending.lower_getattr(ListOffsetArrayType, "id")
def lower_id(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if tpe.idtpe == numba.none:
        return context.make_optional_none(builder, identity.IdentityType(numba.int32[:, :]))
    else:
        if context.enable_nrt:
            context.nrt.incref(builder, tpe.idtpe, proxyin.id)
        return proxyin.id
