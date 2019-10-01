# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl
import numba.typing.ctypes_utils

import awkward1.layout
from .._numba import cpu, util, content

@numba.extending.typeof_impl.register(awkward1.layout.ListArray32)
@numba.extending.typeof_impl.register(awkward1.layout.ListArray64)
def typeof(val, c):
    return ListArrayType(numba.typeof(numpy.asarray(val.starts)), numba.typeof(numpy.asarray(val.stops)), numba.typeof(val.content), numba.typeof(val.id))

class ListArrayType(content.ContentType):
    def __init__(self, startstpe, stopstpe, contenttpe, idtpe):
        assert startstpe == stopstpe
        super(ListArrayType, self).__init__(name="ListArray{}Type({}, id={})".format(startstpe.dtype.bitwidth, contenttpe.name, idtpe.name))
        self.startstpe = startstpe
        self.contenttpe = contenttpe
        self.idtpe = idtpe

    @property
    def stopstpe(self):
        return self.startstpe

    def bitwidth(self):
        return self.startstpe.dtype.bitwidth

    @property
    def ndim(self):
        return 1 + self.contenttpe.ndim

    def getitem(self, wheretpe, isadvanced):
        import awkward1._numba.listoffsetarray

        headtpe = wheretpe.types[0]
        tailtpe = numba.types.Tuple(wheretpe.types[1:])
        if isinstance(headtpe, numba.types.Integer):
            return self.contenttpe.getitem(tailtpe, isadvanced)
        elif isinstance(headtpe, numba.types.SliceType):
            return awkward1._numba.listoffsetarray.ListOffsetArrayType(self.startstpe, self.contenttpe.getitem(tailtpe, isadvanced), self.idtpe)
        elif isinstance(headtpe, numba.types.EllipsisType):
            raise NotImplementedError("ellipsis")
        elif isinstance(headtpe, type(numba.typeof(numpy.newaxis))):
            raise NotImplementedError("newaxis")
        elif isinstance(headtpe, numba.types.Array) and not isadvanced:
            if headtpe.ndim != 1:
                raise NotImplementedError("array.ndim != 1")
            return awkward1._numba.listoffsetarray.ListOffsetArrayType(self.startstpe, self.contenttpe.getitem(tailtpe, True), self.idtpe)
        elif isinstance(headtpe, numba.types.Array):
            return self.contenttpe.getitem(tailtpe, True)
        else:
            raise AssertionError(headtpe)

    @property
    def lower_len(self):
        return lower_len

    @property
    def lower_getitem_int(self):
        return lower_getitem_int

@numba.extending.register_model(ListArrayType)
class ListArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("starts", fe_type.startstpe),
                   ("stops", fe_type.stopstpe),
                   ("content", fe_type.contenttpe)]
        if fe_type.idtpe != numba.none:
            members.append(("id", fe_type.idtpe))
        super(ListArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(ListArrayType)
def unbox(tpe, obj, c):
    asarray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numpy.asarray))
    starts_obj = c.pyapi.object_getattr_string(obj, "starts")
    stops_obj = c.pyapi.object_getattr_string(obj, "stops")
    content_obj = c.pyapi.object_getattr_string(obj, "content")
    startsarray_obj = c.pyapi.call_function_objargs(asarray_obj, (starts_obj,))
    stopsarray_obj = c.pyapi.call_function_objargs(asarray_obj, (stops_obj,))
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.starts = c.pyapi.to_native_value(tpe.startstpe, startsarray_obj).value
    proxyout.stops = c.pyapi.to_native_value(tpe.stopstpe, stopsarray_obj).value
    proxyout.content = c.pyapi.to_native_value(tpe.contenttpe, content_obj).value
    c.pyapi.decref(asarray_obj)
    c.pyapi.decref(starts_obj)
    c.pyapi.decref(stops_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(startsarray_obj)
    c.pyapi.decref(stopsarray_obj)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.object_getattr_string(obj, "id")
        proxyout.id = c.pyapi.to_native_value(tpe.idtpe, id_obj).value
        c.pyapi.decref(id_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(ListArrayType)
def box(tpe, val, c):
    if tpe.bitwidth() == 32:
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Index32))
        ListArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.ListArray32))
    elif tpe.bitwidth() == 64:
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Index64))
        ListArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.ListArray64))
    else:
        raise AssertionError("unrecognized bitwidth")
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    startsarray_obj = c.pyapi.from_native_value(tpe.startstpe, proxyin.starts, c.env_manager)
    stopsarray_obj = c.pyapi.from_native_value(tpe.stopstpe, proxyin.stops, c.env_manager)
    content_obj = c.pyapi.from_native_value(tpe.contenttpe, proxyin.content, c.env_manager)
    starts_obj = c.pyapi.call_function_objargs(Index_obj, (startsarray_obj,))
    stops_obj = c.pyapi.call_function_objargs(Index_obj, (stopsarray_obj,))
    c.pyapi.decref(Index_obj)
    c.pyapi.decref(startsarray_obj)
    c.pyapi.decref(stopsarray_obj)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.from_native_value(tpe.idtpe, proxyin.id, c.env_manager)
        out = c.pyapi.call_function_objargs(ListArray_obj, (starts_obj, stops_obj, content_obj, id_obj))
        c.pyapi.decref(id_obj)
    else:
        out = c.pyapi.call_function_objargs(ListArray_obj, (starts_obj, stops_obj, content_obj))
    c.pyapi.decref(ListArray_obj)
    c.pyapi.decref(starts_obj)
    c.pyapi.decref(stops_obj)
    c.pyapi.decref(content_obj)
    return out

@numba.extending.lower_builtin(len, ListArrayType)
def lower_len(context, builder, sig, args):
    rettpe, (tpe,) = sig.return_type, sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    startslen = numba.targets.arrayobj.array_len(context, builder, numba.intp(tpe.startstpe), (proxyin.starts,))
    return startslen

@numba.extending.lower_builtin(operator.getitem, ListArrayType, numba.types.BaseTuple)
def lower_getitem_tuple(context, builder, sig, args):
    rettpe, (arraytpe, wheretpe) = sig.return_type, sig.args
    arrayval, whereval = args

    wheretpe2 = util._typing_maskarrays_to_indexarrays(wheretpe)
    util.maskarrays_to_indexarrays.compile(wheretpe2(wheretpe))
    cres = util.maskarrays_to_indexarrays.overloads[(wheretpe,)]
    whereval2 = context.call_internal(builder, cres.fndesc, wheretpe2(wheretpe), (whereval,))

    wheretpe3 = util._typing_broadcast_arrays(wheretpe2)
    util.broadcast_arrays.compile(wheretpe3(wheretpe2))
    cres2 = util.broadcast_arrays.overloads[(wheretpe2,)]
    whereval3 = context.call_internal(builder, cres2.fndesc, wheretpe3(wheretpe2), (whereval2,))

    return lower_getitem_next(context, builder, rettpe, arraytpe, wheretpe3, arrayval, whereval3, None)

def lower_getitem_next(context, builder, rettpe, arraytpe, wheretpe, arrayval, whereval, advanced):
    if len(wheretpe.types) == 0:
        if context.enable_nrt:
            context.nrt.incref(builder, rettpe, arrayval)
        return arrayval

    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)
    lenstarts = numba.targets.arrayobj.array_len(context, builder, numba.types.intp(arraytpe.startstpe), (proxyin.starts,))
    lenstops = numba.targets.arrayobj.array_len(context, builder, numba.types.intp(arraytpe.stopstpe), (proxyin.stops,))
    lencontent = arraytpe.contenttpe.lower_len(context, builder, numba.types.intp(arraytpe.contenttpe), (proxyin.content,))
    with builder.if_then(builder.icmp_signed("<", lenstops, lenstarts), likely=False):
        context.call_conv.return_user_exc(builder, ValueError, ("len(stops) < len(starts)",))

    headtpe = wheretpe.types[0]
    tailtpe = numba.types.Tuple(wheretpe.types[1:])
    headval = numba.cgutils.unpack_tuple(builder, whereval)[0]
    tailval = context.make_tuple(builder, tailtpe, numba.cgutils.unpack_tuple(builder, whereval)[1:])

    if isinstance(headtpe, numba.types.Integer):
        raise NotImplementedError("ListArray.getitem_next(int)")

    elif isinstance(headtpe, numba.types.SliceType):
        raise NotImplementedError("ListArray.getitem_next(slice)")

    elif isinstance(headtpe, numba.types.EllipsisType):
        raise NotImplementedError("ListArray.getitem_next(ellipsis)")

    elif isinstance(headtpe, type(numba.typeof(numpy.newaxis))):
        raise NotImplementedError("ListArray.getitem_next(newaxis)")

    elif isinstance(headtpe, numba.types.Array) and advanced is None:
        if headtpe.ndim != 1:
            raise NotImplementedError("array.ndim != 1")

        flathead = numba.targets.arrayobj.array_ravel(context, builder, numba.types.Array(numba.int64, 1, "C")(headtpe), (headval,))
        lenflathead = numba.targets.arrayobj.array_len(context, builder, numba.types.intp(numba.types.Array(numba.int64, 1, "C")), (flathead,))

        lencarry = builder.mul(lenstarts, lenflathead)
        lenoffsets = builder.add(lenstarts, context.get_constant(numba.intp, 1))

        nextcarry = numba.targets.arrayobj.numpy_empty_nd(context, builder, numba.types.Array(numba.int64, 1, "C")(numba.types.intp), (lencarry,))
        nextadvanced = numba.targets.arrayobj.numpy_empty_nd(context, builder, numba.types.Array(numba.int64, 1, "C")(numba.types.intp), (lencarry,))
        nextoffsets = numba.targets.arrayobj.numpy_empty_nd(context, builder, numba.types.Array(numba.int64, 1, "C")(numba.types.intp), (lenoffsets,))

        proxynextcarry = numba.cgutils.create_struct_proxy(numba.types.Array(numba.int64, 1, "C"))(context, builder, value=nextcarry)
        proxynextadvanced = numba.cgutils.create_struct_proxy(numba.types.Array(numba.int64, 1, "C"))(context, builder, value=nextadvanced)
        proxynextoffsets = numba.cgutils.create_struct_proxy(numba.types.Array(numba.int64, 1, "C"))(context, builder, value=nextoffsets)
        proxystarts = numba.cgutils.create_struct_proxy(arraytpe.startstpe)(context, builder, value=proxyin.starts)
        proxystops = numba.cgutils.create_struct_proxy(arraytpe.stopstpe)(context, builder, value=proxyin.stops)
        proxyflathead = numba.cgutils.create_struct_proxy(numba.types.Array(numba.int64, 1, "C"))(context, builder, value=flathead)

        fcn = cpu.kernels.awkward_listarray64_getitem_next_array_64
        fcntpe = context.get_function_pointer_type(fcn.numbatpe)
        fcnval = context.add_dynamic_addr(builder, fcn.numbatpe.get_pointer(fcn), info=fcn.name)
        fcnptr = builder.bitcast(fcnval, fcntpe)

        err = context.call_function_pointer(builder, fcnptr,
            (proxynextoffsets.data,
             proxynextcarry.data,
             proxynextadvanced.data,
             proxystarts.data,
             proxystops.data,
             proxyflathead.data,
             context.get_constant(numba.int64, 0),
             context.get_constant(numba.int64, 0),
             lenstarts,
             lenflathead,
             lencontent))

        proxyout = numba.cgutils.create_struct_proxy(rettpe)(context, builder)
        proxyout.offsets = nextoffsets
        proxyout.content = proxyin.content
        out = proxyout._getvalue()
        if context.enable_nrt:
            context.nrt.incref(builder, rettpe, out)
        return out

    elif isinstance(headtpe, numba.types.Array):
        raise NotImplementedError("ListArray.getitem_next(advanced Array)")

    else:
        raise AssertionError(headtpe)
