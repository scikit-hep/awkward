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

    @property
    def bitwidth(self):
        return self.startstpe.dtype.bitwidth

    @property
    def ndim(self):
        return 1 + self.contenttpe.ndim

    def getitem_int(self):
        return self.contenttpe

    def getitem_range(self):
        return self

    def getitem_tuple(self, wheretpe):
        nexttpe = ListArrayType(numba.types.Array(numba.int64, 1, "C"), numba.types.Array(numba.int64, 1, "C"), self, numba.none)
        out = nexttpe.getitem_next(wheretpe, False)
        return out.getitem_int()

    def getitem_next(self, wheretpe, isadvanced):
        import awkward1._numba.listoffsetarray
        if len(wheretpe.types) == 0:
            return self
        headtpe = wheretpe.types[0]
        tailtpe = numba.types.Tuple(wheretpe.types[1:])
        if isinstance(headtpe, numba.types.Integer):
            return self.contenttpe.getitem_next(tailtpe, isadvanced)
        elif isinstance(headtpe, numba.types.SliceType):
            return awkward1._numba.listoffsetarray.ListOffsetArrayType(self.startstpe, self.contenttpe.getitem_next(tailtpe, isadvanced), self.idtpe)
        elif isinstance(headtpe, numba.types.EllipsisType):
            raise NotImplementedError("ellipsis")
        elif isinstance(headtpe, type(numba.typeof(numpy.newaxis))):
            raise NotImplementedError("newaxis")
        elif isinstance(headtpe, numba.types.Array) and not isadvanced:
            if headtpe.ndim != 1:
                raise NotImplementedError("array.ndim != 1")
            contenttpe = self.contenttpe.carry().getitem_next(tailtpe, True)
            return awkward1._numba.listoffsetarray.ListOffsetArrayType(self.startstpe, contenttpe, self.idtpe)
        elif isinstance(headtpe, numba.types.Array):
            return self.contenttpe.getitem_next(tailtpe, True)
        else:
            raise AssertionError(headtpe)

    def carry(self):
        return self

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
    if tpe.bitwidth == 32:
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Index32))
        ListArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.ListArray32))
    elif tpe.bitwidth == 64:
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

@numba.extending.lower_builtin(operator.getitem, ListArrayType, numba.types.Integer)
def lower_getitem_int(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    if tpe.bitwidth == 32:
        starttpe, stoptpe = numba.int32, numba.int32
    elif tpe.bitwidth == 64:
        starttpe, stoptpe = numba.int64, numba.int64

    start = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, starttpe(tpe.startstpe, wheretpe), (proxyin.starts, whereval))
    stop = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, stoptpe(tpe.stopstpe, wheretpe), (proxyin.stops, whereval))
    proxyslice = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    if tpe.bitwidth < numba.intp.bitwidth:
        proxyslice.start = builder.zext(start, context.get_value_type(numba.intp))
        proxyslice.stop = builder.zext(stop, context.get_value_type(numba.intp))
    elif tpe.bitwidth == numba.intp.bitwidth:
        proxyslice.start = start
        proxyslice.stop = stop
    elif tpe.bitwidth > numba.intp.bitwidth:
        proxyslice.start = builder.trunc(start, context.get_value_type(numba.intp))
        proxyslice.stop = builder.trunc(stop, context.get_value_type(numba.intp))
    proxyslice.step = context.get_constant(numba.intp, 1)

    fcn = context.get_function(operator.getitem, rettpe(tpe.contenttpe, numba.types.slice2_type))
    return fcn(builder, (proxyin.content, proxyslice._getvalue()))

@numba.extending.lower_builtin(operator.getitem, ListArrayType, numba.types.slice2_type)
def lower_getitem_range(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    proxyout = numba.cgutils.create_struct_proxy(tpe)(context, builder)
    proxyout.starts = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.startstpe(tpe.startstpe, wheretpe), (proxyin.starts, whereval))
    proxyout.stops = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.stopstpe(tpe.stopstpe, wheretpe), (proxyin.stops, whereval))
    proxyout.content = proxyin.content
    if not isinstance(tpe.idtpe, numba.types.NoneType):
        raise NotImplementedError("id is not None")

    out = proxyout._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out

@numba.extending.lower_builtin(operator.getitem, ListArrayType, numba.types.BaseTuple)
def lower_getitem_tuple(context, builder, sig, args):
    rettpe, (arraytpe, wheretpe1) = sig.return_type, sig.args
    arrayval, whereval1 = args

    wheretpe2 = util._typing_maskarrays_to_indexarrays(wheretpe1)
    util.maskarrays_to_indexarrays.compile(wheretpe2(wheretpe1))
    cres = util.maskarrays_to_indexarrays.overloads[(wheretpe1,)]
    whereval2 = context.call_internal(builder, cres.fndesc, wheretpe2(wheretpe1), (whereval1,))

    wheretpe3 = util._typing_broadcast_arrays(wheretpe2)
    util.broadcast_arrays.compile(wheretpe3(wheretpe2))
    cres2 = util.broadcast_arrays.overloads[(wheretpe2,)]
    whereval3 = context.call_internal(builder, cres2.fndesc, wheretpe3(wheretpe2), (whereval2,))

    length = lower_len(context, builder, numba.intp(arraytpe), (arrayval,))
    if numba.int64.bitwidth > numba.intp.bitwidth:
        length = builder.zext(length, context.get_value_type(numba.int64))

    nexttpe = ListArrayType(numba.types.Array(numba.int64, 1, "C"), numba.types.Array(numba.int64, 1, "C"), arraytpe, numba.types.none)
    proxynext = numba.cgutils.create_struct_proxy(nexttpe)(context, builder)
    proxynext.starts = numba.targets.arrayobj.numpy_empty_nd(context, builder, numba.types.Array(numba.int64, 1, "C")(numba.intp), (context.get_constant(numba.intp, 1),))
    proxynext.stops = numba.targets.arrayobj.numpy_empty_nd(context, builder, numba.types.Array(numba.int64, 1, "C")(numba.intp), (context.get_constant(numba.intp, 1),))
    numba.targets.arrayobj.store_item(context, builder, numba.types.Array(numba.int64, 1, "C"), context.get_constant(numba.int64, 0), numba.targets.arrayobj.make_array(numba.types.Array(numba.int64, 1, "C"))(context, builder, proxynext.starts).data)
    numba.targets.arrayobj.store_item(context, builder, numba.types.Array(numba.int64, 1, "C"), length, numba.targets.arrayobj.make_array(numba.types.Array(numba.int64, 1, "C"))(context, builder, proxynext.stops).data)
    proxynext.content = arrayval
    nextval = proxynext._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, nexttpe, nextval)

    outtpe = nexttpe.getitem_next(wheretpe3, False)
    outval = nexttpe.lower_getitem_next(context, builder, nexttpe, wheretpe3, nextval, whereval3, None)

    return outtpe.lower_getitem_int(context, builder, rettpe(outtpe, numba.int64), (outval, context.get_constant(numba.int64, 0)))

def lower_getitem_next(context, builder, arraytpe, wheretpe, arrayval, whereval, advanced):
    import awkward1._numba.listoffsetarray

    if len(wheretpe.types) == 0:
        if context.enable_nrt:
            context.nrt.incref(builder, arraytpe, arrayval)
        return arrayval

    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)
    lenstarts = numba.targets.arrayobj.array_len(context, builder, numba.intp(arraytpe.startstpe), (proxyin.starts,))
    lenstops = numba.targets.arrayobj.array_len(context, builder, numba.intp(arraytpe.stopstpe), (proxyin.stops,))
    lencontent = arraytpe.contenttpe.lower_len(context, builder, numba.intp(arraytpe.contenttpe), (proxyin.content,))
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

        print("before ravel, headtpe", headtpe, "headtpe.dtype", headtpe.dtype)
        print("              headval", headval)

        flathead = numba.targets.arrayobj.array_ravel(context, builder, numba.types.Array(numba.int64, 1, "C")(headtpe), (headval,))
        lenflathead = numba.targets.arrayobj.array_len(context, builder, numba.intp(numba.types.Array(numba.int64, 1, "C")), (flathead,))

        lencarry = builder.mul(lenstarts, lenflathead)
        lenoffsets = builder.add(lenstarts, context.get_constant(numba.intp, 1))

        nextcarry = numba.targets.arrayobj.numpy_empty_nd(context, builder, numba.types.Array(numba.int64, 1, "C")(numba.intp), (lencarry,))
        nextadvanced = numba.targets.arrayobj.numpy_empty_nd(context, builder, numba.types.Array(numba.int64, 1, "C")(numba.intp), (lencarry,))
        nextoffsets = numba.targets.arrayobj.numpy_empty_nd(context, builder, numba.types.Array(numba.int64, 1, "C")(numba.intp), (lenoffsets,))

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

        nexttpe = arraytpe.contenttpe.carry()
        nextval = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, numba.types.Array(numba.int64, 1, "C"), proxyin.content, nextcarry)

        contenttpe = nexttpe.getitem_next(tailtpe, True)
        contentval = nexttpe.lower_getitem_next(context, builder, nexttpe, tailtpe, nextval, tailval, nextadvanced)

        if not isinstance(arraytpe.idtpe, numba.types.NoneType):
            raise NotImplementedError("array.id is not None")

        outtpe = awkward1._numba.listoffsetarray.ListOffsetArrayType(arraytpe.startstpe, contenttpe, arraytpe.idtpe)
        proxyout = numba.cgutils.create_struct_proxy(outtpe)(context, builder)
        proxyout.offsets = nextoffsets
        proxyout.content = contentval
        outval = proxyout._getvalue()
        if context.enable_nrt:
            context.nrt.incref(builder, arraytpe.startstpe, nextoffsets)
            context.nrt.incref(builder, outtpe, outval)
        return outval

    elif isinstance(headtpe, numba.types.Array):
        raise NotImplementedError("ListArray.getitem_next(advanced Array)")

    else:
        raise AssertionError(headtpe)

def lower_carry(context, builder, arraytpe, carrytpe, arrayval, carryval):
    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)
    proxyout = numba.cgutils.create_struct_proxy(arraytpe)(context, builder)
    proxyout.starts = numba.targets.arrayobj.fancy_getitem_array(context, builder, arraytpe.startstpe(arraytpe.startstpe, carrytpe), (proxyin.starts, carryval))
    proxyout.stops = numba.targets.arrayobj.fancy_getitem_array(context, builder, arraytpe.stopstpe(arraytpe.stopstpe, carrytpe), (proxyin.stops, carryval))
    proxyout.content = proxyin.content
    if not isinstance(arraytpe.idtpe, numba.types.NoneType):
        raise NotImplementedError("array.id is not None")
    outval = proxyout._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, arraytpe, outval)
    return outval
