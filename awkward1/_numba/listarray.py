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
        headtpe = wheretpe.types[0]
        tailtpe = numba.types.Tuple(wheretpe.types[1:])
        if isinstance(headtpe, numba.types.Integer):
            return self.contenttpe.getitem(tailtpe, isadvanced)
        elif isinstance(headtpe, numba.types.SliceType):
            return ListOffsetArrayType(self.startstpe, self.contenttpe.getitem(tailtpe, isadvanced), self.idtpe)
        elif isinstance(headtpe, numba.types.EllipsisType):
            raise NotImplementedError("ellipsis")
        elif isinstance(headtpe, numba.typeof(numpy.newaxis)):
            raise NotImplementedError("newaxis")
        elif isinstance(headtpe, numba.types.Array) and not isadvanced:
            if headtpe.ndim != 1:
                raise NotImplementedError("array.ndim != 1")
            return ListOffsetArrayType(self.startstpe, self.contenttpe.getitem(tailtpe, True), self.idtpe)
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
