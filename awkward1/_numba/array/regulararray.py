# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl

import awkward1.layout
from ..._numba import cpu, util, content

@numba.extending.typeof_impl.register(awkward1.layout.RegularArray)
def typeof(val, c):
    return RegularArrayType(numba.typeof(val.content), numba.typeof(val.id))

class RegularArrayType(content.ContentType):
    def __init__(self, contenttpe, idtpe):
        super(RegularArrayType, self).__init__(name="RegularArrayType({}, id={})".format(contenttpe.name, idtpe.name))
        self.contenttpe = contenttpe
        self.idtpe = idtpe

    @property
    def ndim(self):
        return 1 + self.contenttpe.ndim

    def getitem_int(self):
        return self.contenttpe

    def getitem_range(self):
        return self

    def getitem_tuple(self, wheretpe):
        nexttpe = RegularArrayType(self, numba.none)
        out = nexttpe.getitem_next(wheretpe, False)
        return out.getitem_int()

    def getitem_next(self, wheretpe, isadvanced):
        if len(wheretpe.types) == 0:
            return self
        headtpe = wheretpe.types[0]
        tailtpe = numba.types.Tuple(wheretpe.types[1:])

        if isinstance(headtpe, numba.types.Integer):
            return self.contenttpe.carry().getitem_next(tailtpe, isadvanced)

        elif isinstance(headtpe, numba.types.SliceType):
            contenttpe = self.contenttpe.carry().getitem_next(tailtpe, isadvanced)
            return RegularArrayType(contenttpe, self.idtype)

        elif isinstance(headtpe, numba.types.EllipsisType):
            raise NotImplementedError("ellipsis")

        elif isinstance(headtpe, type(numba.typeof(numpy.newaxis))):
            raise NotImplementedError("newaxis")

        elif isinstance(headtpe, numba.types.Array):
            if headtpe.ndim != 1:
                raise NotImplementedError("array.ndim != 1")
            contenttpe = self.contenttpe.carry().getitem_next(tailtpe, True)
            if not isadvanced:
                return RegularArrayType(contenttpe, self.idtpe)
            else:
                return contenttpe

        else:
            raise AssertionError(headtpe)

    def carry(self):
        return RegularArrayType(self.contenttpe, self.idtpe)

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

@numba.extending.register_model(RegularArrayType)
class RegularArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("content", fe_type.contenttpe),
                   ("size", numba.int64)]
        if fe_type.idtpe != numba.none:
            members.append(("id", fe_type.idtpe))
        super(RegularArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(RegularArrayType)
def unbox(tpe, obj, c):
    content_obj = c.pyapi.object_getattr_string(obj, "content")
    size_obj = c.pyapi.object_getattr_string(obj, "size")
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.content = c.pyapi.to_native_value(tpe.contenttpe, content_obj).value
    proxyout.size = c.pyapi.to_native_value(numba.int64, size_obj).value
    c.pyapi.decref(content_obj)
    c.pyapi.decref(size_obj)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.object_getattr_string(obj, "id")
        proxyout.id = c.pyapi.to_native_value(tpe.idtpe, id_obj).value
        c.pyapi.decref(id_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(RegularArrayType)
def box(tpe, val, c):
    RegularArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.RegularArray))
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    content_obj = c.pyapi.from_native_value(tpe.contenttpe, proxyin.content, c.env_manager)
    size_obj = c.pyapi.long_from_longlong(proxyin.size)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.from_native_value(tpe.idtpe, proxyin.id, c.env_manager)
        out = c.pyapi.call_function_objargs(RegularArray_obj, (content_obj, size_obj, id_obj))
        c.pyapi.decreef(id_obj)
    else:
        out = c.pyapi.call_function_objargs(RegularArray_obj, (content_obj, size_obj))
    c.pyapi.decref(RegularArray_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(size_obj)
    return out

@numba.extending.lower_builtin(len, RegularArrayType)
def lower_len(context, builder, sig, args):
    rettpe, (tpe,) = sig.return_type, sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    innerlen = tpe.contenttpe.lower_len(context, builder, rettpe(tpe.contenttpe), (proxyin.content,))
    size = util.cast(context, builder, numba.int64, numba.intp, proxyin.size)
    return builder.sdiv(innerlen, size)

@numba.extending.lower_builtin(operator.getitem, RegularArrayType, numba.types.Integer)
def lower_getitem_int(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args
    whereval, wheretpe = util.cast(context, builder, wheretpe, numba.int64, whereval), numba.int64
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    start = builder.mul(whereval, proxyin.size)
    stop = builder.mul(builder.add(whereval, context.get_constant(numba.int64, 1)), proxyin.size)
    proxyslice = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxyslice.start = util.cast(context, builder, numba.int64, numba.intp, start)
    proxyslice.stop = util.cast(context, builder, numba.int64, numba.intp, stop)
    proxyslice.step = context.get_constant(numba.intp, 1)

    outtpe = tpe.contenttpe.getitem_range()
    return tpe.contenttpe.lower_getitem_range(context, builder, outtpe(tpe.contenttpe, numba.types.slice2_type), (proxyin.content, proxyslice._getvalue()))

@numba.extending.lower_builtin(operator.getitem, RegularArrayType, numba.types.slice2_type)
def lower_getitem_range(context, builder, sig, args):
    import awkward1._numba.identity

    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    size = util.cast(context, builder, numba.intp, numba.int64, proxyin.size)

    proxyslicein = context.make_helper(builder, wheretpe, value=whereval)
    numba.targets.slicing.guard_invalid_slice(context, builder, wheretpe, proxyslicein)
    numba.targets.slicing.fix_slice(builder, proxyslicein, lower_len(context, builder, numba.intp(tpe), (val,)))

    proxysliceout = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxysliceout.start = builder.mul(proxyslicein.start, size)
    proxysliceout.stop = builder.mul(proxyslicein.stop, size)
    proxysliceout.step = builder.mul(proxyslicein.step, size)

    proxyout = numba.cgutils.create_struct_proxy(tpe)(context, builder)
    proxyout.content = tpe.contenttpe.lower_getitem_range(context, builder, rettpe.contenttpe(tpe.contenttpe, numba.types.slice2_type), (proxyin.content, proxysliceout._getvalue()))
    proxyout.size = proxyin.size
    if tpe.idtpe != numba.none:
        proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, tpe.idtpe, wheretpe, proxyin.id, whereval)

    out = proxyout._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out
