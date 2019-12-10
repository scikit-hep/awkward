# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl
import numba.typing.ctypes_utils

import awkward1.layout
from ..._numba import cpu, util, identity, content

@numba.extending.typeof_impl.register(awkward1.layout.NumpyArray)
def typeof(val, c):
    type = val.type
    while isinstance(type, (awkward1.layout.ArrayType, awkward1.layout.RegularType)):
        type = type.type
    return NumpyArrayType(numba.typeof(numpy.asarray(val)), numba.typeof(val.id), numba.typeof(type))

class NumpyArrayType(content.ContentType):
    def __init__(self, arraytpe, idtpe, typetpe):
        super(NumpyArrayType, self).__init__(name="ak::NumpyArrayType({0}, id={1}, type={2})".format(arraytpe.name, idtpe.name, typetpe.name))
        self.arraytpe = arraytpe
        self.idtpe = idtpe
        self.typetpe = typetpe

    @property
    def ndim(self):
        return self.arraytpe.ndim

    def getitem_int(self):
        return self.getitem_tuple(numba.int64)

    def getitem_range(self):
        return self.getitem_tuple(numba.types.slice2_type)

    def getitem_str(self):
        raise IndexError("cannot slice NumpyArray with str (Record field name)")

    def getitem_tuple(self, wheretpe):
        outtpe = numba.typing.arraydecl.get_array_index_type(self.arraytpe, wheretpe).result
        if isinstance(outtpe, numba.types.Array):
            return NumpyArrayType(outtpe, self.idtpe, self.typetpe)
        else:
            return outtpe

    def getitem_next(self, wheretpe, isadvanced):
        if len(wheretpe.types) > self.arraytpe.ndim:
            raise IndexError("too many dimensions in slice")
        if any(isinstance(x, numba.types.StringLiteral) for x in wheretpe):
            raise IndexError("cannot slice NumpyArray with str (Record field name)")

        if isadvanced:
            numreduce = sum(1 if isinstance(x, (numba.types.Integer, numba.types.Array)) else 0 for x in wheretpe.types)
        else:
            numreduce = sum(1 if isinstance(x, numba.types.Integer) else 0 for x in wheretpe.types)
        if numreduce < self.arraytpe.ndim:
            return NumpyArrayType(numba.types.Array(self.arraytpe.dtype, self.arraytpe.ndim - numreduce, self.arraytpe.layout), self.idtpe, self.typetpe)
        elif numreduce == self.arraytpe.ndim:
            return self.arraytpe.dtype
        else:
            assert False

    def carry(self):
        return self

    @property
    def lower_len(self):
        return lower_len

    @property
    def lower_getitem_nothing(self):
        return None

    @property
    def lower_getitem_int(self):
        return lower_getitem

    @property
    def lower_getitem_range(self):
        return lower_getitem

    @property
    def lower_getitem_next(self):
        return lower_getitem_next

    @property
    def lower_carry(self):
        return lower_carry

@numba.extending.register_model(NumpyArrayType)
class NumpyArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("array", fe_type.arraytpe)]
        if fe_type.idtpe != numba.none:
            members.append(("id", fe_type.idtpe))
        super(NumpyArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(NumpyArrayType)
def unbox(tpe, obj, c):
    asarray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numpy.asarray))
    array_obj = c.pyapi.call_function_objargs(asarray_obj, (obj,))
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.array = c.pyapi.to_native_value(tpe.arraytpe, array_obj).value
    c.pyapi.decref(asarray_obj)
    c.pyapi.decref(array_obj)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.object_getattr_string(obj, "id")
        proxyout.id = c.pyapi.to_native_value(tpe.idtpe, id_obj).value
        c.pyapi.decref(id_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(NumpyArrayType)
def box(tpe, val, c):
    NumpyArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.NumpyArray))
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    array_obj = c.pyapi.from_native_value(tpe.arraytpe, proxyin.array, c.env_manager)
    args = [array_obj]
    if tpe.idtpe != numba.none:
        args.append(c.pyapi.from_native_value(tpe.idtpe, proxyin.id, c.env_manager))
    else:
        args.append(c.pyapi.make_none())
    if tpe.typetpe != numba.none:
        args.append(c.pyapi.unserialize(c.pyapi.serialize_object(tpe.typetpe.type)))
    else:
        args.append(c.pyapi.make_none())
    out = c.pyapi.call_function_objargs(NumpyArray_obj, args)
    for x in args:
        c.pyapi.decref(x)
    c.pyapi.decref(NumpyArray_obj)
    return out

@numba.extending.lower_builtin(len, NumpyArrayType)
def lower_len(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    return numba.targets.arrayobj.array_len(context, builder, numba.intp(tpe.arraytpe), (proxyin.array,))

@numba.extending.lower_builtin(operator.getitem, NumpyArrayType, numba.types.Integer)
@numba.extending.lower_builtin(operator.getitem, NumpyArrayType, numba.types.SliceType)
@numba.extending.lower_builtin(operator.getitem, NumpyArrayType, numba.types.Array)
@numba.extending.lower_builtin(operator.getitem, NumpyArrayType, numba.types.List)
@numba.extending.lower_builtin(operator.getitem, NumpyArrayType, numba.types.ArrayCompatible)
@numba.extending.lower_builtin(operator.getitem, NumpyArrayType, numba.types.EllipsisType)
@numba.extending.lower_builtin(operator.getitem, NumpyArrayType, type(numba.typeof(numpy.newaxis)))
@numba.extending.lower_builtin(operator.getitem, NumpyArrayType, numba.types.BaseTuple)
def lower_getitem(context, builder, sig, args):
    import awkward1._numba.identity

    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    if not isinstance(wheretpe, (numba.types.Integer, numba.types.SliceType)):
        if not isinstance(wheretpe, numba.types.BaseTuple):
            wheretpe = numba.types.Tuple((wheretpe,))
            whereval = context.make_tuple(builder, wheretpe, (whereval,))
        wheretpe, whereval = util.preprocess_slicetuple(context, builder, wheretpe, whereval)

    if isinstance(rettpe, NumpyArrayType):
        signature = rettpe.arraytpe(tpe.arraytpe, wheretpe)
    else:
        signature = rettpe(tpe.arraytpe, wheretpe)

    if isinstance(wheretpe, numba.types.BaseTuple):
        out = numba.targets.arrayobj.getitem_array_tuple(context, builder, signature, (proxyin.array, whereval))
    else:
        out = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, signature, (proxyin.array, whereval))

    if isinstance(rettpe, NumpyArrayType):
        proxyout = numba.cgutils.create_struct_proxy(rettpe)(context, builder)
        proxyout.array = out
        if rettpe.idtpe != numba.none:
            proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, rettpe.idtpe, wheretpe, proxyin.id, whereval)
        return proxyout._getvalue()
    else:
        return out

def lower_getitem_next(context, builder, arraytpe, wheretpe, arrayval, whereval, advanced):
    if len(wheretpe.types) == 0:
        return arrayval
    headtpe = wheretpe.types[0]
    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)

    if isinstance(headtpe, numba.types.Array) and advanced is not None:
        tailtpe = numba.types.Tuple(wheretpe.types[1:])
        headval = numba.cgutils.unpack_tuple(builder, whereval)[0]
        tailval = context.make_tuple(builder, tailtpe, numba.cgutils.unpack_tuple(builder, whereval)[1:])
        if headtpe.ndim != 1:
            raise NotImplementedError("array.ndim != 1")
        if arraytpe.arraytpe.ndim < 2:
            raise TypeError("too many dimensions in slice")

        shapeval = numba.targets.arrayobj.make_array(arraytpe.arraytpe)(context, builder, proxyin.array).shape
        shapeunpacked = numba.cgutils.unpack_tuple(builder, shapeval)
        lenself, skip = shapeunpacked[:2]
        lennext = builder.mul(lenself, skip)

        carry = numba.targets.arrayobj.numpy_arange_1(context, builder, util.index64tpe(numba.intp), (lenself,))

        flathead = numba.targets.arrayobj.array_flatten(context, builder, util.index64tpe(headtpe), (headval,))
        lenflathead = util.arraylen(context, builder, util.index64tpe, flathead)

        util.call(context, builder, cpu.kernels.awkward_regularize_arrayslice_64,
            (util.arrayptr(context, builder, util.index64tpe, flathead),
             util.cast(context, builder, numba.intp, numba.int64, lenflathead),
             util.cast(context, builder, numba.intp, numba.int64, skip)),
            "in {0}, indexing error".format(arraytpe.shortname))

        nextcarry = util.newindex64(context, builder, numba.intp, lenself)
        util.call(context, builder, cpu.kernels.awkward_numpyarray_getitem_next_array_advanced_64,
            (util.arrayptr(context, builder, util.index64tpe, nextcarry),
             util.arrayptr(context, builder, util.index64tpe, carry),
             util.arrayptr(context, builder, util.index64tpe, advanced),
             util.arrayptr(context, builder, util.index64tpe, flathead),
             util.cast(context, builder, numba.intp, numba.int64, lenself),
             util.cast(context, builder, numba.intp, numba.int64, skip)),
            "in {0}, indexing error".format(arraytpe.shortname))

        nextshapetpe = numba.types.UniTuple(numba.intp, arraytpe.arraytpe.ndim - 1)
        nextshapeval = context.make_tuple(builder, nextshapetpe, [lennext] + shapeunpacked[2:])

        nextarraytpe = numba.types.Array(arraytpe.arraytpe.dtype, arraytpe.arraytpe.ndim - 1, arraytpe.arraytpe.layout)
        nextarrayval = numba.targets.arrayobj.array_reshape(context, builder, nextarraytpe(arraytpe.arraytpe, nextshapetpe), (proxyin.array, nextshapeval))

        nexttpe = NumpyArrayType(nextarraytpe, arraytpe.idtpe, arraytpe.typetpe)
        proxynext = numba.cgutils.create_struct_proxy(nexttpe)(context, builder)
        proxynext.array = nextarrayval
        if arraytpe.idtpe != numba.none:
            proxynext.id = proxyin.id

        outval = lower_carry(context, builder, nexttpe, util.index64tpe, proxynext._getvalue(), nextcarry)
        return lower_getitem_next(context, builder, nexttpe, tailtpe, outval, tailval, advanced)

    else:
        proxyslice = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
        proxyslice.start = context.get_constant(numba.intp, 0)
        proxyslice.stop = util.arraylen(context, builder, arraytpe.arraytpe, proxyin.array, totpe=numba.intp)
        proxyslice.step = context.get_constant(numba.intp, 1)
        wheretpe = numba.types.Tuple((numba.types.slice2_type,) + wheretpe.types)
        whereval = context.make_tuple(builder, wheretpe, [proxyslice._getvalue()] + numba.cgutils.unpack_tuple(builder, whereval))

        outtpe = numba.typing.arraydecl.get_array_index_type(arraytpe.arraytpe, wheretpe).result
        outval = numba.targets.arrayobj.getitem_array_tuple(context, builder, outtpe(arraytpe.arraytpe, wheretpe), (proxyin.array, whereval))

        if isinstance(outtpe, numba.types.Array):
            proxyout = numba.cgutils.create_struct_proxy(NumpyArrayType(outtpe, arraytpe.idtpe, arraytpe.typetpe))(context, builder)
            proxyout.array = outval
            if arraytpe.idtpe != numba.none:
                proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, arraytpe.idtpe, wheretpe, proxyin.id, whereval)
            return proxyout._getvalue()
        else:
            return out

def lower_carry(context, builder, arraytpe, carrytpe, arrayval, carryval):
    import awkward1._numba.identity

    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)

    proxyout = numba.cgutils.create_struct_proxy(arraytpe)(context, builder)
    proxyout.array = numba.targets.arrayobj.fancy_getitem_array(context, builder, arraytpe.arraytpe(arraytpe.arraytpe, carrytpe), (proxyin.array, carryval))
    if arraytpe.idtpe != numba.none:
        proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, arraytpe.idtpe, carrytpe, proxyin.id, carryval)
    return proxyout._getvalue()

@numba.typing.templates.infer_getattr
class type_methods(numba.typing.templates.AttributeTemplate):
    key = NumpyArrayType

    def generic_resolve(self, tpe, attr):
        if attr == "id":
            if tpe.idtpe == numba.none:
                return numba.optional(identity.IdentityType(numba.int32[:, :]))
            else:
                return tpe.idtpe

@numba.extending.lower_getattr(NumpyArrayType, "id")
def lower_id(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if tpe.idtpe == numba.none:
        return context.make_optional_none(builder, identity.IdentityType(numba.int32[:, :]))
    else:
        if context.enable_nrt:
            context.nrt.incref(builder, tpe.idtpe, proxyin.id)
        return proxyin.id
