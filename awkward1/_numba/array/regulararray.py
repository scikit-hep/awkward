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

    def getitem_str(self, key):
        return RegularArrayType(self.contenttpe.getitem_str(key), self.idtpe)

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
            return RegularArrayType(contenttpe, self.idtpe)

        elif isinstance(headtpe, numba.types.StringLiteral):
            ### FIXME: this is the next one

            raise NotImplementedError("string-literal")

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
        return RegularArrayType(self.contenttpe.carry(), self.idtpe)

    @property
    def lower_len(self):
        return lower_len

    @property
    def lower_getitem_nothing(self):
        return content.lower_getitem_nothing

    @property
    def lower_getitem_int(self):
        return lower_getitem_int

    @property
    def lower_getitem_range(self):
        return lower_getitem_range

    @property
    def lower_getitem_str(self):
        return lower_getitem_str

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
        c.pyapi.decref(id_obj)
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
    size = util.cast(context, builder, numba.int64, numba.intp, proxyin.size)

    proxyslicein = context.make_helper(builder, wheretpe, value=whereval)
    numba.targets.slicing.guard_invalid_slice(context, builder, wheretpe, proxyslicein)
    numba.targets.slicing.fix_slice(builder, proxyslicein, util.arraylen(context, builder, tpe, val))

    proxysliceout = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxysliceout.start = builder.mul(proxyslicein.start, size)
    proxysliceout.stop = builder.mul(proxyslicein.stop, size)
    proxysliceout.step = context.get_constant(numba.intp, 1)

    proxyout = numba.cgutils.create_struct_proxy(tpe)(context, builder)
    proxyout.content = tpe.contenttpe.lower_getitem_range(context, builder, rettpe.contenttpe(tpe.contenttpe, numba.types.slice2_type), (proxyin.content, proxysliceout._getvalue()))
    proxyout.size = proxyin.size
    if tpe.idtpe != numba.none:
        proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, tpe.idtpe, wheretpe, proxyin.id, whereval)

    out = proxyout._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out

@numba.extending.lower_builtin(operator.getitem, RegularArrayType, numba.types.StringLiteral)
def lower_getitem_str(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    proxyout = numba.cgutils.create_struct_proxy(rettpe)(context, builder)
    proxyout.size = proxyin.size
    proxyout.content = tpe.contenttpe.lower_getitem_str(context, builder, rettpe.contenttpe(tpe.contenttpe, wheretpe), (proxyin.content, whereval))
    if tpe.idtpe != numba.none:
        proxyout.id = proxyin.id

    out = proxyout._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out

@numba.extending.lower_builtin(operator.getitem, RegularArrayType, numba.types.BaseTuple)
def lower_getitem_tuple(context, builder, sig, args):
    return content.lower_getitem_tuple(context, builder, sig, args)

@numba.extending.lower_builtin(operator.getitem, RegularArrayType, numba.types.Array)
@numba.extending.lower_builtin(operator.getitem, RegularArrayType, numba.types.List)
@numba.extending.lower_builtin(operator.getitem, RegularArrayType, numba.types.ArrayCompatible)
@numba.extending.lower_builtin(operator.getitem, RegularArrayType, numba.types.EllipsisType)
@numba.extending.lower_builtin(operator.getitem, RegularArrayType, type(numba.typeof(numpy.newaxis)))
def lower_getitem_other(context, builder, sig, args):
    return content.lower_getitem_other(context, builder, sig, args)

def lower_getitem_next(context, builder, arraytpe, wheretpe, arrayval, whereval, advanced):
    if len(wheretpe.types) == 0:
        return arrayval

    headtpe = wheretpe.types[0]
    tailtpe = numba.types.Tuple(wheretpe.types[1:])
    headval = numba.cgutils.unpack_tuple(builder, whereval)[0]
    tailval = context.make_tuple(builder, tailtpe, numba.cgutils.unpack_tuple(builder, whereval)[1:])

    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)
    leng = util.arraylen(context, builder, arraytpe, arrayval, totpe=numba.int64)

    if isinstance(headtpe, numba.types.Integer):
        assert advanced is None
        nextcarry = util.newindex64(context, builder, numba.int64, leng)
        util.call(context, builder, cpu.kernels.awkward_regulararray_getitem_next_at_64,
            (util.arrayptr(context, builder, util.index64tpe, nextcarry),
             util.cast(context, builder, headtpe, numba.int64, headval),
             leng,
             proxyin.size),
            "in {}, indexing error".format(arraytpe.shortname))
        nextcontenttpe = arraytpe.contenttpe.carry()
        nextcontentval = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, util.index64tpe, proxyin.content, nextcarry)
        return nextcontenttpe.lower_getitem_next(context, builder, nextcontenttpe, tailtpe, nextcontentval, tailval, advanced)

    elif isinstance(headtpe, numba.types.SliceType):
        proxyslicein = context.make_helper(builder, headtpe, value=headval)
        numba.targets.slicing.guard_invalid_slice(context, builder, headtpe, proxyslicein)
        numba.targets.slicing.fix_slice(builder, proxyslicein, util.cast(context, builder, numba.int64, numba.intp, proxyin.size))
        nextsize = util.cast(context, builder, numba.intp, numba.int64, numba.targets.slicing.get_slice_length(builder, proxyslicein))

        nextcarry = util.newindex64(context, builder, numba.int64, builder.mul(leng, nextsize))
        util.call(context, builder, cpu.kernels.awkward_regulararray_getitem_next_range_64,
            (util.arrayptr(context, builder, util.index64tpe, nextcarry),
             util.cast(context, builder, numba.intp, numba.int64, proxyslicein.start),
             util.cast(context, builder, numba.intp, numba.int64, proxyslicein.step),
             leng,
             proxyin.size,
             nextsize),
            "in {}, indexing error".format(arraytpe.shortname))

        nextcontenttpe = arraytpe.contenttpe.carry()
        nextcontentval = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, util.index64tpe, proxyin.content, nextcarry)

        if advanced is None:
            outcontenttpe = nextcontenttpe.getitem_next(tailtpe, False)
            outcontentval = nextcontenttpe.lower_getitem_next(context, builder, nextcontenttpe, tailtpe, nextcontentval, tailval, advanced)

        else:
            nextadvanced = util.newindex64(context, builder, numba.int64, builder.mul(leng, nextsize))
            util.call(context, builder, cpu.kernels.awkward_regulararray_getitem_next_range_spreadadvanced_64,
                (util.arrayptr(context, builder, util.index64tpe, nextadvanced),
                 util.arrayptr(context, builder, util.index64tpe, advanced),
                 leng,
                 nextsize),
                "in {}, indexing error".format(arraytpe.shortname))

            outcontenttpe = nextcontenttpe.getitem_next(tailtpe, True)
            outcontentval = nextcontenttpe.lower_getitem_next(context, builder, nextcontenttpe, tailtpe, nextcontentval, tailval, nextadvanced)

        outtpe = RegularArrayType(outcontenttpe, arraytpe.idtpe)
        proxyout = numba.cgutils.create_struct_proxy(outtpe)(context, builder)
        proxyout.content = outcontentval
        proxyout.size = nextsize
        if arraytpe.idtpe != numba.none:
            proxyout.id = proxyin.id
        return proxyout._getvalue()

    elif isinstance(headtpe, numba.types.StringLiteral):
        raise NotImplementedError("RegularArray.getitem_next(StringLiteral)")

    elif isinstance(headtpe, numba.types.EllipsisType):
        raise NotImplementedError("RegularArray.getitem_next(ellipsis)")

    elif isinstance(headtpe, type(numba.typeof(numpy.newaxis))):
        raise NotImplementedError("RegularArray.getitem_next(newaxis)")

    elif isinstance(headtpe, numba.types.Array):
        if headtpe.ndim != 1:
            raise NotImplementedError("array.ndim != 1")

        flathead = numba.targets.arrayobj.array_ravel(context, builder, util.index64tpe(headtpe), (headval,))
        lenflathead = util.arraylen(context, builder, util.index64tpe, flathead, totpe=numba.int64)
        regular_flathead = util.newindex64(context, builder, numba.int64, lenflathead)

        util.call(context, builder, cpu.kernels.awkward_regulararray_getitem_next_array_regularize_64,
            (util.arrayptr(context, builder, util.index64tpe, regular_flathead),
             util.arrayptr(context, builder, util.index64tpe, flathead),
             lenflathead,
             proxyin.size),
            "in {}, indexing error".format(arraytpe.shortname))

        if advanced is None:
            lencarry = builder.mul(leng, lenflathead)
            nextcarry = util.newindex64(context, builder, numba.int64, lencarry)
            nextadvanced = util.newindex64(context, builder, numba.int64, lencarry)

            util.call(context, builder, cpu.kernels.awkward_regulararray_getitem_next_array_64,
                (util.arrayptr(context, builder, util.index64tpe, nextcarry),
                 util.arrayptr(context, builder, util.index64tpe, nextadvanced),
                 util.arrayptr(context, builder, util.index64tpe, regular_flathead),
                 leng,
                 lenflathead,
                 proxyin.size),
                "in {}, indexing error".format(arraytpe.shortname))

            nexttpe = arraytpe.contenttpe.carry()
            nextval = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, util.index64tpe, proxyin.content, nextcarry)

            contenttpe = nexttpe.getitem_next(tailtpe, True)
            contentval = nexttpe.lower_getitem_next(context, builder, nexttpe, tailtpe, nextval, tailval, nextadvanced)

            outtpe = RegularArrayType(contenttpe, arraytpe.idtpe)
            proxyout = numba.cgutils.create_struct_proxy(outtpe)(context, builder)
            proxyout.content = contentval
            proxyout.size = lenflathead
            if outtpe.idtpe != numba.none:
                proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, outtpe.idtpe, util.index64tpe, proxyin.id, flathead)
            return proxyout._getvalue()

        else:
            nextcarry = util.newindex64(context, builder, numba.int64, leng)
            nextadvanced = util.newindex64(context, builder, numba.int64, leng)

            util.call(context, builder, cpu.kernels.awkward_regulararray_getitem_next_array_advanced_64,
                (util.arrayptr(context, builder, util.index64tpe, nextcarry),
                 util.arrayptr(context, builder, util.index64tpe, nextadvanced),
                 util.arrayptr(context, builder, util.index64tpe, advanced),
                 util.arrayptr(context, builder, util.index64tpe, regular_flathead),
                 leng,
                 lenflathead,
                 proxyin.size),
                "in {}, indexing error".format(arraytpe.shortname))

            nexttpe = arraytpe.contenttpe.carry()
            nextval = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, util.index64tpe, proxyin.content, nextcarry)

            outtpe = nexttpe.getitem_next(tailtpe, True)
            return nexttpe.lower_getitem_next(context, builder, nexttpe, tailtpe, nextval, tailval, nextadvanced)

    else:
        raise AssertionError(headtpe)

def lower_carry(context, builder, arraytpe, carrytpe, arrayval, carryval):
    import awkward1._numba.identity

    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)

    lencarry = util.arraylen(context, builder, carrytpe, carryval, totpe=numba.int64)
    nextcarry = util.newindex64(context, builder, numba.int64, builder.mul(lencarry, proxyin.size))
    util.call(context, builder, cpu.kernels.awkward_regulararray_getitem_carry_64,
        (util.arrayptr(context, builder, util.index64tpe, nextcarry),
         util.arrayptr(context, builder, carrytpe, carryval),
         lencarry,
         proxyin.size),
        "in {}, indexing error".format(arraytpe.shortname))
    nextcontent = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, util.index64tpe, proxyin.content, nextcarry)

    rettpe = arraytpe.carry()
    proxyout = numba.cgutils.create_struct_proxy(rettpe)(context, builder)
    proxyout.content = nextcontent
    proxyout.size = proxyin.size
    if rettpe.idtpe != numba.none:
        proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, rettpe.idtpe, carrytpe, proxyin.id, carryval)

    return proxyout._getvalue()

@numba.typing.templates.infer_getattr
class type_methods(numba.typing.templates.AttributeTemplate):
    key = RegularArrayType

    def generic_resolve(self, tpe, attr):
        if attr == "content":
            return tpe.contenttpe

        elif attr == "size":
            return numba.int64

        elif attr == "id":
            if tpe.idtpe == numba.none:
                return numba.optional(identity.IdentityType(numba.int32[:, :]))
            else:
                return tpe.idtpe

@numba.extending.lower_getattr(RegularArrayType, "content")
def lower_content(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if context.enable_nrt:
        context.nrt.incref(builder, tpe.contenttpe, proxyin.content)
    return proxyin.content

@numba.extending.lower_getattr(RegularArrayType, "size")
def lower_size(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    return proxyin.size

@numba.extending.lower_getattr(RegularArrayType, "id")
def lower_id(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if tpe.idtpe == numba.none:
        return context.make_optional_none(builder, identity.IdentityType(numba.int32[:, :]))
    else:
        if context.enable_nrt:
            context.nrt.incref(builder, tpe.idtpe, proxyin.id)
        return proxyin.id
