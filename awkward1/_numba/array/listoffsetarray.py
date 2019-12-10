# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl

import awkward1.layout
from ..._numba import cpu, util, content

@numba.extending.typeof_impl.register(awkward1.layout.ListOffsetArray32)
@numba.extending.typeof_impl.register(awkward1.layout.ListOffsetArrayU32)
@numba.extending.typeof_impl.register(awkward1.layout.ListOffsetArray64)
def typeof(val, c):
    type = val.type
    if isinstance(type, awkward1.layout.ArrayType):
        type = type.type
    return ListOffsetArrayType(numba.typeof(numpy.asarray(val.offsets)), numba.typeof(val.content), numba.typeof(val.id), numba.typeof(type))

class ListOffsetArrayType(content.ContentType):
    def __init__(self, offsetstpe, contenttpe, idtpe, typetpe):
        super(ListOffsetArrayType, self).__init__(name="ak::ListOffsetArray{0}{1}Type({2}, id={3}, type={4})".format("" if offsetstpe.dtype.signed else "U", offsetstpe.dtype.bitwidth, contenttpe.name, idtpe.name, typetpe.name))
        self.offsetstpe = offsetstpe
        self.contenttpe = contenttpe
        self.idtpe = idtpe
        self.typetpe = typetpe

    @property
    def bitwidth(self):
        return self.offsetstpe.dtype.bitwidth

    @property
    def indexname(self):
        return ("" if self.offsetstpe.dtype.signed else "U") + str(self.offsetstpe.dtype.bitwidth)

    @property
    def ndim(self):
        return 1 + self.contenttpe.ndim

    def getitem_int(self):
        return self.contenttpe

    def getitem_range(self):
        return self

    def getitem_str(self, key):
        return ListOffsetArrayType(self.offsetstpe, self.contenttpe.getitem_str(key), self.idtpe, numba.none)   # FIXME: Type::none()

    def getitem_tuple(self, wheretpe):
        import awkward1._numba.array.listarray
        nexttpe = awkward1._numba.array.listarray.ListArrayType(util.index64tpe, util.index64tpe, self, numba.none, numba.none)   # FIXME: Type::none()
        out = nexttpe.getitem_next(wheretpe, False)
        return out.getitem_int()

    def getitem_next(self, wheretpe, isadvanced):
        import awkward1._numba.array.regulararray
        if len(wheretpe.types) == 0:
            return self
        headtpe = wheretpe.types[0]
        tailtpe = numba.types.Tuple(wheretpe.types[1:])

        if isinstance(headtpe, numba.types.Integer):
            return self.contenttpe.carry().getitem_next(tailtpe, isadvanced)

        elif isinstance(headtpe, numba.types.SliceType):
            contenttpe = self.contenttpe.carry().getitem_next(tailtpe, isadvanced)
            return ListOffsetArrayType(util.indextpe(self.indexname), contenttpe, self.idtpe, numba.none)   # FIXME: Type::none()

        elif isinstance(headtpe, numba.types.StringLiteral):
            return self.getitem_str(headtpe.literal_value).getitem_next(tailtpe, isadvanced)

        elif isinstance(headtpe, numba.types.EllipsisType):
            raise NotImplementedError("ellipsis")

        elif isinstance(headtpe, type(numba.typeof(numpy.newaxis))):
            raise NotImplementedError("newaxis")

        elif isinstance(headtpe, numba.types.Array):
            if headtpe.ndim != 1:
                raise NotImplementedError("array.ndim != 1")
            contenttpe = self.contenttpe.carry().getitem_next(tailtpe, True)
            if not isadvanced:
                return awkward1._numba.array.regulararray.RegularArrayType(contenttpe, self.idtpe, numba.none)   # FIXME: Type::none()
            else:
                return contenttpe

        else:
            raise AssertionError(headtpe)

    def carry(self):
        import awkward1._numba.array.listarray
        return awkward1._numba.array.listarray.ListArrayType(self.offsetstpe, self.offsetstpe, self.contenttpe, self.idtpe, numba.none)   # FIXME: Type::none()

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
    if tpe.indexname == "64":
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Index64))
        ListOffsetArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.ListOffsetArray64))
    elif tpe.indexname == "32":
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Index32))
        ListOffsetArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.ListOffsetArray32))
    elif tpe.indexname == "U32":
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.IndexU32))
        ListOffsetArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.ListOffsetArrayU32))
    else:
        raise AssertionError("unrecognized index type: {0}".format(tpe.indexname))
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    offsetsarray_obj = c.pyapi.from_native_value(tpe.offsetstpe, proxyin.offsets, c.env_manager)
    content_obj = c.pyapi.from_native_value(tpe.contenttpe, proxyin.content, c.env_manager)
    offsets_obj = c.pyapi.call_function_objargs(Index_obj, (offsetsarray_obj,))
    c.pyapi.decref(Index_obj)
    c.pyapi.decref(offsetsarray_obj)
    args = [offsets_obj, content_obj]
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.from_native_value(tpe.idtpe, proxyin.id, c.env_manager)
        args.append(id_obj)
    else:
        args.append(c.pyapi.make_none())
    if tpe.typetpe != numba.none:
        args.append(c.pyapi.unserialize(c.pyapi.serialize_object(tpe.typetpe.type)))
    else:
        args.append(c.pyapi.make_none())
    out = c.pyapi.call_function_objargs(ListOffsetArray_obj, args)
    if tpe.idtpe != numba.none:
        c.pyapi.decref(id_obj)
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

    if isinstance(wheretpe, numba.types.Literal):
        wherevalp1_tpe = numba.types.IntegerLiteral(wheretpe.literal_value + 1)
        wherevalp1 = whereval
    else:
        wherevalp1_tpe = wheretpe
        wherevalp1 = builder.add(whereval, context.get_constant(wheretpe, 1))

    start = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.offsetstpe.dtype(tpe.offsetstpe, wheretpe), (proxyin.offsets, whereval))
    stop = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.offsetstpe.dtype(tpe.offsetstpe, wherevalp1_tpe), (proxyin.offsets, wherevalp1))
    proxyslice = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxyslice.start = util.cast(context, builder, tpe.offsetstpe.dtype, numba.intp, start)
    proxyslice.stop = util.cast(context, builder, tpe.offsetstpe.dtype, numba.intp, stop)
    proxyslice.step = context.get_constant(numba.intp, 1)

    outtpe = tpe.contenttpe.getitem_range()
    return tpe.contenttpe.lower_getitem_range(context, builder, outtpe(tpe.contenttpe, numba.types.slice2_type), (proxyin.content, proxyslice._getvalue()))

@numba.extending.lower_builtin(operator.getitem, ListOffsetArrayType, numba.types.slice2_type)
def lower_getitem_range(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    proxyslicein = numba.cgutils.create_struct_proxy(wheretpe)(context, builder, value=whereval)
    numba.targets.slicing.fix_slice(builder, proxyslicein, tpe.lower_len(context, builder, numba.intp(tpe), (val,)))

    proxysliceout = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxysliceout.start = proxyslicein.start
    proxysliceout.stop = builder.add(proxyslicein.stop, context.get_constant(numba.intp, 1))
    proxysliceout.step = context.get_constant(numba.intp, 1)

    proxyout = numba.cgutils.create_struct_proxy(tpe)(context, builder)
    proxyout.offsets = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.offsetstpe(tpe.offsetstpe, numba.types.slice2_type), (proxyin.offsets, proxysliceout._getvalue()))
    proxyout.content = proxyin.content
    if tpe.idtpe != numba.none:
        proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, tpe.idtpe, wheretpe, proxyin.id, whereval)

    out = proxyout._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out

@numba.extending.lower_builtin(operator.getitem, ListOffsetArrayType, numba.types.StringLiteral)
def lower_getitem_str(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    proxyout = numba.cgutils.create_struct_proxy(rettpe)(context, builder)
    proxyout.offsets = proxyin.offsets
    proxyout.content = tpe.contenttpe.lower_getitem_str(context, builder, rettpe.contenttpe(tpe.contenttpe, wheretpe), (proxyin.content, whereval))
    if tpe.idtpe != numba.none:
        proxyout.id = proxyin.id

    out = proxyout._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out

@numba.extending.lower_builtin(operator.getitem, ListOffsetArrayType, numba.types.BaseTuple)
def lower_getitem_tuple(context, builder, sig, args):
    return content.lower_getitem_tuple(context, builder, sig, args)

@numba.extending.lower_builtin(operator.getitem, ListOffsetArrayType, numba.types.Array)
@numba.extending.lower_builtin(operator.getitem, ListOffsetArrayType, numba.types.List)
@numba.extending.lower_builtin(operator.getitem, ListOffsetArrayType, numba.types.ArrayCompatible)
@numba.extending.lower_builtin(operator.getitem, ListOffsetArrayType, numba.types.EllipsisType)
@numba.extending.lower_builtin(operator.getitem, ListOffsetArrayType, type(numba.typeof(numpy.newaxis)))
def lower_getitem_other(context, builder, sig, args):
    return content.lower_getitem_other(context, builder, sig, args)

def starts_stops(context, builder, offsetstpe, offsetsval, lenstarts, lenoffsets):
    proxyslicestarts = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxyslicestarts.start = context.get_constant(numba.intp, 0)
    proxyslicestarts.stop = util.cast(context, builder, lenstarts.type, numba.intp, lenstarts)
    proxyslicestarts.step = context.get_constant(numba.intp, 1)
    slicestarts = proxyslicestarts._getvalue()

    proxyslicestops = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxyslicestops.start = context.get_constant(numba.intp, 1)
    proxyslicestops.stop = util.cast(context, builder, lenoffsets.type, numba.intp, lenoffsets)
    proxyslicestops.step = context.get_constant(numba.intp, 1)
    slicestops = proxyslicestops._getvalue()

    starts = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, offsetstpe(offsetstpe, numba.types.slice2_type), (offsetsval, slicestarts))
    stops = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, offsetstpe(offsetstpe, numba.types.slice2_type), (offsetsval, slicestops))
    return starts, stops

def lower_getitem_next(context, builder, arraytpe, wheretpe, arrayval, whereval, advanced):
    import awkward1._numba.array.listarray

    if len(wheretpe.types) == 0:
        return arrayval

    headtpe = wheretpe.types[0]
    tailtpe = numba.types.Tuple(wheretpe.types[1:])
    headval = numba.cgutils.unpack_tuple(builder, whereval)[0]
    tailval = context.make_tuple(builder, tailtpe, numba.cgutils.unpack_tuple(builder, whereval)[1:])

    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)
    lenoffsets = util.arraylen(context, builder, arraytpe.offsetstpe, proxyin.offsets, totpe=numba.int64)
    lenstarts = builder.sub(lenoffsets, context.get_constant(numba.int64, 1))
    lencontent = util.arraylen(context, builder, arraytpe.contenttpe, proxyin.content, totpe=numba.int64)

    starts, stops = starts_stops(context, builder, tpe.offsetstpe, proxyin.offsets, lenstarts, lenoffsets)

    if isinstance(headtpe, numba.types.Integer):
        assert advanced is None
        if arraytpe.indexname == "64":
            kernel = cpu.kernels.awkward_listarray64_getitem_next_at_64
        elif arraytpe.indexname == "32":
            kernel = cpu.kernels.awkward_listarray32_getitem_next_at_64
        elif arraytpe.indexname == "U32":
            kernel = cpu.kernels.awkward_listarrayU32_getitem_next_at_64
        else:
            raise AssertionError("unrecognized index type: {0}".format(arraytpe.indexname))

        nextcarry = util.newindex64(context, builder, numba.int64, lenstarts)
        util.call(context, builder, kernel,
            (util.arrayptr(context, builder, util.index64tpe, nextcarry),
             util.arrayptr(context, builder, arraytpe.startstpe, starts),
             util.arrayptr(context, builder, arraytpe.stopstpe, stops),
             lenstarts,
             context.get_constant(numba.int64, 0),
             context.get_constant(numba.int64, 0),
             util.cast(context, builder, headtpe, numba.int64, headval)),
            "in {0}, indexing error".format(arraytpe.shortname))
        nextcontenttpe = arraytpe.contenttpe.carry()
        nextcontentval = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, util.index64tpe, proxyin.content, nextcarry)
        return nextcontenttpe.lower_getitem_next(context, builder, nextcontenttpe, tailtpe, nextcontentval, tailval, advanced)

    elif isinstance(headtpe, numba.types.SliceType):
        proxyslicein = numba.cgutils.create_struct_proxy(headtpe)(context, builder, value=headval)

        if arraytpe.indexname == "64":
            determine_carrylength = cpu.kernels.awkward_listarray64_getitem_next_range_carrylength
            fill_carry = cpu.kernels.awkward_listarray64_getitem_next_range_64
            determine_total = cpu.kernels.awkward_listarray64_getitem_next_range_counts_64
            fill_nextadvanced = cpu.kernels.awkward_listarray64_getitem_next_range_spreadadvanced_64
        elif arraytpe.indexname == "32":
            determine_carrylength = cpu.kernels.awkward_listarray32_getitem_next_range_carrylength
            fill_carry = cpu.kernels.awkward_listarray32_getitem_next_range_64
            determine_total = cpu.kernels.awkward_listarray32_getitem_next_range_counts_64
            fill_nextadvanced = cpu.kernels.awkward_listarray32_getitem_next_range_spreadadvanced_64
        elif arraytpe.indexname == "U32":
            determine_carrylength = cpu.kernels.awkward_listarrayU32_getitem_next_range_carrylength
            fill_carry = cpu.kernels.awkward_listarrayU32_getitem_next_range_64
            determine_total = cpu.kernels.awkward_listarrayU32_getitem_next_range_counts_64
            fill_nextadvanced = cpu.kernels.awkward_listarrayU32_getitem_next_range_spreadadvanced_64
        else:
            raise AssertionError("unrecognized index type: {0}".format(arraytpe.indexname))

        carrylength = numba.cgutils.alloca_once(builder, context.get_value_type(numba.int64))
        util.call(context, builder, determine_carrylength,
            (carrylength,
             util.arrayptr(context, builder, arraytpe.offsetstpe, starts),
             util.arrayptr(context, builder, arraytpe.offsetstpe, stops),
             lenstarts,
             context.get_constant(numba.int64, 0),
             context.get_constant(numba.int64, 0),
             util.cast(context, builder, numba.intp, numba.int64, proxyslicein.start),
             util.cast(context, builder, numba.intp, numba.int64, proxyslicein.stop),
             util.cast(context, builder, numba.intp, numba.int64, proxyslicein.step)),
            "in {0}, indexing error".format(arraytpe.shortname))

        nextoffsets = util.newindex(arraytpe.indexname, context, builder, numba.int64, builder.add(lenstarts, context.get_constant(numba.int64, 1)))
        nextcarry = util.newindex64(context, builder, numba.int64, builder.load(carrylength))
        util.call(context, builder, fill_carry,
            (util.arrayptr(context, builder, util.indextpe(arraytpe.indexname), nextoffsets),
             util.arrayptr(context, builder, util.index64tpe, nextcarry),
             util.arrayptr(context, builder, arraytpe.offsetstpe, starts),
             util.arrayptr(context, builder, arraytpe.offsetstpe, stops),
             lenstarts,
             context.get_constant(numba.int64, 0),
             context.get_constant(numba.int64, 0),
             util.cast(context, builder, numba.intp, numba.int64, proxyslicein.start),
             util.cast(context, builder, numba.intp, numba.int64, proxyslicein.stop),
             util.cast(context, builder, numba.intp, numba.int64, proxyslicein.step)),
            "in {0}, indexing error".format(arraytpe.shortname))

        nextcontenttpe = arraytpe.contenttpe.carry()
        nextcontentval = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, util.index64tpe, proxyin.content, nextcarry)

        if advanced is None:
            outcontenttpe = nextcontenttpe.getitem_next(tailtpe, False)
            outcontentval = nextcontenttpe.lower_getitem_next(context, builder, nextcontenttpe, tailtpe, nextcontentval, tailval, advanced)

        else:
            total = numba.cgutils.alloca_once(builder, context.get_value_type(numba.int64))
            util.call(context, builder, determine_total,
                (total,
                 util.arrayptr(context, builder, util.indextpe(arraytpe.indexname), nextoffsets),
                 lenstarts),
                "in {0}, indexing error".format(arraytpe.shortname))

            nextadvanced = util.newindex64(context, builder, numba.int64, builder.load(total))
            util.call(context, builder, fill_nextadvanced,
                (util.arrayptr(context, builder, util.index64tpe, nextadvanced),
                 util.arrayptr(context, builder, util.index64tpe, advanced),
                 util.arrayptr(context, builder, util.indextpe(arraytpe.indexname), nextoffsets),
                 lenstarts),
                "in {0}, indexing error".format(arraytpe.shortname))

            outcontenttpe = nextcontenttpe.getitem_next(tailtpe, True)
            outcontentval = nextcontenttpe.lower_getitem_next(context, builder, nextcontenttpe, tailtpe, nextcontentval, tailval, nextadvanced)

        outtpe = awkward1._numba.array.listoffsetarray.ListOffsetArrayType(util.indextpe(arraytpe.indexname), outcontenttpe, arraytpe.idtpe, numba.none)   # FIXME: Type::none()
        proxyout = numba.cgutils.create_struct_proxy(outtpe)(context, builder)
        proxyout.offsets = nextoffsets
        proxyout.content = outcontentval
        if arraytpe.idtpe != numba.none:
            proxyout.id = proxyin.id
        return proxyout._getvalue()

    elif isinstance(headtpe, numba.types.StringLiteral):
        nexttpe = arraytpe.getitem_str(headtpe.literal_value)
        nextval = lower_getitem_str(context, builder, nexttpe(arraytpe, headtpe), (arrayval, headval))
        return lower_getitem_next(context, builder, nexttpe, tailtpe, nextval, tailval, advanced)

    elif isinstance(headtpe, numba.types.EllipsisType):
        raise NotImplementedError("ListOffsetArray.getitem_next(ellipsis)")

    elif isinstance(headtpe, type(numba.typeof(numpy.newaxis))):
        raise NotImplementedError("ListOffsetArray.getitem_next(newaxis)")

    elif isinstance(headtpe, numba.types.Array):
        if headtpe.ndim != 1:
            raise NotImplementedError("array.ndim != 1")

        flathead = numba.targets.arrayobj.array_ravel(context, builder, util.int64tep(headtpe), (headval,))
        lenflathead = util.arraylen(context, builder, util.int64tpe, flathead, totpe=numba.int64)

        if advanced is None:
            if arraytpe.indexname == "64":
                kernel = cpu.kernels.awkward_listarray64_getitem_next_array_64
            elif arraytpe.indexname == "32":
                kernel = cpu.kernels.awkward_listarray32_getitem_next_array_64
            elif arraytpe.indexname == "U32":
                kernel = cpu.kernels.awkward_listarrayU32_getitem_next_array_64
            else:
                raise AssertionError("unrecognized index type: {0}".format(arraytpe.indexname))

            lencarry = builder.mul(lenstarts, lenflathead)

            nextcarry = util.newindex64(context, builder, numba.int64, lencarry)
            nextadvanced = util.newindex64(context, builder, numba.int64, lencarry)
            util.call(context, builder, kernel,
                (util.arrayptr(context, builder, util.index64tpe, nextcarry),
                 util.arrayptr(context, builder, util.index64tpe, nextadvanced),
                 util.arrayptr(context, builder, arraytpe.offsetstpe, starts),
                 util.arrayptr(context, builder, arraytpe.offsetstpe, stops),
                 util.arrayptr(context, builder, util.index64tpe, flathead),
                 context.get_constant(numba.int64, 0),
                 context.get_constant(numba.int64, 0),
                 lenstarts,
                 lenflathead,
                 lencontent),
                "in {0}, indexing error".format(arraytpe.shortname))

            nexttpe = arraytpe.contenttpe.carry()
            nextval = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, util.index64tpe, proxyin.content, nextcarry)

            contenttpe = nexttpe.getitem_next(tailtpe, True)
            contentval = nexttpe.lower_getitem_next(context, builder, nexttpe, tailtpe, nextval, tailval, nextadvanced)

            outtpe = awkward1._numba.array.regulararray.RegularArrayType(contenttpe, arraytpe.idtpe, numba.none)   # FIXME: Type::none()
            proxyout = numba.cgutils.create_struct_proxy(outtpe)(context, builder)
            proxyout.content = contentval
            proxyout.size = lenflathead
            if outtpe.idtpe != numba.none:
                proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, outtpe.idtpe, util.index64tpe, proxyin.id, flathead)
            return proxyout._getvalue()

        else:
            if arraytpe.indexname == "64":
                kernel = cpu.kernels.awkward_listarray64_getitem_next_array_advanced_64
            elif arraytpe.indexname == "32":
                kernel = cpu.kernels.awkward_listarray32_getitem_next_array_advanced_64
            elif arraytpe.indexname == "U32":
                kernel = cpu.kernels.awkward_listarrayU32_getitem_next_array_advanced_64
            else:
                raise AssertionError("unrecognized index type: {0}".format(arraytpe.indexname))

            nextcarry = util.newindex64(context, builder, numba.int64, lenstarts)
            nextadvanced = util.newindex64(context, builder, numba.int64, lenstarts)
            util.call(context, builder, kernel,
                (util.arrayptr(context, builder, util.index64tpe, nextcarry),
                 util.arrayptr(context, builder, util.index64tpe, nextadvanced),
                 util.arrayptr(context, builder, arraytpe.offsetstpe, starts),
                 util.arrayptr(context, builder, arraytpe.offsetstpe, stops),
                 util.arrayptr(context, builder, util.index64tpe, flathead),
                 util.arrayptr(context, builder, util.index64tpe, advanced),
                 context.get_constant(numba.int64, 0),
                 context.get_constant(numba.int64, 0),
                 lenstarts,
                 lenflathead,
                 lencontent),
                "in {0}, indexing error".format(arraytpe.shortname))

            nexttpe = arraytpe.contenttpe.carry()
            nextval = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, util.index64tpe, proxyin.content, nextcarry)

            outtpe = nexttpe.getitem_next(tailtpe, True)
            return nexttpe.lower_getitem_next(context, builder, nexttpe, tailtpe, nextval, tailval, nextadvanced)

    else:
        raise AssertionError(headtpe)

def lower_carry(context, builder, arraytpe, carrytpe, arrayval, carryval):
    import awkward1._numba.array.listarray

    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)
    lenoffsets = util.arraylen(context, builder, arraytpe.offsetstpe, proxyin.offsets, totpe=numba.int64)
    lenstarts = builder.sub(lenoffsets, context.get_constant(numba.int64, 1))

    starts, stops = starts_stops(context, builder, arraytpe.offsetstpe, proxyin.offsets, lenstarts, lenoffsets)

    proxyout = numba.cgutils.create_struct_proxy(awkward1._numba.array.listarray.ListArrayType(arraytpe.offsetstpe, arraytpe.offsetstpe, arraytpe.contenttpe, arraytpe.idtpe, numba.none))(context, builder)   # FIXME: Type::none()
    proxyout.starts = numba.targets.arrayobj.fancy_getitem_array(context, builder, arraytpe.offsetstpe(arraytpe.offsetstpe, carrytpe), (starts, carryval))
    proxyout.stops = numba.targets.arrayobj.fancy_getitem_array(context, builder, arraytpe.offsetstpe(arraytpe.offsetstpe, carrytpe), (stops, carryval))

    proxyout.content = proxyin.content
    if arraytpe.idtpe != numba.none:
        proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, arraytpe.idtpe, carrytpe, proxyin.id, carryval)
    return proxyout._getvalue()

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
