# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json

import numba

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


@numba.extending.typeof_impl.register(ak.contents.Content)
@numba.extending.typeof_impl.register(ak.index.Index)
@numba.extending.typeof_impl.register(ak.record.Record)
def fake_typeof(obj, c):
    raise TypeError(
        "{} objects cannot be passed directly into Numba-compiled functions; "
        "construct a high-level ak.Array or ak.Record instead".format(
            type(obj).__name__
        )
    )


class ContentType(numba.types.Type):
    @classmethod
    def from_form_index(cls, index_string):
        if index_string == "i8":
            return numba.types.Array(numba.int8, 1, "C")
        elif index_string == "u8":
            return numba.types.Array(numba.uint8, 1, "C")
        elif index_string == "i32":
            return numba.types.Array(numba.int32, 1, "C")
        elif index_string == "u32":
            return numba.types.Array(numba.uint32, 1, "C")
        elif index_string == "i64":
            return numba.types.Array(numba.int64, 1, "C")
        else:
            raise AssertionError(f"unrecognized Form index type: {index_string!r}")

    def IndexOf(self, arraytype):
        if arraytype.dtype.bitwidth == 8 and arraytype.dtype.signed:
            return ak.index.Index8
        elif arraytype.dtype.bitwidth == 8:
            return ak.index.IndexU8
        elif arraytype.dtype.bitwidth == 32 and arraytype.dtype.signed:
            return ak.index.Index32
        elif arraytype.dtype.bitwidth == 32:
            return ak.index.IndexU32
        elif arraytype.dtype.bitwidth == 64:
            return ak.index.Index64
        else:
            raise ak._errors.wrap_error(
                AssertionError(f"no Index* type for array: {arraytype}")
            )

    def getitem_at_check(self, viewtype):
        typer = ak._util.numba_array_typer(viewtype.type, viewtype.behavior)
        if typer is None:
            return self.getitem_at(viewtype)
        else:
            return typer(viewtype)

    def getitem_range(self, viewtype):
        return ak._connect.numba.arrayview.wrap(self, viewtype, None)

    def getitem_field(self, viewtype, key):
        if self.has_field(key):
            return ak._connect.numba.arrayview.wrap(
                self, viewtype, viewtype.fields + (key,)
            )
        else:
            raise TypeError(f"array does not have a field with key {repr(key)}")

    def lower_getitem_at_check(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        lower = ak._util.numba_array_lower(viewtype.type, viewtype.behavior)
        if lower is not None:
            atval = regularize_atval(
                context, builder, viewproxy, attype, atval, wrapneg, checkbounds
            )
            return lower(
                context, builder, rettype, viewtype, viewval, viewproxy, attype, atval
            )
        else:
            return self.lower_getitem_at(
                context,
                builder,
                rettype,
                viewtype,
                viewval,
                viewproxy,
                attype,
                atval,
                wrapneg,
                checkbounds,
            )

    def lower_getitem_range(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        start,
        stop,
        wrapneg,
    ):
        length = builder.sub(viewproxy.stop, viewproxy.start)

        regular_start = numba.core.cgutils.alloca_once_value(builder, start)
        regular_stop = numba.core.cgutils.alloca_once_value(builder, stop)

        if wrapneg:
            with builder.if_then(
                builder.icmp_signed("<", start, context.get_constant(numba.intp, 0))
            ):
                builder.store(builder.add(start, length), regular_start)
            with builder.if_then(
                builder.icmp_signed("<", stop, context.get_constant(numba.intp, 0))
            ):
                builder.store(builder.add(stop, length), regular_stop)

        with builder.if_then(
            builder.icmp_signed(
                "<", builder.load(regular_start), context.get_constant(numba.intp, 0)
            )
        ):
            builder.store(context.get_constant(numba.intp, 0), regular_start)
        with builder.if_then(
            builder.icmp_signed(">", builder.load(regular_start), length)
        ):
            builder.store(length, regular_start)

        with builder.if_then(
            builder.icmp_signed(
                "<", builder.load(regular_stop), builder.load(regular_start)
            )
        ):
            builder.store(builder.load(regular_start), regular_stop)
        with builder.if_then(
            builder.icmp_signed(">", builder.load(regular_stop), length)
        ):
            builder.store(length, regular_stop)

        proxyout = context.make_helper(builder, rettype)
        proxyout.pos = viewproxy.pos
        proxyout.start = builder.add(viewproxy.start, builder.load(regular_start))
        proxyout.stop = builder.add(viewproxy.start, builder.load(regular_stop))
        proxyout.arrayptrs = viewproxy.arrayptrs
        proxyout.pylookup = viewproxy.pylookup
        return proxyout._getvalue()

    def lower_getitem_field(self, context, builder, viewtype, viewval, key):
        return viewval


def castint(context, builder, fromtype, totype, val):
    import llvmlite.ir.types

    if isinstance(fromtype, llvmlite.ir.types.IntType):
        if fromtype.width == 8:
            fromtype = numba.int8
        elif fromtype.width == 16:
            fromtype = numba.int16
        elif fromtype.width == 32:
            fromtype = numba.int32
        elif fromtype.width == 64:
            fromtype = numba.int64
    if not isinstance(fromtype, numba.types.Integer):
        raise AssertionError(f"unrecognized integer type: {repr(fromtype)}")

    if fromtype.bitwidth < totype.bitwidth:
        if fromtype.signed:
            return builder.sext(val, context.get_value_type(totype))
        else:
            return builder.zext(val, context.get_value_type(totype))
    elif fromtype.bitwidth > totype.bitwidth:
        return builder.trunc(val, context.get_value_type(totype))
    else:
        return val


def type_bitwidth(numbatype):
    if isinstance(numbatype, numba.types.Boolean):
        return 8
    elif isinstance(numbatype, (numba.types.NPDatetime, numba.types.NPTimedelta)):
        return 64
    else:
        return numbatype.bitwidth


def posat(context, builder, pos, offset):
    return builder.add(pos, context.get_constant(numba.intp, offset))


def getat(context, builder, baseptr, offset, rettype=None):
    ptrtype = None
    if rettype is not None:
        ptrtype = context.get_value_type(numba.types.CPointer(rettype))
        bitwidth = type_bitwidth(rettype)
    else:
        bitwidth = numba.intp.bitwidth
    byteoffset = builder.mul(offset, context.get_constant(numba.intp, bitwidth // 8))
    out = builder.load(
        numba.core.cgutils.pointer_add(builder, baseptr, byteoffset, ptrtype)
    )
    if rettype is not None and isinstance(rettype, numba.types.Boolean):
        return builder.icmp_signed(
            "!=",
            out,
            context.get_constant(numba.int8, 0),
        )
    else:
        return out


def regularize_atval(context, builder, viewproxy, attype, atval, wrapneg, checkbounds):
    atval = castint(context, builder, attype, numba.intp, atval)

    if not attype.signed:
        wrapneg = False

    if wrapneg or checkbounds:
        length = builder.sub(viewproxy.stop, viewproxy.start)

        if wrapneg:
            regular_atval = numba.core.cgutils.alloca_once_value(builder, atval)
            with builder.if_then(
                builder.icmp_signed("<", atval, context.get_constant(numba.intp, 0))
            ):
                builder.store(builder.add(atval, length), regular_atval)
            atval = builder.load(regular_atval)

        if checkbounds:
            with builder.if_then(
                builder.or_(
                    builder.icmp_signed(
                        "<", atval, context.get_constant(numba.intp, 0)
                    ),
                    builder.icmp_signed(">=", atval, length),
                )
            ):
                context.call_conv.return_user_exc(
                    builder, ValueError, ("slice index out of bounds",)
                )

    return castint(context, builder, atval.type, numba.intp, atval)


class NumpyArrayType(ContentType, ak._lookup.NumpyLookup):
    @classmethod
    def from_form(cls, form):
        t = numba.from_dtype(ak.types.numpytype.primitive_to_dtype(form.primitive))
        arraytype = numba.types.Array(t, 1, "C")
        return NumpyArrayType(arraytype, form.parameters)

    def __init__(self, arraytype, parameters):
        super().__init__(
            name=f"ak.NumpyArrayType({arraytype.name}, {json.dumps(parameters)})"
        )
        self.arraytype = arraytype
        self.parameters = parameters

    def has_field(self, key):
        return False

    def getitem_at(self, viewtype):
        return self.arraytype.dtype

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.ARRAY)
        arrayptr = getat(context, builder, viewproxy.arrayptrs, whichpos)
        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )
        arraypos = builder.add(viewproxy.start, atval)
        return getat(context, builder, arrayptr, arraypos, rettype=rettype)

    @property
    def ndim(self):
        return self.arraytype.ndim

    @property
    def inner_dtype(self):
        return self.arraytype.dtype

    @property
    def is_optiontype(self):
        return False

    @property
    def is_recordtype(self):
        return False


class RegularArrayType(ContentType, ak._lookup.RegularLookup):
    @classmethod
    def from_form(cls, form):
        return RegularArrayType(
            ak._connect.numba.arrayview.tonumbatype(form.content),
            form.size,
            form.parameters,
        )

    def __init__(self, contenttype, size, parameters):
        super().__init__(
            name=f"ak.RegularArrayType({contenttype.name}, {size}, {json.dumps(parameters)})"
        )
        self.contenttype = contenttype
        self.size = size
        self.parameters = parameters

    def has_field(self, key):
        return self.contenttype.has_field(key)

    def getitem_at(self, viewtype):
        return ak._connect.numba.arrayview.wrap(self.contenttype, viewtype, None)

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        # FIXME: what about ZEROS_LENGTH? Is that handled correctly?

        size = context.get_constant(numba.intp, self.size)
        start = builder.mul(builder.add(viewproxy.start, atval), size)
        stop = builder.add(start, size)

        proxyout = context.make_helper(builder, rettype)
        proxyout.pos = nextpos
        proxyout.start = start
        proxyout.stop = stop
        proxyout.arrayptrs = viewproxy.arrayptrs
        proxyout.pylookup = viewproxy.pylookup
        return proxyout._getvalue()

    @property
    def ndim(self):
        return 1 + self.contenttype.ndim

    @property
    def inner_dtype(self):
        return self.contenttype.inner_dtype

    @property
    def is_optiontype(self):
        return False

    @property
    def is_recordtype(self):
        return False


class ListArrayType(ContentType, ak._lookup.ListLookup):
    @classmethod
    def from_form(cls, form):
        if isinstance(form, ak.forms.ListForm):
            index_string = form.starts
        else:
            index_string = form.offsets

        return ListArrayType(
            cls.from_form_index(index_string),
            ak._connect.numba.arrayview.tonumbatype(form.content),
            form.parameters,
        )

    def __init__(self, indextype, contenttype, parameters):
        super().__init__(
            name=f"ak.ListArrayType({indextype.name}, {contenttype.name}, {json.dumps(parameters)})"
        )
        self.indextype = indextype
        self.contenttype = contenttype
        self.parameters = parameters

    def has_field(self, key):
        return self.contenttype.has_field(key)

    def getitem_at(self, viewtype):
        return ak._connect.numba.arrayview.wrap(self.contenttype, viewtype, None)

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        startspos = posat(context, builder, viewproxy.pos, self.STARTS)
        startsptr = getat(context, builder, viewproxy.arrayptrs, startspos)
        startsarraypos = builder.add(viewproxy.start, atval)
        start = getat(
            context, builder, startsptr, startsarraypos, rettype=self.indextype.dtype
        )

        stopspos = posat(context, builder, viewproxy.pos, self.STOPS)
        stopsptr = getat(context, builder, viewproxy.arrayptrs, stopspos)
        stopsarraypos = builder.add(viewproxy.start, atval)
        stop = getat(
            context, builder, stopsptr, stopsarraypos, rettype=self.indextype.dtype
        )

        proxyout = context.make_helper(builder, rettype)
        proxyout.pos = nextpos
        proxyout.start = castint(
            context, builder, self.indextype.dtype, numba.intp, start
        )
        proxyout.stop = castint(
            context, builder, self.indextype.dtype, numba.intp, stop
        )
        proxyout.arrayptrs = viewproxy.arrayptrs
        proxyout.pylookup = viewproxy.pylookup
        return proxyout._getvalue()

    @property
    def ndim(self):
        return 1 + self.contenttype.ndim

    @property
    def inner_dtype(self):
        return self.contenttype.inner_dtype

    @property
    def is_optiontype(self):
        return False

    @property
    def is_recordtype(self):
        return False


class IndexedArrayType(ContentType, ak._lookup.IndexedLookup):
    @classmethod
    def from_form(cls, form):
        return IndexedArrayType(
            cls.from_form_index(form.index),
            ak._connect.numba.arrayview.tonumbatype(form.content),
            form.parameters,
        )

    def __init__(self, indextype, contenttype, parameters):
        super().__init__(
            name=f"ak.IndexedArrayType({indextype.name}, {contenttype.name}, {json.dumps(parameters)})"
        )
        self.indextype = indextype
        self.contenttype = contenttype
        self.parameters = parameters

    def has_field(self, key):
        return self.contenttype.has_field(key)

    def getitem_at(self, viewtype):
        viewtype = ak._connect.numba.arrayview.wrap(self.contenttype, viewtype, None)
        return self.contenttype.getitem_at_check(viewtype)

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        indexpos = posat(context, builder, viewproxy.pos, self.INDEX)
        indexptr = getat(context, builder, viewproxy.arrayptrs, indexpos)
        indexarraypos = builder.add(viewproxy.start, atval)
        nextat = getat(
            context, builder, indexptr, indexarraypos, rettype=self.indextype.dtype
        )

        nextviewtype = ak._connect.numba.arrayview.wrap(
            self.contenttype, viewtype, None
        )
        proxynext = context.make_helper(builder, nextviewtype)
        proxynext.pos = nextpos
        proxynext.start = context.get_constant(numba.intp, 0)
        proxynext.stop = builder.add(
            castint(context, builder, self.indextype.dtype, numba.intp, nextat),
            context.get_constant(numba.intp, 1),
        )
        proxynext.arrayptrs = viewproxy.arrayptrs
        proxynext.pylookup = viewproxy.pylookup

        return self.contenttype.lower_getitem_at_check(
            context,
            builder,
            rettype,
            nextviewtype,
            proxynext._getvalue(),
            proxynext,
            numba.intp,
            nextat,
            False,
            False,
        )

    @property
    def ndim(self):
        return self.contenttype.ndim

    @property
    def inner_dtype(self):
        return self.contenttype.inner_dtype

    @property
    def is_optiontype(self):
        return False

    @property
    def is_recordtype(self):
        return self.contenttype.is_recordtype


class IndexedOptionArrayType(ContentType, ak._lookup.IndexedOptionLookup):
    @classmethod
    def from_form(cls, form):
        return IndexedOptionArrayType(
            cls.from_form_index(form.index),
            ak._connect.numba.arrayview.tonumbatype(form.content),
            form.parameters,
        )

    def __init__(self, indextype, contenttype, parameters):
        super().__init__(
            name=f"ak.IndexedOptionArrayType({indextype.name}, {contenttype.name}, {json.dumps(parameters)})"
        )
        self.indextype = indextype
        self.contenttype = contenttype
        self.parameters = parameters

    def has_field(self, key):
        return self.contenttype.has_field(key)

    def getitem_at(self, viewtype):
        viewtype = ak._connect.numba.arrayview.wrap(self.contenttype, viewtype, None)
        return numba.types.optional(self.contenttype.getitem_at_check(viewtype))

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        indexpos = posat(context, builder, viewproxy.pos, self.INDEX)
        indexptr = getat(context, builder, viewproxy.arrayptrs, indexpos)
        indexarraypos = builder.add(viewproxy.start, atval)
        nextat = getat(
            context, builder, indexptr, indexarraypos, rettype=self.indextype.dtype
        )

        output = context.make_helper(builder, rettype)

        with builder.if_else(
            builder.icmp_signed(
                "<", nextat, context.get_constant(self.indextype.dtype, 0)
            )
        ) as (isnone, isvalid):
            with isnone:
                output.valid = numba.core.cgutils.false_bit
                output.data = numba.core.cgutils.get_null_value(output.data.type)

            with isvalid:
                nextviewtype = ak._connect.numba.arrayview.wrap(
                    self.contenttype, viewtype, None
                )
                proxynext = context.make_helper(builder, nextviewtype)
                proxynext.pos = nextpos
                proxynext.start = context.get_constant(numba.intp, 0)
                proxynext.stop = builder.add(
                    castint(context, builder, self.indextype.dtype, numba.intp, nextat),
                    context.get_constant(numba.intp, 1),
                )
                proxynext.arrayptrs = viewproxy.arrayptrs
                proxynext.pylookup = viewproxy.pylookup

                outdata = self.contenttype.lower_getitem_at_check(
                    context,
                    builder,
                    rettype.type,
                    nextviewtype,
                    proxynext._getvalue(),
                    proxynext,
                    numba.intp,
                    nextat,
                    False,
                    False,
                )

                output.valid = numba.core.cgutils.true_bit
                output.data = outdata

        return output._getvalue()

    @property
    def ndim(self):
        return self.contenttype.ndim

    @property
    def inner_dtype(self):
        return None

    @property
    def is_optiontype(self):
        return True

    @property
    def is_recordtype(self):
        return self.contenttype.is_recordtype


class ByteMaskedArrayType(ContentType, ak._lookup.ByteMaskedLookup):
    @classmethod
    def from_form(cls, form):
        return ByteMaskedArrayType(
            cls.from_form_index(form.mask),
            ak._connect.numba.arrayview.tonumbatype(form.content),
            form.valid_when,
            form.parameters,
        )

    def __init__(self, masktype, contenttype, valid_when, parameters):
        super().__init__(
            name=f"ak.ByteMaskedArrayType({masktype.name}, {contenttype.name}, {valid_when}, {json.dumps(parameters)})"
        )
        self.masktype = masktype
        self.contenttype = contenttype
        self.valid_when = valid_when
        self.parameters = parameters

    def has_field(self, key):
        return self.contenttype.has_field(key)

    def getitem_at(self, viewtype):
        viewtype = ak._connect.numba.arrayview.wrap(self.contenttype, viewtype, None)
        return numba.types.optional(self.contenttype.getitem_at_check(viewtype))

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        maskpos = posat(context, builder, viewproxy.pos, self.MASK)
        maskptr = getat(context, builder, viewproxy.arrayptrs, maskpos)
        maskarraypos = builder.add(viewproxy.start, atval)
        byte = getat(
            context, builder, maskptr, maskarraypos, rettype=self.masktype.dtype
        )

        output = context.make_helper(builder, rettype)

        with builder.if_else(
            builder.icmp_signed(
                "==",
                builder.icmp_signed("!=", byte, context.get_constant(numba.int8, 0)),
                context.get_constant(numba.int8, int(self.valid_when)),
            )
        ) as (isvalid, isnone):
            with isvalid:
                nextviewtype = ak._connect.numba.arrayview.wrap(
                    self.contenttype, viewtype, None
                )
                proxynext = context.make_helper(builder, nextviewtype)
                proxynext.pos = nextpos
                proxynext.start = viewproxy.start
                proxynext.stop = viewproxy.stop
                proxynext.arrayptrs = viewproxy.arrayptrs
                proxynext.pylookup = viewproxy.pylookup

                outdata = self.contenttype.lower_getitem_at_check(
                    context,
                    builder,
                    rettype.type,
                    nextviewtype,
                    proxynext._getvalue(),
                    proxynext,
                    numba.intp,
                    atval,
                    False,
                    False,
                )

                output.valid = numba.core.cgutils.true_bit
                output.data = outdata

            with isnone:
                output.valid = numba.core.cgutils.false_bit
                output.data = numba.core.cgutils.get_null_value(output.data.type)

        return output._getvalue()

    @property
    def ndim(self):
        return self.contenttype.ndim

    @property
    def inner_dtype(self):
        return None

    @property
    def is_optiontype(self):
        return True

    @property
    def is_recordtype(self):
        return self.contenttype.is_recordtype


class BitMaskedArrayType(ContentType, ak._lookup.BitMaskedLookup):
    @classmethod
    def from_form(cls, form):
        return BitMaskedArrayType(
            cls.from_form_index(form.mask),
            ak._connect.numba.arrayview.tonumbatype(form.content),
            form.valid_when,
            form.lsb_order,
            form.parameters,
        )

    def __init__(self, masktype, contenttype, valid_when, lsb_order, parameters):
        super().__init__(
            name=f"ak.BitMaskedArrayType({masktype.name}, {contenttype.name}, {valid_when}, {lsb_order}, {json.dumps(parameters)})"
        )
        self.masktype = masktype
        self.contenttype = contenttype
        self.valid_when = valid_when
        self.lsb_order = lsb_order
        self.parameters = parameters

    def has_field(self, key):
        return self.contenttype.has_field(key)

    def getitem_at(self, viewtype):
        viewtype = ak._connect.numba.arrayview.wrap(self.contenttype, viewtype, None)
        return numba.types.optional(self.contenttype.getitem_at_check(viewtype))

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )
        startatval = builder.add(viewproxy.start, atval)
        bitatval = builder.sdiv(startatval, context.get_constant(numba.intp, 8))
        shiftval = castint(
            context,
            builder,
            numba.intp,
            numba.uint8,
            builder.srem(startatval, context.get_constant(numba.intp, 8)),
        )

        maskpos = posat(context, builder, viewproxy.pos, self.MASK)
        maskptr = getat(context, builder, viewproxy.arrayptrs, maskpos)
        maskarraypos = bitatval
        byte = getat(
            context, builder, maskptr, maskarraypos, rettype=self.masktype.dtype
        )
        if self.lsb_order:
            # ((byte >> ((uint8_t)shift)) & ((uint8_t)1))
            asbool = builder.and_(
                builder.lshr(byte, shiftval), context.get_constant(numba.uint8, 1)
            )
        else:
            # ((byte << ((uint8_t)shift)) & ((uint8_t)128))
            asbool = builder.and_(
                builder.shl(byte, shiftval), context.get_constant(numba.uint8, 128)
            )

        output = context.make_helper(builder, rettype)

        with builder.if_else(
            builder.icmp_signed(
                "==",
                builder.icmp_signed("!=", asbool, context.get_constant(numba.uint8, 0)),
                context.get_constant(numba.uint8, int(self.valid_when)),
            )
        ) as (isvalid, isnone):
            with isvalid:
                nextviewtype = ak._connect.numba.arrayview.wrap(
                    self.contenttype, viewtype, None
                )
                proxynext = context.make_helper(builder, nextviewtype)
                proxynext.pos = nextpos
                proxynext.start = viewproxy.start
                proxynext.stop = viewproxy.stop
                proxynext.arrayptrs = viewproxy.arrayptrs
                proxynext.pylookup = viewproxy.pylookup

                outdata = self.contenttype.lower_getitem_at_check(
                    context,
                    builder,
                    rettype.type,
                    nextviewtype,
                    proxynext._getvalue(),
                    proxynext,
                    numba.intp,
                    atval,
                    False,
                    False,
                )

                output.valid = numba.core.cgutils.true_bit
                output.data = outdata

            with isnone:
                output.valid = numba.core.cgutils.false_bit
                output.data = numba.core.cgutils.get_null_value(output.data.type)

        return output._getvalue()

    @property
    def ndim(self):
        return self.contenttype.ndim

    @property
    def inner_dtype(self):
        return None

    @property
    def is_optiontype(self):
        return True

    @property
    def is_recordtype(self):
        return self.contenttype.is_recordtype


class UnmaskedArrayType(ContentType, ak._lookup.UnmaskedLookup):
    @classmethod
    def from_form(cls, form):
        return UnmaskedArrayType(
            ak._connect.numba.arrayview.tonumbatype(form.content),
            form.parameters,
        )

    def __init__(self, contenttype, parameters):
        super().__init__(
            name=f"ak.UnmaskedArrayType({contenttype.name}, {json.dumps(parameters)})"
        )
        self.contenttype = contenttype
        self.parameters = parameters

    def has_field(self, key):
        return self.contenttype.has_field(key)

    def getitem_at(self, viewtype):
        viewtype = ak._connect.numba.arrayview.wrap(self.contenttype, viewtype, None)
        return numba.types.optional(self.contenttype.getitem_at_check(viewtype))

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        output = context.make_helper(builder, rettype)

        nextviewtype = ak._connect.numba.arrayview.wrap(
            self.contenttype, viewtype, None
        )
        proxynext = context.make_helper(builder, nextviewtype)
        proxynext.pos = nextpos
        proxynext.start = viewproxy.start
        proxynext.stop = viewproxy.stop
        proxynext.arrayptrs = viewproxy.arrayptrs
        proxynext.pylookup = viewproxy.pylookup

        outdata = self.contenttype.lower_getitem_at_check(
            context,
            builder,
            rettype.type,
            nextviewtype,
            proxynext._getvalue(),
            proxynext,
            numba.intp,
            atval,
            False,
            False,
        )

        output.valid = numba.core.cgutils.true_bit
        output.data = outdata

        return output._getvalue()

    @property
    def ndim(self):
        return self.contenttype.ndim

    @property
    def inner_dtype(self):
        return None

    @property
    def is_optiontype(self):
        return True

    @property
    def is_recordtype(self):
        return self.contenttype.is_recordtype


class RecordArrayType(ContentType, ak._lookup.RecordLookup):
    @classmethod
    def from_form(cls, form):
        return RecordArrayType(
            [ak._connect.numba.arrayview.tonumbatype(x) for x in form.contents],
            None if form.is_tuple else form.fields,
            form.parameters,
        )

    def __init__(self, contenttypes, fields, parameters):
        tmp1 = ", ".join(x.name for x in contenttypes)
        tmp2 = "," if len(contenttypes) == 1 else ""
        tmp3 = "None" if fields is None else repr(tuple(fields))
        super().__init__(
            name=f"ak.RecordArrayType(({tmp1}{tmp2}), {tmp3}, {json.dumps(parameters)})"
        )
        self.contenttypes = contenttypes
        self.fields = fields
        self.parameters = parameters

    def fieldindex(self, key):
        out = -1
        if self.fields is not None:
            for i, x in enumerate(self.fields):
                if x == key:
                    out = i
                    break
        if out == -1:
            try:
                out = int(key)
            except ValueError:
                return None
            if not 0 <= out < len(self.contenttypes):
                return None
        return out

    def has_field(self, key):
        return self.fieldindex(key) is not None

    def getitem_at_check(self, viewtype):
        out = self.getitem_at(viewtype)
        if isinstance(out, ak._connect.numba.arrayview.RecordViewType):
            typer = ak._util.numba_record_typer(
                out.arrayviewtype.type, out.arrayviewtype.behavior
            )
            if typer is not None:
                return typer(out)
        return out

    def getitem_at(self, viewtype):
        if len(viewtype.fields) == 0:
            return ak._connect.numba.arrayview.RecordViewType(viewtype)
        else:
            key = viewtype.fields[0]
            index = self.fieldindex(key)
            if index is None:
                if self.fields is None:
                    raise ValueError(
                        "no field {} in tuples with {} fields".format(
                            repr(key), len(self.contenttypes)
                        )
                    )
                else:
                    raise ValueError(
                        "no field {} in records with fields: [{}]".format(
                            repr(key), ", ".join(repr(x) for x in self.fields)
                        )
                    )
            contenttype = self.contenttypes[index]
            subviewtype = ak._connect.numba.arrayview.wrap(
                contenttype, viewtype, viewtype.fields[1:]
            )
            return contenttype.getitem_at_check(subviewtype)

    def getitem_field(self, viewtype, key):
        index = self.fieldindex(key)
        if index is None:
            if self.fields is None:
                raise ValueError(
                    "no field {} in tuples with {} fields".format(
                        repr(key), len(self.contenttypes)
                    )
                )
            else:
                raise ValueError(
                    "no field {} in records with fields: [{}]".format(
                        repr(key), ", ".join(repr(x) for x in self.fields)
                    )
                )
        contenttype = self.contenttypes[index]
        subviewtype = ak._connect.numba.arrayview.wrap(contenttype, viewtype, None)
        return contenttype.getitem_range(subviewtype)

    def getitem_field_record(self, recordviewtype, key):
        index = self.fieldindex(key)
        if index is None:
            if self.fields is None:
                raise ValueError(
                    "no field {} in tuple with {} fields".format(
                        repr(key), len(self.contenttypes)
                    )
                )
            else:
                raise ValueError(
                    "no field {} in record with fields: [{}]".format(
                        repr(key), ", ".join(repr(x) for x in self.fields)
                    )
                )
        contenttype = self.contenttypes[index]
        subviewtype = ak._connect.numba.arrayview.wrap(
            contenttype, recordviewtype, None
        )
        return contenttype.getitem_at_check(subviewtype)

    def lower_getitem_at_check(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        out = self.lower_getitem_at(
            context,
            builder,
            rettype,
            viewtype,
            viewval,
            viewproxy,
            attype,
            atval,
            wrapneg,
            checkbounds,
        )
        baretype = self.getitem_at(viewtype)
        if isinstance(baretype, ak._connect.numba.arrayview.RecordViewType):
            lower = ak._util.numba_record_lower(
                baretype.arrayviewtype.type, baretype.arrayviewtype.behavior
            )
            if lower is not None:
                return lower(context, builder, rettype(baretype), (out,))
        return out

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        if len(viewtype.fields) == 0:
            proxyout = context.make_helper(
                builder, ak._connect.numba.arrayview.RecordViewType(viewtype)
            )
            proxyout.arrayview = viewval
            proxyout.at = atval
            return proxyout._getvalue()

        else:
            index = self.fieldindex(viewtype.fields[0])
            contenttype = self.contenttypes[index]

            whichpos = posat(context, builder, viewproxy.pos, self.CONTENTS + index)
            nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

            nextviewtype = ak._connect.numba.arrayview.wrap(
                contenttype, viewtype, viewtype.fields[1:]
            )
            proxynext = context.make_helper(builder, nextviewtype)
            proxynext.pos = nextpos
            proxynext.start = viewproxy.start
            proxynext.stop = builder.add(
                atval, builder.add(viewproxy.start, context.get_constant(numba.intp, 1))
            )
            proxynext.arrayptrs = viewproxy.arrayptrs
            proxynext.pylookup = viewproxy.pylookup

            return contenttype.lower_getitem_at_check(
                context,
                builder,
                rettype,
                nextviewtype,
                proxynext._getvalue(),
                proxynext,
                numba.intp,
                atval,
                False,
                False,
            )

    def lower_getitem_field(self, context, builder, viewtype, viewval, key):
        viewproxy = context.make_helper(builder, viewtype, viewval)

        index = self.fieldindex(key)
        contenttype = self.contenttypes[index]

        whichpos = posat(context, builder, viewproxy.pos, self.CONTENTS + index)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        proxynext = context.make_helper(builder, contenttype.getitem_range(viewtype))
        proxynext.pos = nextpos
        proxynext.start = viewproxy.start
        proxynext.stop = viewproxy.stop
        proxynext.arrayptrs = viewproxy.arrayptrs
        proxynext.pylookup = viewproxy.pylookup

        return proxynext._getvalue()

    def lower_getitem_field_record(
        self, context, builder, recordviewtype, recordviewval, key
    ):
        arrayviewtype = recordviewtype.arrayviewtype
        recordviewproxy = context.make_helper(builder, recordviewtype, recordviewval)
        arrayviewval = recordviewproxy.arrayview
        arrayviewproxy = context.make_helper(builder, arrayviewtype, arrayviewval)

        index = self.fieldindex(key)
        contenttype = self.contenttypes[index]

        whichpos = posat(context, builder, arrayviewproxy.pos, self.CONTENTS + index)
        nextpos = getat(context, builder, arrayviewproxy.arrayptrs, whichpos)

        proxynext = context.make_helper(
            builder, contenttype.getitem_range(arrayviewtype)
        )
        proxynext.pos = nextpos
        proxynext.start = arrayviewproxy.start
        proxynext.stop = builder.add(
            recordviewproxy.at,
            builder.add(arrayviewproxy.start, context.get_constant(numba.intp, 1)),
        )
        proxynext.arrayptrs = arrayviewproxy.arrayptrs
        proxynext.pylookup = arrayviewproxy.pylookup

        nextviewtype = ak._connect.numba.arrayview.wrap(
            contenttype, arrayviewtype, None
        )

        rettype = self.getitem_field_record(recordviewtype, key)

        return contenttype.lower_getitem_at_check(
            context,
            builder,
            rettype,
            nextviewtype,
            proxynext._getvalue(),
            proxynext,
            numba.intp,
            recordviewproxy.at,
            False,
            False,
        )

    @property
    def is_tuple(self):
        return self.fields is None

    @property
    def ndim(self):
        return 1

    @property
    def inner_dtype(self):
        return None

    @property
    def is_optiontype(self):
        return False

    @property
    def is_recordtype(self):
        return True


class UnionArrayType(ContentType, ak._lookup.UnionLookup):
    @classmethod
    def from_form(cls, form):
        return UnionArrayType(
            cls.from_form_index(form.tags),
            cls.from_form_index(form.index),
            [ak._connect.numba.arrayview.tonumbatype(x) for x in form.contents],
            form.parameters,
        )

    def __init__(self, tagstype, indextype, contenttypes, parameters):
        tmp1 = ", ".join(x.name for x in contenttypes)
        tmp2 = "," if len(contenttypes) == 1 else ""
        super().__init__(
            name=f"ak.UnionArrayType({tagstype.name}, {indextype.name}, ({tmp1}{tmp2}), {json.dumps(parameters)})"
        )
        self.tagstype = tagstype
        self.indextype = indextype
        self.contenttypes = contenttypes
        self.parameters = parameters

    def has_field(self, key):
        return any(x.has_field(key) for x in self.contenttypes)

    def getitem_at(self, viewtype):
        if not all(isinstance(x, RecordArrayType) for x in self.contenttypes):
            raise TypeError("union types cannot be accessed in Numba")

    def getitem_range(self, viewtype):
        if not all(isinstance(x, RecordArrayType) for x in self.contenttypes):
            raise TypeError("union types cannot be accessed in Numba")

    def getitem_field(self, viewtype, key):
        if not all(isinstance(x, RecordArrayType) for x in self.contenttypes):
            raise TypeError("union types cannot be accessed in Numba")

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        raise NotImplementedError(
            type(self).__name__ + ".lower_getitem_at not implemented"
        )

    def lower_getitem_range(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        start,
        stop,
        wrapneg,
    ):
        raise NotImplementedError(
            type(self).__name__ + ".lower_getitem_range not implemented"
        )

    def lower_getitem_field(self, context, builder, viewtype, viewval, viewproxy, key):
        raise NotImplementedError(
            type(self).__name__ + ".lower_getitem_field not implemented"
        )

    @property
    def ndim(self):
        out = None
        for contenttype in self.contenttypes:
            if out is None:
                out = contenttype.ndim
            elif out != contenttype.ndim:
                return None
        return out

    @property
    def inner_dtype(self):
        context = numba.core.typing.Context()
        return context.unify_types(*[x.inner_dtype for x in self.contenttypes])

    @property
    def is_optiontype(self):
        return any(x.is_optiontype for x in self.contents)

    @property
    def is_recordtype(self):
        if all(x.is_recordtype for x in self.contents):
            return True
        elif all(not x.is_recordtype for x in self.contents):
            return False
        else:
            return None


def inner_dtype_of_form(form):
    if form is None:
        return None

    elif isinstance(form, ak.forms.NumpyForm):
        return numba.from_dtype(ak.types.numpytype.primitive_to_dtype(form.primitive))

    elif isinstance(form, ak.forms.EmptyForm):
        return numba.types.float64

    elif isinstance(
        form,
        (
            ak.forms.RegularForm,
            ak.forms.ListForm,
            ak.forms.ListOffsetForm,
            ak.forms.IndexedForm,
        ),
    ):
        return inner_dtype_of_form(form.content)

    elif isinstance(
        form,
        (
            ak.forms.RecordForm,
            ak.forms.IndexedOptionForm,
            ak.forms.ByteMaskedForm,
            ak.forms.BitMaskedForm,
            ak.forms.UnmaskedForm,
        ),
    ):
        return None

    elif isinstance(form, ak.forms.UnionForm):
        context = numba.core.typing.Context()
        return context.unify_types(*[inner_dtype_of_form(x) for x in form.contents])

    else:
        raise AssertionError(f"unrecognized Form type: {type(form)}")


def optiontype_of_form(form):
    if form is None:
        return None

    elif isinstance(
        form,
        (
            ak.forms.NumpyForm,
            ak.forms.EmptyForm,
            ak.forms.RegularForm,
            ak.forms.ListForm,
            ak.forms.ListOffsetForm,
            ak.forms.IndexedForm,
            ak.forms.RecordForm,
        ),
    ):
        return False

    elif isinstance(
        form,
        (
            ak.forms.IndexedOptionForm,
            ak.forms.ByteMaskedForm,
            ak.forms.BitMaskedForm,
            ak.forms.UnmaskedForm,
        ),
    ):
        return False

    elif isinstance(form, ak.forms.UnionForm):
        return any(optiontype_of_form(x) for x in form.contents)

    else:
        raise AssertionError(f"unrecognized Form type: {type(form)}")


def recordtype_of_form(form):
    if form is None:
        return None

    elif isinstance(
        form,
        (
            ak.forms.NumpyForm,
            ak.forms.EmptyForm,
            ak.forms.RegularForm,
            ak.forms.ListForm,
            ak.forms.ListOffsetForm,
        ),
    ):
        return False

    elif isinstance(
        form,
        (
            ak.forms.IndexedForm,
            ak.forms.IndexedOptionForm,
            ak.forms.ByteMaskedForm,
            ak.forms.BitMaskedForm,
            ak.forms.UnmaskedForm,
        ),
    ):
        return recordtype_of_form(form.content)

    elif isinstance(form, (ak.forms.RecordForm,)):
        return True

    elif isinstance(form, ak.forms.UnionForm):
        return any(recordtype_of_form(x) for x in form.contents)

    else:
        raise AssertionError(f"unrecognized Form type: {type(form)}")
