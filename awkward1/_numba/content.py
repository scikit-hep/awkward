# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba

from .._numba import cpu, util, identity

class ContentType(numba.types.Type):
    @property
    def shortname(self):
        return self.name[:self.name.index("Type")]

@numba.typing.templates.infer_global(len)
class type_len(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            arraytpe, = args
            if isinstance(arraytpe, ContentType):
                return numba.typing.templates.signature(numba.types.intp, arraytpe)

@numba.typing.templates.infer_global(operator.getitem)
class type_getitem(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            arraytpe, wheretpe = args

            if isinstance(arraytpe, ContentType):
                original_wheretpe = wheretpe
                if isinstance(wheretpe, numba.types.Integer):
                    return numba.typing.templates.signature(arraytpe.getitem_int(), arraytpe, original_wheretpe)
                if isinstance(wheretpe, numba.types.SliceType) and not wheretpe.has_step:
                    return numba.typing.templates.signature(arraytpe.getitem_range(), arraytpe, original_wheretpe)
                if isinstance(wheretpe, numba.types.StringLiteral):
                    return numba.typing.templates.signature(arraytpe.getitem_str(wheretpe.literal_value), arraytpe, original_wheretpe)

                if not isinstance(wheretpe, numba.types.BaseTuple):
                    wheretpe = numba.types.Tuple((wheretpe,))

                wheretpe = util.typing_regularize_slice(wheretpe)
                self.check_slice_types(wheretpe)

                return numba.typing.templates.signature(arraytpe.getitem_tuple(wheretpe), arraytpe, original_wheretpe)

    @staticmethod
    def check_slice_types(wheretpe):
        if any(not isinstance(t, (numba.types.Integer, numba.types.SliceType, numba.types.EllipsisType, type(numba.typeof(numpy.newaxis)), numba.types.StringLiteral)) and not (isinstance(t, numba.types.Array) and isinstance(t.dtype, (numba.types.Boolean, numba.types.Integer))) for t in wheretpe.types):
            raise TypeError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`), integer or boolean arrays (possibly jagged), and constant strings (known at compile-time) are valid indices")


def lower_getitem_nothing(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    proxyslice = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxyslice.start = context.get_constant(numba.intp, 0)
    proxyslice.stop = context.get_constant(numba.intp, 0)
    proxyslice.step = context.get_constant(numba.intp, 1)
    outtpe = tpe.contenttpe.getitem_range()
    return tpe.contenttpe.lower_getitem_range(context, builder, outtpe(tpe.contenttpe, numba.types.slice2_type), (proxyin.content, proxyslice._getvalue()))

def lower_getitem_tuple(context, builder, sig, args):
    import awkward1._numba.array.listarray
    import awkward1._numba.array.regulararray

    rettpe, (arraytpe, wheretpe) = sig.return_type, sig.args
    arrayval, whereval = args

    wheretpe, whereval = util.preprocess_slicetuple(context, builder, wheretpe, whereval)

    nexttpe = awkward1._numba.array.regulararray.RegularArrayType(arraytpe, numba.types.none)
    outtpe = nexttpe.getitem_next(wheretpe, False)
    if outtpe.lower_getitem_nothing is None:
        nexttpe = awkward1._numba.array.listarray.ListArrayType(util.index64tpe, util.index64tpe, arraytpe, numba.types.none)
        outtpe = nexttpe.getitem_next(wheretpe, False)

        length = util.arraylen(context, builder, arraytpe, arrayval, totpe=numba.int64)
        proxynext = numba.cgutils.create_struct_proxy(nexttpe)(context, builder)
        proxynext.starts = util.newindex64(context, builder, numba.int64, context.get_constant(numba.int64, 1))
        proxynext.stops = util.newindex64(context, builder, numba.int64, context.get_constant(numba.int64, 1))
        numba.targets.arrayobj.store_item(context, builder, util.index64tpe, context.get_constant(numba.int64, 0), util.arrayptr(context, builder, util.index64tpe, proxynext.starts))
        numba.targets.arrayobj.store_item(context, builder, util.index64tpe, length, util.arrayptr(context, builder, util.index64tpe, proxynext.stops))
        proxynext.content = arrayval
        nextval = proxynext._getvalue()

        outval = nexttpe.lower_getitem_next(context, builder, nexttpe, wheretpe, nextval, whereval, None)

        return outtpe.lower_getitem_int(context, builder, rettpe(outtpe, numba.int64), (outval, context.get_constant(numba.int64, 0)))

    else:
        proxynext = numba.cgutils.create_struct_proxy(nexttpe)(context, builder)
        proxynext.content = arrayval
        proxynext.size = util.arraylen(context, builder, arraytpe, arrayval, totpe=numba.int64)
        nextval = proxynext._getvalue()

        outval = nexttpe.lower_getitem_next(context, builder, nexttpe, wheretpe, nextval, whereval, None)

        lenout = util.arraylen(context, builder, outtpe, outval)
        outputptr = numba.cgutils.alloca_once(builder, context.get_value_type(outtpe.getitem_int()))
        with builder.if_else(builder.icmp_signed("==", lenout, context.get_constant(numba.intp, 0)), likely=False) as (nothing, something):
            with nothing:
                builder.store(outtpe.lower_getitem_nothing(context, builder, outtpe, outval), outputptr)
            with something:
                builder.store(outtpe.lower_getitem_int(context, builder, rettpe(outtpe, numba.int64), (outval, context.get_constant(numba.int64, 0))), outputptr)
        return builder.load(outputptr)

def lower_getitem_other(context, builder, sig, args):
    rettpe, (arraytpe, wheretpe) = sig.return_type, sig.args
    arrayval, whereval = args
    wrappedtpe = numba.types.Tuple((wheretpe,))
    wrappedval = context.make_tuple(builder, wrappedtpe, (whereval,))
    return lower_getitem_tuple(context, builder, rettpe(arraytpe, wrappedtpe), (arrayval, wrappedval))
