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

                if not isinstance(wheretpe, numba.types.BaseTuple):
                    wheretpe = numba.types.Tuple((wheretpe,))

                wheretpe = util.typing_regularize_slice(wheretpe)

                if any(not isinstance(t, (numba.types.Integer, numba.types.SliceType, numba.types.EllipsisType, type(numba.typeof(numpy.newaxis)), numba.types.Array)) for t in wheretpe.types):
                    raise TypeError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`), and integer or boolean arrays (possibly jagged) are valid indices")

                return numba.typing.templates.signature(arraytpe.getitem_tuple(wheretpe), arraytpe, original_wheretpe)

def lower_getitem_tuple(context, builder, sig, args):
    rettpe, (arraytpe, wheretpe) = sig.return_type, sig.args
    arrayval, whereval = args

    wheretpe, whereval = util.preprocess_slicetuple(context, builder, wheretpe, whereval)
    nexttpe, nextval = util.wrap_for_slicetuple(context, builder, arraytpe, arrayval)

    outtpe = nexttpe.getitem_next(wheretpe, False)
    outval = nexttpe.lower_getitem_next(context, builder, nexttpe, wheretpe, nextval, whereval, None)

    return outtpe.lower_getitem_int(context, builder, rettpe(outtpe, numba.int64), (outval, context.get_constant(numba.int64, 0)))

def lower_getitem_other(context, builder, sig, args):
    rettpe, (arraytpe, wheretpe) = sig.return_type, sig.args
    arrayval, whereval = args
    wrappedtpe = numba.types.Tuple((wheretpe,))
    wrappedval = context.make_tuple(builder, wrappedtpe, (whereval,))
    return lower_getitem_tuple(context, builder, rettpe(arraytpe, wrappedtpe), (arrayval, wrappedval))
