# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba

from .._numba import cpu, identity

class ContentType(numba.types.Type):
    pass

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

                if any(isinstance(t, numba.types.Array) and t.ndim == 1 for t in wheretpe.types):
                    newwhere = ()
                    for t in wheretpe:
                        if isinstance(t, numba.types.Integer):
                            newwhere = newwhere + (numba.int64[:],)
                        elif isinstance(t, numba.types.Array) and isinstance(t.dtype, numba.types.Integer):
                            newwhere = newwhere + (numba.types.Array(numba.int64, t.ndim, "C"),)
                        elif isinstance(t, numba.types.Array) and isinstance(t.dtype, numba.types.Boolean):
                            for i in range(t.ndim):
                                newwhere = newwhere + (numba.int64[:],)
                        elif isinstance(t, numba.types.Array):
                            raise TypeError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`), and integer or boolean arrays (possibly jagged) are valid indices")
                        else:
                            newwhere = newwhere + (t,)
                    where = newwhere

                if any(not isinstance(t, (numba.types.Integer, numba.types.SliceType, numba.types.EllipsisType, type(numba.typeof(numpy.newaxis)), numba.types.Array)) for t in wheretpe.types):
                    raise TypeError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`), and integer or boolean arrays (possibly jagged) are valid indices")

                return numba.typing.templates.signature(arraytpe.getitem_tuple(wheretpe), arraytpe, original_wheretpe)
