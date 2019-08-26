# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numba

AtType = numba.int64
IndexType = numba.int32

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
                if not isinstance(wheretpe, numba.types.BaseTuple):
                    wheretpe = numba.types.Tuple((wheretpe,))
                if len(wheretpe.types) == 0:
                    return arraytpe
                if any(isinstance(x, numba.types.Array) and x.ndim == 1 for x in wheretpe.types):
                    wheretpe = numba.types.Tuple(tuple(numba.types.Array(x, 1, "C") if isinstance(x, numba.types.Integer) else x for x in wheretpe))
                return numba.typing.templates.signature(arraytpe.getitem(wheretpe), arraytpe, original_wheretpe)
