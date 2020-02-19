# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import operator

import numpy
import numba

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
                if isinstance(wheretpe, numba.types.Integer):
                    return numba.typing.templates.signature(arraytpe.getitem_at(), arraytpe, wheretpe)
                elif isinstance(wheretpe, numba.types.SliceType) and not wheretpe.has_step:
                    return numba.typing.templates.signature(arraytpe.getitem_range(), arraytpe, wheretpe)
                elif isinstance(wheretpe, numba.types.StringLiteral):
                    return numba.typing.templates.signature(arraytpe.getitem_field(), arraytpe, wheretpe)
                else:
                    raise TypeError("Awkward-Numba only supports int, start:stop, and \"field\" slices")
