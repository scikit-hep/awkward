# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import numbers
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward1._util
import awkward1.layout
import awkward1._npfunctions
import awkward1.operations.convert

@awkward1._npfunctions.implements(numpy.where)
def where(condition, *args):
    import awkward1.highlevel

    condition = awkward1.operations.convert.tonumpy(condition)

    if len(args) == 0:
        out = numpy.nonzero(condition)
        return tuple(awkward1.highlevel.Array(x) for x in out)

    elif len(args) == 1:
        raise ValueError("either both or neither of x and y should be given")

    elif len(args) == 2:
        if len(condition.shape) != 1:
            raise NotImplementedError("FIXME: ak.where(condition, x, y) where condition is not 1-d")

        x = awkward1.operations.convert.tolayout(args[0])
        y = awkward1.operations.convert.tolayout(args[1])

        tags = (condition == 0)
        assert tags.itemsize == 1
        index = numpy.empty(len(tags), dtype=numpy.int64)
        index = numpy.arange(len(condition), dtype=numpy.int64)

        tags = awkward1.layout.Index8(tags.view(numpy.int8))
        index = awkward1.layout.Index64(index)
        # FIXME: call "simplify" when it exists
        return awkward1.highlevel.Array(awkward1.layout.UnionArray8_64(tags, index, [x, y]))

    else:
        raise TypeError("where() takes from 1 to 3 positional arguments but {0} were given".format(len(args) + 1))

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numbers", "Iterable", "numpy", "awkward1")]
