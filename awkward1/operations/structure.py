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
import awkward1._numpy
import awkward1.operations.convert

def isna(array):
    import awkward1.highlevel
    
    def apply(layout):
        if isinstance(layout, awkward1.layout.EmptyArray):
            return apply(awkward.layout.NumpyArray(numpy.array([])))

        elif isinstance(layout, (awkward1.layout.IndexedArray32, awkward1.layout.IndexedArrayU32, awkward1.layout.IndexedArray64)):
            return apply(layout.project())

        elif isinstance(layout, (awkward1.layout.UnionArray8_32, awkward1.layout.UnionArray8_U32, awkward1.layout.UnionArray8_64)):
            contents = [apply(layout.project(i)) for i in range(len(layout))]
            out = numpy.empty(len(layout), dtype=numpy.bool_)
            tags = numpy.asarray(layout.tags)
            for tag, content in enumerate(contents):
                out[tags == tag] = content
            return out

        elif isinstance(layout, (awkward1.layout.IndexedOptionArray32, awkward1.layout.IndexedOptionArray64)):
            index = numpy.asarray(layout.index)
            return (index < 0)

        else:
            return numpy.zeros(len(layout), dtype=numpy.bool_)

    return awkward1.highlevel.Array(apply(awkward1.operations.convert.tolayout(array)))

def notna(array):
    return ~isna(array)

@awkward1._numpy.implements(numpy.concatenate)
def concatenate(arrays, axis=0):
    import awkward1.highlevel

    if axis != 0:
        raise NotImplementedError("axis={0}".format(axis))

    contents = [awkward1.operations.convert.tolayout(x) for x in arrays]

    tags = numpy.empty(sum(len(x) for x in contents), dtype=numpy.int8)
    index = numpy.empty(len(tags), dtype=numpy.int64)
    start = 0
    for tag, x in enumerate(contents):
        tags[start : start + len(x)] = tag
        index[start : start + len(x)] = numpy.arange(len(x), dtype=numpy.int64)
        start += len(x)

    tags = awkward1.layout.Index8(tags)
    index = awkward1.layout.Index64(index)
    out = awkward1.layout.UnionArray8_64(tags, index, contents)
    return awkward1.highlevel.Array(out)

@awkward1._numpy.implements(numpy.where)
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
