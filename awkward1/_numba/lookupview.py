# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import numba

import awkward1.layout
import awkward1.operations.convert
import awkward1._util
import awkward1._numba.layout

class Lookup(object):
    def __init__(self, layout):
        postable = []
        arrays = []
        identities = []
        tolookup(layout, postable, arrays, identities)
        self.postable = numpy.array(postable, dtype=numpy.int64)
        self.arrays = tuple(arrays)
        self.identities = tuple(identities)

def tolookup(layout, postable, arrays, identities):
    if isinstance(layout, awkward1.layout.NumpyArray):
        return awkward1._numba.layout.NumpyArrayType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, awkward1.layout.RegularArray):
        return awkward1._numba.layout.RegularArrayType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, (awkward1.layout.ListArray32, awkward1.layout.ListArrayU32, awkward1.layout.ListArray64, awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArrayU32, awkward1.layout.ListOffsetArray64)):
        return awkward1._numba.layout.ListArrayType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, (awkward1.layout.IndexedArray32, awkward1.layout.IndexedArrayU32, awkward1.layout.IndexedArray64)):
        return awkward1._numba.layout.IndexedArrayType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, (awkward1.layout.IndexedOptionArray32, awkward1.layout.IndexedOptionArray64)):
        return awkward1._numba.layout.IndexedOptionArrayType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, awkward1.layout.RecordArray):
        return awkward1._numba.layout.RecordArrayType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, awkward1.layout.Record):
        return awkward1._numba.layout.RecordType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, (awkward1.layout.UnionArray8_32, awkward1.layout.UnionArray8_U32, awkward1.layout.UnionArray8_64)):
        return awkward1._numba.layout.UnionArrayType.tolookup(layout, postable, arrays, identities)

    else:
        raise AssertionError("unrecognized layout type: {0}".format(type(layout)))

class LookupType(numba.types.Type):
    pass

class View(object):
    @classmethod
    def fromarray(self, array):
        behavior = awkward1._util.behaviorof(array)
        layout = awkward1.operations.convert.tolayout(array, allowrecord=True, allowother=False, numpytype=(numpy.number,))
        layout = awkward1.operations.convert.regularize_numpyarray(layout, allowempty=False, highlevel=False)
        return View(behavior, Lookup(layout), 0, 0, len(layout), (), numba.typeof(layout))

    def __init__(self, behavior, lookup, pos, start, stop, fields, type):
        self.behavior = behavior
        self.lookup = lookup
        self.pos = pos
        self.start = start
        self.stop = stop
        self.fields = fields
        self.type = type

    def toarray(self):
        layout = self.type.tolayout(self.lookup, self.pos, self.fields)
        return awkward1._util.wrap(layout[self.start:self.stop], self.behavior)
