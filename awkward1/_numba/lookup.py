# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import numba

import awkward1.layout
import awkward1.highlevel
import awkward1.operations.convert
import awkward1._util

class Lookup(object):
    def __init__(self, array):
        self.behavior = awkward1._util.behaviorof(array)
        layout = awkward1.operations.convert.tolayout(array, allowrecord=True, allowother=False, numpytype=(numpy.number,))
        lookup = []
        arrays = []
        identities = []
        tolookup(layout, lookup, arrays, identities)
        self.lookup = numpy.array(lookup, dtype=numpy.int64)
        self.arrays = arrays
        self.identities = identities

class View(object):
    def __init__(self, lookup, pos, start, stop, fields, type):
        self.lookup = lookup
        self.pos = pos
        self.start = start
        self.stop = stop
        self.fields = fields
        self.type = type

def tolookup(layout, lookup, arrays, identities):
    if isinstance(layout, awkward1.layout.NumpyArray):
        return NumpyArrayType.tolookup(layout, lookup, arrays, identities)

    elif isinstance(layout, awkward1.layout.RegularArray):
        return RegularArrayType.tolookup(layout, lookup, arrays, identities)

    elif isinstance(layout, (awkward1.layout.ListArray32, awkward1.layout.ListArrayU32, awkward1.layout.ListArray64, awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArrayU32, awkward1.layout.ListOffsetArray64)):
        return ListArrayType.tolookup(layout, lookup, arrays, identities)

    elif isinstance(layout, (awkward1.layout.IndexedArray32, awkward1.layout.IndexedArrayU32, awkward1.layout.IndexedArray64)):
        return IndexedArrayType.tolookup(layout, lookup, arrays, identities)

    elif isinstance(layout, (awkward1.layout.IndexedOptionArray32, awkward1.layout.IndexedOptionArrayU32, awkward1.layout.IndexedOptionArray64)):
        return IndexedOptionArrayType.tolookup(layout, lookup, arrays, identities)

    elif isinstance(layout, awkward1.layout.RecordArray):
        return RecordArrayType.tolookup(layout, lookup, arrays, identities)

    elif isinstance(layout, awkward1.layout.Record):
        return RecordType.tolookup(layout, lookup, arrays, identities)

    elif isinstance(layout, (awkward1.layout.UnionArray8_32, awkward1.layout.UnionArray8_U32, awkward1.layout.UnionArray8_64)):
        return UnionType.tolookup(layout, lookup, arrays, identities)

    else:
        raise AssertionError("unrecognized layout type: {0}".format(type(layout)))

def totype(layout):
    if isinstance(layout, awkward1.layout.NumpyArray):
        return NumpyArrayType(layout)

    elif isinstance(layout, awkward1.layout.RegularArray):
        return RegularArrayType(layout)

    elif isinstance(layout, (awkward1.layout.ListArray32, awkward1.layout.ListArrayU32, awkward1.layout.ListArray64, awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArrayU32, awkward1.layout.ListOffsetArray64)):
        return ListArrayType(layout)

    elif isinstance(layout, (awkward1.layout.IndexedArray32, awkward1.layout.IndexedArrayU32, awkward1.layout.IndexedArray64)):
        return IndexedArrayType(layout)

    elif isinstance(layout, (awkward1.layout.IndexedOptionArray32, awkward1.layout.IndexedOptionArrayU32, awkward1.layout.IndexedOptionArray64)):
        return IndexedOptionArrayType(layout)

    elif isinstance(layout, awkward1.layout.RecordArray):
        return RecordArrayType(layout)

    elif isinstance(layout, awkward1.layout.Record):
        return RecordType(layout)

    elif isinstance(layout, (awkward1.layout.UnionArray8_32, awkward1.layout.UnionArray8_U32, awkward1.layout.UnionArray8_64)):
        return UnionType(layout)

    else:
        raise AssertionError("unrecognized layout type: {0}".format(type(layout)))

class LookupType(numba.types.Type):
    pass

class ContentType(numba.types.Type):
    @classmethod
    def tolookup_identities(cls, layout, lookup, identities):
        lookup.append(None)
        if layout.identities is None:
            lookup[-1] = -1
        else:
            lookup[-1] = len(identities)
            identities.append(layout.identities)
            
class NumpyArrayType(ContentType):
    # 0: identities
    # 1: array

    @classmethod
    def tolookup(cls, layout, lookup, arrays, identities):
        if len(numpy.asarray(layout).shape) > 1:
            return tolookup(layout.toRegularArray(), lookup, arrays, identities)
        else:
            pos = len(lookup)
            cls.tolookup_identities(layout, lookup, identities)
            lookup.append(len(arrays))
            arrays.append(numpy.asarray(layout))
            return pos

    def __init__(self, arraytype, identitiestype, parameters):
        super(NumpyArrayType, self).__init__(name="awkward1.NumpyArrayType({0}, {1}, {2})".format(arraytype.name, identitiestype.name, parameters))
        self.arraytype = arraytype
        self.identitiestype = identitiestype
        self.parameters = parameters

class RegularArrayType(ContentType):
    # 0: identities
    # 1: content pos

    @classmethod
    def tolookup(cls, layout, lookup, arrays, identities):
        pos = len(lookup)
        cls.tolookup_identities(layout, lookup, identities)
        lookup.append(None)
        lookup[pos + 1] = tolookup(layout.content, lookup, arrays, identities)
        return pos

class ListArrayType(ContentType):
    # 0: identities
    # 1: starts array
    # 2: stops array
    # 3: content pos

    @classmethod
    def tolookup(cls, layout, lookup, arrays, identities):
        if isinstance(layout, (awkward1.layout.ListArray32, awkward1.layout.ListArrayU32, awkward1.layout.ListArray64)):
            starts = numpy.asarray(layout.starts)
            stops = numpy.asarray(layout.stops)
        elif isinstance(layout, (awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArrayU32, awkward1.layout.ListOffsetArray64)):
            offsets = numpy.asarray(layout.offsets)
            starts = offsets[:-1]
            stops = offsets[1:]

        pos = len(lookup)
        cls.tolookup_identities(layout, lookup, identities)
        lookup.append(len(arrays))
        arrays.append(starts)
        lookup.append(len(arrays))
        arrays.append(stops)
        lookup.append(None)
        lookup[pos + 3] = tolookup(layout.content, lookup, arrays, identities)
        return pos

class IndexedArrayType(ContentType):
    # 0: identities
    # 1: index array
    # 2: content pos

    @classmethod
    def tolookup(cls, layout, lookup, arrays, identities):
        pos = len(lookup)
        cls.tolookup_identities(layout, lookup, identities)
        lookup.append(len(arrays))
        arrays.append(numpy.asarray(layout.index))
        lookup.append(None)
        lookup[pos + 2] = tolookup(layout.content, lookup, arrays, identities)
        return pos

class IndexedOptionArrayType(ContentType):
    # 0: identities
    # 1: index array
    # 2: content pos

    @classmethod
    def tolookup(cls, layout, lookup, arrays, identities):
        pos = len(lookup)
        cls.tolookup_identities(layout, lookup, identities)
        lookup.append(len(arrays))
        arrays.append(numpy.asarray(layout.index))
        lookup.append(None)
        lookup[pos + 2] = tolookup(layout.content, lookup, arrays, identities)
        return pos

class RecordArrayType(ContentType):
    # 0: identities
    # i + 1: contents pos

    @classmethod
    def tolookup(cls, layout, lookup, arrays, identities):
        pos = len(lookup)
        cls.tolookup_identities(layout, lookup, identities)
        lookup.extend([None] * layout.numfields)
        for i, content in layout.contents:
            lookup[pos + i + 1] = tolookup(content, lookup, arrays, identities)
        return pos

class RecordType(ContentType):
    # 0: at
    # 1: record pos

    @classmethod
    def tolookup(cls, layout, lookup, arrays, identities):
        pos = len(lookup)
        lookup.append(layout.at)
        lookup.append(None)
        lookup[pos + 1] = tolookup(layout.array, lookup, arrays, identities)
        return pos

class UnionArrayType(ContentType):
    # 0: identities
    # 1: tags array
    # 2: index array
    # i + 3: contents pos

    @classmethod
    def tolookup(cls, layout, lookup, arrays, identities):
        pos = len(lookup)
        cls.tolookup_identities(layout, lookup, identities)
        lookup.append(len(arrays))
        arrays.append(numpy.asarray(layout.tags))
        lookup.append(len(arrays))
        arrays.append(numpy.asarray(layout.index))
        lookup.extend([None] * layout.numcontents)
        for i, content in layout.contents:
            lookup[pos + i + 3] = tolookup(content, lookup, arrays, identities)
        return pos
