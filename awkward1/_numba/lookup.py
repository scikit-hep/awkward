# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import json

import numpy
import numba

import awkward1.layout
import awkward1.highlevel
import awkward1.operations.convert
import awkward1._util

class View(object):
    @classmethod
    def fromarray(self, array):
        behavior = awkward1._util.behaviorof(array)
        layout = awkward1.operations.convert.tolayout(array, allowrecord=True, allowother=False, numpytype=(numpy.number,))
        return View(behavior, Lookup(layout), 0, 0, len(array), (), numba.typeof(layout))

    def __init__(self, behavior, lookup, pos, start, stop, fields, type):
        self.behavior = behavior
        self.lookup = lookup
        self.pos = pos
        self.start = start
        self.stop = stop
        self.fields = fields
        self.type = type

    def toarray(self):
        layout = self.type.tolayout(self.lookup, self.pos, (self.start, self.stop), self.fields)
        return awkward1._util.wrap(layout, self.behavior)

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
        return NumpyArrayType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, awkward1.layout.RegularArray):
        return RegularArrayType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, (awkward1.layout.ListArray32, awkward1.layout.ListArrayU32, awkward1.layout.ListArray64, awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArrayU32, awkward1.layout.ListOffsetArray64)):
        return ListArrayType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, (awkward1.layout.IndexedArray32, awkward1.layout.IndexedArrayU32, awkward1.layout.IndexedArray64)):
        return IndexedArrayType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, (awkward1.layout.IndexedOptionArray32, awkward1.layout.IndexedOptionArray64)):
        return IndexedOptionArrayType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, awkward1.layout.RecordArray):
        return RecordArrayType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, awkward1.layout.Record):
        return RecordType.tolookup(layout, postable, arrays, identities)

    elif isinstance(layout, (awkward1.layout.UnionArray8_32, awkward1.layout.UnionArray8_U32, awkward1.layout.UnionArray8_64)):
        return UnionType.tolookup(layout, postable, arrays, identities)

    else:
        raise AssertionError("unrecognized layout type: {0}".format(type(layout)))

@numba.extending.typeof_impl.register(awkward1.layout.NumpyArray)
def typeof(obj, c):
    return NumpyArrayType(numba.typeof(numpy.asarray(obj)), numba.typeof(obj.identities), json.dumps(obj.parameters))

@numba.extending.typeof_impl.register(awkward1.layout.RegularArray)
def typeof(obj, c):
    return RegularArrayType(numba.typeof(obj.content), obj.size, numba.typeof(obj.identities), json.dumps(obj.parameters))

@numba.extending.typeof_impl.register(awkward1.layout.ListArray32)
@numba.extending.typeof_impl.register(awkward1.layout.ListArrayU32)
@numba.extending.typeof_impl.register(awkward1.layout.ListArray64)
@numba.extending.typeof_impl.register(awkward1.layout.ListOffsetArray32)
@numba.extending.typeof_impl.register(awkward1.layout.ListOffsetArrayU32)
@numba.extending.typeof_impl.register(awkward1.layout.ListOffsetArray64)
def typeof(obj, c):
    return ListArrayType(numba.typeof(numpy.asarray(obj.starts)), numba.typeof(obj.content), numba.typeof(obj.identities), json.dumps(obj.parameters))

@numba.extending.typeof_impl.register(awkward1.layout.IndexedArray32)
@numba.extending.typeof_impl.register(awkward1.layout.IndexedArrayU32)
@numba.extending.typeof_impl.register(awkward1.layout.IndexedArray64)
def typeof(obj, c):
    return IndexedArrayType(numba.typeof(numpy.asarray(obj.index)), numba.typeof(obj.content), numba.typeof(obj.identities), json.dumps(obj.parameters))

@numba.extending.typeof_impl.register(awkward1.layout.IndexedOptionArray32)
@numba.extending.typeof_impl.register(awkward1.layout.IndexedOptionArray64)
def typeof(obj, c):
    return IndexedOptionArrayType(numba.typeof(numpy.asarray(obj.index)), numba.typeof(obj.content), numba.typeof(obj.identities), json.dumps(obj.parameters))

@numba.extending.typeof_impl.register(awkward1.layout.RecordArray)
def typeof(obj, c):
    return RecordArrayType(tuple(numba.typeof(x) for x in obj.contents), numba.typeof(obj.identities), json.dumps(obj.parameters))

@numba.extending.typeof_impl.register(awkward1.layout.Record)
def typeof(obj, c):
    return RecordType(numba.typeof(obj.array))

@numba.extending.typeof_impl.register(awkward1.layout.UnionArray8_32)
@numba.extending.typeof_impl.register(awkward1.layout.UnionArray8_U32)
@numba.extending.typeof_impl.register(awkward1.layout.UnionArray8_64)
def typeof(obj, c):
    return UnionArrayType(numba.typeof(numpy.asarray(obj.tags)), numba.typeof(numpy.asarray(obj.index)), tuple(numba.typeof(x) for x in obj.contents), numba.typeof(obj.identities), json.dumps(obj.parameters))

class LookupType(numba.types.Type):
    pass

class ContentType(numba.types.Type):
    @classmethod
    def tolookup_identities(cls, layout, postable, identities):
        postable.append(None)
        if layout.identities is None:
            postable[-1] = -1
        else:
            postable[-1] = len(identities)
            identities.append(layout.identities)

    def IndexOf(self, arraytype):
        if arraytype.dtype.bitwidth == 8 and arraytype.dtype.signed:
            return awkward1.layout.Index8
        elif arraytype.dtype.bitwidth == 8:
            return awkward1.layout.IndexU8
        elif arraytype.dtype.bitwidth == 32 and arraytype.dtype.signed:
            return awkward1.layout.Index32
        elif arraytype.dtype.bitwidth == 32:
            return awkward1.layout.IndexU32
        elif arraytype.dtype.bitwidth == 64 and arraytype.dtype.signed:
            return awkward1.layout.Index64
        else:
            raise AssertionError("no Index* type for array: {0}".format(arraytype))

class NumpyArrayType(ContentType):
    # 0: identities
    # 1: array

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        if len(numpy.asarray(layout).shape) > 1:
            return tolookup(layout.toRegularArray(), postable, arrays, identities)
        else:
            pos = len(postable)
            cls.tolookup_identities(layout, postable, identities)
            postable.append(len(arrays))
            arrays.append(numpy.asarray(layout))
            return pos

    def __init__(self, arraytype, identitiestype, parameters):
        super(NumpyArrayType, self).__init__(name="awkward1.NumpyArrayType({0}, {1}, {2})".format(arraytype.name, identitiestype.name, repr(parameters)))
        self.arraytype = arraytype
        self.identitiestype = identitiestype
        self.parameters = parameters

    def tolayout(self, lookup, pos, startstop, fields):
        assert fields == ()
        out = awkward1.layout.NumpyArray(lookup.arrays[lookup.postable[pos + 1]])
        if startstop is None:
            return out
        else:
            start, stop = startstop
            return out[start:stop]

class RegularArrayType(ContentType):
    # 0: identities
    # 1: content pos

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        pos = len(postable)
        cls.tolookup_identities(layout, postable, identities)
        postable.append(None)
        postable[pos + 1] = tolookup(layout.content, postable, arrays, identities)
        return pos

    def __init__(self, contenttype, size, identitiestype, parameters):
        super(RegularArrayType, self).__init__(name="awkward1.RegularArrayType({0}, {1}, {2}, {3})".format(contenttype.name, size, identitiestype.name, repr(parameters)))
        self.contenttype = contenttype
        self.size = size
        self.identitiestype = identitiestype
        self.parameters = parameters

    def tolayout(self, lookup, pos, startstop, fields):
        content = self.contenttype.tolayout(lookup, lookup.postable[pos + 1], None, fields)
        out = awkward1.layout.RegularArray(content, size)
        if startstop is None:
            return out
        else:
            start, stop = startstop
            return out[start:stop]

class ListArrayType(ContentType):
    # 0: identities
    # 1: starts array
    # 2: stops array
    # 3: content pos

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        if isinstance(layout, (awkward1.layout.ListArray32, awkward1.layout.ListArrayU32, awkward1.layout.ListArray64)):
            starts = numpy.asarray(layout.starts)
            stops = numpy.asarray(layout.stops)
        elif isinstance(layout, (awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArrayU32, awkward1.layout.ListOffsetArray64)):
            offsets = numpy.asarray(layout.offsets)
            starts = offsets[:-1]
            stops = offsets[1:]

        pos = len(postable)
        cls.tolookup_identities(layout, postable, identities)
        postable.append(len(arrays))
        arrays.append(starts)
        postable.append(len(arrays))
        arrays.append(stops)
        postable.append(None)
        postable[pos + 3] = tolookup(layout.content, postable, arrays, identities)
        return pos

    def __init__(self, indextype, contenttype, identitiestype, parameters):
        super(ListArrayType, self).__init__(name="awkward1.ListArrayType({0}, {1}, {2}, {3})".format(indextype.name, contenttype.name, identitiestype.name, repr(parameters)))
        self.indextype = indextype
        self.contenttype = contenttype
        self.identitiestype = identitiestype
        self.parameters = parameters

    def ListArrayOf(self, arraytype):
        if arraytype.dtype.bitwidth == 32 and arraytype.dtype.signed:
            return awkward1.layout.ListArray32
        elif arraytype.dtype.bitwidth == 32:
            return awkward1.layout.ListArrayU32
        elif arraytype.dtype.bitwidth == 64 and arraytype.dtype.signed:
            return awkward1.layout.ListArray64
        else:
            raise AssertionError("no ListArray* type for array: {0}".format(arraytype))

    def tolayout(self, lookup, pos, startstop, fields):
        starts = self.IndexOf(self.indextype)(lookup.arrays[lookup.postable[pos + 1]])
        stops = self.IndexOf(self.indextype)(lookup.arrays[lookup.postable[pos + 2]])
        content = self.contenttype.tolayout(lookup, lookup.postable[pos + 3], None, fields)
        out = self.ListArrayOf(self.indextype)(starts, stops, content)
        if startstop is None:
            return out
        else:
            start, stop = startstop
            return out[start:stop]

class IndexedArrayType(ContentType):
    # 0: identities
    # 1: index array
    # 2: content pos

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        pos = len(postable)
        cls.tolookup_identities(layout, postable, identities)
        postable.append(len(arrays))
        arrays.append(numpy.asarray(layout.index))
        postable.append(None)
        postable[pos + 2] = tolookup(layout.content, postable, arrays, identities)
        return pos

    def __init__(self, indextype, contenttype, identitiestype, parameters):
        super(IndexedArrayType, self).__init__(name="awkward1.IndexedArrayType({0}, {1}, {2}, {3})".format(indextype.name, contenttype.name, identitiestype.name, repr(parameters)))
        self.indextype = indextype
        self.contenttype = contenttype
        self.identitiestype = identitiestype
        self.parameters = parameters

class IndexedOptionArrayType(ContentType):
    # 0: identities
    # 1: index array
    # 2: content pos

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        pos = len(postable)
        cls.tolookup_identities(layout, postable, identities)
        postable.append(len(arrays))
        arrays.append(numpy.asarray(layout.index))
        postable.append(None)
        postable[pos + 2] = tolookup(layout.content, postable, arrays, identities)
        return pos

    def __init__(self, indextype, contenttype, identitiestype, parameters):
        super(IndexedOptionArrayType, self).__init__(name="awkward1.IndexedOptionArrayType({0}, {1}, {2}, {3})".format(indextype.name, contenttype.name, identitiestype.name, repr(parameters)))
        self.indextype = indextype
        self.contenttype = contenttype
        self.identitiestype = identitiestype
        self.parameters = parameters

class RecordArrayType(ContentType):
    # 0: identities
    # i + 1: contents pos

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        pos = len(postable)
        cls.tolookup_identities(layout, postable, identities)
        postable.extend([None] * layout.numfields)
        for i, content in layout.contents:
            postable[pos + i + 1] = tolookup(content, postable, arrays, identities)
        return pos

    def __init__(self, contenttypes, identitiestype, parameters):
        super(RecordArrayType, self).__init__(name="awkward1.RecordArrayType(({0}{1}), {2}, {3})".format(", ".join(x.name for x in contenttypes), "," if len(contenttypes) == 1 else "", identitiestype.name, repr(parameters)))
        self.contenttypes = contenttypes
        self.identitiestype = identitiestype
        self.parameters = parameters

class RecordType(ContentType):
    # 0: at
    # 1: record pos

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        pos = len(postable)
        postable.append(layout.at)
        postable.append(None)
        postable[pos + 1] = tolookup(layout.array, postable, arrays, identities)
        return pos

    def __init__(self, arraytype):
        super(RecordType, self).__init__(name="awkward1.RecordType({0})".format(arraytype.name))
        self.arraytype = arraytype

class UnionArrayType(ContentType):
    # 0: identities
    # 1: tags array
    # 2: index array
    # i + 3: contents pos

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        pos = len(postable)
        cls.tolookup_identities(layout, postable, identities)
        postable.append(len(arrays))
        arrays.append(numpy.asarray(layout.tags))
        postable.append(len(arrays))
        arrays.append(numpy.asarray(layout.index))
        postable.extend([None] * layout.numcontents)
        for i, content in layout.contents:
            postable[pos + i + 3] = tolookup(content, postable, arrays, identities)
        return pos

    def __init__(self, tagstype, indextype, contenttypes, identitiestype, parameters):
        super(UnionArrayType, self).__init__(name="awkward1.UnionArrayType({0}, {1}, ({2}{3}), {4}, {5})".format(tagstype.name, indextype.name, ", ".join(x.name for x in contenttypes), "," if len(contenttypes) == 1 else "", identitiestype.name, repr(parameters)))
        self.tagstype = tagstype
        self.indextype = indextype
        self.contenttypes = contenttypes
        self.identitiestype = identitiestype
        self.parameters = parameters
