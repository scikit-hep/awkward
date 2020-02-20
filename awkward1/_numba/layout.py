# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import json

import numpy
import numba

import awkward1.layout

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
    return RecordArrayType(tuple(numba.typeof(x) for x in obj.contents), obj.recordlookup, numba.typeof(obj.identities), json.dumps(obj.parameters))

@numba.extending.typeof_impl.register(awkward1.layout.Record)
def typeof(obj, c):
    return RecordType(numba.typeof(obj.array))

@numba.extending.typeof_impl.register(awkward1.layout.UnionArray8_32)
@numba.extending.typeof_impl.register(awkward1.layout.UnionArray8_U32)
@numba.extending.typeof_impl.register(awkward1.layout.UnionArray8_64)
def typeof(obj, c):
    return UnionArrayType(numba.typeof(numpy.asarray(obj.tags)), numba.typeof(numpy.asarray(obj.index)), tuple(numba.typeof(x) for x in obj.contents), numba.typeof(obj.identities), json.dumps(obj.parameters))

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
    IDENTITIES = 0
    ARRAY = 1

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        array = numpy.asarray(layout)
        assert len(array.shape) == 1
        pos = len(postable)
        cls.tolookup_identities(layout, postable, identities)
        postable.append(len(arrays))
        arrays.append(array)
        return pos

    def __init__(self, arraytype, identitiestype, parameters):
        super(NumpyArrayType, self).__init__(name="awkward1.NumpyArrayType({0}, {1}, {2})".format(arraytype.name, identitiestype.name, repr(parameters)))
        self.arraytype = arraytype
        self.identitiestype = identitiestype
        self.parameters = parameters

    def tolayout(self, lookup, pos, fields):
        assert fields == ()
        return awkward1.layout.NumpyArray(lookup.arrays[lookup.postable[pos + self.ARRAY]])

class RegularArrayType(ContentType):
    IDENTITIES = 0
    CONTENT = 1

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        import awkward1._numba.arrayview

        pos = len(postable)
        cls.tolookup_identities(layout, postable, identities)
        postable.append(None)
        postable[pos + cls.CONTENT] = awkward1._numba.arrayview.tolookup(layout.content, postable, arrays, identities)
        return pos

    def __init__(self, contenttype, size, identitiestype, parameters):
        super(RegularArrayType, self).__init__(name="awkward1.RegularArrayType({0}, {1}, {2}, {3})".format(contenttype.name, size, identitiestype.name, repr(parameters)))
        self.contenttype = contenttype
        self.size = size
        self.identitiestype = identitiestype
        self.parameters = parameters

    def tolayout(self, lookup, pos, fields):
        content = self.contenttype.tolayout(lookup, lookup.postable[pos + self.CONTENT], fields)
        return awkward1.layout.RegularArray(content, self.size)

class ListArrayType(ContentType):
    IDENTITIES = 0
    STARTS = 1
    STOPS = 2
    CONTENT = 3

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        import awkward1._numba.arrayview

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
        postable[pos + cls.CONTENT] = awkward1._numba.arrayview.tolookup(layout.content, postable, arrays, identities)
        return pos

    def __init__(self, indextype, contenttype, identitiestype, parameters):
        super(ListArrayType, self).__init__(name="awkward1.ListArrayType({0}, {1}, {2}, {3})".format(indextype.name, contenttype.name, identitiestype.name, repr(parameters)))
        self.indextype = indextype
        self.contenttype = contenttype
        self.identitiestype = identitiestype
        self.parameters = parameters

    def ListArrayOf(self):
        if self.indextype.dtype.bitwidth == 32 and self.indextype.dtype.signed:
            return awkward1.layout.ListArray32
        elif self.indextype.dtype.bitwidth == 32:
            return awkward1.layout.ListArrayU32
        elif self.indextype.dtype.bitwidth == 64 and self.indextype.dtype.signed:
            return awkward1.layout.ListArray64
        else:
            raise AssertionError("no ListArray* type for array: {0}".format(indextype))

    def tolayout(self, lookup, pos, fields):
        starts = self.IndexOf(self.indextype)(lookup.arrays[lookup.postable[pos + self.STARTS]])
        stops = self.IndexOf(self.indextype)(lookup.arrays[lookup.postable[pos + self.STOPS]])
        content = self.contenttype.tolayout(lookup, lookup.postable[pos + self.CONTENT], fields)
        return self.ListArrayOf()(starts, stops, content)

class IndexedArrayType(ContentType):
    IDENTITIES = 0
    INDEX = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        import awkward1._numba.arrayview

        pos = len(postable)
        cls.tolookup_identities(layout, postable, identities)
        postable.append(len(arrays))
        arrays.append(numpy.asarray(layout.index))
        postable.append(None)
        postable[pos + cls.CONTENT] = awkward1._numba.arrayview.tolookup(layout.content, postable, arrays, identities)
        return pos

    def __init__(self, indextype, contenttype, identitiestype, parameters):
        super(IndexedArrayType, self).__init__(name="awkward1.IndexedArrayType({0}, {1}, {2}, {3})".format(indextype.name, contenttype.name, identitiestype.name, repr(parameters)))
        self.indextype = indextype
        self.contenttype = contenttype
        self.identitiestype = identitiestype
        self.parameters = parameters

    def IndexedArrayOf(self):
        if self.indextype.dtype.bitwidth == 32 and self.indextype.dtype.signed:
            return awkward1.layout.IndexedArray32
        elif self.indextype.dtype.bitwidth == 32:
            return awkward1.layout.IndexedArrayU32
        elif self.indextype.dtype.bitwidth == 64 and self.indextype.dtype.signed:
            return awkward1.layout.IndexedArray64
        else:
            raise AssertionError("no IndexedArray* type for array: {0}".format(self.indextype))

    def tolayout(self, lookup, pos, fields):
        index = self.IndexOf(self.indextype)(lookup.arrays[lookup.postable[pos + self.INDEX]])
        content = self.contenttype.tolayout(lookup, lookup.postable[pos + self.CONTENT], fields)
        return self.IndexedArrayOf()(index, content)

class IndexedOptionArrayType(ContentType):
    IDENTITIES = 0
    INDEX = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        import awkward1._numba.arrayview

        pos = len(postable)
        cls.tolookup_identities(layout, postable, identities)
        postable.append(len(arrays))
        arrays.append(numpy.asarray(layout.index))
        postable.append(None)
        postable[pos + cls.CONTENT] = awkward1._numba.arrayview.tolookup(layout.content, postable, arrays, identities)
        return pos

    def __init__(self, indextype, contenttype, identitiestype, parameters):
        super(IndexedOptionArrayType, self).__init__(name="awkward1.IndexedOptionArrayType({0}, {1}, {2}, {3})".format(indextype.name, contenttype.name, identitiestype.name, repr(parameters)))
        self.indextype = indextype
        self.contenttype = contenttype
        self.identitiestype = identitiestype
        self.parameters = parameters

    def IndexedOptionArrayOf(self):
        if self.indextype.dtype.bitwidth == 32 and self.indextype.dtype.signed:
            return awkward1.layout.IndexedOptionArray32
        elif self.indextype.dtype.bitwidth == 64 and self.indextype.dtype.signed:
            return awkward1.layout.IndexedOptionArray64
        else:
            raise AssertionError("no IndexedOptionArray* type for array: {0}".format(self.indextype))

    def tolayout(self, lookup, pos, fields):
        index = self.IndexOf(self.indextype)(lookup.arrays[lookup.postable[pos + self.INDEX]])
        content = self.contenttype.tolayout(lookup, lookup.postable[pos + self.CONTENT], fields)
        return self.IndexedOptionArrayOf()(index, content)

class RecordArrayType(ContentType):
    IDENTITIES = 0
    CONTENTS = 1

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        import awkward1._numba.arrayview

        pos = len(postable)
        cls.tolookup_identities(layout, postable, identities)
        postable.extend([None] * layout.numfields)
        for i, content in enumerate(layout.contents):
            postable[pos + cls.CONTENTS + i] = awkward1._numba.arrayview.tolookup(content, postable, arrays, identities)
        return pos

    def __init__(self, contenttypes, recordlookup, identitiestype, parameters):
        super(RecordArrayType, self).__init__(name="awkward1.RecordArrayType(({0}{1}), ({2}), {3}, {4})".format(", ".join(x.name for x in contenttypes), "," if len(contenttypes) == 1 else "", "None" if recordlookup is None else repr(tuple(recordlookup)), identitiestype.name, repr(parameters)))
        self.contenttypes = contenttypes
        self.recordlookup = recordlookup
        self.identitiestype = identitiestype
        self.parameters = parameters

    def fieldindex(self, key):
        out = -1
        if self.recordlookup is not None:
            for x in recordlookup:
                if x == key:
                    break
        if out == -1:
            try:
                out = int(key)
            except ValueError:
                return None
            if not 0 <= out < len(self.contenttypes):
                return None
        return out

    def tolayout(self, lookup, pos, fields):
        if len(fields) > 0:
            index = self.fieldindex(fields[0])
            assert index is not None
            return self.contenttypes[index].tolayout(lookup, lookup.postable[pos + self.CONTENTS + index], fields[1:])
        else:
            contents = []
            for i, contenttype in enumerate(self.contenttypes):
                layout = contenttype.tolayout(lookup, lookup.postable[pos + self.CONTENTS + i], fields)
                contents.append(layout)
            if len(contents) == 0:
                return awkward1.layout.RecordArray(numpy.iinfo(numpy.int64).max, self.recordlookup is None)
            else:
                return awkward1.layout.RecordArray(contents, self.recordlookup)

class UnionArrayType(ContentType):
    IDENTITIES = 0
    TAGS = 1
    INDEX = 2
    CONTENTS = 3

    @classmethod
    def tolookup(cls, layout, postable, arrays, identities):
        import awkward1._numba.arrayview

        pos = len(postable)
        cls.tolookup_identities(layout, postable, identities)
        postable.append(len(arrays))
        arrays.append(numpy.asarray(layout.tags))
        postable.append(len(arrays))
        arrays.append(numpy.asarray(layout.index))
        postable.extend([None] * layout.numcontents)
        for i, content in enumerate(layout.contents):
            postable[pos + cls.CONTENTS + i] = awkward1._numba.arrayview.tolookup(content, postable, arrays, identities)
        return pos

    def __init__(self, tagstype, indextype, contenttypes, identitiestype, parameters):
        super(UnionArrayType, self).__init__(name="awkward1.UnionArrayType({0}, {1}, ({2}{3}), {4}, {5})".format(tagstype.name, indextype.name, ", ".join(x.name for x in contenttypes), "," if len(contenttypes) == 1 else "", identitiestype.name, repr(parameters)))
        self.tagstype = tagstype
        self.indextype = indextype
        self.contenttypes = contenttypes
        self.identitiestype = identitiestype
        self.parameters = parameters

    def UnionArrayOf(self):
        if self.tagstype.dtype.bitwidth == 8 and self.tagstype.dtype.signed:
            if self.indextype.dtype.bitwidth == 32 and self.indextype.dtype.signed:
                return awkward1.layout.UnionArray8_32
            elif self.indextype.dtype.bitwidth == 32:
                return awkward1.layout.UnionArray8_U32
            elif self.indextype.dtype.bitwidth == 64 and self.indextype.dtype.signed:
                return awkward1.layout.UnionArray8_64
            else:
                raise AssertionError("no UnionArray* type for index array: {0}".format(self.indextype))
        else:
            raise AssertionError("no UnionArray* type for tags array: {0}".format(self.tagstype))

    def tolayout(self, lookup, pos, fields):
        tags = self.IndexOf(self.tagstype)(lookup.arrays[lookup.postable[pos + self.TAGS]])
        index = self.IndexOf(self.indextype)(lookup.arrays[lookup.postable[pos + self.INDEX]])
        contents = []
        for i, contenttype in enumerate(self.contenttypes):
            layout = contenttype.tolayout(lookup, lookup.postable[pos + self.CONTENTS + i], fields)
            contents.append(layout)
        return self.UnionArrayOf()(tags, index, contents)
