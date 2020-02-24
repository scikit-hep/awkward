# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import json

import numpy
import numba
import llvmlite.ir.types

import awkward1.layout

import awkward1._numba.arrayview

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
    def tolookup_identities(cls, layout, positions, arrays):
        if layout.identities is None:
            positions.append(-1)
        else:
            arrays.append(numpy.asarray(layout.identities))
            positions.append(arrays[-1])

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

    def getitem_range(self, viewtype):
        return awkward1._numba.arrayview.ArrayViewType(self, viewtype.behavior, viewtype.fields)

    def getitem_field(self, viewtype, key):
        if self.hasfield(key):
            return awkward1._numba.arrayview.ArrayViewType(self, viewtype.behavior, viewtype.fields + (key,))
        else:
            raise TypeError("array does not have a field with key {0}".format(repr(key)))

    def lower_getitem_range(self, context, builder, rettype, viewtype, viewval, viewproxy, start, stop, wrapneg):
        print(type(self).__name__, "lower range", viewtype)

        length = builder.sub(viewproxy.stop, viewproxy.start)

        regular_start = numba.cgutils.alloca_once_value(builder, start)
        regular_stop = numba.cgutils.alloca_once_value(builder, stop)

        if wrapneg:
            with builder.if_then(builder.icmp_signed("<", start, context.get_constant(numba.intp, 0))):
                builder.store(builder.add(start, length), regular_start)
            with builder.if_then(builder.icmp_signed("<", stop, context.get_constant(numba.intp, 0))):
                builder.store(builder.add(stop, length), regular_stop)

        with builder.if_then(builder.icmp_signed("<", builder.load(regular_start), context.get_constant(numba.intp, 0))):
            builder.store(context.get_constant(numba.intp, 0), regular_start)
        with builder.if_then(builder.icmp_signed(">", builder.load(regular_start), length)):
            builder.store(length, regular_start)

        with builder.if_then(builder.icmp_signed("<", builder.load(regular_stop), builder.load(regular_start))):
            builder.store(builder.load(regular_start), regular_stop)
        with builder.if_then(builder.icmp_signed(">", builder.load(regular_stop), length)):
            builder.store(length, regular_stop)

        proxyout = context.make_helper(builder, rettype)
        proxyout.pos       = viewproxy.pos
        proxyout.start     = builder.add(viewproxy.start, builder.load(regular_start))
        proxyout.stop      = builder.add(viewproxy.start, builder.load(regular_stop))
        proxyout.arrayptrs = viewproxy.arrayptrs
        proxyout.pylookup  = viewproxy.pylookup
        return proxyout._getvalue()

    def lower_getitem_field(self, context, builder, viewtype, viewval, key):
        print(type(self).__name__, "lower field", viewtype)

        return viewval

def castint(context, builder, fromtype, totype, val):
    if isinstance(fromtype, llvmlite.ir.types.IntType):
        if fromtype.width == 8:
            fromtype = numba.int8
        elif fromtype.width == 16:
            fromtype = numba.int16
        elif fromtype.width == 32:
            fromtype = numba.int32
        elif fromtype.width == 64:
            fromtype = numba.int64
    if not isinstance(fromtype, numba.types.Integer):
        raise AssertionError("unrecognized integer type: {0}".format(repr(fromtype)))
    if fromtype.bitwidth < totype.bitwidth:
        if fromtype.signed:
            return builder.sext(val, context.get_value_type(totype))
        else:
            return builder.zext(val, context.get_value_type(totype))
    elif fromtype.bitwidth > totype.bitwidth:
        return builder.trunc(val, context.get_value_type(totype))
    else:
        return val

def posat(context, builder, pos, offset):
    return builder.add(pos, context.get_constant(numba.intp, offset))

def getat(context, builder, baseptr, offset, rettype=None):
    ptrtype = None
    if rettype is not None:
        ptrtype = context.get_value_type(numba.types.CPointer(rettype))
        bitwidth = rettype.bitwidth
    else:
        bitwidth = numba.intp.bitwidth
    byteoffset = builder.mul(offset, context.get_constant(numba.intp, bitwidth // 8))
    return builder.load(numba.cgutils.pointer_add(builder, baseptr, byteoffset, ptrtype))

def regularize_atval(context, builder, viewproxy, attype, atval, wrapneg, checkbounds):
    atval = castint(context, builder, attype, numba.intp, atval)

    if not attype.signed:
        wrapneg = False

    if wrapneg or checkbounds:
        length = builder.sub(viewproxy.stop, viewproxy.start)

        if wrapneg:
            regular_atval = numba.cgutils.alloca_once_value(builder, atval)
            with builder.if_then(builder.icmp_signed("<", atval, context.get_constant(numba.intp, 0))):
                builder.store(builder.add(atval, length), regular_atval)
            atval = builder.load(regular_atval)

        if checkbounds:
            with builder.if_then(builder.or_(builder.icmp_signed("<", atval, context.get_constant(numba.intp, 0)),
                                             builder.icmp_signed(">=", atval, length))):
                context.call_conv.return_user_exc(builder, ValueError, ("slice index out of bounds",))

    return castint(context, builder, atval.type, numba.intp, atval)

class NumpyArrayType(ContentType):
    IDENTITIES = 0
    ARRAY = 1

    @classmethod
    def tolookup(cls, layout, positions, arrays):
        array = numpy.asarray(layout)
        assert len(array.shape) == 1
        pos = len(positions)
        cls.tolookup_identities(layout, positions, arrays)
        positions.append(array)
        arrays.append(array)
        return pos

    def __init__(self, arraytype, identitiestype, parameters):
        super(NumpyArrayType, self).__init__(name="awkward1.NumpyArrayType({0}, {1}, {2})".format(arraytype.name, identitiestype.name, repr(parameters)))
        self.arraytype = arraytype
        self.identitiestype = identitiestype
        self.parameters = parameters

    def tolayout(self, lookup, pos, fields):
        assert fields == ()
        return awkward1.layout.NumpyArray(lookup.arrays[lookup.positions[pos + self.ARRAY]])

    def hasfield(self, key):
        return False

    def getitem_at(self, viewtype):
        return self.arraytype.dtype

    def lower_getitem_at(self, context, builder, rettype, viewtype, viewval, viewproxy, attype, atval, wrapneg, checkbounds):
        print(type(self).__name__, "lower at", viewtype)

        whichpos = posat(context, builder, viewproxy.pos, self.ARRAY)
        arrayptr = getat(context, builder, viewproxy.arrayptrs, whichpos)
        atval = regularize_atval(context, builder, viewproxy, attype, atval, wrapneg, checkbounds)
        arraypos = builder.add(viewproxy.start, atval)
        return getat(context, builder, arrayptr, arraypos, rettype)
        
class RegularArrayType(ContentType):
    IDENTITIES = 0
    CONTENT = 1

    @classmethod
    def tolookup(cls, layout, positions, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, arrays)
        positions.append(None)
        positions[pos + cls.CONTENT] = awkward1._numba.arrayview.tolookup(layout.content, positions, arrays)
        return pos

    def __init__(self, contenttype, size, identitiestype, parameters):
        super(RegularArrayType, self).__init__(name="awkward1.RegularArrayType({0}, {1}, {2}, {3})".format(contenttype.name, size, identitiestype.name, repr(parameters)))
        self.contenttype = contenttype
        self.size = size
        self.identitiestype = identitiestype
        self.parameters = parameters

    def tolayout(self, lookup, pos, fields):
        content = self.contenttype.tolayout(lookup, lookup.positions[pos + self.CONTENT], fields)
        return awkward1.layout.RegularArray(content, self.size)

    def hasfield(self, key):
        return self.contenttype.hasfield(key)

    def getitem_at(self, viewtype):
        return awkward1._numba.arrayview.ArrayViewType(self.contenttype, viewtype.behavior, viewtype.fields)

    def lower_getitem_at(self, context, builder, rettype, viewtype, viewval, viewproxy, attype, atval, wrapneg, checkbounds):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(context, builder, viewproxy, attype, atval, wrapneg, checkbounds)

        size = context.get_constant(numba.intp, self.size)
        start = builder.mul(builder.add(viewproxy.start, atval), size)
        stop  = builder.add(start, size)

        proxyout = context.make_helper(builder, rettype)
        proxyout.pos       = nextpos
        proxyout.start     = start
        proxyout.stop      = stop
        proxyout.arrayptrs = viewproxy.arrayptrs
        proxyout.pylookup  = viewproxy.pylookup
        return proxyout._getvalue()

class ListArrayType(ContentType):
    IDENTITIES = 0
    STARTS = 1
    STOPS = 2
    CONTENT = 3

    @classmethod
    def tolookup(cls, layout, positions, arrays):
        if isinstance(layout, (awkward1.layout.ListArray32, awkward1.layout.ListArrayU32, awkward1.layout.ListArray64)):
            starts = numpy.asarray(layout.starts)
            stops = numpy.asarray(layout.stops)
        elif isinstance(layout, (awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArrayU32, awkward1.layout.ListOffsetArray64)):
            offsets = numpy.asarray(layout.offsets)
            starts = offsets[:-1]
            stops = offsets[1:]

        pos = len(positions)
        cls.tolookup_identities(layout, positions, arrays)
        positions.append(starts)
        arrays.append(starts)
        positions.append(stops)
        arrays.append(stops)
        positions.append(None)
        positions[pos + cls.CONTENT] = awkward1._numba.arrayview.tolookup(layout.content, positions, arrays)
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
        starts = self.IndexOf(self.indextype)(lookup.arrays[lookup.positions[pos + self.STARTS]])
        stops = self.IndexOf(self.indextype)(lookup.arrays[lookup.positions[pos + self.STOPS]])
        content = self.contenttype.tolayout(lookup, lookup.positions[pos + self.CONTENT], fields)
        return self.ListArrayOf()(starts, stops, content)

    def hasfield(self, key):
        return self.contenttype.hasfield(key)

    def getitem_at(self, viewtype):
        return awkward1._numba.arrayview.ArrayViewType(self.contenttype, viewtype.behavior, viewtype.fields)

    def lower_getitem_at(self, context, builder, rettype, viewtype, viewval, viewproxy, attype, atval, wrapneg, checkbounds):
        print(type(self).__name__, "lower at", viewtype)

        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(context, builder, viewproxy, attype, atval, wrapneg, checkbounds)

        startspos = posat(context, builder, viewproxy.pos, self.STARTS)
        startsptr = getat(context, builder, viewproxy.arrayptrs, startspos)
        startsarraypos = builder.add(viewproxy.start, atval)
        start = getat(context, builder, startsptr, startsarraypos, self.indextype.dtype)

        stopspos = posat(context, builder, viewproxy.pos, self.STOPS)
        stopsptr = getat(context, builder, viewproxy.arrayptrs, stopspos)
        stopsarraypos = builder.add(viewproxy.start, atval)
        stop = getat(context, builder, stopsptr, stopsarraypos, self.indextype.dtype)

        proxyout = context.make_helper(builder, rettype)
        proxyout.pos       = nextpos
        proxyout.start     = castint(context, builder, self.indextype.dtype, numba.intp, start)
        proxyout.stop      = castint(context, builder, self.indextype.dtype, numba.intp, stop)
        proxyout.arrayptrs = viewproxy.arrayptrs
        proxyout.pylookup  = viewproxy.pylookup
        return proxyout._getvalue()

class IndexedArrayType(ContentType):
    IDENTITIES = 0
    INDEX = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, positions, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, arrays)
        arrays.append(numpy.asarray(layout.index))
        positions.append(arrays[-1])
        positions.append(None)
        positions[pos + cls.CONTENT] = awkward1._numba.arrayview.tolookup(layout.content, positions, arrays)
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
        index = self.IndexOf(self.indextype)(lookup.arrays[lookup.positions[pos + self.INDEX]])
        content = self.contenttype.tolayout(lookup, lookup.positions[pos + self.CONTENT], fields)
        return self.IndexedArrayOf()(index, content)

    def hasfield(self, key):
        return self.contenttype.hasfield(key)

    def getitem_at(self, viewtype):
        return self.contenttype.getitem_at(viewtype)

    def lower_getitem_at(self, context, builder, rettype, viewtype, viewval, viewproxy, attype, atval, wrapneg, checkbounds):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(context, builder, viewproxy, attype, atval, wrapneg, checkbounds)

        indexpos = posat(context, builder, viewproxy.pos, self.INDEX)
        indexptr = getat(context, builder, viewproxy.arrayptrs, indexpos)
        indexarraypos = builder.add(viewproxy.start, atval)
        nextat = getat(context, builder, indexptr, indexarraypos, self.indextype.dtype)

        nextviewtype = awkward1._numba.arrayview.ArrayViewType(self.contenttype, viewtype.behavior, viewtype.fields)
        proxynext = context.make_helper(builder, nextviewtype)
        proxynext.pos       = nextpos
        proxynext.start     = viewproxy.start
        proxynext.stop      = builder.add(castint(context, builder, self.indextype.dtype, numba.intp, nextat), builder.add(viewproxy.start, context.get_constant(numba.intp, 1)))
        proxynext.arrayptrs = viewproxy.arrayptrs
        proxynext.pylookup  = viewproxy.pylookup

        return self.contenttype.lower_getitem_at(context, builder, rettype, nextviewtype, proxynext._getvalue(), proxynext, numba.intp, nextat, False, False)

class IndexedOptionArrayType(ContentType):
    IDENTITIES = 0
    INDEX = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, positions, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, arrays)
        arrays.append(numpy.asarray(layout.index))
        positions.append(arrays[-1])
        positions.append(None)
        positions[pos + cls.CONTENT] = awkward1._numba.arrayview.tolookup(layout.content, positions, arrays)
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
        index = self.IndexOf(self.indextype)(lookup.arrays[lookup.positions[pos + self.INDEX]])
        content = self.contenttype.tolayout(lookup, lookup.positions[pos + self.CONTENT], fields)
        return self.IndexedOptionArrayOf()(index, content)

    def hasfield(self, key):
        return self.contenttype.hasfield(key)

    def getitem_at(self, viewtype):
        return numba.types.optional(self.contenttype.getitem_at(viewtype))

    def lower_getitem_at(self, context, builder, rettype, viewtype, viewval, viewproxy, attype, atval, wrapneg, checkbounds):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(context, builder, viewproxy, attype, atval, wrapneg, checkbounds)

        indexpos = posat(context, builder, viewproxy.pos, self.INDEX)
        indexptr = getat(context, builder, viewproxy.arrayptrs, indexpos)
        indexarraypos = builder.add(viewproxy.start, atval)
        nextat = getat(context, builder, indexptr, indexarraypos, self.indextype.dtype)

        output = context.make_helper(builder, rettype)

        with builder.if_else(builder.icmp_signed("<", nextat, context.get_constant(self.indextype.dtype, 0))) as (isnone, isvalid):
            with isnone:
                output.valid = numba.cgutils.false_bit
                output.data = numba.cgutils.get_null_value(output.data.type)

            with isvalid:
                nextviewtype = awkward1._numba.arrayview.ArrayViewType(self.contenttype, viewtype.behavior, viewtype.fields)
                proxynext = context.make_helper(builder, nextviewtype)
                proxynext.pos       = nextpos
                proxynext.start     = viewproxy.start
                proxynext.stop      = builder.add(castint(context, builder, self.indextype.dtype, numba.intp, nextat), builder.add(viewproxy.start, context.get_constant(numba.intp, 1)))
                proxynext.arrayptrs = viewproxy.arrayptrs
                proxynext.pylookup  = viewproxy.pylookup

                outdata = self.contenttype.lower_getitem_at(context, builder, rettype.type, nextviewtype, proxynext._getvalue(), proxynext, numba.intp, nextat, False, False)

                output.valid = numba.cgutils.true_bit
                output.data = outdata

        return output._getvalue()

class RecordArrayType(ContentType):
    IDENTITIES = 0
    CONTENTS = 1

    @classmethod
    def tolookup(cls, layout, positions, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, arrays)
        positions.extend([None] * layout.numfields)
        for i, content in enumerate(layout.contents):
            positions[pos + cls.CONTENTS + i] = awkward1._numba.arrayview.tolookup(content, positions, arrays)
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
            for i, x in enumerate(self.recordlookup):
                if x == key:
                    out = i
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
            return self.contenttypes[index].tolayout(lookup, lookup.positions[pos + self.CONTENTS + index], fields[1:])
        else:
            contents = []
            for i, contenttype in enumerate(self.contenttypes):
                layout = contenttype.tolayout(lookup, lookup.positions[pos + self.CONTENTS + i], fields)
                contents.append(layout)
            if len(contents) == 0:
                return awkward1.layout.RecordArray(numpy.iinfo(numpy.int64).max, self.recordlookup is None)
            else:
                return awkward1.layout.RecordArray(contents, self.recordlookup)

    def hasfield(self, key):
        return self.fieldindex(key) is not None

    def getitem_at(self, viewtype):
        if len(viewtype.fields) == 0:
            return awkward1._numba.arrayview.RecordViewType(viewtype)
        else:
            key = viewtype.fields[0]
            index = self.fieldindex(key)
            if index is None:
                if self.recordlookup is None:
                    raise ValueError("no field {0} in tuples with {1} fields".format(repr(key), len(self.contenttypes)))
                else:
                    raise ValueError("no field {0} in records with fields: [{1}]".format(repr(key), ", ".join(repr(x) for x in self.recordlookup)))
            contenttype = self.contenttypes[index]
            subviewtype = awkward1._numba.arrayview.ArrayViewType(contenttype, viewtype.behavior, viewtype.fields[1:])
            return contenttype.getitem_at(subviewtype)

    def getitem_field(self, viewtype, key):
        index = self.fieldindex(key)
        if index is None:
            if self.recordlookup is None:
                raise ValueError("no field {0} in tuples with {1} fields".format(repr(key), len(self.contenttypes)))
            else:
                raise ValueError("no field {0} in records with fields: [{1}]".format(repr(key), ", ".join(repr(x) for x in self.recordlookup)))
        contenttype = self.contenttypes[index]
        subviewtype = awkward1._numba.arrayview.ArrayViewType(contenttype, viewtype.behavior, viewtype.fields)
        return contenttype.getitem_range(subviewtype)

    def getitem_field_record(self, recordviewtype, key):
        index = self.fieldindex(key)
        if index is None:
            if self.recordlookup is None:
                raise ValueError("no field {0} in tuple with {1} fields".format(repr(key), len(self.contenttypes)))
            else:
                raise ValueError("no field {0} in record with fields: [{1}]".format(repr(key), ", ".join(repr(x) for x in self.recordlookup)))
        contenttype = self.contenttypes[index]
        subviewtype = awkward1._numba.arrayview.ArrayViewType(contenttype, recordviewtype.arrayviewtype.behavior, recordviewtype.arrayviewtype.fields)
        return contenttype.getitem_at(subviewtype)

    def lower_getitem_at(self, context, builder, rettype, viewtype, viewval, viewproxy, attype, atval, wrapneg, checkbounds):
        print(type(self).__name__, "lower at", viewtype)

        atval = regularize_atval(context, builder, viewproxy, attype, atval, wrapneg, checkbounds)

        if len(viewtype.fields) == 0:
            proxyout = context.make_helper(builder, awkward1._numba.arrayview.RecordViewType(viewtype))
            proxyout.arrayview = viewval
            proxyout.at        = atval
            return proxyout._getvalue()

        else:
            index = self.fieldindex(viewtype.fields[0])
            contenttype = self.contenttypes[index]

            whichpos = posat(context, builder, viewproxy.pos, self.CONTENTS + index)
            nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

            nextviewtype = awkward1._numba.arrayview.ArrayViewType(contenttype, viewtype.behavior, viewtype.fields[1:])
            proxynext = context.make_helper(builder, nextviewtype)
            proxynext.pos       = nextpos
            proxynext.start     = viewproxy.start
            proxynext.stop      = builder.add(atval, builder.add(viewproxy.start, context.get_constant(numba.intp, 1)))
            proxynext.arrayptrs = viewproxy.arrayptrs
            proxynext.pylookup  = viewproxy.pylookup

            return contenttype.lower_getitem_at(context, builder, rettype, nextviewtype, proxynext._getvalue(), proxynext, numba.intp, atval, False, False)

    def lower_getitem_field(self, context, builder, viewtype, viewval, key):
        print(type(self).__name__, "lower field", viewtype, key)

        viewproxy = context.make_helper(builder, viewtype, viewval)

        index = self.fieldindex(key)
        contenttype = self.contenttypes[index]

        whichpos = posat(context, builder, viewproxy.pos, self.CONTENTS + index)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        proxynext = context.make_helper(builder, contenttype.getitem_range(viewtype))
        proxynext.pos       = nextpos
        proxynext.start     = viewproxy.start
        proxynext.stop      = viewproxy.stop
        proxynext.arrayptrs = viewproxy.arrayptrs
        proxynext.pylookup  = viewproxy.pylookup

        return proxynext._getvalue()

    def lower_getitem_field_record(self, context, builder, recordviewtype, recordviewval, key):
        print(type(self).__name__, "lower field record", recordviewtype, key)

        arrayviewtype = recordviewtype.arrayviewtype
        recordviewproxy = context.make_helper(builder, recordviewtype, recordviewval)
        arrayviewval = recordviewproxy.arrayview
        arrayviewproxy = context.make_helper(builder, arrayviewtype, arrayviewval)

        index = self.fieldindex(key)
        contenttype = self.contenttypes[index]

        whichpos = posat(context, builder, arrayviewproxy.pos, self.CONTENTS + index)
        nextpos = getat(context, builder, arrayviewproxy.arrayptrs, whichpos)

        proxynext = context.make_helper(builder, contenttype.getitem_range(arrayviewtype))
        proxynext.pos       = nextpos
        proxynext.start     = arrayviewproxy.start
        proxynext.stop      = builder.add(recordviewproxy.at, builder.add(arrayviewproxy.start, context.get_constant(numba.intp, 1)))
        proxynext.arrayptrs = arrayviewproxy.arrayptrs
        proxynext.pylookup  = arrayviewproxy.pylookup

        nextviewtype = awkward1._numba.arrayview.ArrayViewType(contenttype, arrayviewtype.behavior, arrayviewtype.fields)

        rettype = self.getitem_field_record(recordviewtype, key)

        return contenttype.lower_getitem_at(context, builder, rettype, nextviewtype, proxynext._getvalue(), proxynext, numba.intp, recordviewproxy.at, False, False)

class UnionArrayType(ContentType):
    IDENTITIES = 0
    TAGS = 1
    INDEX = 2
    CONTENTS = 3

    @classmethod
    def tolookup(cls, layout, positions, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, arrays)
        arrays.append(numpy.asarray(layout.tags))
        positions.append(arrays[-1])
        arrays.append(numpy.asarray(layout.index))
        positions.append(arrays[-1])
        positions.extend([None] * layout.numcontents)
        for i, content in enumerate(layout.contents):
            positions[pos + cls.CONTENTS + i] = awkward1._numba.arrayview.tolookup(content, positions, arrays)
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
        tags = self.IndexOf(self.tagstype)(lookup.arrays[lookup.positions[pos + self.TAGS]])
        index = self.IndexOf(self.indextype)(lookup.arrays[lookup.positions[pos + self.INDEX]])
        contents = []
        for i, contenttype in enumerate(self.contenttypes):
            layout = contenttype.tolayout(lookup, lookup.positions[pos + self.CONTENTS + i], fields)
            contents.append(layout)
        return self.UnionArrayOf()(tags, index, contents)

    def hasfield(self, key):
        return any(x.hasfield(key) for x in self.contenttypes)

    def getitem_at(self, viewtype):
        raise NotImplementedError(type(self).__name__ + ".getitem_at not implemented")

    def getitem_range(self, viewtype):
        raise NotImplementedError(type(self).__name__ + ".getitem_range not implemented")

    def getitem_field(self, viewtype, key):
        raise NotImplementedError(type(self).__name__ + ".getitem_field not implemented")

    def lower_getitem_at(self, context, builder, rettype, viewtype, viewval, viewproxy, attype, atval, wrapneg, checkbounds):
        raise NotImplementedError(type(self).__name__ + ".lower_getitem_at not implemented")

    def lower_getitem_range(self, context, builder, rettype, viewtype, viewval, viewproxy, start, stop, wrapneg):
        raise NotImplementedError(type(self).__name__ + ".lower_getitem_range not implemented")

    def lower_getitem_field(self, context, builder, viewtype, viewval, viewproxy, key):
        raise NotImplementedError(type(self).__name__ + ".lower_getitem_field not implemented")
