# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import codecs

import numpy

import awkward1.highlevel
import awkward1.operations.convert

class ByteBehavior(awkward1.highlevel.Array):
    def __bytes__(self):
        return numpy.asarray(self.layout).tostring()

    def __str__(self):
        return str(self.__bytes__())

    def __repr__(self):
        return repr(self.__bytes__())

    def __iter__(self):
        for x in self.__bytes__():
            yield x

class CharBehavior(awkward1.highlevel.Array):
    def __bytes__(self):
        return numpy.asarray(self.layout).tostring()

    def __str__(self):
        return self.__bytes__().decode("utf-8", "surrogateescape")

    def __repr__(self):
        return repr(self.__bytes__().decode("utf-8", "surrogateescape"))

    def __iter__(self):
        for x in self.__str__():
            yield x

awkward1.behavior["byte"] = ByteBehavior
awkward1.behavior["__typestr__", "byte"] = "byte"
awkward1.behavior["char"] = CharBehavior
awkward1.behavior["__typestr__", "char"] = "char"

class ByteStringBehavior(awkward1.highlevel.Array):
    def __iter__(self):
        for x in super(ByteStringBehavior, self).__iter__():
            yield x.__bytes__()

class StringBehavior(awkward1.highlevel.Array):
    def __iter__(self):
        for x in super(StringBehavior, self).__iter__():
            yield x.__str__()

awkward1.behavior["bytestring"] = ByteStringBehavior
awkward1.behavior["__typestr__", "bytestring"] = "bytes"
awkward1.behavior["string"] = StringBehavior
awkward1.behavior["__typestr__", "string"] = "string"

def string_equal(one, two):
    # first condition: string lengths must be the same
    counts1 = numpy.asarray(one.count(axis=-1))
    counts2 = numpy.asarray(two.count(axis=-1))

    out = (counts1 == counts2)

    # only compare characters in strings that are possibly equal (same length)
    possible = numpy.logical_and(out, counts1)
    possible_counts = counts1[possible]

    chars1 = numpy.asarray(one[possible].flatten())
    chars2 = numpy.asarray(two[possible].flatten())
    samechars = (chars1 == chars2)

    # ufunc.reduceat requires a weird "offsets" that
    #    (a) lacks a final value (end of array is taken as boundary)
    #    (b) fails on Windows if it's not 32-bit
    #    (c) starts with a zero, which cumsum does not provide
    #    (d) doesn't handle offset[i] == offset[i + 1] with an identity
    dtype = numpy.int32 if awkward1._util.win else numpy.int64
    offsets = numpy.empty(len(possible_counts), dtype=dtype)
    offsets[0] = 0
    numpy.cumsum(possible_counts[:-1], out=offsets[1:])

    reduced = numpy.bitwise_and.reduceat(samechars, offsets)

    # update strings of the same length with a verdict about their characters
    out[possible] = reduced

    return awkward1.layout.NumpyArray(out)

awkward1.behavior[numpy.equal, "bytestring", "bytestring"] = string_equal
awkward1.behavior[numpy.equal, "string", "string"] = string_equal

def string_numba_typer(viewtype):
    import numba
    return numba.types.string

def string_numba_lower(context, builder, rettype, viewtype, viewval, viewproxy, attype, atval):
    import numba
    import llvmlite.llvmpy.core
    import awkward1._numba.layout

    whichpos = awkward1._numba.layout.posat(context, builder, viewproxy.pos, viewtype.type.CONTENT)
    nextpos = awkward1._numba.layout.getat(context, builder, viewproxy.arrayptrs, whichpos)

    whichnextpos = awkward1._numba.layout.posat(context, builder, nextpos, viewtype.type.contenttype.ARRAY)

    startspos = awkward1._numba.layout.posat(context, builder, viewproxy.pos, viewtype.type.STARTS)
    startsptr = awkward1._numba.layout.getat(context, builder, viewproxy.arrayptrs, startspos)
    startsarraypos = builder.add(viewproxy.start, atval)
    start = awkward1._numba.layout.getat(context, builder, startsptr, startsarraypos, viewtype.type.indextype.dtype)

    stopspos = awkward1._numba.layout.posat(context, builder, viewproxy.pos, viewtype.type.STOPS)
    stopsptr = awkward1._numba.layout.getat(context, builder, viewproxy.arrayptrs, stopspos)
    stopsarraypos = builder.add(viewproxy.start, atval)
    stop = awkward1._numba.layout.getat(context, builder, stopsptr, stopsarraypos, viewtype.type.indextype.dtype)

    baseptr = awkward1._numba.layout.getat(context, builder, viewproxy.arrayptrs, whichnextpos)
    rawptr = builder.add(baseptr, awkward1._numba.castint(context, builder, viewtype.type.indextype.dtype, numba.intp, start))
    rawptr_cast = builder.inttoptr(rawptr, llvmlite.llvmpy.core.Type.pointer(llvmlite.llvmpy.core.Type.int(numba.intp.bitwidth // 8)))
    strsize = builder.sub(stop, start)
    strsize_cast = awkward1._numba.castint(context, builder, viewtype.type.indextype.dtype, numba.intp, strsize)

    pyapi = context.get_python_api(builder)
    gil = pyapi.gil_ensure()

    strptr = builder.bitcast(rawptr_cast, pyapi.cstring)
    kind = context.get_constant(numba.types.int32, pyapi.py_unicode_1byte_kind)
    pystr = pyapi.string_from_kind_and_data(kind, strptr, strsize_cast)

    out = pyapi.to_native_value(rettype, pystr).value

    pyapi.gil_release(gil)

    return out

# awkward1.behavior["__numba_typer__", "bytestring"] = string_numba_typer
# awkward1.behavior["__numba_lower__", "bytestring"] = string_numba_lower
awkward1.behavior["__numba_typer__", "string"] = string_numba_typer
awkward1.behavior["__numba_lower__", "string"] = string_numba_lower
