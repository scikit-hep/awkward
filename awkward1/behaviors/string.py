# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import codecs

import numpy

import awkward1.highlevel
import awkward1.operations.convert

class CharBehavior(awkward1.highlevel.Array):
    def __bytes__(self):
        return numpy.asarray(self.layout).tostring()

    def __str__(self):
        encoding = self.layout.type.parameters.get("encoding")
        if encoding is None:
            return str(self.__bytes__())
        else:
            return self.__bytes__().decode(encoding, "surrogateescape")

    def __repr__(self):
        encoding = self.layout.type.parameters.get("encoding")
        if encoding is None:
            return repr(self.__bytes__())
        else:
            return repr(self.__bytes__().decode(encoding, "surrogateescape"))

    def __iter__(self):
        for x in str(self):
            yield x

awkward1.behavior["char"] = CharBehavior
byte = awkward1.types.PrimitiveType("uint8", {"__array__": "char", "__typestr__": "byte", "encoding": None})
utf8 = awkward1.types.PrimitiveType("uint8", {"__array__": "char", "__typestr__": "utf8", "encoding": "utf-8"})

class StringBehavior(awkward1.highlevel.Array):
    def __iter__(self):
        if self.layout.type.type.parameters.get("encoding") is None:
            for x in super(StringBehavior, self).__iter__():
                yield x.__bytes__()
        else:
            for x in super(StringBehavior, self).__iter__():
                yield x.__str__()

awkward1.behavior["string"] = StringBehavior
bytestring = awkward1.types.ListType(byte, {"__array__": "string", "__typestr__": "bytes"})
string = awkward1.types.ListType(utf8, {"__array__": "string", "__typestr__": "string"})

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

awkward1.behavior[numpy.equal, "string", "string"] = string_equal
