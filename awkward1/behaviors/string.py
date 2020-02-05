# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

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
byte = awkward1.layout.PrimitiveType("uint8", {"__array__": "char", "__typestr__": "byte", "encoding": None})
utf8 = awkward1.layout.PrimitiveType("uint8", {"__array__": "char", "__typestr__": "utf8", "encoding": "utf-8"})

class StringBehavior(awkward1.highlevel.Array):
    def __iter__(self):
        if self.layout.type.type.parameters.get("encoding") is None:
            for x in super(StringBehavior, self).__iter__():
                yield x.__bytes__()
        else:
            for x in super(StringBehavior, self).__iter__():
                yield x.__str__()

awkward1.behavior["string"] = StringBehavior
bytestring = awkward1.layout.ListType(byte, {"__array__": "string", "__typestr__": "bytes"})
string = awkward1.layout.ListType(utf8, {"__array__": "string", "__typestr__": "string"})

def string_equal(one, two):
    # FIXME: this needs a much better implementation;
    # It's here just to demonstrate overloading.
    
    counts1 = numpy.asarray(one.count())
    counts2 = numpy.asarray(two.count())

    # out = (counts1 == counts2)

    # print("out", out)

    # possible = numpy.logical_and(out, counts1)
    # numpossible = numpy.count_nonzero(possible)

    # print("possible", possible)
    # print("counts1", counts1)
    # print("counts1[possible]", counts1[possible])

    # offsets = numpy.empty(numpossible, dtype=numpy.int64)
    # offsets[0] = 0
    # print(numpy.cumsum(counts1[possible]))

    # print("offsets", offsets)

    # # chars1 = one[possible].flatten()
    # # chars2 = two[possible].flatten()
    # # samechars = (chars1 == chars2)

    # # print("samechars", samechars)




    # raise Exception
    
    counts_equal = (counts1 == counts2)
    contents_equal = numpy.empty_like(counts_equal)
    for i, (x, y) in enumerate(zip(one, two)):
        contents_equal[i] = numpy.array_equal(numpy.asarray(x), numpy.asarray(y))

    return awkward1.layout.NumpyArray(counts_equal & contents_equal)

awkward1.behavior[numpy.equal, "string", "string"] = string_equal
