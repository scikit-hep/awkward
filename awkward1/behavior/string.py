# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import codecs

import numpy

import awkward1.highlevel

class CharBehavior(awkward1.highlevel.Array):
    @staticmethod
    def typestr(baretype, parameters):
        encoding = parameters.get("encoding")
        if encoding is None:
            return "char"
        elif codecs.getdecoder(encoding) is codecs.getdecoder("utf-8"):
            return "utf8"
        else:
            return "encoded[{0}]".format(repr(encoding))

    def __bytes__(self):
        return numpy.asarray(self.layout).tostring()

    def __str__(self):
        encoding = self.type.nolength().parameters.get("encoding")
        if encoding is None:
            return str(self.__bytes__())
        else:
            return self.__bytes__().decode(encoding)

    def __repr__(self):
        encoding = self.type.nolength().parameters.get("encoding")
        if encoding is None:
            return repr(self.__bytes__())
        else:
            return repr(self.__bytes__().decode(encoding))

    def __iter__(self):
        for x in str(self):
            yield x

class StringBehavior(awkward1.highlevel.Array):
    @staticmethod
    def typestr(baretype, parameters):
        encoding = baretype.inner().parameters.get("encoding")
        if encoding is None:
            return "bytes"
        elif codecs.getdecoder(encoding) is codecs.getdecoder("utf-8"):
            return "string"
        else:
            return "string[{0}]".format(repr(encoding))

    def __iter__(self):
        if self.type.nolength().inner().parameters.get("encoding") is None:
            for x in super(StringBehavior, self).__iter__():
                yield x.__bytes__()
        else:
            for x in super(StringBehavior, self).__iter__():
                yield x.__str__()

    def __eq__(self, other):
        raise NotImplementedError("return one boolean per string, not lists of booleans per character")

char = awkward1.layout.DressedType(awkward1.layout.PrimitiveType("uint8"), CharBehavior)
utf8 = awkward1.layout.DressedType(awkward1.layout.PrimitiveType("uint8"), CharBehavior, encoding="utf-8")

bytestring = awkward1.layout.DressedType(awkward1.layout.ListType(char), StringBehavior)
string = awkward1.layout.DressedType(awkward1.layout.ListType(utf8), StringBehavior)
