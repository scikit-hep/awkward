# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import codecs

import numpy

import awkward1.highlevel

class Char(awkward1.highlevel.Array):
    @staticmethod
    def typestr(baretype, parameters):
        encoding = parameters.get("encoding")
        if encoding is None:
            return "char"
        elif codecs.getdecoder(encoding) is codecs.getdecoder("utf-8"):
            return "utf8"
        else:
            return "encoded[{0}]".format(repr(encoding))

    def __str__(self):
        out = numpy.asarray(self.layout).tostring()
        encoding = self.type.parameters.get("encoding")
        if encoding is None:
            return str(out)
        else:
            return out.decode(encoding)

    def __repr__(self):
        out = numpy.asarray(self.layout).tostring()
        encoding = self.type.nolength().parameters.get("encoding")
        if encoding is None:
            return repr(out)
        else:
            return repr(out.decode(encoding))

class String(awkward1.highlevel.Array):
    @staticmethod
    def typestr(baretype, parameters):
        encoding = baretype.inner().parameters.get("encoding")
        if encoding is None:
            return "bytes"
        elif codecs.getdecoder(encoding) is codecs.getdecoder("utf-8"):
            return "string"
        else:
            return "string[{0}]".format(repr(encoding))

    def __eq__(self, other):
        raise NotImplementedError("return one boolean per string, not lists of booleans per character")

char = awkward1.layout.DressedType(awkward1.layout.PrimitiveType("uint8"), Char)
utf8 = awkward1.layout.DressedType(awkward1.layout.PrimitiveType("uint8"), Char, encoding="utf-8")

bytestring = awkward1.layout.DressedType(awkward1.layout.ListType(char), String)
string = awkward1.layout.DressedType(awkward1.layout.ListType(utf8), String)
