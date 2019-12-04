# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import codecs

import numpy

import awkward1.highlevel

class String(awkward1.highlevel.Array):
    @staticmethod
    def typestr(parameters):
        encoding = parameters.get("encoding")
        if encoding is None:
            return "bytes"
        elif codecs.getdecoder(encoding) is codecs.getdecoder("utf-8"):
            return "string"
        else:
            return "string[{0}]".format(repr(encoding))

    def __str__(self):
        out = numpy.asarray(self.layout).tostring()
        encoding = self.type.parameters.get("encoding")
        if encoding is None:
            return str(out)
        else:
            return out.decode(encoding)

    def __repr__(self):
        out = numpy.asarray(self.layout).tostring()
        encoding = self.type.parameters.get("encoding")
        if encoding is None:
            return repr(out)
        else:
            return repr(out.decode(encoding))
