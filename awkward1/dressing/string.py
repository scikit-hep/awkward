# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numpy

import awkward1.highlevel

class String(awkward1.highlevel.Array):
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
