# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import awkward1.highlevel

class String(awkward1.highlevel.Array):
    def __str__(self):
        return str("".join(chr(x) for x in self))

    def __repr__(self):
        return repr("".join(chr(x) for x in self))
