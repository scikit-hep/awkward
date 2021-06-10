# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numbers

import awkward as ak

class Record(object):
    def __init__(self, array, at):
        assert isinstance(array, ak._v2.contents.recordarray.RecordArray)
        # FIXME this is not a correct way to check as len of array is min of the new recordarry but the index can be larger as it's the index of the original RecordArray
        # if 0 > at or at >= len(array):
        #     raise IndexError("array index out of bounds")
        self._array = array
        self._at = at

    @property
    def array(self):
        return self._array

    @property
    def at(self):
        return self._at

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<Record>\n"]
        out.append(indent + "    <at>" + str(self._at) + "</at>\n")
        out.append(self._array._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</Record>")
        out.append(post)
        return "".join(out)

    def __getitem__(self, where):
        if isinstance(where, numbers.Integral) or isinstance(where, slice):
            raise TypeError(
                "scalar Record can only be sliced by field name (string); try "
                + repr(where)
            )
        elif isinstance(where, str):
            return self._array._getitem_field(where)
        elif isinstance(where, Iterable) and all(isinstance(x, str) for x in where):
            return self._array._getitem_fields(where)
        else:
            raise AssertionError(where)
