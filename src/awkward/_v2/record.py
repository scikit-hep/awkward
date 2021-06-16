# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numbers

import awkward as ak
from awkward._v2.contents.content import Content


np = ak.nplike.NumpyMetadata.instance()


class Record(object):
    def __init__(self, array, at):
        assert isinstance(array, ak._v2.contents.recordarray.RecordArray)
        if 0 <= at < len(array):
            self._array = array
            self._at = at
        else:
            raise ValueError(
                "Record at={0} for array of length {1}".format(at, len(array))
            )

    @property
    def array(self):
        return self._array

    @property
    def at(self):
        return self._at

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<Record at="]
        out.append(repr(str(self._at)))
        out.append(">\n")
        out.append(self._array._repr(indent + "    ", "<array>", "</array>\n"))
        out.append(indent)
        out.append("</Record>")
        out.append(post)
        return "".join(out)

    def __getitem__(self, where):
        if isinstance(where, numbers.Integral):
            raise IndexError("scalar Record cannot be sliced by an integer")

        elif isinstance(where, slice):
            raise IndexError("scalar Record cannot be sliced by a range slice (`:`)")

        elif isinstance(where, str):
            return self._getitem_field(where)

        elif where is np.newaxis:
            raise IndexError("scalar Record cannot be sliced by np.newaxis (`None`)")

        elif where is Ellipsis:
            raise IndexError("scalar Record cannot be sliced by an ellipsis (`...`)")

        elif isinstance(where, tuple):
            raise NotImplementedError("needs _getitem_next")

        elif isinstance(where, ak.highlevel.Array):
            raise IndexError("scalar Record cannot be sliced by an array")

        elif isinstance(where, Content):
            raise IndexError("scalar Record cannot be sliced by an array")

        elif isinstance(where, Iterable) and all(isinstance(x, str) for x in where):
            return self._getitem_fields(where)

        elif isinstance(where, Iterable):
            raise IndexError("scalar Record cannot be sliced by an array")

        else:
            raise TypeError(
                "only field name (str) or names (non-tuple iterable of str) "
                "are valid indices for slicing a scalar record, not\n\n    "
                + repr(where)
            )

    def _getitem_field(self, where):
        return self._array._getitem_field(where)._getitem_at(self._at)

    def _getitem_fields(self, where):
        return self._array._getitem_fields(where)._getitem_at(self._at)
