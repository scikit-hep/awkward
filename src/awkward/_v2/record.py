# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2.contents.content import Content

np = ak.nplike.NumpyMetadata.instance()


class Record(object):
    def __init__(self, array, at):
        if not isinstance(array, ak._v2.contents.recordarray.RecordArray):
            raise TypeError(
                "Record 'array' must be a RecordArray, not {0}".format(repr(array))
            )
        if not ak._util.isint(at):
            raise TypeError(
                "Record 'at' must be an integer, not {0}".format(repr(array))
            )
        if 0 <= at < len(array):
            self._array = array
            self._at = at
        else:
            raise ValueError(
                "Record 'at' must be >= 0 and < len(array) == {0}, not {1}".format(
                    len(array), at
                )
            )

    @property
    def array(self):
        return self._array

    @property
    def at(self):
        return self._at

    @property
    def fields(self):
        return self._array.fields

    @property
    def is_tuple(self):
        return self._array.is_tuple

    @property
    def as_tuple(self):
        return Record(self._array.as_tuple, self._at)

    @property
    def contents(self):
        out = []
        for field in self._array.fields:
            out.append(self._array[field][self._at])
        return out

    def content(self, index_or_field):
        return self._array.content(index_or_field)[self._at]

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

    def validityerror(self, path="layout.array"):
        return self._array.validityerror(path)

    @property
    def parameters(self):
        return self._array.parameters

    def parameter(self, key):
        return self._array.parameter(key)

    def purelist_parameter(self, key):
        return self._array.purelist_parameter(key)

    @property
    def purelist_isregular(self):
        return self._array.purelist_isregular

    @property
    def purelist_depth(self):
        return 0

    @property
    def minmax_depth(self):
        mindepth, maxdepth = self._array.minmax_depth
        return mindepth - 1, maxdepth - 1

    @property
    def branch_depth(self):
        branch, depth = self._array.branch_depth
        return branch, depth - 1

    def __getitem__(self, where):
        if ak._util.isint(where):
            raise IndexError("scalar Record cannot be sliced by an integer")

        elif isinstance(where, slice):
            raise IndexError("scalar Record cannot be sliced by a range slice (`:`)")

        elif ak._util.isstr(where):
            return self._getitem_field(where)

        elif where is np.newaxis:
            raise IndexError("scalar Record cannot be sliced by np.newaxis (`None`)")

        elif where is Ellipsis:
            raise IndexError("scalar Record cannot be sliced by an ellipsis (`...`)")

        elif isinstance(where, tuple) and len(where) == 0:
            return self

        elif isinstance(where, tuple) and len(where) == 1:
            return self.__getitem__(where[0])

        elif isinstance(where, tuple) and ak._util.isstr(where[0]):
            return self._getitem_field(where[0]).__getitem__(where[1:])

        elif isinstance(where, ak.highlevel.Array):
            raise IndexError("scalar Record cannot be sliced by an array")

        elif isinstance(where, ak.layout.Content):
            raise IndexError("scalar Record cannot be sliced by an array")

        elif isinstance(where, Content):
            raise IndexError("scalar Record cannot be sliced by an array")

        elif isinstance(where, Iterable) and all(ak._util.isstr(x) for x in where):
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

    def packed(self):
        if len(self._array) == 1:
            return Record(self._array.packed(), self._at)
        else:
            return Record(self._array[self._at : self._at + 1].packed(), 0)

    def to_list(self, behavior=None):
        cls = ak._v2._util.recordclass(self._array, behavior)
        if cls is not ak._v2.highlevel.Record:
            return cls(self)

        return self._array[self._at : self._at + 1].to_list(behavior)[0]
