# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

from awkward._v2.contents.content import Content


class EmptyArray(Content):
    def __init__(self):
        pass

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        return indent + pre + "<EmptyArray/>" + post

    def __len__(self):
        return 0

    def _getitem_at(self, where):
        raise IndexError("array of type Empty has no index " + repr(where))

    def _getitem_range(self, where):
        return EmptyArray()

    def _getitem_field(self, where):
        raise IndexError("field " + repr(where) + " not found")

    def _getitem_fields(self, where):
        raise IndexError("fields " + repr(where) + " not found")
