# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content
from awkward._v2.index import Index

np = ak.nplike.NumpyMetadata.instance()


class IndexedOptionArray(Content):
    def __init__(self, index, content):
        assert isinstance(index, Index) and index.dtype in (
            np.dtype(np.int32),
            np.dtype(np.int64),
        )
        assert isinstance(content, Content)
        self._index = index
        self._content = content

    def __len__(self):
        return len(self._index)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<IndexedOptionArray>\n"]
        out.append(indent + "    <index>" + str(self._index._data) + "</index>\n")
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</IndexedOptionArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        if self._index[where] < 0:
            return None
        else:
            return self._content[self._index[where]]

    def _getitem_range(self, where):
        return IndexedOptionArray(
            Index(self._index[where.start : where.stop]), self._content
        )

    def _getitem_field(self, where):
        return IndexedOptionArray(self._index, self._content[where])

    def _getitem_fields(self, where):
        return IndexedOptionArray(self._index, self._content[where])
