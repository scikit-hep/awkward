# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content
from awkward._v2.index import Index

np = ak.nplike.NumpyMetadata.instance()


class UnionArray(Content):
    def __init__(self, tags, index, contents):
        assert isinstance(tags, Index) and tags._T == np.int8
        assert isinstance(index, Index) and index._T in (np.int32, np.uint32, np.int64)
        assert isinstance(contents, list)
        assert len(index) >= len(tags)  # usually equal
        self._tags = tags
        self._index = index
        self._contents = contents

    def __len__(self):
        return len(self._tags)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<UnionArray>\n"]
        out.append(indent + "    <tags>" + str(self._tags._data) + "</tags>\n")
        out.append(indent + "    <index>" + str(self._index._data) + "</index>\n")
        for x in self._contents:
            out.append(x._repr(indent + "    ", "<content i=\"" + repr(self._contents.index(x)) + "\">", "</content>\n"))
        out.append(indent)
        out.append("</UnionArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        return self._contents[self._tags[where]][self._index[where]]

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        return UnionArray(
            Index(self._tags[start:stop]), Index(self._index[start:stop]), self._contents
        )

    def _getitem_field(self, where):
        return UnionArray(self._tags, self._index, [x[where] for x in self._contents])

    def _getitem_fields(self, where):
        return UnionArray(self._tags, self._index, [x[where] for x in self._contents])
