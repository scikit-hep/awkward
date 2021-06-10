# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content
from awkward._v2.index import Index

np = ak.nplike.NumpyMetadata.instance()


class ListOffsetArray(Content):
    def __init__(self, offsets, content):
        assert isinstance(offsets, Index) and offsets._T in (
            np.int32,
            np.uint32,
            np.int64,
        )
        assert isinstance(content, Content)
        assert len(offsets) != 0
        self._offsets = offsets
        self._content = content

    def __len__(self):
        return len(self._offsets) - 1

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ListOffsetArray>\n"]
        out.append(indent + "    <offsets>" + " ".join(str(x) for x in self._offsets) + "</offsets>\n")
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</ListOffsetArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        return self._content[self._offsets[where] : self._offsets[where + 1]]

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        offsets = self._offsets[start : stop + 1]
        if len(offsets) == 0:
            offsets = [0]
        return ListOffsetArray(Index(offsets), self._content)

    def _getitem_field(self, where):
        return ListOffsetArray(self._offsets, self._content[where])

    def _getitem_fields(self, where):
        return ListOffsetArray(self._offsets, self._content[where])
