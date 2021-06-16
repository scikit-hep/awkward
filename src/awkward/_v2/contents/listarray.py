# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content
from awkward._v2.index import Index

np = ak.nplike.NumpyMetadata.instance()


class ListArray(Content):
    def __init__(self, starts, stops, content):
        assert isinstance(starts, Index) and starts.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        )
        assert isinstance(stops, Index) and starts.dtype == stops.dtype
        assert isinstance(content, Content)
        assert len(stops) >= len(starts)  # usually equal
        self._starts = starts
        self._stops = stops
        self._content = content

    @property
    def starts(self):
        return self._starts

    @property
    def stops(self):
        return self._stops

    @property
    def content(self):
        return self._content

    def __len__(self):
        return len(self._starts)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ListArray len="]
        out.append(repr(str(len(self))))
        out.append(">\n")
        out.append(self._starts._repr(indent + "    ", "<starts>", "</starts>\n"))
        out.append(self._stops._repr(indent + "    ", "<stops>", "</stops>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</ListArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        return self._content[self._starts[where] : self._stops[where]]

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        starts = Index(self._starts[start:stop])
        stops = Index(self._stops[start:stop])
        return ListArray(starts, stops, self._content)

    def _getitem_field(self, where):
        return ListArray(self._starts, self._stops, self._content[where])

    def _getitem_fields(self, where):
        return ListArray(self._starts, self._stops, self._content[where])
