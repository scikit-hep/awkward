# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.content import Content
from awkward._v2.index import Index

np = ak.nplike.NumpyMetadata.instance()


class ListArray(Content):
    def __init__(self, starts, stops, content):
        assert isinstance(starts, Index) and starts._T in (
            np.int32,
            np.uint32,
            np.int64,
        )
        assert isinstance(stops, Index) and starts._T == stops._T
        assert isinstance(content, Content)
        assert len(stops) >= len(starts)  # usually equal
        self.starts = starts
        self.stops = stops
        self.content = content

    def __len__(self):
        return len(self.starts)

    def __repr__(self):
        return (
            "ListArray("
            + repr(self.starts)
            + ", "
            + repr(self.stops)
            + ", "
            + repr(self.content)
            + ")"
        )

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        return self.content[self.starts[where] : self.stops[where]]

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        starts = Index(self.starts[start:stop])
        stops = Index(self.stops[start:stop])
        return ListArray(starts, stops, self.content)

    def _getitem_field(self, where):
        return ListArray(self.starts, self.stops, self.content[where])

    def _getitem_fields(self, where):
        return ListArray(self.starts, self.stops, self.content[where])
