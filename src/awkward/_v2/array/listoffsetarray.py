# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.content import Content
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
        self.offsets = offsets
        self.content = content

    def __len__(self):
        return len(self.offsets) - 1

    def __repr__(self):
        return "ListOffsetArray(" + repr(self.offsets) + ", " + repr(self.content) + ")"

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        return self.content[self.offsets[where] : self.offsets[where + 1]]

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        offsets = self.offsets[start : stop + 1]
        if len(offsets) == 0:
            offsets = [0]
        return ListOffsetArray(Index(offsets), self.content)

    def _getitem_field(self, where):
        return ListOffsetArray(self.offsets, self.content[where])

    def _getitem_fields(self, where):
        return ListOffsetArray(self.offsets, self.content[where])
