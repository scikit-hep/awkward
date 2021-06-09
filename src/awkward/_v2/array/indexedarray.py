# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.content import Content
from awkward._v2.index import Index

np = ak.nplike.NumpyMetadata.instance()


class IndexedArray(Content):
    def __init__(self, index, content):
        assert isinstance(index, Index) and index._T in (np.int32, np.uint32, np.int64)
        assert isinstance(content, Content)
        self.index = index
        self.content = content

    def __len__(self):
        return len(self.index)

    def __repr__(self):
        return "IndexedArray(" + repr(self.index) + ", " + repr(self.content) + ")"

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        return self.content[self.index[where]]

    def _getitem_range(self, where):
        return IndexedArray(Index(self.index[where.start : where.stop]), self.content)

    def _getitem_field(self, where):
        return IndexedArray(self.index, self.content[where])

    def _getitem_fields(self, where):
        return IndexedArray(self.index, self.content[where])
