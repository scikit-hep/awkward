# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.content import Content
from awkward._v2.index import Index

np = ak.nplike.NumpyMetadata.instance()


class UnionArray(Content):
    def __init__(self, tags, index, contents):
        assert isinstance(tags, Index) and tags._T == np.int8
        assert isinstance(index, Index) and index._T in (np.int32, np.uint32, np.int64)
        assert isinstance(contents, list)
        assert len(index) >= len(tags)  # usually equal
        self.tags = tags
        self.index = index
        self.contents = contents

    def __len__(self):
        return len(self.tags)

    def __repr__(self):
        return (
            "UnionArray("
            + repr(self.tags)
            + ", "
            + repr(self.index)
            + ", ["
            + ", ".join(repr(x) for x in self.contents)
            + "])"
        )

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        return self.contents[self.tags[where]][self.index[where]]

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        return UnionArray(
            Index(self.tags[start:stop]), Index(self.index[start:stop]), self.contents
        )

    def _getitem_field(self, where):
        return UnionArray(self.tags, self.index, [x[where] for x in self.contents])

    def _getitem_fields(self, where):
        return UnionArray(self.tags, self.index, [x[where] for x in self.contents])
