# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.content import Content
from awkward._v2.index import Index

np = ak.nplike.NumpyMetadata.instance()


class ByteMaskedArray(Content):
    def __init__(self, mask, content, valid_when):
        assert isinstance(mask, Index) and mask._T == np.int8
        assert isinstance(content, Content)
        assert isinstance(valid_when, bool)
        assert len(mask) <= len(content)
        self.mask = mask
        self.content = content
        self.valid_when = valid_when

    def __len__(self):
        return len(self.mask)

    # def __repr__(self):
    #     return (
    #         "ByteMaskedArray("
    #         + repr(self.mask)
    #         + ", "
    #         + repr(self.content)
    #         + ", "
    #         + repr(self.valid_when)
    #         + ")"
    #     )

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        if self.mask[where] == self.valid_when:
            return self.content[where]
        else:
            return None

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        return ByteMaskedArray(
            Index(self.mask[start:stop]),
            self.content[start:stop],
            valid_when=self.valid_when,
        )

    def _getitem_field(self, where):
        return ByteMaskedArray(
            self.mask, self.content[where], valid_when=self.valid_when
        )

    def _getitem_fields(self, where):
        return ByteMaskedArray(
            self.mask, self.content[where], valid_when=self.valid_when
        )
