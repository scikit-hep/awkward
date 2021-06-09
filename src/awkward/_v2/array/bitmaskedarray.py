# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import numbers

from awkward._v2.content import Content
from awkward._v2.index import Index
from awkward._v2.array.bytemaskedarray import ByteMaskedArray

import numpy as np


class BitMaskedArray(Content):
    def __init__(self, mask, content, valid_when, length, lsb_order):
        assert isinstance(mask, Index) and mask._T == np.uint8
        assert isinstance(content, Content)
        assert isinstance(valid_when, bool)
        assert isinstance(length, numbers.Integral) and length >= 0
        assert isinstance(lsb_order, bool)
        assert len(mask) <= len(content)
        self.mask = mask
        self.content = content
        self.valid_when = valid_when
        self.length = length
        self.lsb_order = lsb_order

    def __len__(self):
        return self.length

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        if self.lsb_order:
            bit = bool(self.mask[where // 8] & (1 << (where % 8)))
        else:
            bit = bool(self.mask[where // 8] & (128 >> (where % 8)))
        if bit == self.valid_when:
            return self.content[where]
        else:
            return None

    def _getitem_range(self, where):
        # In general, slices must convert BitMaskedArray to ByteMaskedArray.
        # FIXME this will return an array of bools, but now the first argument is of type Index and not List as before and Index doesn't have bool as an accepted data type
        bytemask = np.unpackbits(
            self.mask, bitorder=("little" if self.lsb_order else "big")
        ).view(np.bool_)
        start, stop, step = where.indices(len(self))
        return ByteMaskedArray(
            Index(bytemask[start:stop]),
            self.content[start:stop],
            valid_when=self.valid_when,
        )

    def _getitem_field(self, where):
        return BitMaskedArray(
            self.mask,
            self.content[where],
            valid_when=self.valid_when,
            length=self.length,
            lsb_order=self.lsb_order,
        )

    def _getitem_fields(self, where):
        return BitMaskedArray(
            self.mask,
            self.content[where],
            valid_when=self.valid_when,
            length=self.length,
            lsb_order=self.lsb_order,
        )
