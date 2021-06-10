# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import numbers

from awkward._v2.contents.content import Content
from awkward._v2.index import Index
from awkward._v2.contents.bytemaskedarray import ByteMaskedArray

import numpy as np


class BitMaskedArray(Content):
    def __init__(self, mask, content, valid_when, length, lsb_order):
        assert isinstance(mask, Index) and mask._T == np.uint8
        assert isinstance(content, Content)
        assert isinstance(valid_when, bool)
        assert isinstance(length, numbers.Integral) and length >= 0
        assert isinstance(lsb_order, bool)
        assert len(mask) <= len(content)
        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._length = length
        self._lsb_order = lsb_order

    def __len__(self):
        return self._length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<BitMaskedArray>\n"]
        out.append(
            indent + "    <valid_when>" + str(self._valid_when) + "</valid_when>\n"
        )
        out.append(indent + "    <length>" + str(self._length) + "</valid_when>\n")
        out.append(indent + "    <lsb_order>" + str(self._lsb_order) + "</lsb_order>\n")
        out.append(indent + "    <mask>" + str(self._mask._data) + "</mask>\n")
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</BitMaskedArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        if self._lsb_order:
            bit = bool(self._mask[where // 8] & (1 << (where % 8)))
        else:
            bit = bool(self._mask[where // 8] & (128 >> (where % 8)))
        if bit == self._valid_when:
            return self._content[where]
        else:
            return None

    def _getitem_range(self, where):
        # In general, slices must convert BitMaskedArray to ByteMaskedArray.
        if self._lsb_order:
            bytemask = np.unpackbits(self._mask).reshape(-1, 8)[:, ::-1].reshape(-1)
        else:
            bytemask = np.unpackbits(self._mask)
        start, stop, step = where.indices(len(self))
        return ByteMaskedArray(
            Index(bytemask[start:stop]),
            self._content[start:stop],
            valid_when=self._valid_when,
        )

    def _getitem_field(self, where):
        return BitMaskedArray(
            self._mask,
            self._content[where],
            valid_when=self._valid_when,
            length=self._length,
            lsb_order=self._lsb_order,
        )

    def _getitem_fields(self, where):
        return BitMaskedArray(
            self._mask,
            self._content[where],
            valid_when=self._valid_when,
            length=self._length,
            lsb_order=self._lsb_order,
        )
