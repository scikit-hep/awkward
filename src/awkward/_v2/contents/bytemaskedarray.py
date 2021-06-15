# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content
from awkward._v2.index import Index

np = ak.nplike.NumpyMetadata.instance()


class ByteMaskedArray(Content):
    def __init__(self, mask, content, valid_when):
        assert isinstance(mask, Index) and mask.dtype == np.dtype(np.uint8)
        assert isinstance(content, Content)
        assert isinstance(valid_when, bool)
        assert len(mask) <= len(content)
        self._mask = mask
        self._content = content
        self._valid_when = valid_when

    def __len__(self):
        return len(self._mask)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ByteMaskedArray>\n"]
        out.append(
            indent + "    <valid_when>" + str(self._valid_when) + "</valid_when>\n"
        )
        out.append(indent + "    <mask>" + str(self._mask._data) + "</mask>\n")
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</ByteMaskedArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        if self._mask[where] == self._valid_when:
            return self._content[where]
        else:
            return None

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        return ByteMaskedArray(
            Index(self._mask[start:stop]),
            self._content[start:stop],
            valid_when=self._valid_when,
        )

    def _getitem_field(self, where):
        return ByteMaskedArray(
            self._mask, self._content[where], valid_when=self._valid_when
        )

    def _getitem_fields(self, where):
        return ByteMaskedArray(
            self._mask, self._content[where], valid_when=self._valid_when
        )
