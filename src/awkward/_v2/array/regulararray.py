# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import numbers

from awkward._v2.content import Content


class RegularArray(Content):
    def __init__(self, content, size, zeros_length=0):
        assert isinstance(content, Content)
        assert isinstance(size, numbers.Integral)
        assert isinstance(zeros_length, numbers.Integral)
        assert size >= 0
        if size != 0:
            length = len(content) // size  # floor division
        else:
            assert zeros_length >= 0
            length = zeros_length
        self.content = content
        self.size = size
        self.length = length

    def __len__(self):
        return self.length

    def __repr__(self):
        if self.size == 0:
            zeros_length = ", " + repr(self.length)
        else:
            zeros_length = ""
        return (
            "RegularArray("
            + repr(self.content)
            + ", "
            + repr(self.size)
            + zeros_length
            + ")"
        )

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        return self.content[(where) * self.size : (where + 1) * self.size]

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        zeros_length = stop - start
        start *= self.size
        stop *= self.size
        return RegularArray(self.content[start:stop], self.size, zeros_length)

    def _getitem_field(self, where):
        return RegularArray(self.content[where], self.size, self.length)

    def _getitem_fields(self, where):
        return RegularArray(self.content[where], self.size, self.length)
