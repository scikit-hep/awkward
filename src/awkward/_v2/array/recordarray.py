# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import numbers

import awkward as ak
from awkward._v2.content import Content
from awkward._v2.record import Record

np = ak.nplike.NumpyMetadata.instance()


class RecordArray(Content):
    def __init__(self, contents, recordlookup, length=None):
        assert isinstance(contents, list)
        if length is None:
            assert len(contents) != 0
            length = min([len(x) for x in contents])
        assert isinstance(length, numbers.Integral)
        for x in contents:
            assert isinstance(x, Content)
            assert len(x) >= length
        assert recordlookup is None or isinstance(recordlookup, list)
        if isinstance(recordlookup, list):
            assert len(recordlookup) == len(contents)
            for x in recordlookup:
                assert isinstance(x, str)
        self.contents = contents
        self.recordlookup = recordlookup
        self.length = length

    def __len__(self):
        return self.length

    def __repr__(self):
        return (
            "RecordArray(["
            + ", ".join(repr(x) for x in self.contents)
            + "], "
            + repr(self.recordlookup)
            + ", "
            + repr(self.length)
            + ")"
        )

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        record = [x[where] for x in self.contents]
        if self.recordlookup is None:
            return Record(zip((str(x) for x in range(len(record))), record))
        else:
            return Record(zip(self.recordlookup, record))

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        if len(self.contents) == 0:
            start = min(max(start, 0), self.length)
            stop = min(max(stop, 0), self.length)
            if stop < start:
                stop = start
            return RecordArray([], self.recordlookup, stop - start)
        else:
            return RecordArray(
                [x[start:stop] for x in self.contents],
                self.recordlookup,
                stop - start,
            )

    def _getitem_field(self, where):
        if self.recordlookup is None:
            try:
                i = int(where)
            except ValueError:
                pass
            else:
                if i < len(self.contents):
                    return self.contents[i][: len(self)]
        else:
            try:
                i = self.recordlookup.index(where)
            except ValueError:
                pass
            else:
                return self.contents[i][: len(self)]
        raise IndexError("field " + repr(where) + " not found")

    # FIXME should it return a list?
    def _getitem_fields(self, where):
        if self.recordlookup is None:
            try:
                i = int(where)
            except ValueError:
                pass
            else:
                if i < len(self.contents):
                    return self.contents[i][: len(self)]
        else:
            i = []
            try:
                for field in where:
                    i.append(self.recordlookup.index(field))
            except ValueError:
                pass
            else:
                out = []
                for elem in i:
                    out.append(self.contents[elem][: len(self)])
                return out
        raise IndexError("fields " + repr(where) + " not found")
