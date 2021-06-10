# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import numbers

import numpy as np

from awkward._v2.contents.content import Content
from awkward._v2.record import Record

class RecordArray(Content):
    def __init__(self, contents, recordlookup, length=None):
        assert isinstance(contents, list)
        if length is None:
            assert len(contents) != 0
            length = min([len(x) for x in contents])
        assert isinstance(length, numbers.Integral)
        for x in contents:
            # FIXME this conflicts with return of recordarray and building Content
            # assert isinstance(x, Content)
            assert len(x) >= length
        assert recordlookup is None or isinstance(recordlookup, list)
        if isinstance(recordlookup, list):
            assert len(recordlookup) == len(contents)
            for x in recordlookup:
                assert isinstance(x, str)
        self._contents = contents
        self._recordlookup = recordlookup
        self._length = length

    def __len__(self):
        return self._length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<RecordArray>\n"]
        out.append(indent + "    <length>" + str(self._length) + "</length>\n")
        for x in self._contents:
            out.append(x._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</RecordArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        record = [np.asarray([x[where]]) for x in self._contents]
        length = min([len(x) for x in record]) if len(record) > 0 else 0
        if self._recordlookup is None:
            return Record(
                RecordArray(record, [str(x) for x in range(len(record))], length), where
            )
        else:
            return Record(RecordArray(record, self._recordlookup, length), where)

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        if len(self._contents) == 0:
            start = min(max(start, 0), self._length)
            stop = min(max(stop, 0), self._length)
            if stop < start:
                stop = start
            return RecordArray([], self._recordlookup, stop - start)
        else:
            return RecordArray(
                [x[start:stop] for x in self._contents],
                self._recordlookup,
                stop - start,
            )

    def _getitem_field(self, where):
        if self._recordlookup is None:
            try:
                i = int(where)
            except ValueError:
                pass
            else:
                if i < len(self._contents):
                    return self._contents[i][: len(self)]
        else:
            try:
                i = self._recordlookup.index(where)
            except ValueError:
                pass
            else:
                return self._contents[i][: len(self)]
        raise IndexError("field " + repr(where) + " not found")

    def _getitem_fields(self, where):
        if self._recordlookup is None:
            try:
                i = [int(x) for x in where]
            except ValueError:
                pass
            else:
                out = [
                    self._contents[elem][: len(self)]
                    for elem in i
                    if elem < len(self._contents)
                ]
                return RecordArray(out, list(where), len(where))
        else:
            try:
                i = [self._recordlookup.index(field) for field in where]
            except ValueError:
                pass
            else:
                out = [self._contents[elem][: len(self)] for elem in i]
                return RecordArray(out, list(where), len(where))
        raise IndexError("fields " + repr(where) + " not found")
