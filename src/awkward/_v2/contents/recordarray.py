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

    @property
    def contents(self):
        return self._contents

    @property
    def recordlookup(self):
        return self._recordlookup

    @property
    def is_tuple(self):
        return self._recordlookup is None

    def __len__(self):
        return self._length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<RecordArray len="]
        out.append(repr(str(len(self))))
        out.append(" is_tuple=")
        out.append(repr(str(self.is_tuple)))
        out.append(">\n")
        if self._recordlookup is None:
            for i, x in enumerate(self._contents):
                out.append("{0}    <content index={1}>\n".format(indent, repr(str(i))))
                out.append(x._repr(indent + "        ", "", "\n"))
                out.append("{0}    </content>\n".format(indent))
        else:
            for i, x in enumerate(self._contents):
                out.append(
                    "{0}    <content index={1} key={2}>\n".format(
                        indent, repr(str(i)), repr(self._recordlookup[i])
                    )
                )
                out.append(x._repr(indent + "        ", "", "\n"))
                out.append("{0}    </content>\n".format(indent))
        out.append(indent)
        out.append("</RecordArray>")
        out.append(post)
        return "".join(out)

    def index_to_key(self, index):
        if 0 <= index < len(self._contents):
            if self._recordlookup is None:
                return str(index)
            else:
                return self._recordlookup[index]
        else:
            raise IndexError(
                "no index {0} in record with {1} fields".format(
                    index, len(self._contents)
                )
            )

    def key_to_index(self, key):
        if self._recordlookup is None:
            try:
                i = int(key)
            except ValueError:
                pass
            else:
                if 0 <= i < len(self._contents):
                    return i
        else:
            try:
                i = self._recordlookup.index(key)
            except ValueError:
                pass
            else:
                return i
        raise IndexError(
            "no field {0} in record with {1} fields".format(
                repr(key), len(self._contents)
            )
        )

    def content(self, index_or_key):
        if isinstance(index_or_key, numbers.Integral):
            index = index_or_key
        else:
            index = self.key_to_index(index_or_key)
        return self._contents[index][: self._length]

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
        return self.content(where)

    def _getitem_fields(self, where):
        indexes = [self.key_to_index(key) for key in where]
        if self._recordlookup is None:
            return RecordArray([self.content(i) for i in indexes], None)
        else:
            return RecordArray(
                [self.content(i) for i in indexes],
                [self._recordlookup[i] for i in indexes],
            )
