# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2.contents.content import Content
from awkward._v2.record import Record


class RecordArray(Content):
    def __init__(self, contents, keys, length=None, identifier=None, parameters=None):
        if not isinstance(contents, Iterable):
            raise TypeError(
                "{0} 'contents' must be iterable, not {1}".format(
                    type(self).__name__, repr(contents)
                )
            )
        if not isinstance(contents, list):
            contents = list(contents)

        if len(contents) == 0 and length is None:
            raise TypeError(
                "{0} if len(contents) == 0, a 'length' must be specified".format(
                    type(self).__name__
                )
            )
        elif length is None:
            length = min(len(x) for x in contents)
        if not (ak._util.isint(length) and length >= 0):
            raise TypeError(
                "{0} 'length' must be a non-negative integer or None, not {1}".format(
                    type(self).__name__, repr(length)
                )
            )
        for content in contents:
            if not isinstance(content, Content):
                raise TypeError(
                    "{0} all 'contents' must be Content subclasses, not {1}".format(
                        type(self).__name__, repr(content)
                    )
                )
            if not len(content) >= length:
                raise ValueError(
                    "{0} len(content) ({1}) must be <= length ({2}) for all 'contents'".format(
                        type(self).__name__, len(content), length
                    )
                )

        if isinstance(keys, Iterable):
            if not isinstance(keys, list):
                keys = list(keys)
            if not all(ak._util.isstr(x) for x in keys):
                raise TypeError(
                    "{0} 'keys' must all be strings, not {1}".format(
                        type(self).__name__, repr(keys)
                    )
                )
            if not len(contents) == len(keys):
                raise ValueError(
                    "{0} len(contents) ({1}) must be equal to len(keys) ({2})".format(
                        type(self).__name__, len(contents), len(keys)
                    )
                )
        elif keys is not None:
            raise TypeError(
                "{0} 'keys' must be iterable or None, not {1}".format(
                    type(self).__name__, repr(keys)
                )
            )

        self._contents = contents
        self._keys = keys
        self._length = length
        self._init(identifier, parameters)

    @property
    def contents(self):
        return self._contents

    @property
    def keys(self):
        return self._keys

    @property
    def is_tuple(self):
        return self._keys is None

    @property
    def form(self):
        return ak._v2.forms.RecordForm(
            [x.form for x in self._contents],
            self._keys,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

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
        if self._keys is None:
            for i, x in enumerate(self._contents):
                out.append("{0}    <content index={1}>\n".format(indent, repr(str(i))))
                out.append(x._repr(indent + "        ", "", "\n"))
                out.append("{0}    </content>\n".format(indent))
        else:
            for i, x in enumerate(self._contents):
                out.append(
                    "{0}    <content index={1} key={2}>\n".format(
                        indent, repr(str(i)), repr(self._keys[i])
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
            if self._keys is None:
                return str(index)
            else:
                return self._keys[index]
        else:
            raise IndexError(
                "no index {0} in record with {1} fields".format(
                    index, len(self._contents)
                )
            )

    def key_to_index(self, key):
        if self._keys is None:
            try:
                i = int(key)
            except ValueError:
                pass
            else:
                if 0 <= i < len(self._contents):
                    return i
        else:
            try:
                i = self._keys.index(key)
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
        if ak._util.isint(index_or_key):
            index = index_or_key
        elif ak._util.isstr(index_or_key):
            index = self.key_to_index(index_or_key)
        else:
            raise TypeError(
                "index_or_key must be an integer (index) or string (key), not {0}".format(
                    repr(index_or_key)
                )
            )
        return self._contents[index][: self._length]

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        return Record(self, where)

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        if len(self._contents) == 0:
            start = min(max(start, 0), self._length)
            stop = min(max(stop, 0), self._length)
            if stop < start:
                stop = start
            return RecordArray([], self._keys, stop - start)
        else:
            return RecordArray(
                [x[start:stop] for x in self._contents],
                self._keys,
                stop - start,
            )

    def _getitem_field(self, where):
        return self.content(where)

    def _getitem_fields(self, where):
        indexes = [self.key_to_index(key) for key in where]
        if self._keys is None:
            return RecordArray([self.content(i) for i in indexes], None)
        else:
            return RecordArray(
                [self.content(i) for i in indexes],
                [self._keys[i] for i in indexes],
            )
