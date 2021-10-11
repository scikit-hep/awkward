# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2.record import Record
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.recordform import RecordForm

np = ak.nplike.NumpyMetadata.instance()


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
        if self._keys is None:
            return [str(i) for i in range(len(self._contents))]
        else:
            return self._keys

    @property
    def is_tuple(self):
        return self._keys is None

    @property
    def as_tuple(self):
        return RecordArray(
            self._contents,
            None,
            length=self._length,
            identifier=None,
            parameters=None,
        )

    @property
    def nplike(self):
        for content in self._contents:
            nplike = content.nonvirtual_nplike
            if nplike is not None:
                return nplike
        else:
            for content in self._contents:
                return content.nplike
            else:
                return ak.nplike.Numpy.instance()

    @property
    def nonvirtual_nplike(self):
        for content in self._contents:
            nplike = content.nonvirtual_nplike
            if nplike is not None:
                return nplike
        else:
            return None

    Form = RecordForm

    @property
    def form(self):
        return self.Form(
            [x.form for x in self._contents],
            self._keys,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    @property
    def typetracer(self):
        return RecordArray(
            [x.typetracer for x in self._contents],
            self._keys,
            self._length,
            self._typetracer_identifier(),
            self._parameters,
        )

    def __len__(self):
        return self._length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<RecordArray is_tuple="]
        out.append(repr(json.dumps(self.is_tuple)))
        out.append(" len=")
        out.append(repr(str(len(self))))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")

        if self._keys is None:
            for i, x in enumerate(self._contents):
                out.append("{0}    <content index={1}>\n".format(indent, repr(str(i))))
                out.append(x._repr(indent + "        ", "", "\n"))
                out.append(indent + "    </content>\n")
        else:
            for i, x in enumerate(self._contents):
                out.append(
                    "{0}    <content index={1} key={2}>\n".format(
                        indent, repr(str(i)), repr(self._keys[i])
                    )
                )
                out.append(x._repr(indent + "        ", "", "\n"))
                out.append(indent + "    </content>\n")

        out.append(indent + "</RecordArray>")
        out.append(post)
        return "".join(out)

    def index_to_key(self, index):
        return self.Form.index_to_key(self, index)

    def key_to_index(self, key):
        return self.Form.key_to_index(self, key)

    def haskey(self, key):
        return self.Form.haskey(self, key)

    def content(self, index_or_key):
        return self.Form.content(self, index_or_key)[: self._length]

    def _getitem_nothing(self):
        return self._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)) and self.nplike.known_shape:
            raise NestedIndexError(self, where)
        return Record(self, where)

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        if len(self._contents) == 0:
            start = min(max(start, 0), self._length)
            stop = min(max(stop, 0), self._length)
            if stop < start:
                stop = start
            return RecordArray(
                [],
                self._keys,
                stop - start,
                self._range_identifier(start, stop),
                self._parameters,
            )
        else:
            nextslice = slice(start, stop)
            return RecordArray(
                [x._getitem_range(nextslice) for x in self._contents],
                self._keys,
                stop - start,
                self._range_identifier(start, stop),
                self._parameters,
            )

    def _getitem_field(self, where, only_fields=()):
        if len(only_fields) == 0:
            return self.content(where)

        else:
            nexthead, nexttail = ak._v2._slicing.headtail(only_fields)
            if ak._util.isstr(nexthead):
                return self.content(where)._getitem_field(nexthead, nexttail)
            else:
                return self.content(where)._getitem_fields(nexthead, nexttail)

    def _getitem_fields(self, where, only_fields=()):
        indexes = [self.key_to_index(key) for key in where]
        if self._keys is None:
            keys = None
        else:
            keys = [self._keys[i] for i in indexes]

        if len(only_fields) == 0:
            contents = [self.content(i) for i in indexes]
        else:
            nexthead, nexttail = ak._v2._slicing.headtail(only_fields)
            if ak._util.isstr(nexthead):
                contents = [
                    self.content(i)._getitem_field(nexthead, nexttail) for i in indexes
                ]
            else:
                contents = [
                    self.content(i)._getitem_fields(nexthead, nexttail) for i in indexes
                ]

        return RecordArray(
            contents,
            keys,
            self._fields_identifier(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        if allow_lazy:
            nplike, where = carry.nplike, carry.data

            if allow_lazy != "copied":
                where = where.copy()

            negative = where < 0
            if nplike.any(negative, prefer=False):
                where[negative] += self._length

            if nplike.any(where >= self._length, prefer=False):
                if issubclass(exception, NestedIndexError):
                    raise exception(self, where)
                else:
                    raise exception("index out of range")

            nextindex = ak._v2.index.Index64(where)
            return ak._v2.contents.indexedarray.IndexedArray(
                nextindex,
                self,
                self._carry_identifier(carry, exception),
                None,
            )

        else:
            contents = [
                self.content(i)._carry(carry, allow_lazy, exception)
                for i in range(len(self._contents))
            ]

            # if issubclass(carry.dtype.type, np.integer):
            #     length = len(carry)
            # else:
            #     length = len(self)
            assert issubclass(carry.dtype.type, np.integer)
            length = len(carry)

            return RecordArray(
                contents,
                self._keys,
                length,
                self._carry_identifier(carry, exception),
                self._parameters,
            )

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        contents = []
        for i in range(len(self._contents)):
            contents.append(
                self.content(i)._getitem_next_jagged(
                    slicestarts, slicestops, slicecontent, tail
                )
            )
        return RecordArray(contents, self._keys, self._length)

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            nexthead, nexttail = ak._v2._slicing.headtail(tail)

            contents = []
            for i in range(len(self._contents)):
                contents.append(self.content(i)._getitem_next(head, (), advanced))

            parameters = None
            if (
                isinstance(
                    head,
                    (
                        slice,
                        ak._v2.contents.ListOffsetArray,
                        ak._v2.contents.IndexedOptionArray,
                    ),
                )
                or head is Ellipsis
                or advanced is None
            ):
                parameters = self._parameters
            next = RecordArray(contents, self._keys, None, self._identifier, parameters)
            return next._getitem_next(nexthead, nexttail, advanced)

    def _localindex(self, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        else:
            contents = []
            for content in self._contents:
                contents.append(content._localindex(posaxis, depth))
            return RecordArray(
                contents, self._keys, len(self), self._identifier, self._parameters
            )

    def _argsort_next(
        self,
        negaxis,
        starts,
        shifts,
        parents,
        outlength,
        ascending,
        stable,
        kind,
        order,
    ):
        if self._keys is None or len(self._keys) == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))
        else:
            raise NotImplementedError

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        if self._keys is None or len(self._keys) == 0:
            return ak._v2.contents.NumpyArray(self.nplike.instance().empty(0, np.int64))

        contents = []
        for content in self._contents:
            contents.append(
                content._sort_next(
                    negaxis,
                    starts,
                    parents,
                    outlength,
                    ascending,
                    stable,
                    kind,
                    order,
                )
            )
        return RecordArray(contents, self._keys, len(self), None, self._parameters)

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            contents = []
            for content in self._contents:
                contents.append(
                    content._combinations(
                        n, replacement, recordlookup, parameters, posaxis, depth
                    )
                )
            return ak._v2.contents.recordarray.RecordArray(
                contents, recordlookup, len(self), self._identifier, self._parameters
            )

    def _reduce_next(
        self,
        reducer,
        negaxis,
        starts,
        shifts,
        parents,
        outlength,
        mask,
        keepdims,
    ):
        contents = []
        for content in self._contents:
            contents.append(
                content[: self._length]._reduce_next(
                    reducer,
                    negaxis,
                    starts,
                    shifts,
                    parents,
                    outlength,
                    mask,
                    keepdims,
                )
            )

        return ak._v2.contents.RecordArray(
            contents,
            self._keys,
            outlength,
            None,
            None,
        )

    def _validityerror(self, path):
        for i in range(len(self.contents)):
            if len(self.contents[i]) < len(self):
                return 'at {0} ("{1}"): len(field({2})) < len(recordarray)'.format(
                    path, type(self), i
                )
        for i in range(len(self.contents)):
            sub = self.contents[i].validityerror(path + ".field({0})".format(i))
            if sub != "":
                return sub
        return ""
