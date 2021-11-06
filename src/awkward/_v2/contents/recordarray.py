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
from awkward._v2.forms.form import _parameters_equal

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
            return content.nplike
        else:
            return ak.nplike.Numpy.instance()

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

    def mergeable(self, other, mergebool=True):
        if not _parameters_equal(self._parameters, other._parameters):
            return False

        if isinstance(
            other,
            (
                ak._v2.contents.emptyarray.EmptyArray,
                ak._v2.contents.unionarray.UnionArray,
            ),
        ):
            return True

        elif isinstance(
            other,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):
            return self.mergeable(other.content, mergebool)

        if isinstance(other, ak._v2.contents.recordarray.RecordArray):
            if self.is_tuple and other.is_tuple:
                if len(self.contents) == len(other.contents):
                    for i in range(len(self.contents)):
                        if not self.contents[i].mergeable(other.contents[i], mergebool):
                            return False
                    return True

            elif not self.is_tuple and not other.is_tuple:
                self_keys = self.keys.copy()
                other_keys = other.keys.copy()
                self_keys.sort()
                other_keys.sort()
                if self_keys == other_keys:
                    for key in self_keys:
                        if not self[key].mergeable(other[key], mergebool):
                            return False
                    return True
            return False
        else:
            return False

    def mergemany(self, others):
        if len(others) == 0:
            return self

        head, tail = self._merging_strategy(others)

        parameters = {}
        headless = head[1:]

        for_each_field = []
        for field in self.contents:
            trimmed = field[0 : len(self)]
            for_each_field.append([field])

        if self.is_tuple:
            for array in headless:
                parameters = dict(self.parameters.items() & array.parameters.items())

                if isinstance(array, ak._v2.contents.recordarray.RecordArray):
                    if self.is_tuple:
                        if len(self.contents) == len(array.contents):
                            for i in range(len(self.contents)):
                                field = array[self.index_to_key(i)]
                                for_each_field[i].append(field[0 : len(array)])
                        else:
                            raise ValueError(
                                "cannot merge tuples with different numbers of fields"
                            )
                    else:
                        raise ValueError("cannot merge tuple with non-tuple record")
                elif isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                    pass
                else:
                    raise AssertionError(
                        "cannot merge "
                        + type(self).__name__
                        + " with "
                        + type(array).__name__
                    )

        else:
            these_keys = self.keys.copy()
            these_keys.sort()

            for array in headless:
                if isinstance(array, ak._v2.contents.recordarray.RecordArray):
                    if not array.is_tuple:
                        those_keys = array.keys.copy()
                        those_keys.sort()

                        if these_keys == those_keys:
                            for i in range(len(self.contents)):
                                field = array[self.index_to_key(i)]

                                trimmed = field[0 : len(array)]
                                for_each_field[i].append(trimmed)
                        else:
                            raise AssertionError(
                                "cannot merge records with different sets of field names"
                            )
                    else:
                        raise AssertionError("cannot merge non-tuple record with tuple")

                elif isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                    pass
                else:
                    raise AssertionError(
                        "cannot merge "
                        + type(self).__name__
                        + " with "
                        + type(array).__name__
                    )

        nextcontents = []
        minlength = -1

        for forfield in for_each_field:
            tail_forfield = forfield[1:]
            merged = forfield[0].mergemany(tail_forfield)

            nextcontents.append(merged)

            if minlength == -1 or len(merged) < minlength:
                minlength = len(merged)

        next = RecordArray(nextcontents, self._keys, minlength, None, parameters)

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed.mergemany(tail[1:])

        raise NotImplementedError(
            "not implemented: " + type(self).__name__ + " ::mergemany"
        )

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
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
        posaxis = self.axis_wrap_if_negative(axis)
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

    def _rpad(self, target, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.rpad_axis0(target, False)
        else:
            contents = []
            for content in self._contents:
                contents.append(content._rpad(target, posaxis, depth))
            if len(contents) == 0:
                return ak._v2.contents.recordarray.RecordArray(
                    contents,
                    self.recordlookup,
                    len(self),
                    self._identifier,
                    self._parameters,
                )
            else:
                return ak._v2.contents.recordarray.RecordArray(
                    contents,
                    self.recordlookup,
                    self._identifier,
                    self._parameters,
                )

    def _rpad_and_clip(self, target, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.rpad_axis0(target, True)
        else:
            contents = []
            for content in self._contents:
                contents.append(content._rpad_and_clip(target, posaxis, depth))
            if len(contents) == 0:
                return ak._v2.contents.recordarray.RecordArray(
                    contents,
                    self.recordlookup,
                    len(self),
                    self._identifier,
                    self._parameters,
                )
            else:
                return ak._v2.contents.recordarray.RecordArray(
                    contents,
                    self.recordlookup,
                    self._identifier,
                    self._parameters,
                )
