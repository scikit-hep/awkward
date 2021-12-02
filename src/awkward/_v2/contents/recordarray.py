# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json
import copy

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
    is_RecordType = True

    def __init__(self, contents, fields, length=None, identifier=None, parameters=None):
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

        if isinstance(fields, Iterable):
            if not isinstance(fields, list):
                fields = list(fields)
            if not all(ak._util.isstr(x) for x in fields):
                raise TypeError(
                    "{0} 'fields' must all be strings, not {1}".format(
                        type(self).__name__, repr(fields)
                    )
                )
            if not len(contents) == len(fields):
                raise ValueError(
                    "{0} len(contents) ({1}) must be equal to len(fields) ({2})".format(
                        type(self).__name__, len(contents), len(fields)
                    )
                )
        elif fields is not None:
            raise TypeError(
                "{0} 'fields' must be iterable or None, not {1}".format(
                    type(self).__name__, repr(fields)
                )
            )

        self._contents = contents
        self._fields = fields
        self._length = length
        self._init(identifier, parameters)

    @property
    def contents(self):
        return self._contents

    @property
    def fields(self):
        if self._fields is None:
            return [str(i) for i in range(len(self._contents))]
        else:
            return self._fields

    @property
    def is_tuple(self):
        return self._fields is None

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

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.Form(
            [x._form_with_key(getkey) for x in self._contents],
            self._fields,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, nplike):
        assert isinstance(form, self.Form)
        if self._fields is None:
            for i, content in enumerate(self._contents):
                content._to_buffers(form.content(i), getkey, container, nplike)
        else:
            for field, content in zip(self._fields, self._contents):
                content._to_buffers(form.content(field), getkey, container, nplike)

    @property
    def typetracer(self):
        return RecordArray(
            [x.typetracer for x in self._contents],
            self._fields,
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

        if self._fields is None:
            for i, x in enumerate(self._contents):
                out.append("{0}    <content index={1}>\n".format(indent, repr(str(i))))
                out.append(x._repr(indent + "        ", "", "\n"))
                out.append(indent + "    </content>\n")
        else:
            for i, x in enumerate(self._contents):
                out.append(
                    "{0}    <content index={1} field={2}>\n".format(
                        indent, repr(str(i)), repr(self._fields[i])
                    )
                )
                out.append(x._repr(indent + "        ", "", "\n"))
                out.append(indent + "    </content>\n")

        out.append(indent + "</RecordArray>")
        out.append(post)
        return "".join(out)

    def merge_parameters(self, parameters):
        return RecordArray(
            self._contents,
            self._fields,
            self._length,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
        )

    def index_to_field(self, index):
        return self.Form.index_to_field(self, index)

    def field_to_index(self, field):
        return self.Form.field_to_index(self, field)

    def has_field(self, field):
        return self.Form.has_field(self, field)

    def content(self, index_or_field):
        out = self.Form.content(self, index_or_field)
        if len(out) == self._length:
            return out
        else:
            return out[: self._length]

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
                self._fields,
                stop - start,
                self._range_identifier(start, stop),
                self._parameters,
            )
        else:
            nextslice = slice(start, stop)
            return RecordArray(
                [x._getitem_range(nextslice) for x in self._contents],
                self._fields,
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
        indexes = [self.field_to_index(field) for field in where]
        if self._fields is None:
            fields = None
        else:
            fields = [self._fields[i] for i in indexes]

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
            fields,
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
                self._fields,
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
        return RecordArray(contents, self._fields, self._length)

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
            next = RecordArray(
                contents, self._fields, None, self._identifier, parameters
            )
            return next._getitem_next(nexthead, nexttail, advanced)

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            single = ak._v2.index.Index64.empty(1, self.nplike)
            single[0] = len(self)
            singleton = ak._v2.contents.numpyarray.NumpyArray(single)
            contents = [singleton] * len(self._contents)

            record = ak._v2.contents.recordarray.RecordArray(
                contents, self._fields, 1, None, self._parameters
            )
            return record[0]
        else:
            contents = []
            for content in self._contents:
                contents.append(content.num(posaxis, depth))
            return ak._v2.contents.recordarray.RecordArray(
                contents, self._fields, self._length, None, self._parameters
            )

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            raise np.AxisError(self, "axis=0 not allowed for flatten")

        elif posaxis == depth + 1:
            raise ValueError(
                "arrays of records cannot be flattened (but their contents can be; try a different 'axis')"
            )

        else:
            contents = []
            for content in self._contents:
                trimmed = content._getitem_range(slice(0, len(self)))
                offsets, flattened = trimmed._offsets_and_flattened(posaxis, depth)
                if len(offsets) != 0:
                    raise AssertionError(
                        "RecordArray content with axis > depth + 1 returned a non-empty offsets from offsets_and_flattened"
                    )
                contents.append(flattened)
            offsets = ak._v2.index.Index64.zeros(1, self.nplike, dtype=np.int64)
            return (
                offsets,
                RecordArray(
                    contents, self._fields, self._length, None, self._parameters
                ),
            )

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
                self_fields = self.fields.copy()
                other_fields = other.fields.copy()
                self_fields.sort()
                other_fields.sort()
                if self_fields == other_fields:
                    for field in self_fields:
                        if not self[field].mergeable(other[field], mergebool):
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
                parameters = ak._v2._util.merge_parameters(
                    self._parameters, array._parameters
                )

                if isinstance(array, ak._v2.contents.recordarray.RecordArray):
                    if self.is_tuple:
                        if len(self.contents) == len(array.contents):
                            for i in range(len(self.contents)):
                                field = array[self.index_to_field(i)]
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
            these_fields = self._fields.copy()
            these_fields.sort()

            for array in headless:
                if isinstance(array, ak._v2.contents.recordarray.RecordArray):
                    if not array.is_tuple:
                        those_fields = array._fields.copy()
                        those_fields.sort()

                        if these_fields == those_fields:
                            for i in range(len(self.contents)):
                                field = array[self.index_to_field(i)]

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
        minlength = None
        for forfield in for_each_field:
            tail_forfield = forfield[1:]
            merged = forfield[0].mergemany(tail_forfield)

            nextcontents.append(merged)

            if minlength is None or len(merged) < minlength:
                minlength = len(merged)

        if minlength is None:
            minlength = len(self)
            for x in others:
                minlength += len(x)

        next = RecordArray(nextcontents, self._fields, minlength, None, parameters)

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

    def fillna(self, value):
        contents = []
        for content in self._contents:
            contents.append(content.fillna(value))
        return RecordArray(
            contents, self._fields, self._length, self._identifier, self._parameters
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
                contents, self._fields, len(self), self._identifier, self._parameters
            )

    def numbers_to_type(self, name):
        contents = []
        for x in self._contents:
            contents.append(x.numbers_to_type(name))
        return ak._v2.contents.recordarray.RecordArray(
            contents, self._fields, self._length, self._identifier, self._parameters
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        for content in self._contents:
            if not content._is_unique(negaxis, starts, parents, outlength):
                return False
        return True

    def _unique(self, negaxis, starts, parents, outlength):
        raise NotImplementedError

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
        if self._fields is None or len(self._fields) == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))
        else:
            raise NotImplementedError

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        if self._fields is None or len(self._fields) == 0:
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
        return RecordArray(contents, self._fields, len(self), None, self._parameters)

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
            self._fields,
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

    def _nbytes_part(self):
        result = 0
        for content in self.contents:
            result = result + content._nbytes_part()
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def _rpad(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.rpad_axis0(target, clip)
        else:
            contents = []
            for content in self._contents:
                contents.append(content._rpad(target, posaxis, depth, clip))
            if len(contents) == 0:
                return ak._v2.contents.recordarray.RecordArray(
                    contents,
                    self._fields,
                    len(self),
                    identifier=self._identifier,
                    parameters=self._parameters,
                )
            else:
                return ak._v2.contents.recordarray.RecordArray(
                    contents,
                    self._fields,
                    identifier=self._identifier,
                    parameters=self._parameters,
                )

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        values = [
            (x if len(x) == length else x[:length])._to_arrow(
                pyarrow, mask_node, validbytes, length, options
            )
            for x in self._contents
        ]

        types = pyarrow.struct(
            [
                pyarrow.field(self.index_to_field(i), values[i].type).with_nullable(
                    x.is_OptionType
                )
                for i, x in enumerate(self._contents)
            ]
        )

        return pyarrow.Array.from_buffers(
            ak._v2._connect.pyarrow.to_awkwardarrow_type(
                types, options["extensionarray"], mask_node, self
            ),
            length,
            [ak._v2._connect.pyarrow.to_validbits(validbytes)],
            children=values,
        )

    def _to_numpy(self, allow_missing):
        if self.fields is None:
            return self.nplike.empty(len(self), dtype=[])
        contents = [x._to_numpy(allow_missing) for x in self._contents]
        if any(len(x.shape) != 1 for x in contents):
            raise ValueError("cannot convert {0} into np.ndarray".format(self))
        out = self.nplike.empty(
            len(contents[0]),
            dtype=[(str(n), x.dtype) for n, x in zip(self.fields, contents)],
        )
        mask = None
        for n, x in zip(self.fields, contents):
            if isinstance(x, self.nplike.ma.MaskedArray):
                if mask is None:
                    mask = self.nplike.ma.zeros(
                        len(self), [(n, np.bool_) for n in self.fields]
                    )
                if x.mask is not None:
                    mask[n] |= x.mask
            out[n] = x

        if mask is not None:
            out = self.nplike.ma.MaskedArray(out, mask)

        return out

    def _completely_flatten(self, nplike, options):
        if options["flatten_records"]:
            out = []
            for content in self._contents:
                out.extend(content[: self._length]._completely_flatten(nplike, options))
            return out
        else:
            in_function = ""
            if options["function_name"] is not None:
                in_function = " in " + options["function_name"]
            raise TypeError(
                "cannot combine record fields{0} unless flatten_records=True".format(
                    in_function
                )
            )

    def _recursively_apply(
        self, action, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                return RecordArray(
                    [
                        content._recursively_apply(
                            action,
                            depth,
                            copy.copy(depth_context),
                            lateral_context,
                            options,
                        )
                        for content in self._contents
                    ],
                    self._fields,
                    self._length,
                    self._identifier,
                    self._parameters if options["keep_parameters"] else None,
                )

        else:

            def continuation():
                for content in self._contents:
                    content._recursively_apply(
                        action,
                        depth,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    )

        result = action(
            self,
            depth=depth,
            depth_context=depth_context,
            lateral_context=lateral_context,
            continuation=continuation,
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise AssertionError(result)

    def packed(self):
        return RecordArray(
            [
                x.packed() if len(x) == self._length else x[: self._length].packed()
                for x in self._contents
            ],
            self._fields,
            self._length,
            self._identifier,
            self._parameters,
        )

    def _to_list(self, behavior):
        out = self._to_list_custom(behavior)
        if out is not None:
            return out

        cls = ak._v2._util.recordclass(self, behavior)
        if cls is not ak._v2.highlevel.Record:
            length = self._length
            out = [None] * length
            for i in range(length):
                out[i] = cls(self[i])
            return out

        if self.is_tuple:
            contents = [x._to_list(behavior) for x in self._contents]
            length = self._length
            out = [None] * length
            for i in range(length):
                out[i] = tuple(x[i] for x in contents)
            return out

        else:
            fields = self._fields
            contents = [x._to_list(behavior) for x in self._contents]
            length = self._length
            out = [None] * length
            for i in range(length):
                out[i] = dict(zip(fields, [x[i] for x in contents]))
            return out
