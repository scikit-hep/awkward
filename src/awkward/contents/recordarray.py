# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy
import json
from collections.abc import Iterable

import awkward as ak
from awkward._util import unset
from awkward.contents.content import Content
from awkward.forms.recordform import RecordForm
from awkward.record import Record
from awkward.typing import Final, Self

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


class RecordArray(Content):
    is_record = True

    @property
    def is_leaf(self):
        return len(self._contents) == 0

    def __init__(
        self,
        contents,
        fields,
        length=None,
        *,
        parameters=None,
        backend=None,
    ):
        if not isinstance(contents, Iterable):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'contents' must be iterable, not {}".format(
                        type(self).__name__, repr(contents)
                    )
                )
            )
        if not isinstance(contents, list):
            contents = list(contents)

        if len(contents) == 0 and length is None:
            raise ak._errors.wrap_error(
                TypeError(
                    "{} if len(contents) == 0, a 'length' must be specified".format(
                        type(self).__name__
                    )
                )
            )
        elif length is None:
            lengths = [x.length for x in contents]
            if any(isinstance(x, ak._typetracer.UnknownLengthType) for x in contents):
                length = ak._typetracer.UnknownLength
            else:
                length = min(lengths)
        if not isinstance(length, ak._typetracer.UnknownLengthType) and not (
            ak._util.is_integer(length) and length >= 0
        ):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'length' must be a non-negative integer or None, not {}".format(
                        type(self).__name__, repr(length)
                    )
                )
            )
        for content in contents:
            if not isinstance(content, Content):
                raise ak._errors.wrap_error(
                    TypeError(
                        "{} all 'contents' must be Content subclasses, not {}".format(
                            type(self).__name__, repr(content)
                        )
                    )
                )
            if content.length < length:
                raise ak._errors.wrap_error(
                    ValueError(
                        "{} len(content) ({}) must be >= length ({}) for all 'contents'".format(
                            type(self).__name__, content.length, length
                        )
                    )
                )

        if isinstance(fields, Iterable):
            if not isinstance(fields, list):
                fields = list(fields)
            if not all(isinstance(x, str) for x in fields):
                raise ak._errors.wrap_error(
                    TypeError(
                        "{} 'fields' must all be strings, not {}".format(
                            type(self).__name__, repr(fields)
                        )
                    )
                )
            if not len(contents) == len(fields):
                raise ak._errors.wrap_error(
                    ValueError(
                        "{} len(contents) ({}) must be equal to len(fields) ({})".format(
                            type(self).__name__, len(contents), len(fields)
                        )
                    )
                )
        elif fields is not None:
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'fields' must be iterable or None, not {}".format(
                        type(self).__name__, repr(fields)
                    )
                )
            )
        for content in contents:
            if backend is None:
                backend = content.backend
            elif backend is not content.backend:
                raise ak._errors.wrap_error(
                    TypeError(
                        "{} 'contents' must use the same array library (backend): {} vs {}".format(
                            type(self).__name__,
                            type(backend).__name__,
                            type(content.backend).__name__,
                        )
                    )
                )
        if backend is None:
            backend = ak._backends.NumpyBackend.instance()

        self._contents = contents
        self._fields = fields
        self._length = length
        self._init(parameters, backend)

    @property
    def contents(self):
        return self._contents

    @property
    def fields(self):
        if self._fields is None:
            return [str(i) for i in range(len(self._contents))]
        else:
            return self._fields

    form_cls: Final = RecordForm

    def copy(
        self,
        contents=unset,
        fields=unset,
        length=unset,
        *,
        parameters=unset,
        backend=unset,
    ):
        return RecordArray(
            self._contents if contents is unset else contents,
            self._fields if fields is unset else fields,
            self._length if length is unset else length,
            parameters=self._parameters if parameters is unset else parameters,
            backend=self._backend if backend is unset else backend,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            contents=[copy.deepcopy(x, memo) for x in self._contents],
            fields=copy.deepcopy(self._fields, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(
        cls,
        contents,
        fields,
        length=None,
        *,
        parameters=None,
        backend=None,
    ):
        return cls(contents, fields, length, parameters=parameters, backend=backend)

    @property
    def is_tuple(self):
        return self._fields is None

    def to_tuple(self) -> Self:
        return RecordArray(
            self._contents, None, self._length, parameters=None, backend=self._backend
        )

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.form_cls(
            [x._form_with_key(getkey) for x in self._contents],
            self._fields,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, backend):
        assert isinstance(form, self.form_cls)
        if self._fields is None:
            for i, content in enumerate(self._contents):
                content._to_buffers(form.content(i), getkey, container, backend)
        else:
            for field, content in zip(self._fields, self._contents):
                content._to_buffers(form.content(field), getkey, container, backend)

    def _to_typetracer(self, forget_length: bool) -> Self:
        backend = ak._backends.TypeTracerBackend.instance()
        contents = [x._to_typetracer(forget_length) for x in self._contents]
        return RecordArray(
            contents,
            self._fields,
            ak._typetracer.UnknownLength if forget_length else self._length,
            parameters=self._parameters,
            backend=backend,
        )

    def _touch_data(self, recursive):
        if recursive:
            for x in self._contents:
                x._touch_data(recursive)

    def _touch_shape(self, recursive):
        if recursive:
            for x in self._contents:
                x._touch_shape(recursive)

    @property
    def length(self):
        return self._length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<RecordArray is_tuple="]
        out.append(repr(json.dumps(self.is_tuple)))
        out.append(" len=")
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")

        if self._fields is None:
            for i, x in enumerate(self._contents):
                out.append(f"{indent}    <content index={repr(str(i))}>\n")
                out.append(x._repr(indent + "        ", "", "\n"))
                out.append(indent + "    </content>\n")
        else:
            for i, x in enumerate(self._contents):
                out.append(
                    "{}    <content index={} field={}>\n".format(
                        indent, repr(str(i)), repr(self._fields[i])
                    )
                )
                out.append(x._repr(indent + "        ", "", "\n"))
                out.append(indent + "    </content>\n")

        out.append(indent + "</RecordArray>")
        out.append(post)
        return "".join(out)

    def index_to_field(self, index):
        return self.form_cls.index_to_field(self, index)

    def field_to_index(self, field):
        return self.form_cls.field_to_index(self, field)

    def has_field(self, field):
        return self.form_cls.has_field(self, field)

    def content(self, index_or_field):
        out = self.form_cls.content(self, index_or_field)
        if out.length == self._length:
            return out
        else:
            return out[: self._length]

    def _getitem_nothing(self):
        return self._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if self._backend.nplike.known_shape and where < 0:
            where += self.length

        if where < 0 or where >= self.length:
            raise ak._errors.index_error(self, where)
        return Record(self, where)

    def _getitem_range(self, where):
        if not self._backend.nplike.known_shape:
            self._touch_shape(recursive=False)
            return self

        start, stop, step = where.indices(self.length)
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
                parameters=self._parameters,
                backend=self._backend,
            )
        else:
            nextslice = slice(start, stop)
            return RecordArray(
                [x._getitem_range(nextslice) for x in self._contents],
                self._fields,
                stop - start,
                parameters=self._parameters,
                backend=self._backend,
            )

    def _getitem_field(self, where, only_fields=()):
        if len(only_fields) == 0:
            return self.content(where)

        else:
            nexthead, nexttail = ak._slicing.headtail(only_fields)
            if isinstance(nexthead, str):
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
            nexthead, nexttail = ak._slicing.headtail(only_fields)
            if isinstance(nexthead, str):
                contents = [
                    self.content(i)._getitem_field(nexthead, nexttail) for i in indexes
                ]
            else:
                contents = [
                    self.content(i)._getitem_fields(nexthead, nexttail) for i in indexes
                ]

        return RecordArray(
            contents, fields, self._length, parameters=None, backend=self._backend
        )

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak.index.Index)

        if allow_lazy:
            where = carry.data

            if allow_lazy != "copied":
                where = where.copy()

            negative = where < 0
            if self._backend.index_nplike.any(negative, prefer=False):
                where[negative] += self._length

            if self._backend.index_nplike.any(where >= self._length, prefer=False):
                raise ak._errors.index_error(self, where)

            nextindex = ak.index.Index64(where, nplike=self._backend.index_nplike)
            return ak.contents.IndexedArray(nextindex, self, parameters=None)

        else:
            contents = [
                self.content(i)._carry(carry, allow_lazy)
                for i in range(len(self._contents))
            ]

            # if issubclass(carry.dtype.type, np.integer):
            #     length = carry.length
            # else:
            #     length = self.length
            assert issubclass(carry.dtype.type, np.integer)
            length = carry.length

            return RecordArray(
                contents,
                self._fields,
                length,
                parameters=self._parameters,
                backend=self._backend,
            )

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        contents = []
        for i in range(len(self._contents)):
            contents.append(
                self.content(i)._getitem_next_jagged(
                    slicestarts, slicestops, slicecontent, tail
                )
            )
        return RecordArray(
            contents, self._fields, self._length, parameters=None, backend=self._backend
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif isinstance(head, ak.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            nexthead, nexttail = ak._slicing.headtail(tail)

            contents = []
            for i in range(len(self._contents)):
                contents.append(self.content(i)._getitem_next(head, (), advanced))

            parameters = None
            if (
                isinstance(
                    head,
                    (
                        slice,
                        ak.contents.ListOffsetArray,
                        ak.contents.IndexedOptionArray,
                    ),
                )
                or head is Ellipsis
                or advanced is None
            ):
                parameters = self._parameters
            next = RecordArray(
                contents,
                self._fields,
                None,
                parameters=parameters,
                backend=self._backend,
            )
            return next._getitem_next(nexthead, nexttail, advanced)

    def _offsets_and_flattened(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            raise ak._errors.wrap_error(np.AxisError("axis=0 not allowed for flatten"))

        elif posaxis is not None and posaxis + 1 == depth + 1:
            raise ak._errors.wrap_error(
                ValueError(
                    "arrays of records cannot be flattened (but their contents can be; try a different 'axis')"
                )
            )

        else:
            contents = []
            for content in self._contents:
                trimmed = content._getitem_range(slice(0, self.length))
                offsets, flattened = trimmed._offsets_and_flattened(axis, depth)
                if self._backend.nplike.known_shape and offsets.length != 0:
                    raise ak._errors.wrap_error(
                        AssertionError(
                            "RecordArray content with axis > depth + 1 returned a non-empty offsets from offsets_and_flattened"
                        )
                    )
                contents.append(flattened)
            offsets = ak.index.Index64.zeros(
                1,
                nplike=self._backend.index_nplike,
                dtype=np.int64,
            )
            return (
                offsets,
                RecordArray(
                    contents,
                    self._fields,
                    self._length,
                    parameters=None,
                    backend=self._backend,
                ),
            )

    def _mergeable_next(self, other, mergebool):
        if isinstance(
            other,
            (
                ak.contents.IndexedArray,
                ak.contents.IndexedOptionArray,
                ak.contents.ByteMaskedArray,
                ak.contents.BitMaskedArray,
                ak.contents.UnmaskedArray,
            ),
        ):
            return self._mergeable(other.content, mergebool)

        if isinstance(other, RecordArray):
            if self.is_tuple and other.is_tuple:
                if len(self._contents) == len(other._contents):
                    for self_cont, other_cont in zip(self._contents, other._contents):
                        if not self_cont._mergeable(other_cont, mergebool):
                            return False

                    return True

            elif not self.is_tuple and not other.is_tuple:
                if set(self._fields) != set(other._fields):
                    return False

                for i, field in enumerate(self._fields):
                    x = self._contents[i]
                    y = other._contents[other.field_to_index(field)]
                    if not x._mergeable(y, mergebool):
                        return False
                return True

            else:
                return False

        else:
            return False

    def _mergemany(self, others):
        if len(others) == 0:
            return self

        head, tail = self._merging_strategy(others)

        parameters = self._parameters
        headless = head[1:]

        for_each_field = []
        for field in self.contents:
            trimmed = field[0 : self.length]
            for_each_field.append([trimmed])

        if self.is_tuple:
            for array in headless:
                parameters = ak._util.merge_parameters(
                    parameters, array._parameters, True
                )

                if isinstance(array, ak.contents.RecordArray):
                    if self.is_tuple:
                        if len(self.contents) == len(array.contents):
                            for i in range(len(self.contents)):
                                field = array[self.index_to_field(i)]
                                for_each_field[i].append(field[0 : array.length])
                        else:
                            raise ak._errors.wrap_error(
                                ValueError(
                                    "cannot merge tuples with different numbers of fields"
                                )
                            )
                    else:
                        raise ak._errors.wrap_error(
                            ValueError("cannot merge tuple with non-tuple record")
                        )
                elif isinstance(array, ak.contents.EmptyArray):
                    pass
                else:
                    raise ak._errors.wrap_error(
                        AssertionError(
                            "cannot merge "
                            + type(self).__name__
                            + " with "
                            + type(array).__name__
                        )
                    )

        else:
            these_fields = self._fields.copy()
            these_fields.sort()

            for array in headless:
                parameters = ak._util.merge_parameters(
                    parameters, array._parameters, True
                )

                if isinstance(array, ak.contents.RecordArray):
                    if not array.is_tuple:
                        those_fields = array._fields.copy()
                        those_fields.sort()

                        if these_fields == those_fields:
                            for i in range(len(self.contents)):
                                field = array[self.index_to_field(i)]

                                trimmed = field[0 : array.length]
                                for_each_field[i].append(trimmed)
                        else:
                            raise ak._errors.wrap_error(
                                AssertionError(
                                    "cannot merge records with different sets of field names"
                                )
                            )
                    else:
                        raise ak._errors.wrap_error(
                            AssertionError("cannot merge non-tuple record with tuple")
                        )

                elif isinstance(array, ak.contents.EmptyArray):
                    pass
                else:
                    raise ak._errors.wrap_error(
                        AssertionError(
                            "cannot merge "
                            + type(self).__name__
                            + " with "
                            + type(array).__name__
                        )
                    )

        nextcontents = []
        minlength = None
        for forfield in for_each_field:
            merged = forfield[0]._mergemany(forfield[1:])

            nextcontents.append(merged)

            if minlength is None or merged.length < minlength:
                minlength = merged.length

        if minlength is None:
            minlength = self.length
            for x in others:
                minlength += x.length

        next = RecordArray(
            nextcontents,
            self._fields,
            minlength,
            parameters=parameters,
            backend=self._backend,
        )

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed._mergemany(tail[1:])

    def _fill_none(self, value: Content) -> Content:
        contents = []
        for content in self._contents:
            contents.append(content._fill_none(value))
        return RecordArray(
            contents,
            self._fields,
            self._length,
            parameters=self._parameters,
            backend=self._backend,
        )

    def _local_index(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        else:
            contents = []
            for content in self._contents:
                contents.append(content._local_index(axis, depth))
            return RecordArray(
                contents,
                self._fields,
                self.length,
                parameters=self._parameters,
                backend=self._backend,
            )

    def _numbers_to_type(self, name):
        contents = []
        for x in self._contents:
            contents.append(x._numbers_to_type(name))
        return ak.contents.RecordArray(
            contents,
            self._fields,
            self._length,
            parameters=self._parameters,
            backend=self._backend,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        for content in self._contents:
            if not content._is_unique(negaxis, starts, parents, outlength):
                return False
        return True

    def _unique(self, negaxis, starts, parents, outlength):
        raise ak._errors.wrap_error(NotImplementedError)

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
        raise ak._errors.wrap_error(NotImplementedError)

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        if self._fields is None or len(self._fields) == 0:
            return ak.contents.NumpyArray(
                self._backend.nplike.instance().empty(0, np.int64),
                parameters=None,
                backend=self._backend,
            )

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
        return RecordArray(
            contents,
            self._fields,
            self._length,
            parameters=self._parameters,
            backend=self._backend,
        )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            contents = []
            for content in self._contents:
                contents.append(
                    content._combinations(
                        n, replacement, recordlookup, parameters, axis, depth
                    )
                )
            return ak.contents.RecordArray(
                contents,
                recordlookup,
                self.length,
                parameters=self._parameters,
                backend=self._backend,
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
        behavior,
    ):
        reducer_recordclass = ak._util.reducer_recordclass(reducer, self, behavior)
        if reducer_recordclass is None:
            raise ak._errors.wrap_error(
                TypeError(
                    "no ak.{} overloads for custom types: {}".format(
                        reducer.name, ", ".join(self._fields)
                    )
                )
            )
        else:
            raise ak._errors.wrap_error(
                NotImplementedError(
                    "overloading reducers for RecordArrays has not been implemented yet"
                )
            )

    def _validity_error(self, path):
        for i, cont in enumerate(self.contents):
            if cont.length < self.length:
                return f'at {path} ("{type(self)}"): len(field({i})) < len(recordarray)'
        for i, cont in enumerate(self.contents):
            sub = cont._validity_error(f"{path}.field({i})")
            if sub != "":
                return sub
        return ""

    def _nbytes_part(self):
        result = 0
        for content in self.contents:
            result = result + content._nbytes_part()
        return result

    def _pad_none(self, target, axis, depth, clip):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._pad_none_axis0(target, clip)
        else:
            contents = []
            for content in self._contents:
                contents.append(content._pad_none(target, axis, depth, clip))
            if len(contents) == 0:
                return ak.contents.RecordArray(
                    contents,
                    self._fields,
                    self._length,
                    parameters=self._parameters,
                    backend=self._backend,
                )
            else:
                return ak.contents.RecordArray(
                    contents,
                    self._fields,
                    self._length,
                    parameters=self._parameters,
                    backend=self._backend,
                )

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        values = [
            (x if x.length == length else x[:length])._to_arrow(
                pyarrow, mask_node, validbytes, length, options
            )
            for x in self._contents
        ]

        types = pyarrow.struct(
            [
                pyarrow.field(self.index_to_field(i), values[i].type).with_nullable(
                    x.is_option
                )
                for i, x in enumerate(self._contents)
            ]
        )

        return pyarrow.Array.from_buffers(
            ak._connect.pyarrow.to_awkwardarrow_type(
                types,
                options["extensionarray"],
                options["record_is_scalar"],
                mask_node,
                self,
            ),
            length,
            [ak._connect.pyarrow.to_validbits(validbytes)],
            children=values,
        )

    def _to_numpy(self, allow_missing):
        if self.fields is None:
            return self._backend.nplike.empty(self.length, dtype=[])
        contents = [x._to_numpy(allow_missing) for x in self._contents]
        if any(len(x.shape) != 1 for x in contents):
            raise ak._errors.wrap_error(
                ValueError(f"cannot convert {self} into np.ndarray")
            )
        out = self._backend.nplike.empty(
            self.length,
            dtype=[(str(n), x.dtype) for n, x in zip(self.fields, contents)],
        )
        mask = None
        for n, x in zip(self.fields, contents):
            if isinstance(x, self._backend.nplike.ma.MaskedArray):
                if mask is None:
                    mask = self._backend.index_nplike.ma.zeros(
                        self.length, [(n, np.bool_) for n in self.fields]
                    )
                if x.mask is not None:
                    mask[n] |= x.mask
            out[n] = x

        if mask is not None:
            out = self._backend.nplike.ma.MaskedArray(out, mask)

        return out

    def _completely_flatten(self, backend, options):
        if options["flatten_records"]:
            out = []
            for content in self._contents:
                out.extend(
                    content[: self._length]._completely_flatten(backend, options)
                )
            return out
        else:
            in_function = ""
            if options["function_name"] is not None:
                in_function = " in " + options["function_name"]
            raise ak._errors.wrap_error(
                TypeError(
                    "cannot combine record fields{} unless flatten_records=True".format(
                        in_function
                    )
                )
            )

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if self._backend.nplike.known_shape:
            contents = [x[: self._length] for x in self._contents]
        else:
            self._touch_data(recursive=False)
            contents = self._contents

        if options["return_array"]:

            def continuation():
                if not options["allow_records"]:
                    raise ak._errors.wrap_error(
                        ValueError(
                            f"cannot broadcast records in {options['function_name']}"
                        )
                    )
                return RecordArray(
                    [
                        content._recursively_apply(
                            action,
                            behavior,
                            depth,
                            copy.copy(depth_context),
                            lateral_context,
                            options,
                        )
                        for content in contents
                    ],
                    self._fields,
                    self._length,
                    parameters=self._parameters if options["keep_parameters"] else None,
                    backend=self._backend,
                )

        else:

            def continuation():
                for content in contents:
                    content._recursively_apply(
                        action,
                        behavior,
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
            behavior=behavior,
            backend=self._backend,
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise ak._errors.wrap_error(AssertionError(result))

    def to_packed(self) -> Self:
        return RecordArray(
            [
                x.to_packed()
                if x.length == self._length
                else x[: self._length].to_packed()
                for x in self._contents
            ],
            self._fields,
            self._length,
            parameters=self._parameters,
            backend=self._backend,
        )

    def _to_list(self, behavior, json_conversions):
        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        if self.is_tuple and json_conversions is None:
            contents = [x._to_list(behavior, json_conversions) for x in self._contents]
            length = self._length
            out = [None] * length
            for i in range(length):
                out[i] = tuple(x[i] for x in contents)
            return out

        else:
            fields = self._fields
            if fields is None:
                fields = [str(i) for i in range(len(self._contents))]
            contents = [x._to_list(behavior, json_conversions) for x in self._contents]
            length = self._length
            out = [None] * length
            for i in range(length):
                out[i] = dict(zip(fields, [x[i] for x in contents]))
            return out

    def to_backend(self, backend: ak._backends.Backend) -> Self:
        contents = [content.to_backend(backend) for content in self._contents]
        return RecordArray(
            contents,
            self._fields,
            length=self._length,
            parameters=self._parameters,
            backend=backend,
        )

    def _is_equal_to(self, other, index_dtype, numpyarray):
        return (
            self.fields == other.fields
            and len(self.contents) == len(other.contents)
            and all(
                [
                    self.contents[i].is_equal_to(
                        other.contents[i], index_dtype, numpyarray
                    )
                    for i in range(len(self.contents))
                ]
            )
        )
