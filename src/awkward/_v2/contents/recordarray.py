# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json
import copy

from collections.abc import Iterable

import awkward as ak
from awkward._v2.record import Record
from awkward._v2.contents.content import Content, unset
from awkward._v2.forms.recordform import RecordForm

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class RecordArray(Content):
    is_RecordType = True

    def copy(
        self,
        contents=unset,
        fields=unset,
        length=unset,
        identifier=unset,
        parameters=unset,
        nplike=unset,
    ):
        return RecordArray(
            self._contents if contents is unset else contents,
            self._fields if fields is unset else fields,
            self._length if length is unset else length,
            self._identifier if identifier is unset else identifier,
            self._parameters if parameters is unset else parameters,
            self._nplike if nplike is unset else nplike,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            contents=[copy.deepcopy(x, memo) for x in self._contents],
            fields=copy.deepcopy(self._fields, memo),
            identifier=copy.deepcopy(self._identifier, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    def __init__(
        self,
        contents,
        fields,
        length=None,
        identifier=None,
        parameters=None,
        nplike=None,
    ):
        if not isinstance(contents, Iterable):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'contents' must be iterable, not {}".format(
                        type(self).__name__, repr(contents)
                    )
                )
            )
        if not isinstance(contents, list):
            contents = list(contents)

        if len(contents) == 0 and length is None:
            raise ak._v2._util.error(
                TypeError(
                    "{} if len(contents) == 0, a 'length' must be specified".format(
                        type(self).__name__
                    )
                )
            )
        elif length is None:
            lengths = [x.length for x in contents]
            if any(
                isinstance(x, ak._v2._typetracer.UnknownLengthType) for x in contents
            ):
                length = ak._v2._typetracer.UnknownLength
            else:
                length = min(lengths)
        if not isinstance(length, ak._v2._typetracer.UnknownLengthType) and not (
            ak._util.isint(length) and length >= 0
        ):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'length' must be a non-negative integer or None, not {}".format(
                        type(self).__name__, repr(length)
                    )
                )
            )
        for content in contents:
            if not isinstance(content, Content):
                raise ak._v2._util.error(
                    TypeError(
                        "{} all 'contents' must be Content subclasses, not {}".format(
                            type(self).__name__, repr(content)
                        )
                    )
                )
            if content.length < length:
                raise ak._v2._util.error(
                    ValueError(
                        "{} len(content) ({}) must be >= length ({}) for all 'contents'".format(
                            type(self).__name__, content.length, length
                        )
                    )
                )

        if isinstance(fields, Iterable):
            if not isinstance(fields, list):
                fields = list(fields)
            if not all(ak._util.isstr(x) for x in fields):
                raise ak._v2._util.error(
                    TypeError(
                        "{} 'fields' must all be strings, not {}".format(
                            type(self).__name__, repr(fields)
                        )
                    )
                )
            if not len(contents) == len(fields):
                raise ak._v2._util.error(
                    ValueError(
                        "{} len(contents) ({}) must be equal to len(fields) ({})".format(
                            type(self).__name__, len(contents), len(fields)
                        )
                    )
                )
        elif fields is not None:
            raise ak._v2._util.error(
                TypeError(
                    "{} 'fields' must be iterable or None, not {}".format(
                        type(self).__name__, repr(fields)
                    )
                )
            )
        if nplike is None:
            for content in contents:
                if nplike is None:
                    nplike = content.nplike
                    break
                elif nplike is not content.nplike:
                    raise ak._v2._util.error(
                        TypeError(
                            "{} 'contents' must use the same array library (nplike): {} vs {}".format(
                                type(self).__name__,
                                type(nplike).__name__,
                                type(content.nplike).__name__,
                            )
                        )
                    )
        if nplike is None:
            nplike = numpy

        self._contents = contents
        self._fields = fields
        self._length = length
        self._init(identifier, parameters, nplike)

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
            self._length,
            None,
            None,
            self._nplike,
        )

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
        tt = ak._v2._typetracer.TypeTracer.instance()
        return RecordArray(
            [x.typetracer for x in self._contents],
            self._fields,
            self._length,
            self._typetracer_identifier(),
            self._parameters,
            tt,
        )

    @property
    def length(self):
        return self._length

    def _forget_length(self):
        return RecordArray(
            [x._forget_length() for x in self._contents],
            self._fields,
            ak._v2._typetracer.UnknownLength,
            self._identifier,
            self._parameters,
            self._nplike,
        )

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

    def merge_parameters(self, parameters):
        return RecordArray(
            self._contents,
            self._fields,
            self._length,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
            self._nplike,
        )

    def index_to_field(self, index):
        return self.Form.index_to_field(self, index)

    def field_to_index(self, field):
        return self.Form.field_to_index(self, field)

    def has_field(self, field):
        return self.Form.has_field(self, field)

    def content(self, index_or_field):
        out = self.Form.content(self, index_or_field)
        if out.length == self._length:
            return out
        else:
            return out[: self._length]

    def _getitem_nothing(self):
        return self._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if self._nplike.known_data and where < 0:
            where += self.length

        if where < 0 or where >= self.length:
            raise ak._v2._util.indexerror(self, where)
        return Record(self, where)

    def _getitem_range(self, where):
        if not self._nplike.known_shape:
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
                self._range_identifier(start, stop),
                self._parameters,
                self._nplike,
            )
        else:
            nextslice = slice(start, stop)
            return RecordArray(
                [x._getitem_range(nextslice) for x in self._contents],
                self._fields,
                stop - start,
                self._range_identifier(start, stop),
                self._parameters,
                self._nplike,
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
            self._length,
            self._fields_identifier(where),
            None,
            self._nplike,
        )

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak._v2.index.Index)

        if allow_lazy:
            where = carry.data

            if allow_lazy != "copied":
                where = where.copy()

            negative = where < 0
            if self._nplike.index_nplike.any(negative, prefer=False):
                where[negative] += self._length

            if self._nplike.index_nplike.any(where >= self._length, prefer=False):
                raise ak._v2._util.indexerror(self, where)

            nextindex = ak._v2.index.Index64(where, nplike=self.nplike)
            return ak._v2.contents.indexedarray.IndexedArray(
                nextindex,
                self,
                self._carry_identifier(carry),
                None,
                self._nplike,
            )

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
                self._carry_identifier(carry),
                self._parameters,
                self._nplike,
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
            contents, self._fields, self._length, None, None, self._nplike
        )

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
                contents, self._fields, None, self._identifier, parameters, self._nplike
            )
            return next._getitem_next(nexthead, nexttail, advanced)

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            npsingle = self._nplike.index_nplike.full((1,), self.length, np.int64)
            single = ak._v2.index.Index64(npsingle, nplike=self._nplike)
            singleton = ak._v2.contents.numpyarray.NumpyArray(
                single, None, None, self._nplike
            )
            contents = [singleton] * len(self._contents)
            record = ak._v2.contents.recordarray.RecordArray(
                contents, self._fields, 1, None, self._parameters, self._nplike
            )
            return record[0]
        else:
            contents = []
            for content in self._contents:
                contents.append(content.num(posaxis, depth))
            return ak._v2.contents.recordarray.RecordArray(
                contents,
                self._fields,
                self._length,
                None,
                self._parameters,
                self._nplike,
            )

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            raise ak._v2._util.error(np.AxisError("axis=0 not allowed for flatten"))

        elif posaxis == depth + 1:
            raise ak._v2._util.error(
                ValueError(
                    "arrays of records cannot be flattened (but their contents can be; try a different 'axis')"
                )
            )

        else:
            contents = []
            for content in self._contents:
                trimmed = content._getitem_range(slice(0, self.length))
                offsets, flattened = trimmed._offsets_and_flattened(posaxis, depth)
                if self._nplike.known_shape and offsets.length != 0:
                    raise ak._v2._util.error(
                        AssertionError(
                            "RecordArray content with axis > depth + 1 returned a non-empty offsets from offsets_and_flattened"
                        )
                    )
                contents.append(flattened)
            offsets = ak._v2.index.Index64.zeros(1, self._nplike, dtype=np.int64)
            return (
                offsets,
                RecordArray(
                    contents,
                    self._fields,
                    self._length,
                    None,
                    {},
                    self._nplike,
                ),
            )

    def _mergeable(self, other, mergebool):
        if isinstance(
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

        if isinstance(other, RecordArray):
            if self.is_tuple and other.is_tuple:
                if len(self._contents) == len(other._contents):
                    for self_cont, other_cont in zip(self._contents, other._contents):
                        if not self_cont.mergeable(other_cont, mergebool):
                            return False

                    return True

            elif not self.is_tuple and not other.is_tuple:
                if set(self._fields) != set(other._fields):
                    return False

                for i, field in enumerate(self._fields):
                    x = self._contents[i]
                    y = other._contents[other.field_to_index(field)]
                    if not x.mergeable(y, mergebool):
                        return False
                return True

            else:
                return False

        else:
            return False

    def mergemany(self, others):
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
                parameters = ak._v2._util.merge_parameters(
                    parameters, array._parameters, True
                )

                if isinstance(array, ak._v2.contents.recordarray.RecordArray):
                    if self.is_tuple:
                        if len(self.contents) == len(array.contents):
                            for i in range(len(self.contents)):
                                field = array[self.index_to_field(i)]
                                for_each_field[i].append(field[0 : array.length])
                        else:
                            raise ak._v2._util.error(
                                ValueError(
                                    "cannot merge tuples with different numbers of fields"
                                )
                            )
                    else:
                        raise ak._v2._util.error(
                            ValueError("cannot merge tuple with non-tuple record")
                        )
                elif isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                    pass
                else:
                    raise ak._v2._util.error(
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
                parameters = ak._v2._util.merge_parameters(
                    parameters, array._parameters, True
                )

                if isinstance(array, ak._v2.contents.recordarray.RecordArray):
                    if not array.is_tuple:
                        those_fields = array._fields.copy()
                        those_fields.sort()

                        if these_fields == those_fields:
                            for i in range(len(self.contents)):
                                field = array[self.index_to_field(i)]

                                trimmed = field[0 : array.length]
                                for_each_field[i].append(trimmed)
                        else:
                            raise ak._v2._util.error(
                                AssertionError(
                                    "cannot merge records with different sets of field names"
                                )
                            )
                    else:
                        raise ak._v2._util.error(
                            AssertionError("cannot merge non-tuple record with tuple")
                        )

                elif isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                    pass
                else:
                    raise ak._v2._util.error(
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
            merged = forfield[0].mergemany(forfield[1:])

            nextcontents.append(merged)

            if minlength is None or merged.length < minlength:
                minlength = merged.length

        if minlength is None:
            minlength = self.length
            for x in others:
                minlength += x.length

        next = RecordArray(
            nextcontents, self._fields, minlength, None, parameters, self._nplike
        )

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed.mergemany(tail[1:])

        raise ak._v2._util.error(
            NotImplementedError(
                "not implemented: " + type(self).__name__ + " ::mergemany"
            )
        )

    def fill_none(self, value):
        contents = []
        for content in self._contents:
            contents.append(content.fill_none(value))
        return RecordArray(
            contents,
            self._fields,
            self._length,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def _local_index(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._local_index_axis0()
        else:
            contents = []
            for content in self._contents:
                contents.append(content._local_index(posaxis, depth))
            return RecordArray(
                contents,
                self._fields,
                self.length,
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def numbers_to_type(self, name):
        contents = []
        for x in self._contents:
            contents.append(x.numbers_to_type(name))
        return ak._v2.contents.recordarray.RecordArray(
            contents,
            self._fields,
            self._length,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        for content in self._contents:
            if not content._is_unique(negaxis, starts, parents, outlength):
                return False
        return True

    def _unique(self, negaxis, starts, parents, outlength):
        raise ak._v2._util.error(NotImplementedError)

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
        raise ak._v2._util.error(NotImplementedError)

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        if self._fields is None or len(self._fields) == 0:
            return ak._v2.contents.NumpyArray(
                self._nplike.instance().empty(0, np.int64), None, None, self._nplike
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
            contents, self._fields, self._length, None, self._parameters, self._nplike
        )

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
                contents,
                recordlookup,
                self.length,
                self._identifier,
                self._parameters,
                self._nplike,
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
        reducer_recordclass = ak._v2._util.reducer_recordclass(reducer, self, behavior)
        if reducer_recordclass is None:
            raise ak._v2._util.error(
                TypeError(
                    "no ak.{} overloads for custom types: {}".format(
                        reducer.name, ", ".join(self._fields)
                    )
                )
            )
        else:
            raise ak._v2._util.error(
                NotImplementedError(
                    "overloading reducers for RecordArrays has not been implemented yet"
                )
            )

    def _validity_error(self, path):
        for i, cont in enumerate(self.contents):
            if cont.length < self.length:
                return f'at {path} ("{type(self)}"): len(field({i})) < len(recordarray)'
        for i, cont in enumerate(self.contents):
            sub = cont.validity_error(f"{path}.field({i})")
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

    def _pad_none(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.pad_none_axis0(target, clip)
        else:
            contents = []
            for content in self._contents:
                contents.append(content._pad_none(target, posaxis, depth, clip))
            if len(contents) == 0:
                return ak._v2.contents.recordarray.RecordArray(
                    contents,
                    self._fields,
                    self._length,
                    self._identifier,
                    self._parameters,
                    self._nplike,
                )
            else:
                return ak._v2.contents.recordarray.RecordArray(
                    contents,
                    self._fields,
                    self._length,
                    self._identifier,
                    self._parameters,
                    self._nplike,
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
                    x.is_OptionType
                )
                for i, x in enumerate(self._contents)
            ]
        )

        return pyarrow.Array.from_buffers(
            ak._v2._connect.pyarrow.to_awkwardarrow_type(
                types,
                options["extensionarray"],
                options["record_is_scalar"],
                mask_node,
                self,
            ),
            length,
            [ak._v2._connect.pyarrow.to_validbits(validbytes)],
            children=values,
        )

    def _to_numpy(self, allow_missing):
        if self.fields is None:
            return self._nplike.empty(self.length, dtype=[])
        contents = [x._to_numpy(allow_missing) for x in self._contents]
        if any(len(x.shape) != 1 for x in contents):
            raise ak._v2._util.error(
                ValueError(f"cannot convert {self} into np.ndarray")
            )
        out = self._nplike.empty(
            contents[0].shape[0],
            dtype=[(str(n), x.dtype) for n, x in zip(self.fields, contents)],
        )
        mask = None
        for n, x in zip(self.fields, contents):
            if isinstance(x, self._nplike.ma.MaskedArray):
                if mask is None:
                    mask = self._nplike.index_nplike.ma.zeros(
                        self.length, [(n, np.bool_) for n in self.fields]
                    )
                if x.mask is not None:
                    mask[n] |= x.mask
            out[n] = x

        if mask is not None:
            out = self._nplike.ma.MaskedArray(out, mask)

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
            raise ak._v2._util.error(
                TypeError(
                    "cannot combine record fields{} unless flatten_records=True".format(
                        in_function
                    )
                )
            )

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if self._nplike.known_shape:
            contents = [x[: self._length] for x in self._contents]
        else:
            contents = self._contents

        if options["return_array"]:

            def continuation():
                if not options["allow_records"]:
                    raise ak._v2._util.error(
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
                    self._identifier,
                    self._parameters if options["keep_parameters"] else None,
                    self._nplike,
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
            nplike=self._nplike,
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise ak._v2._util.error(AssertionError(result))

    def packed(self):
        return RecordArray(
            [
                x.packed() if x.length == self._length else x[: self._length].packed()
                for x in self._contents
            ],
            self._fields,
            self._length,
            self._identifier,
            self._parameters,
            self._nplike,
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

    def _to_nplike(self, nplike):
        contents = [content._to_nplike(nplike) for content in self._contents]
        return RecordArray(
            contents,
            self._fields,
            length=self._length,
            identifier=self._identifier,
            parameters=self._parameters,
            nplike=nplike,
        )

    def _layout_equal(self, other, index_dtype=True, numpyarray=True):
        return (
            self.fields == other.fields
            and len(self.contents) == len(other.contents)
            and all(
                [
                    self.contents[i].layout_equal(
                        other.contents[i], index_dtype, numpyarray
                    )
                    for i in range(len(self.contents))
                ]
            )
        )
