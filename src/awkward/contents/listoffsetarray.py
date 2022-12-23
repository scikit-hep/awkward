# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy

import awkward as ak
from awkward._util import unset
from awkward.contents.content import Content
from awkward.forms.listoffsetform import ListOffsetForm
from awkward.index import Index
from awkward.typing import Final, Self

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


class ListOffsetArray(Content):
    is_list = True

    def __init__(self, offsets, content, *, parameters=None):
        if not isinstance(offsets, Index) and offsets.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        ):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'offsets' must be an Index with dtype in (int32, uint32, int64), "
                    "not {}".format(type(self).__name__, repr(offsets))
                )
            )
        if not isinstance(content, Content):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'content' must be a Content subtype, not {}".format(
                        type(self).__name__, repr(content)
                    )
                )
            )
        if offsets.nplike.known_shape and not offsets.length >= 1:
            raise ak._errors.wrap_error(
                ValueError(
                    "{} len(offsets) ({}) must be >= 1".format(
                        type(self).__name__, offsets.length
                    )
                )
            )

        if parameters is not None and parameters.get("__array__") == "string":
            if not content.is_numpy or not content.parameter("__array__") == "char":
                raise ak._errors.wrap_error(
                    ValueError(
                        "{} is a string, so its 'content' must be uint8 NumpyArray of char, not {}".format(
                            type(self).__name__, repr(content)
                        )
                    )
                )
        if parameters is not None and parameters.get("__array__") == "bytestring":
            if not content.is_numpy or not content.parameter("__array__") == "byte":
                raise ak._errors.wrap_error(
                    ValueError(
                        "{} is a bytestring, so its 'content' must be uint8 NumpyArray of byte, not {}".format(
                            type(self).__name__, repr(content)
                        )
                    )
                )

        assert offsets.nplike is content.backend.index_nplike

        self._offsets = offsets
        self._content = content
        self._init(parameters, content.backend)

    @property
    def offsets(self):
        return self._offsets

    @property
    def content(self):
        return self._content

    form_cls: Final = ListOffsetForm

    def copy(self, offsets=unset, content=unset, *, parameters=unset):
        return ListOffsetArray(
            self._offsets if offsets is unset else offsets,
            self._content if content is unset else content,
            parameters=self._parameters if parameters is unset else parameters,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            offsets=copy.deepcopy(self._offsets, memo),
            content=copy.deepcopy(self._content, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(cls, offsets, content, *, parameters=None):
        return cls(offsets, content, parameters=parameters)

    @property
    def starts(self):
        return self._offsets[:-1]

    @property
    def stops(self):
        return self._offsets[1:]

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.form_cls(
            self._offsets.form,
            self._content._form_with_key(getkey),
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, backend):
        assert isinstance(form, self.form_cls)
        key = getkey(self, form, "offsets")
        container[key] = ak._util.little_endian(self._offsets.raw(backend.index_nplike))
        self._content._to_buffers(form.content, getkey, container, backend)

    def _to_typetracer(self, forget_length: bool) -> Self:
        offsets = self._offsets.to_nplike(ak._typetracer.TypeTracer.instance())
        return ListOffsetArray(
            offsets.forget_length() if forget_length else offsets,
            self._content._to_typetracer(False),
            parameters=self._parameters,
        )

    def _touch_data(self, recursive):
        if not self._backend.index_nplike.known_data:
            self._offsets.data.touch_data()
        if recursive:
            self._content._touch_data(recursive)

    def _touch_shape(self, recursive):
        if not self._backend.index_nplike.known_shape:
            self._offsets.data.touch_shape()
        if recursive:
            self._content._touch_shape(recursive)

    @property
    def length(self):
        return self._offsets.length - 1

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ListOffsetArray len="]
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._offsets._repr(indent + "    ", "<offsets>", "</offsets>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</ListOffsetArray>")
        out.append(post)
        return "".join(out)

    def to_ListOffsetArray64(self, start_at_zero=False):
        if not self._backend.nplike.known_data and (
            start_at_zero or self._offsets.dtype != np.dtype(np.int64)
        ):
            self._touch_data(recursive=False)
            self._content._touch_data(recursive=False)

        if issubclass(self._offsets.dtype.type, np.int64):
            if (
                not self._backend.nplike.known_data
                or not start_at_zero
                or self._offsets[0] == 0
            ):
                return self

            if start_at_zero:
                offsets = ak.index.Index64(
                    self._offsets.raw(self._backend.nplike) - self._offsets[0],
                    nplike=self._backend.index_nplike,
                )
                content = self._content[self._offsets[0] :]
            else:
                offsets, content = self._offsets, self._content

            return ListOffsetArray(offsets, content, parameters=self._parameters)

        else:
            offsets = self._compact_offsets64(start_at_zero)
            return self._broadcast_tooffsets64(offsets)

    def to_RegularArray(self):
        start, stop = self._offsets[0], self._offsets[self._offsets.length - 1]
        content = self._content._getitem_range(slice(start, stop))
        size = ak.index.Index64.empty(1, self._backend.index_nplike)
        assert (
            size.nplike is self._backend.index_nplike
            and self._offsets.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_ListOffsetArray_toRegularArray",
                size.dtype.type,
                self._offsets.dtype.type,
            ](
                size.data,
                self._offsets.data,
                self._offsets.length,
            )
        )

        return ak.contents.RegularArray(
            content, size[0], self._offsets.length - 1, parameters=self._parameters
        )

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if not self._backend.nplike.known_data:
            self._touch_data(recursive=False)
            return self._content._getitem_range(slice(0, 0))

        if where < 0:
            where += self.length
        if not (0 <= where < self.length) and self._backend.nplike.known_shape:
            raise ak._errors.index_error(self, where)
        start, stop = self._offsets[where], self._offsets[where + 1]
        return self._content._getitem_range(slice(start, stop))

    def _getitem_range(self, where):
        if not self._backend.nplike.known_shape:
            self._touch_shape(recursive=False)
            return self

        start, stop, step = where.indices(self.length)
        offsets = self._offsets[start : stop + 1]
        if offsets.length == 0:
            offsets = Index(
                self._backend.index_nplike.array([0], dtype=self._offsets.dtype),
                nplike=self._backend.index_nplike,
            )
        return ListOffsetArray(offsets, self._content, parameters=self._parameters)

    def _getitem_field(self, where, only_fields=()):
        return ListOffsetArray(
            self._offsets,
            self._content._getitem_field(where, only_fields),
            parameters=None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return ListOffsetArray(
            self._offsets,
            self._content._getitem_fields(where, only_fields),
            parameters=None,
        )

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak.index.Index)

        try:
            nextstarts = self.starts[carry.data]
            nextstops = self.stops[carry.data]
        except IndexError as err:
            raise ak._errors.index_error(self, carry.data, str(err)) from err

        return ak.contents.ListArray(
            nextstarts, nextstops, self._content, parameters=self._parameters
        )

    def _compact_offsets64(self, start_at_zero):
        offsets_len = self._offsets.length - 1
        out = ak.index.Index64.empty(offsets_len + 1, self._backend.index_nplike)
        assert (
            out.nplike is self._backend.index_nplike
            and self._offsets.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_ListOffsetArray_compact_offsets",
                out.dtype.type,
                self._offsets.dtype.type,
            ](out.data, self._offsets.data, offsets_len)
        )
        return out

    def _broadcast_tooffsets64(self, offsets):
        if offsets.nplike.known_data and (offsets.length == 0 or offsets[0] != 0):
            raise ak._errors.wrap_error(
                AssertionError(
                    "broadcast_tooffsets64 can only be used with offsets that start at 0, not {}".format(
                        "(empty)" if offsets.length == 0 else str(offsets[0])
                    )
                )
            )

        if offsets.nplike.known_shape and offsets.length - 1 != self.length:
            raise ak._errors.wrap_error(
                AssertionError(
                    "cannot broadcast {} of length {} to length {}".format(
                        type(self).__name__, self.length, offsets.length - 1
                    )
                )
            )

        starts, stops = self.starts, self.stops

        nextcarry = ak.index.Index64.empty(offsets[-1], self._backend.index_nplike)
        assert (
            nextcarry.nplike is self._backend.index_nplike
            and offsets.nplike is self._backend.index_nplike
            and starts.nplike is self._backend.index_nplike
            and stops.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_ListArray_broadcast_tooffsets",
                nextcarry.dtype.type,
                offsets.dtype.type,
                starts.dtype.type,
                stops.dtype.type,
            ](
                nextcarry.data,
                offsets.data,
                offsets.length,
                starts.data,
                stops.data,
                self._content.length,
            )
        )

        nextcontent = self._content._carry(nextcarry, True)

        return ListOffsetArray(offsets, nextcontent, parameters=self._parameters)

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        out = ak.contents.ListArray(
            self.starts, self.stops, self._content, parameters=self._parameters
        )
        return out._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

    def _getitem_next(self, head, tail, advanced):
        advanced = advanced.to_nplike(self._backend.nplike)
        if head == ():
            return self

        elif isinstance(head, int):
            assert advanced is None
            lenstarts = self._offsets.length - 1
            starts, stops = self.starts, self.stops
            nexthead, nexttail = ak._slicing.headtail(tail)
            nextcarry = ak.index.Index64.empty(lenstarts, self._backend.index_nplike)

            assert (
                nextcarry.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
                and stops.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListArray_getitem_next_at",
                    nextcarry.dtype.type,
                    starts.dtype.type,
                    stops.dtype.type,
                ](
                    nextcarry.data,
                    starts.data,
                    stops.data,
                    lenstarts,
                    head,
                ),
                slicer=head,
            )
            nextcontent = self._content._carry(nextcarry, True)
            return nextcontent._getitem_next(nexthead, nexttail, advanced)

        elif isinstance(head, slice):
            nexthead, nexttail = ak._slicing.headtail(tail)
            lenstarts = self._offsets.length - 1
            start, stop, step = head.start, head.stop, head.step

            step = 1 if step is None else step
            start = ak._util.kSliceNone if start is None else start
            stop = ak._util.kSliceNone if stop is None else stop

            carrylength = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert (
                carrylength.nplike is self._backend.index_nplike
                and self.starts.nplike is self._backend.index_nplike
                and self.stops.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListArray_getitem_next_range_carrylength",
                    carrylength.dtype.type,
                    self.starts.dtype.type,
                    self.stops.dtype.type,
                ](
                    carrylength.data,
                    self.starts.data,
                    self.stops.data,
                    lenstarts,
                    start,
                    stop,
                    step,
                ),
                slicer=head,
            )

            if self._starts.dtype == "int64":
                nextoffsets = ak.index.Index64.empty(
                    lenstarts + 1, nplike=self._backend.index_nplike
                )
            elif self._starts.dtype == "int32":
                nextoffsets = ak.index.Index32.empty(
                    lenstarts + 1, nplike=self._backend.index_nplike
                )
            elif self._starts.dtype == "uint32":
                nextoffsets = ak.index.IndexU32.empty(
                    lenstarts + 1, nplike=self._backend.index_nplike
                )
            nextcarry = ak.index.Index64.empty(
                carrylength[0], self._backend.index_nplike
            )

            assert (
                nextoffsets.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
                and self.starts.nplike is self._backend.index_nplike
                and self.stops.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListArray_getitem_next_range",
                    nextoffsets.dtype.type,
                    nextcarry.dtype.type,
                    self.starts.dtype.type,
                    self.stops.dtype.type,
                ](
                    nextoffsets.data,
                    nextcarry.data,
                    self.starts.data,
                    self.stops.data,
                    lenstarts,
                    start,
                    stop,
                    step,
                ),
                slicer=head,
            )

            nextcontent = self._content._carry(nextcarry, True)

            if advanced is None or advanced.length == 0:
                return ak.contents.ListOffsetArray(
                    nextoffsets,
                    nextcontent._getitem_next(nexthead, nexttail, advanced),
                    parameters=self._parameters,
                )

            else:
                total = ak.index.Index64.empty(1, self._backend.index_nplike)
                assert (
                    total.nplike is self._backend.index_nplike
                    and nextoffsets.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_ListArray_getitem_next_range_counts",
                        total.dtype.type,
                        nextoffsets.dtype.type,
                    ](
                        total.data,
                        nextoffsets.data,
                        lenstarts,
                    ),
                    slicer=head,
                )

                nextadvanced = ak.index.Index64.empty(
                    total[0], self._backend.index_nplike
                )
                assert (
                    nextadvanced.nplike is self._backend.index_nplike
                    and advanced.nplike is self._backend.index_nplike
                    and nextoffsets.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_ListArray_getitem_next_range_spreadadvanced",
                        nextadvanced.dtype.type,
                        advanced.dtype.type,
                        nextoffsets.dtype.type,
                    ](
                        nextadvanced.data,
                        advanced.data,
                        nextoffsets.data,
                        lenstarts,
                    ),
                    slicer=head,
                )

                return ak.contents.ListOffsetArray(
                    nextoffsets,
                    nextcontent._getitem_next(nexthead, nexttail, nextadvanced),
                    parameters=self._parameters,
                )

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak.index.Index64):
            nexthead, nexttail = ak._slicing.headtail(tail)
            flathead = self._backend.index_nplike.asarray(head.data.reshape(-1))
            lenstarts = self.starts.length
            regular_flathead = ak.index.Index64(flathead)
            if advanced is None or advanced.length == 0:
                nextcarry = ak.index.Index64.empty(
                    lenstarts * flathead.length, self._backend.index_nplike
                )
                nextadvanced = ak.index.Index64.empty(
                    lenstarts * flathead.length, self._backend.index_nplike
                )
                assert (
                    nextcarry.nplike is self._backend.index_nplike
                    and nextadvanced.nplike is self._backend.index_nplike
                    and regular_flathead.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_ListArray_getitem_next_array",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.data,
                        nextadvanced.data,
                        self.starts.data,
                        self.stops.data,
                        regular_flathead.data,
                        lenstarts,
                        regular_flathead.length,
                        self._content.length,
                    ),
                    slicer=head,
                )
                nextcontent = self._content._carry(nextcarry, True)

                out = nextcontent._getitem_next(nexthead, nexttail, nextadvanced)
                if advanced is None:
                    return ak._slicing.getitem_next_array_wrap(
                        out, head.metadata.get("shape", (head.length,), self.length)
                    )
                else:
                    return out

            else:
                nextcarry = ak.index.Index64.empty(
                    self.length, self._backend.index_nplike
                )
                nextadvanced = ak.index.Index64.empty(
                    self.length, self._backend.index_nplike
                )
                assert (
                    nextcarry.nplike is self._backend.index_nplike
                    and nextadvanced.nplike is self._backend.index_nplike
                    and self.starts.nplike is self._backend.index_nplike
                    and self.stops.nplike is self._backend.index_nplike
                    and regular_flathead.nplike is self._backend.index_nplike
                    and advanced.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_ListArray_getitem_next_array_advanced",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        self.starts.dtype.type,
                        self.stops.dtype.type,
                        regular_flathead.dtype.type,
                        advanced.dtype.type,
                    ](
                        nextcarry.data,
                        nextadvanced.data,
                        self.starts.data,
                        self.stops.data,
                        regular_flathead.data,
                        advanced.data,
                        lenstarts,
                        regular_flathead.length,
                        self._content.length,
                    ),
                    slicer=head,
                )
                nextcontent = self._content._carry(nextcarry, True)
                return nextcontent._getitem_next(nexthead, nexttail, nextadvanced)

        elif isinstance(head, ak.contents.ListOffsetArray):
            listarray = ak.contents.ListArray(
                self.starts, self.stops, self._content, parameters=self._parameters
            )
            return listarray._getitem_next(head, tail, advanced)

        elif isinstance(head, ak.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise ak._errors.wrap_error(AssertionError(repr(head)))

    def _offsets_and_flattened(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            raise ak._errors.wrap_error(np.AxisError("axis=0 not allowed for flatten"))

        elif posaxis is not None and posaxis + 1 == depth + 1:
            listoffsetarray = self.to_ListOffsetArray64(True)
            stop = listoffsetarray.offsets[-1]
            content = listoffsetarray.content._getitem_range(slice(0, stop))
            return (listoffsetarray.offsets, content)

        else:
            inneroffsets, flattened = self._content._offsets_and_flattened(
                axis, depth + 1
            )
            offsets = ak.index.Index64.zeros(
                0,
                nplike=self._backend.index_nplike,
                dtype=np.int64,
            )

            if inneroffsets.length == 0:
                return (
                    offsets,
                    ListOffsetArray(
                        self._offsets, flattened, parameters=self._parameters
                    ),
                )

            elif self._offsets.length == 1:
                tooffsets = ak.index.Index64([inneroffsets[0]])
                return (
                    offsets,
                    ListOffsetArray(tooffsets, flattened, parameters=self._parameters),
                )

            else:
                tooffsets = ak.index.Index64.empty(
                    self._offsets.length,
                    self._backend.index_nplike,
                    dtype=np.int64,
                )
                assert (
                    tooffsets.nplike is self._backend.index_nplike
                    and self._offsets.nplike is self._backend.index_nplike
                    and inneroffsets.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_ListOffsetArray_flatten_offsets",
                        tooffsets.dtype.type,
                        self._offsets.dtype.type,
                        inneroffsets.dtype.type,
                    ](
                        tooffsets.data,
                        self._offsets.data,
                        self._offsets.length,
                        inneroffsets.data,
                        inneroffsets.length,
                    )
                )
                return (
                    offsets,
                    ListOffsetArray(tooffsets, flattened, parameters=self._parameters),
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

        if isinstance(
            other,
            (
                ak.contents.RegularArray,
                ak.contents.ListArray,
                ak.contents.ListOffsetArray,
            ),
        ):
            return self._content._mergeable(other.content, mergebool)

        else:
            return False

    def _mergemany(self, others):
        if len(others) == 0:
            return self
        listarray = ak.contents.ListArray(
            self.starts, self.stops, self._content, parameters=self._parameters
        )
        out = listarray._mergemany(others)

        if all(
            isinstance(x, ListOffsetArray) and x._offsets.dtype == self._offsets.dtype
            for x in others
        ):
            return out.to_ListOffsetArray64(False)
        else:
            return out

    def _fill_none(self, value: Content) -> Content:
        return ListOffsetArray(
            self._offsets, self._content._fill_none(value), parameters=self._parameters
        )

    def _local_index(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        elif posaxis is not None and posaxis + 1 == depth + 1:
            offsets = self._compact_offsets64(True)
            if self._backend.nplike.known_data:
                innerlength = offsets[offsets.length - 1]
            else:
                self._touch_data(recursive=False)
                innerlength = ak._typetracer.UnknownLength
            localindex = ak.index.Index64.empty(innerlength, self._backend.index_nplike)
            assert (
                localindex.nplike is self._backend.index_nplike
                and offsets.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListArray_localindex",
                    localindex.dtype.type,
                    offsets.dtype.type,
                ](
                    localindex.data,
                    offsets.data,
                    offsets.length - 1,
                )
            )
            return ak.contents.ListOffsetArray(
                offsets, ak.contents.NumpyArray(localindex)
            )
        else:
            return ak.contents.ListOffsetArray(
                self._offsets, self._content._local_index(axis, depth + 1)
            )

    def _numbers_to_type(self, name):
        return ak.contents.ListOffsetArray(
            self._offsets,
            self._content._numbers_to_type(name),
            parameters=self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._offsets.length - 1 == 0:
            return True

        branch, depth = self.branch_depth

        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            if branch or (negaxis is not None and negaxis != depth):
                raise ak._errors.wrap_error(
                    ValueError(
                        "array with strings can only be checked on uniqueness with axis=-1"
                    )
                )

            # FIXME: check validity error

            if isinstance(self._content, ak.contents.NumpyArray):
                out, outoffsets = self._content._as_unique_strings(self._offsets)
                out2 = ak.contents.ListOffsetArray(
                    outoffsets, out, parameters=self._parameters
                )
                return out2.length == self.length

        if negaxis is None:
            return self._content._is_unique(negaxis, starts, parents, outlength)

        if not branch and (negaxis == depth):
            return self._content._is_unique(negaxis - 1, starts, parents, outlength)
        else:
            nextparents = ak.index.Index64.empty(
                self._offsets[-1] - self._offsets[0], self._backend.index_nplike
            )

            assert (
                nextparents.nplike is self._backend.index_nplike
                and self._offsets.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.data,
                    self._offsets.data,
                    self._offsets.length - 1,
                )
            )
            starts = self._offsets[:-1]

            return self._content._is_unique(negaxis, starts, nextparents, outlength)

    def _unique(self, negaxis, starts, parents, outlength):
        if self._offsets.length - 1 == 0:
            return self

        branch, depth = self.branch_depth

        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            if branch or (negaxis != depth):
                raise ak._errors.wrap_error(
                    np.AxisError("array with strings can only be sorted with axis=-1")
                )

            # FIXME: check validity error

            if isinstance(self._content, ak.contents.NumpyArray):
                out, nextoffsets = self._content._as_unique_strings(self._offsets)
                return ak.contents.ListOffsetArray(
                    nextoffsets, out, parameters=self._parameters
                )

        if not branch and (negaxis == depth):
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise ak._errors.wrap_error(
                    np.AxisError("array with strings can only be sorted with axis=-1")
                )

            if self._backend.nplike.known_shape and parents.nplike.known_shape:
                assert self._offsets.length - 1 == parents.length

            (
                distincts,
                maxcount,
                maxnextparents,
                nextcarry,
                nextparents,
                nextstarts,
            ) = self._rearrange_prepare_next(outlength, parents)

            nextcontent = self._content._carry(nextcarry, False)
            outcontent = nextcontent._unique(
                negaxis - 1,
                nextstarts,
                nextparents,
                maxnextparents[0] + 1,
            )

            outcarry = ak.index.Index64.empty(
                nextcarry.length, self._backend.index_nplike
            )
            assert (
                outcarry.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListOffsetArray_local_preparenext_64",
                    outcarry.dtype.type,
                    nextcarry.dtype.type,
                ](
                    outcarry.data,
                    nextcarry.data,
                    nextcarry.length,
                )
            )

            return ak.contents.ListOffsetArray(
                outcontent._compact_offsets64(True),
                outcontent._content._carry(outcarry, False),
                parameters=self._parameters,
            )

        else:
            nextparents = ak.index.Index64.empty(
                self._offsets[-1] - self._offsets[0], self._backend.index_nplike
            )

            assert (
                nextparents.nplike is self._backend.index_nplike
                and self._offsets.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.data,
                    self._offsets.data,
                    self._offsets.length - 1,
                )
            )

            trimmed = self._content[self._offsets[0] : self._offsets[-1]]
            outcontent = trimmed._unique(
                negaxis,
                self._offsets[:-1],
                nextparents,
                self._offsets.length - 1,
            )

            if negaxis is None or negaxis == depth - 1:
                return outcontent

            outoffsets = self._compact_offsets64(True)
            return ak.contents.ListOffsetArray(
                outoffsets, outcontent, parameters=self._parameters
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
        branch, depth = self.branch_depth

        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            if branch or (negaxis != depth):
                raise ak._errors.wrap_error(
                    np.AxisError("array with strings can only be sorted with axis=-1")
                )

            # FIXME: check validity error

            if isinstance(self._content, ak.contents.NumpyArray):
                nextcarry = ak.index.Index64.empty(
                    self._offsets.length - 1, self._backend.index_nplike
                )

                self_starts, self_stops = self._offsets[:-1], self._offsets[1:]
                assert (
                    nextcarry.nplike is self._backend.index_nplike
                    and parents.nplike is self._backend.index_nplike
                    and self._content.backend is self._backend
                    and self_starts.nplike is self._backend.index_nplike
                    and self_stops.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_ListOffsetArray_argsort_strings",
                        nextcarry.dtype.type,
                        parents.dtype.type,
                        self._content.dtype.type,
                        self_starts.dtype.type,
                        self_stops.dtype.type,
                    ](
                        nextcarry.data,
                        parents.data,
                        parents.length,
                        self._content._data,
                        self_starts.data,
                        self_stops.data,
                        stable,
                        ascending,
                        True,
                    )
                )
                return ak.contents.NumpyArray(
                    nextcarry, parameters=None, backend=self._backend
                )

        if not branch and (negaxis == depth):
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise ak._errors.wrap_error(
                    np.AxisError("array with strings can only be sorted with axis=-1")
                )

            if self._backend.nplike.known_shape and parents.nplike.known_shape:
                assert self._offsets.length - 1 == parents.length

            (
                distincts,
                maxcount,
                maxnextparents,
                nextcarry,
                nextparents,
                nextstarts,
            ) = self._rearrange_prepare_next(outlength, parents)

            nummissing = ak.index.Index64.empty(maxcount, self._backend.index_nplike)
            missing = ak.index.Index64.empty(
                self._offsets[-1], self._backend.index_nplike
            )
            nextshifts = ak.index.Index64.empty(
                nextcarry.length, self._backend.index_nplike
            )
            assert (
                nummissing.nplike is self._backend.index_nplike
                and missing.nplike is self._backend.index_nplike
                and nextshifts.nplike is self._backend.index_nplike
                and self._offsets.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
            )

            self._handle_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64",
                    nummissing.dtype.type,
                    missing.dtype.type,
                    nextshifts.dtype.type,
                    self._offsets.dtype.type,
                    starts.dtype.type,
                    parents.dtype.type,
                    nextcarry.dtype.type,
                ](
                    nummissing.data,
                    missing.data,
                    nextshifts.data,
                    self._offsets.data,
                    self._offsets.length - 1,
                    starts.data,
                    parents.data,
                    maxcount,
                    nextcarry.length,
                    nextcarry.data,
                )
            )

            nextcontent = self._content._carry(nextcarry, False)
            outcontent = nextcontent._argsort_next(
                negaxis - 1,
                nextstarts,
                nextshifts,
                nextparents,
                nextstarts.length,
                ascending,
                stable,
                kind,
                order,
            )

            outcarry = ak.index.Index64.empty(
                nextcarry.length, self._backend.index_nplike
            )
            assert (
                outcarry.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListOffsetArray_local_preparenext_64",
                    outcarry.dtype.type,
                    nextcarry.dtype.type,
                ](
                    outcarry.data,
                    nextcarry.data,
                    nextcarry.length,
                )
            )

            out_offsets = self._compact_offsets64(True)
            out = outcontent._carry(outcarry, False)
            return ak.contents.ListOffsetArray(
                out_offsets, out, parameters=self._parameters
            )
        else:
            nextparents = ak.index.Index64.empty(
                self._offsets[-1] - self._offsets[0], self._backend.index_nplike
            )

            assert (
                nextparents.nplike is self._backend.index_nplike
                and self._offsets.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.data,
                    self._offsets.data,
                    self._offsets.length - 1,
                )
            )

            trimmed = self._content[self._offsets[0] : self._offsets[-1]]
            outcontent = trimmed._argsort_next(
                negaxis,
                self._offsets[:-1],
                shifts,
                nextparents,
                self._offsets.length - 1,
                ascending,
                stable,
                kind,
                order,
            )
            outoffsets = self._compact_offsets64(True)
            return ak.contents.ListOffsetArray(
                outoffsets, outcontent, parameters=self._parameters
            )

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        branch, depth = self.branch_depth

        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            if branch or (negaxis != depth):
                raise ak._errors.wrap_error(
                    np.AxisError("array with strings can only be sorted with axis=-1")
                )

            # FIXME: check validity error

            if isinstance(self._content, ak.contents.NumpyArray):
                nextcarry = ak.index.Index64.empty(
                    self._offsets.length - 1, self._backend.index_nplike
                )

                starts, stops = self._offsets[:-1], self._offsets[1:]
                assert (
                    nextcarry.nplike is self._backend.index_nplike
                    and parents.nplike is self._backend.index_nplike
                    and self._content.backend is self._backend
                    and starts.nplike is self._backend.index_nplike
                    and stops.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_ListOffsetArray_argsort_strings",
                        nextcarry.dtype.type,
                        parents.dtype.type,
                        self._content.dtype.type,
                        starts.dtype.type,
                        stops.dtype.type,
                    ](
                        nextcarry.data,
                        parents.data,
                        parents.length,
                        self._content._data,
                        starts.data,
                        stops.data,
                        stable,
                        ascending,
                        False,
                    )
                )
                return self._carry(nextcarry, False)

        if not branch and (negaxis == depth):
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise ak._errors.wrap_error(
                    np.AxisError("array with strings can only be sorted with axis=-1")
                )

            if self._backend.nplike.known_shape and parents.nplike.known_shape:
                assert self._offsets.length - 1 == parents.length

            (
                distincts,
                maxcount,
                maxnextparents,
                nextcarry,
                nextparents,
                nextstarts,
            ) = self._rearrange_prepare_next(outlength, parents)

            nextcontent = self._content._carry(nextcarry, False)
            outcontent = nextcontent._sort_next(
                negaxis - 1,
                nextstarts,
                nextparents,
                maxnextparents[0] + 1,
                ascending,
                stable,
                kind,
                order,
            )

            outcarry = ak.index.Index64.empty(
                nextcarry.length, self._backend.index_nplike
            )
            assert (
                outcarry.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListOffsetArray_local_preparenext_64",
                    outcarry.dtype.type,
                    nextcarry.dtype.type,
                ](
                    outcarry.data,
                    nextcarry.data,
                    nextcarry.length,
                )
            )

            return ak.contents.ListOffsetArray(
                self._compact_offsets64(True),
                outcontent._carry(outcarry, False),
                parameters=self._parameters,
            )
        else:
            nextparents = ak.index.Index64.empty(
                self._offsets[-1] - self._offsets[0], self._backend.index_nplike
            )

            assert (
                nextparents.nplike is self._backend.index_nplike
                and self._offsets.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.data,
                    self._offsets.data,
                    self._offsets.length - 1,
                )
            )

            trimmed = self._content[self._offsets[0] : self._offsets[-1]]
            outcontent = trimmed._sort_next(
                negaxis,
                self._offsets[:-1],
                nextparents,
                self._offsets.length - 1,
                ascending,
                stable,
                kind,
                order,
            )
            outoffsets = self._compact_offsets64(True)
            return ak.contents.ListOffsetArray(
                outoffsets, outcontent, parameters=self._parameters
            )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        elif posaxis is not None and posaxis + 1 == depth + 1:
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise ak._errors.wrap_error(
                    ValueError(
                        "ak.combinations does not compute combinations of the characters of a string; please split it into lists"
                    )
                )

            starts = self.starts
            stops = self.stops

            totallen = ak.index.Index64.empty(
                1, self._backend.index_nplike, dtype=np.int64
            )
            offsets = ak.index.Index64.empty(
                self.length + 1,
                self._backend.index_nplike,
                dtype=np.int64,
            )
            assert (
                offsets.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
                and stops.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListArray_combinations_length",
                    totallen.data.dtype.type,
                    offsets.data.dtype.type,
                    starts.data.dtype.type,
                    stops.data.dtype.type,
                ](
                    totallen.data,
                    offsets.data,
                    n,
                    replacement,
                    starts.data,
                    stops.data,
                    self.length,
                )
            )

            tocarryraw = ak.index.Index.empty(
                n, dtype=np.intp, nplike=self._backend.index_nplike
            )
            tocarry = []

            for i in range(n):
                ptr = ak.index.Index64.empty(
                    totallen[0],
                    nplike=self._backend.index_nplike,
                    dtype=np.int64,
                )
                tocarry.append(ptr)
                if self._backend.nplike.known_data:
                    tocarryraw[i] = ptr.ptr

            toindex = ak.index.Index64.empty(
                n, self._backend.index_nplike, dtype=np.int64
            )
            fromindex = ak.index.Index64.empty(
                n, self._backend.index_nplike, dtype=np.int64
            )
            assert (
                toindex.nplike is self._backend.index_nplike
                and fromindex.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
                and stops.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListArray_combinations",
                    np.int64,
                    toindex.data.dtype.type,
                    fromindex.data.dtype.type,
                    starts.data.dtype.type,
                    stops.data.dtype.type,
                ](
                    tocarryraw.data,
                    toindex.data,
                    fromindex.data,
                    n,
                    replacement,
                    starts.data,
                    stops.data,
                    self.length,
                )
            )
            contents = []

            for ptr in tocarry:
                contents.append(self._content._carry(ptr, True))

            recordarray = ak.contents.RecordArray(
                contents,
                recordlookup,
                None,
                parameters=parameters,
                backend=self._backend,
            )
            return ak.contents.ListOffsetArray(
                offsets, recordarray, parameters=self._parameters
            )
        else:
            compact = self.to_ListOffsetArray64(True)
            next = compact._content._combinations(
                n, replacement, recordlookup, parameters, axis, depth + 1
            )
            return ak.contents.ListOffsetArray(
                compact.offsets, next, parameters=self._parameters
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
        if self._offsets.dtype != np.dtype(np.int64) or (
            self._offsets.nplike.known_data and self._offsets[0] != 0
        ):
            next = self.to_ListOffsetArray64(True)
            return next._reduce_next(
                reducer,
                negaxis,
                starts,
                shifts,
                parents,
                outlength,
                mask,
                keepdims,
                behavior,
            )

        branch, depth = self.branch_depth
        globalstarts_length = self._offsets.length - 1

        if not branch and negaxis == depth:
            (
                distincts,
                maxcount,
                maxnextparents,
                nextcarry,
                nextparents,
                nextstarts,
            ) = self._rearrange_prepare_next(outlength, parents)

            outstarts = ak.index.Index64.empty(outlength, self._backend.index_nplike)
            outstops = ak.index.Index64.empty(outlength, self._backend.index_nplike)
            assert (
                outstarts.nplike is self._backend.index_nplike
                and outstops.nplike is self._backend.index_nplike
                and distincts.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",
                    outstarts.dtype.type,
                    outstops.dtype.type,
                    distincts.dtype.type,
                ](
                    outstarts.data,
                    outstops.data,
                    distincts.data,
                    distincts.length,
                    outlength,
                )
            )

            if reducer.needs_position:
                nextshifts = ak.index.Index64.empty(
                    nextcarry.length, self._backend.index_nplike
                )
                nummissing = ak.index.Index64.empty(
                    maxcount, self._backend.index_nplike
                )
                missing = ak.index.Index64.empty(
                    self._offsets[-1], self._backend.index_nplike
                )
                assert (
                    nummissing.nplike is self._backend.index_nplike
                    and missing.nplike is self._backend.index_nplike
                    and nextshifts.nplike is self._backend.index_nplike
                    and self._offsets.nplike is self._backend.index_nplike
                    and starts.nplike is self._backend.index_nplike
                    and parents.nplike is self._backend.index_nplike
                    and nextcarry.nplike is self._backend.index_nplike
                )

                self._handle_error(
                    self._backend[
                        "awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64",
                        nummissing.dtype.type,
                        missing.dtype.type,
                        nextshifts.dtype.type,
                        self._offsets.dtype.type,
                        starts.dtype.type,
                        parents.dtype.type,
                        nextcarry.dtype.type,
                    ](
                        nummissing.data,
                        missing.data,
                        nextshifts.data,
                        self._offsets.data,
                        globalstarts_length,
                        starts.data,
                        parents.data,
                        maxcount,
                        nextcarry.length,
                        nextcarry.data,
                    )
                )
            else:
                nextshifts = None

            nextcontent = self._content._carry(nextcarry, False)
            outcontent = nextcontent._reduce_next(
                reducer,
                negaxis - 1,
                nextstarts,
                nextshifts,
                nextparents,
                maxnextparents[0] + 1,
                mask,
                False,
                behavior,
            )

            out = ak.contents.ListArray(
                outstarts, outstops, outcontent, parameters=None
            )

            if keepdims:
                out = ak.contents.RegularArray(out, 1, self.length, parameters=None)

            return out

        else:
            nextlen = self._offsets[-1] - self._offsets[0]
            nextparents = ak.index.Index64.empty(nextlen, self._backend.index_nplike)

            assert (
                nextparents.nplike is self._backend.index_nplike
                and self._offsets.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.data,
                    self._offsets.data,
                    globalstarts_length,
                )
            )

            trimmed = self._content[self.offsets[0] : self.offsets[-1]]
            nextstarts = self.offsets[:-1]

            outcontent = trimmed._reduce_next(
                reducer,
                negaxis,
                nextstarts,
                shifts,
                nextparents,
                globalstarts_length,
                mask,
                keepdims,
                behavior,
            )

            outoffsets = ak.index.Index64.empty(
                outlength + 1, self._backend.index_nplike
            )
            assert (
                outoffsets.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_outoffsets_64",
                    outoffsets.dtype.type,
                    parents.dtype.type,
                ](
                    outoffsets.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )

            # `outcontent` represents *this* layout in the reduction
            # If this layout survives in the reduction (see `if` below), then we want
            # to ensure that we have a ragged list type (unless it's a `keepdims=True` layout)
            if keepdims and depth == negaxis + 1:
                # Don't convert the `RegularArray()` to a `ListOffsetArray`,
                # means this will be broadcastable
                assert outcontent.is_regular
            elif depth >= negaxis + 2:
                # The *only* >1D list types that we can have as direct children
                # are the `is_list` or `is_regular` types; NumpyArray should be
                # converted to `RegularArray`.
                assert outcontent.is_list or outcontent.is_regular
                outcontent = outcontent.to_ListOffsetArray64(False)

            return ak.contents.ListOffsetArray(outoffsets, outcontent, parameters=None)

    def _rearrange_prepare_next(self, outlength, parents):
        nextlen = self._offsets[-1] - self._offsets[0]
        maxcount = ak.index.Index64.empty(1, self._backend.index_nplike)
        offsetscopy = ak.index.Index64.empty(
            self.offsets.length, self._backend.index_nplike
        )
        assert (
            maxcount.nplike is self._backend.index_nplike
            and offsetscopy.nplike is self._backend.index_nplike
            and self._offsets.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64",
                maxcount.dtype.type,
                offsetscopy.dtype.type,
                self._offsets.dtype.type,
            ](
                maxcount.data,
                offsetscopy.data,
                self._offsets.data,
                self._offsets.length - 1,
            )
        )

        # A "stable" sort is essential for the subsequent steps.
        nextcarry = ak.index.Index64.empty(nextlen, nplike=self._backend.index_nplike)
        nextparents = ak.index.Index64.empty(nextlen, nplike=self._backend.index_nplike)
        maxnextparents = ak.index.Index64.empty(1, self._backend.index_nplike)
        distincts = ak.index.Index64.empty(
            outlength * maxcount[0], self._backend.index_nplike
        )
        assert (
            maxnextparents.nplike is self._backend.index_nplike
            and distincts.nplike is self._backend.index_nplike
            and self._offsets.nplike is self._backend.index_nplike
            and offsetscopy.nplike is self._backend.index_nplike
            and parents.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_ListOffsetArray_reduce_nonlocal_preparenext_64",
                nextcarry.dtype.type,
                nextparents.dtype.type,
                maxnextparents.dtype.type,
                distincts.dtype.type,
                self._offsets.dtype.type,
                offsetscopy.dtype.type,
                parents.dtype.type,
            ](
                nextcarry.data,
                nextparents.data,
                nextlen,
                maxnextparents.data,
                distincts.data,
                distincts.length,
                offsetscopy.data,
                self._offsets.data,
                self._offsets.length - 1,
                parents.data,
                maxcount[0],
            )
        )
        nextstarts = ak.index.Index64.empty(
            maxnextparents[0] + 1, self._backend.index_nplike
        )
        assert (
            nextstarts.nplike is self._backend.index_nplike
            and nextparents.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64",
                nextstarts.dtype.type,
                nextparents.dtype.type,
            ](
                nextstarts.data,
                nextparents.data,
                nextlen,
            )
        )
        return (
            distincts,
            maxcount[0],
            maxnextparents,
            nextcarry,
            nextparents,
            nextstarts,
        )

    def _validity_error(self, path):
        if self.offsets.length < 1:
            return f'at {path} ("{type(self)}"): len(offsets) < 1'
        assert (
            self.starts.nplike is self._backend.index_nplike
            and self.stops.nplike is self._backend.index_nplike
        )
        error = self._backend[
            "awkward_ListArray_validity", self.starts.dtype.type, self.stops.dtype.type
        ](
            self.starts.data,
            self.stops.data,
            self.starts.length,
            self._content.length,
        )
        if error.str is not None:
            if error.filename is None:
                filename = ""
            else:
                filename = " (in compiled code: " + error.filename.decode(
                    errors="surrogateescape"
                ).lstrip("\n").lstrip("(")
            message = error.str.decode(errors="surrogateescape")
            return 'at {} ("{}"): {} at i={}{}'.format(
                path, type(self), message, error.id, filename
            )
        else:
            return self._content._validity_error(path + ".content")

    def _nbytes_part(self):
        return self.offsets._nbytes_part() + self.content._nbytes_part()

    def _pad_none(self, target, axis, depth, clip):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._pad_none_axis0(target, clip)
        if posaxis is not None and posaxis + 1 == depth + 1:
            if not clip:
                tolength = ak.index.Index64.empty(1, self._backend.index_nplike)
                offsets_ = ak.index.Index64.empty(
                    self._offsets.length, self._backend.index_nplike
                )
                assert (
                    offsets_.nplike is self._backend.index_nplike
                    and self._offsets.nplike is self._backend.index_nplike
                    and tolength.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_ListOffsetArray_rpad_length_axis1",
                        offsets_.dtype.type,
                        self._offsets.dtype.type,
                        tolength.dtype.type,
                    ](
                        offsets_.data,
                        self._offsets.data,
                        self._offsets.length - 1,
                        target,
                        tolength.data,
                    )
                )

                outindex = ak.index.Index64.empty(
                    tolength[0], self._backend.index_nplike
                )
                assert (
                    outindex.nplike is self._backend.index_nplike
                    and self._offsets.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_ListOffsetArray_rpad_axis1",
                        outindex.dtype.type,
                        self._offsets.dtype.type,
                    ](
                        outindex.data,
                        self._offsets.data,
                        self._offsets.length - 1,
                        target,
                    )
                )
                next = ak.contents.IndexedOptionArray.simplified(
                    outindex, self._content, parameters=self._parameters
                )
                return ak.contents.ListOffsetArray(
                    offsets_, next, parameters=self._parameters
                )
            else:
                starts_ = ak.index.Index64.empty(
                    self._offsets.length - 1, self._backend.index_nplike
                )
                stops_ = ak.index.Index64.empty(
                    self._offsets.length - 1, self._backend.index_nplike
                )
                assert (
                    starts_.nplike is self._backend.index_nplike
                    and stops_.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_index_rpad_and_clip_axis1",
                        starts_.dtype.type,
                        stops_.dtype.type,
                    ](
                        starts_.data,
                        stops_.data,
                        target,
                        starts_.length,
                    )
                )

                outindex = ak.index.Index64.empty(
                    target * (self._offsets.length - 1), self._backend.index_nplike
                )
                assert (
                    outindex.nplike is self._backend.index_nplike
                    and self._offsets.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_ListOffsetArray_rpad_and_clip_axis1",
                        outindex.dtype.type,
                        self._offsets.dtype.type,
                    ](
                        outindex.data,
                        self._offsets.data,
                        self._offsets.length - 1,
                        target,
                    )
                )
                next = ak.contents.IndexedOptionArray.simplified(
                    outindex, self._content, parameters=self._parameters
                )
                return ak.contents.RegularArray(
                    next,
                    target,
                    self.length,
                    parameters=self._parameters,
                )
        else:
            return ak.contents.ListOffsetArray(
                self._offsets,
                self._content._pad_none(target, axis, depth + 1, clip),
                parameters=self._parameters,
            )

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        is_string = self.parameter("__array__") == "string"
        is_bytestring = self.parameter("__array__") == "bytestring"
        if is_string:
            downsize = options["string_to32"]
        elif is_bytestring:
            downsize = options["bytestring_to32"]
        else:
            downsize = options["list_to32"]
        npoffsets = self._offsets.raw(numpy)
        akcontent = self._content[npoffsets[0] : npoffsets[length]]
        if len(npoffsets) > length + 1:
            npoffsets = npoffsets[: length + 1]
        if npoffsets[0] != 0:
            npoffsets = npoffsets - npoffsets[0]

        # ArrowNotImplementedError: Lists with non-zero length null components
        # are not supported. So make the null'ed lists empty.
        if validbytes is not None:
            nonzeros = npoffsets[1:] != npoffsets[:-1]
            maskedbytes = validbytes == 0
            if numpy.any(maskedbytes & nonzeros):  # null and count > 0
                new_starts = numpy.array(npoffsets[:-1], copy=True)
                new_stops = numpy.array(npoffsets[1:], copy=True)
                new_starts[maskedbytes] = 0
                new_stops[maskedbytes] = 0
                next = ak.contents.ListArray(
                    ak.index.Index(new_starts),
                    ak.index.Index(new_stops),
                    self._content,
                    parameters=self._parameters,
                )
                return next.to_ListOffsetArray64(True)._to_arrow(
                    pyarrow, mask_node, validbytes, length, options
                )

        if issubclass(npoffsets.dtype.type, np.int64):
            if downsize and npoffsets[-1] < np.iinfo(np.int32).max:
                npoffsets = npoffsets.astype(np.int32)

        if issubclass(npoffsets.dtype.type, np.uint32):
            if npoffsets[-1] < np.iinfo(np.int32).max:
                npoffsets = npoffsets.astype(np.int32)
            else:
                npoffsets = npoffsets.astype(np.int64)

        if is_string or is_bytestring:
            assert isinstance(akcontent, ak.contents.NumpyArray)

            if issubclass(npoffsets.dtype.type, np.int32):
                if is_string:
                    string_type = pyarrow.string()
                else:
                    string_type = pyarrow.binary()
            else:
                if is_string:
                    string_type = pyarrow.large_string()
                else:
                    string_type = pyarrow.large_binary()

            return pyarrow.Array.from_buffers(
                ak._connect.pyarrow.to_awkwardarrow_type(
                    string_type,
                    options["extensionarray"],
                    options["record_is_scalar"],
                    mask_node,
                    self,
                ),
                length,
                [
                    ak._connect.pyarrow.to_validbits(validbytes),
                    pyarrow.py_buffer(npoffsets),
                    pyarrow.py_buffer(akcontent._raw(numpy)),
                ],
            )

        else:
            paarray = akcontent._to_arrow(
                pyarrow, None, None, akcontent.length, options
            )

            content_type = pyarrow.list_(paarray.type).value_field.with_nullable(
                akcontent.is_option
            )

            if issubclass(npoffsets.dtype.type, np.int32):
                list_type = pyarrow.list_(content_type)
            else:
                list_type = pyarrow.large_list(content_type)

            return pyarrow.Array.from_buffers(
                ak._connect.pyarrow.to_awkwardarrow_type(
                    list_type,
                    options["extensionarray"],
                    options["record_is_scalar"],
                    mask_node,
                    self,
                ),
                length,
                [
                    ak._connect.pyarrow.to_validbits(validbytes),
                    pyarrow.py_buffer(npoffsets),
                ],
                children=[paarray],
                null_count=ak._connect.pyarrow.to_null_count(
                    validbytes, options["count_nulls"]
                ),
            )

    def _to_numpy(self, allow_missing):
        array_param = self.parameter("__array__")
        if array_param in {"bytestring", "string"}:
            return self._backend.nplike.array(self.to_list())

        return ak.operations.to_numpy(
            self.to_RegularArray(), allow_missing=allow_missing
        )

    def _completely_flatten(self, backend, options):
        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            return [self]
        else:
            flat = self._content[self._offsets[0] : self._offsets[-1]]
            return flat._completely_flatten(backend, options)

    def _drop_none(self):
        if self._content.is_option:
            _, _, none_indexes = self._content._nextcarry_outindex(self._backend)
            new_content = self._content._drop_none()
            return self._rebuild_without_nones(none_indexes, new_content)
        else:
            return self

    def _rebuild_without_nones(self, none_indexes, new_content):
        new_offsets = ak.index.Index64.empty(self._offsets.length, self._backend.nplike)

        assert (
            new_offsets.nplike is self._backend.index_nplike
            and self._offsets.nplike is self._backend.index_nplike
            and none_indexes.nplike is self._backend.index_nplike
        )

        self._handle_error(
            self._backend[
                "awkward_ListOffsetArray_drop_none_indexes",
                new_offsets.dtype.type,
                none_indexes.dtype.type,
                self._offsets.dtype.type,
            ](
                new_offsets.data,
                none_indexes.data,
                self._offsets.data,
                self._offsets.length,
                none_indexes.length,
            )
        )
        return ak.contents.ListOffsetArray(new_offsets, new_content)

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if self._backend.nplike.known_shape and self._backend.nplike.known_data:
            offsetsmin = self._offsets[0]
            offsets = ak.index.Index(
                self._offsets.data - offsetsmin, nplike=self._backend.index_nplike
            )
            content = self._content[offsetsmin : self._offsets[-1]]
        else:
            self._touch_data(recursive=False)
            offsets, content = self._offsets, self._content

        if options["return_array"]:

            def continuation():
                return ListOffsetArray(
                    offsets,
                    content._recursively_apply(
                        action,
                        behavior,
                        depth + 1,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    ),
                    parameters=self._parameters if options["keep_parameters"] else None,
                )

        else:

            def continuation():
                content._recursively_apply(
                    action,
                    behavior,
                    depth + 1,
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
        next = self.to_ListOffsetArray64(True)
        content = next._content.to_packed()
        if content.length != next._offsets[-1]:
            content = content[: next._offsets[-1]]
        return ListOffsetArray(next._offsets, content, parameters=next._parameters)

    def _to_list(self, behavior, json_conversions):
        starts, stops = self.starts, self.stops
        starts_data = starts.raw(numpy)
        stops_data = stops.raw(numpy)[: len(starts_data)]

        nonempty = starts_data != stops_data
        if numpy.count_nonzero(nonempty) == 0:
            mini, maxi = 0, 0
        else:
            mini = starts_data.min()
            maxi = stops_data.max()

        starts_data = starts_data - mini
        stops_data = stops_data - mini

        nextcontent = self._content._getitem_range(slice(mini, maxi))

        if self.parameter("__array__") == "bytestring":
            convert_bytes = (
                None if json_conversions is None else json_conversions["convert_bytes"]
            )
            content = ak._util.tobytes(nextcontent.data)
            out = [None] * starts.length
            if convert_bytes is None:
                for i in range(starts.length):
                    out[i] = content[starts_data[i] : stops_data[i]]
            else:
                for i in range(starts.length):
                    out[i] = convert_bytes(content[starts_data[i] : stops_data[i]])
            return out

        elif self.parameter("__array__") == "string":
            data = nextcontent.data
            if hasattr(data, "tobytes"):

                def tostring(x):
                    return x.tobytes().decode(errors="surrogateescape")

            else:

                def tostring(x):
                    return x.tostring().decode(errors="surrogateescape")

            out = [None] * starts.length
            for i in range(starts.length):
                out[i] = tostring(data[starts_data[i] : stops_data[i]])
            return out

        else:
            out = self._to_list_custom(behavior, json_conversions)
            if out is not None:
                return out

            content = nextcontent._to_list(behavior, json_conversions)
            out = [None] * starts.length

            for i in range(starts.length):
                out[i] = content[starts_data[i] : stops_data[i]]
            return out

    def to_backend(self, backend: ak._backends.Backend) -> Self:
        content = self._content.to_backend(backend)
        offsets = self._offsets.to_nplike(backend.index_nplike)
        return ListOffsetArray(offsets, content, parameters=self._parameters)

    def _awkward_strings_to_nonfinite(self, nonfinit_dict):
        if self.parameter("__array__") == "string":
            strings = self.to_list()
            if any(item in nonfinit_dict for item in strings):
                numbers = self._backend.index_nplike.empty(
                    self.starts.length, np.float64
                )
                has_another_string = False
                for i, val in enumerate(strings):
                    if val in nonfinit_dict:
                        numbers[i] = nonfinit_dict[val]
                    else:
                        numbers[i] = None
                        has_another_string = True

                content = ak.contents.NumpyArray(numbers)

                if has_another_string:
                    union_tags = ak.index.Index8.zeros(
                        content.length, nplike=self._backend.index_nplike
                    )
                    content.backend.nplike.isnan(content._data, union_tags._data)
                    union_index = ak.index.Index64(
                        self._backend.index_nplike.arange(
                            content.length, dtype=np.int64
                        ),
                        nplike=self._backend.index_nplike,
                    )

                    return ak.contents.UnionArray(
                        tags=union_tags,
                        index=union_index,
                        contents=[content, self.to_ListOffsetArray64(True)],
                    )

                return content

    def _is_equal_to(self, other, index_dtype, numpyarray):
        return self.offsets.is_equal_to(
            other.offsets, index_dtype, numpyarray
        ) and self.content.is_equal_to(other.content, index_dtype, numpyarray)
