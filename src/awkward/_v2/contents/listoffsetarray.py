# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import copy

import awkward as ak
from awkward._v2.index import Index
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.listoffsetform import ListOffsetForm
from awkward._v2.forms.form import _parameters_equal

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class ListOffsetArray(Content):
    is_ListType = True

    def __init__(self, offsets, content, identifier=None, parameters=None, nplike=None):
        if not isinstance(offsets, Index) and offsets.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        ):
            raise TypeError(
                "{} 'offsets' must be an Index with dtype in (int32, uint32, int64), "
                "not {}".format(type(self).__name__, repr(offsets))
            )
        if not isinstance(content, Content):
            raise TypeError(
                "{} 'content' must be a Content subtype, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        if ak.nplike.of(offsets).known_shape:
            if not offsets.length >= 1:
                raise ValueError(
                    "{} len(offsets) ({}) must be >= 1".format(
                        type(self).__name__, offsets.length
                    )
                )
        if nplike is None:
            nplike = content.nplike
        if nplike is None:
            nplike = offsets.nplike

        self._offsets = offsets
        self._content = content
        self._init(identifier, parameters, nplike)

    @property
    def starts(self):
        return self._offsets[:-1]

    @property
    def stops(self):
        return self._offsets[1:]

    @property
    def offsets(self):
        return self._offsets

    @property
    def content(self):
        return self._content

    Form = ListOffsetForm

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.Form(
            self._offsets.form,
            self._content._form_with_key(getkey),
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, nplike):
        assert isinstance(form, self.Form)
        key = getkey(self, form, "offsets")
        container[key] = ak._v2._util.little_endian(self._offsets.to(nplike))
        self._content._to_buffers(form.content, getkey, container, nplike)

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return ListOffsetArray(
            ak._v2.index.Index(self._offsets.to(tt)),
            self._content.typetracer,
            self._typetracer_identifier(),
            self._parameters,
            ak._v2._typetracer.TypeTracer.instance(),
        )

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

    def merge_parameters(self, parameters):
        return ListOffsetArray(
            self._offsets,
            self._content,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
            self._nplike,
        )

    def toListOffsetArray64(self, start_at_zero=False):
        if issubclass(self._offsets.dtype.type, np.int64):
            if (
                not self._nplike.known_data
                or not start_at_zero
                or self._offsets[0] == 0
            ):
                return self

            if start_at_zero:
                offsets = ak._v2.index.Index64(
                    self._offsets.to(self._nplike) - self._offsets[0]
                )
                content = self._content[self._offsets[0] :]
            else:
                offsets, content = self._offsets, self._content

            return ListOffsetArray(
                offsets, content, self._identifier, self._parameters, self._nplike
            )

        else:
            offsets = self._compact_offsets64(start_at_zero)
            return self._broadcast_tooffsets64(offsets)

    def toRegularArray(self):
        start, stop = self._offsets[0], self._offsets[self._offsets.length - 1]
        content = self._content._getitem_range(slice(start, stop))
        size = ak._v2.index.Index64.empty(1, self._nplike)
        self._handle_error(
            self._nplike[
                "awkward_ListOffsetArray_toRegularArray",
                size.dtype.type,
                self._offsets.dtype.type,
            ](
                size.to(self._nplike),
                self._offsets.to(self._nplike),
                self._offsets.length,
            )
        )

        return ak._v2.contents.RegularArray(
            content,
            size[0],
            self._offsets.length - 1,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if not self._nplike.known_data:
            return self._content._getitem_range(slice(0, 0))

        if where < 0:
            where += self.length
        if not (0 <= where < self.length) and self._nplike.known_shape:
            raise NestedIndexError(self, where)
        start, stop = self._offsets[where], self._offsets[where + 1]
        return self._content._getitem_range(slice(start, stop))

    def _getitem_range(self, where):
        if not self._nplike.known_shape:
            return self

        start, stop, step = where.indices(self.length)
        offsets = self._offsets[start : stop + 1]
        if offsets.length == 0:
            offsets = Index(self._nplike.array([0], dtype=self._offsets.dtype))
        return ListOffsetArray(
            offsets,
            self._content,
            self._range_identifier(start, stop),
            self._parameters,
            self._nplike,
        )

    def _getitem_field(self, where, only_fields=()):
        return ListOffsetArray(
            self._offsets,
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
            self._nplike,
        )

    def _getitem_fields(self, where, only_fields=()):
        return ListOffsetArray(
            self._offsets,
            self._content._getitem_fields(where, only_fields),
            self._fields_identifier(where),
            None,
            self._nplike,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextstarts = self.starts[carry.data]
            nextstops = self.stops[carry.data]
        except IndexError as err:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, str(err))
            else:
                raise exception(str(err))

        return ak._v2.contents.listarray.ListArray(
            nextstarts,
            nextstops,
            self._content,
            self._carry_identifier(carry, exception),
            self._parameters,
            self._nplike,
        )

    def _compact_offsets64(self, start_at_zero):
        offsets_len = self._offsets.length - 1
        out = ak._v2.index.Index64.empty(offsets_len + 1, self._nplike)
        self._handle_error(
            self._nplike[
                "awkward_ListOffsetArray_compact_offsets",
                out.dtype.type,
                self._offsets.dtype.type,
            ](out.to(self._nplike), self._offsets.to(self._nplike), offsets_len)
        )
        return out

    def _broadcast_tooffsets64(self, offsets):
        if offsets.nplike.known_data and (offsets.length == 0 or offsets[0] != 0):
            raise AssertionError(
                "broadcast_tooffsets64 can only be used with offsets that start at 0, not {}".format(
                    "(empty)" if offsets.length == 0 else str(offsets[0])
                )
            )

        if offsets.nplike.known_shape and offsets.length - 1 != self.length:
            raise AssertionError(
                "cannot broadcast {} of length {} to length {}".format(
                    type(self).__name__, self.length, offsets.length - 1
                )
            )

        if self._identifier is not None:
            identifier = self._identifier[slice(0, offsets.length - 1)]
        else:
            identifier = self._identifier

        starts, stops = self.starts, self.stops

        nextcarry = ak._v2.index.Index64.empty(offsets[-1], self._nplike)
        self._handle_error(
            self._nplike[
                "awkward_ListArray_broadcast_tooffsets",
                nextcarry.dtype.type,
                offsets.dtype.type,
                starts.dtype.type,
                stops.dtype.type,
            ](
                nextcarry.to(self._nplike),
                offsets.to(self._nplike),
                offsets.length,
                starts.to(self._nplike),
                stops.to(self._nplike),
                self._content.length,
            )
        )

        nextcontent = self._content._carry(nextcarry, True, NestedIndexError)

        return ListOffsetArray(
            offsets, nextcontent, identifier, self._parameters, self._nplike
        )

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        out = ak._v2.contents.listarray.ListArray(
            self.starts,
            self.stops,
            self._content,
            self._identifier,
            self._parameters,
            self._nplike,
        )
        return out._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, int):
            assert advanced is None
            lenstarts = self._offsets.length - 1
            starts, stops = self.starts, self.stops
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            nextcarry = ak._v2.index.Index64.empty(lenstarts, self._nplike)

            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_next_at",
                    nextcarry.dtype.type,
                    starts.dtype.type,
                    stops.dtype.type,
                ](
                    nextcarry.to(self._nplike),
                    starts.to(self._nplike),
                    stops.to(self._nplike),
                    lenstarts,
                    head,
                )
            )
            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
            return nextcontent._getitem_next(nexthead, nexttail, advanced)

        elif isinstance(head, slice):
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            lenstarts = self._offsets.length - 1
            start, stop, step = head.start, head.stop, head.step

            step = 1 if step is None else step
            start = ak._util.kSliceNone if start is None else start
            stop = ak._util.kSliceNone if stop is None else stop

            carrylength = ak._v2.index.Index64.empty(1, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_next_range_carrylength",
                    carrylength.dtype.type,
                    self.starts.dtype.type,
                    self.stops.dtype.type,
                ](
                    carrylength.to(self._nplike),
                    self.starts.to(self._nplike),
                    self.stops.to(self._nplike),
                    lenstarts,
                    start,
                    stop,
                    step,
                )
            )

            if self._starts.dtype == "int64":
                nextoffsets = ak._v2.index.Index64.empty(lenstarts + 1, self._nplike)
            elif self._starts.dtype == "int32":
                nextoffsets = ak._v2.index.Index32.empty(lenstarts + 1, self._nplike)
            elif self._starts.dtype == "uint32":
                nextoffsets = ak._v2.index.IndexU32.empty(lenstarts + 1, self._nplike)
            nextcarry = ak._v2.index.Index64.empty(carrylength[0], self._nplike)

            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_next_range",
                    nextoffsets.dtype.type,
                    nextcarry.dtype.type,
                    self.starts.dtype.type,
                    self.stops.dtype.type,
                ](
                    nextoffsets.to(self._nplike),
                    nextcarry.to(self._nplike),
                    self.starts.to(self._nplike),
                    self.stops.to(self._nplike),
                    lenstarts,
                    start,
                    stop,
                    step,
                )
            )

            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)

            if advanced is None or advanced.length == 0:
                return ak._v2.contents.listoffsetarray.ListOffsetArray(
                    nextoffsets,
                    nextcontent._getitem_next(nexthead, nexttail, advanced),
                    self._identifier,
                    self._parameters,
                    self._nplike,
                )

            else:
                total = ak._v2.index.Index64.empty(1, self._nplike)
                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_getitem_next_range_counts",
                        total.dtype.type,
                        nextoffsets.dtype.type,
                    ](
                        total.to(self._nplike),
                        nextoffsets.to(self._nplike),
                        lenstarts,
                    )
                )

                nextadvanced = ak._v2.index.Index64.empty(total[0], self._nplike)
                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_getitem_next_range_spreadadvanced",
                        nextadvanced.dtype.type,
                        advanced.dtype.type,
                        nextoffsets.dtype.type,
                    ](
                        nextadvanced.to(self._nplike),
                        advanced.to(self._nplike),
                        nextoffsets.to(self._nplike),
                        lenstarts,
                    )
                )

                return ak._v2.contents.listoffsetarray.ListOffsetArray(
                    nextoffsets,
                    nextcontent._getitem_next(nexthead, nexttail, nextadvanced),
                    self._identifier,
                    self._parameters,
                    self._nplike,
                )

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.index.Index64):
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            flathead = self._nplike.asarray(head.data.reshape(-1))
            lenstarts = self.starts.length
            regular_flathead = ak._v2.index.Index64(flathead)
            if advanced is None or advanced.length == 0:
                nextcarry = ak._v2.index.Index64.empty(
                    lenstarts * flathead.length, self._nplike
                )
                nextadvanced = ak._v2.index.Index64.empty(
                    lenstarts * flathead.length, self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_getitem_next_array",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.to(self._nplike),
                        nextadvanced.to(self._nplike),
                        self.starts.to(self._nplike),
                        self.stops.to(self._nplike),
                        regular_flathead.to(self._nplike),
                        lenstarts,
                        regular_flathead.length,
                        self._content.length,
                    ),
                    head,
                )
                nextcontent = self._content._carry(nextcarry, True, NestedIndexError)

                out = nextcontent._getitem_next(nexthead, nexttail, nextadvanced)
                if advanced is None:
                    return ak._v2._slicing.getitem_next_array_wrap(
                        out, head.metadata.get("shape", (head.length,))
                    )
                else:
                    return out

            else:
                nextcarry = ak._v2.index.Index64.empty(self.length, self._nplike)
                nextadvanced = ak._v2.index.Index64.empty(self.length, self._nplike)
                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_getitem_next_array_advanced",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        self.starts.dtype.type,
                        self.stops.dtype.type,
                        regular_flathead.dtype.type,
                        advanced.dtype.type,
                    ](
                        nextcarry.to(self._nplike),
                        nextadvanced.to(self._nplike),
                        self.starts.to(self._nplike),
                        self.stops.to(self._nplike),
                        regular_flathead.to(self._nplike),
                        advanced.to(self._nplike),
                        lenstarts,
                        regular_flathead.length,
                        self._content.length,
                    ),
                    head,
                )
                nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
                return nextcontent._getitem_next(nexthead, nexttail, nextadvanced)

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            listarray = ak._v2.contents.listarray.ListArray(
                self.starts,
                self.stops,
                self._content,
                self._identifier,
                self._parameters,
                self._nplike,
            )
            return listarray._getitem_next(head, tail, advanced)

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            out = ak._v2.index.Index64.empty(1, self._nplike)
            out[0] = self.length
            return ak._v2.contents.numpyarray.NumpyArray(out, None, None, self._nplike)[
                0
            ]
        elif posaxis == depth + 1:
            tonum = ak._v2.index.Index64.empty(self.length, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_num",
                    tonum.dtype.type,
                    self.starts.dtype.type,
                    self.stops.dtype.type,
                ](
                    tonum.to(self._nplike),
                    self.starts.to(self._nplike),
                    self.stops.to(self._nplike),
                    self.length,
                )
            )
            return ak._v2.contents.numpyarray.NumpyArray(
                tonum, None, None, self._nplike
            )
        else:
            next = self._content.num(posaxis, depth + 1)
            offsets = self._compact_offsets64(True)
            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                offsets, next, None, self.parameters, self._nplike
            )

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            raise np.AxisError(self, "axis=0 not allowed for flatten")

        elif posaxis == depth + 1:
            listoffsetarray = self.toListOffsetArray64(True)
            stop = listoffsetarray.offsets[-1]
            content = listoffsetarray.content._getitem_range(slice(0, stop))
            return (listoffsetarray.offsets, content)

        else:
            inneroffsets, flattened = self._content._offsets_and_flattened(
                posaxis, depth + 1
            )
            offsets = ak._v2.index.Index64.zeros(0, self._nplike, dtype=np.int64)

            if inneroffsets.length == 0:
                return (
                    offsets,
                    ListOffsetArray(
                        self._offsets, flattened, None, self._parameters, self._nplike
                    ),
                )

            elif self._offsets.length == 1:
                tooffsets = ak._v2.index.Index64([inneroffsets[0]])
                return (
                    offsets,
                    ListOffsetArray(
                        tooffsets, flattened, None, self._parameters, self._nplike
                    ),
                )

            else:
                tooffsets = ak._v2.index.Index64.empty(
                    self._offsets.length, self._nplike, dtype=np.int64
                )
                self._handle_error(
                    self._nplike[
                        "awkward_ListOffsetArray_flatten_offsets",
                        tooffsets.dtype.type,
                        self._offsets.dtype.type,
                        inneroffsets.dtype.type,
                    ](
                        tooffsets.to(self._nplike),
                        self._offsets.to(self._nplike),
                        self._offsets.length,
                        inneroffsets.to(self._nplike),
                        inneroffsets.length,
                    )
                )
                return (
                    offsets,
                    ListOffsetArray(
                        tooffsets, flattened, None, self._parameters, self._nplike
                    ),
                )

    def mergeable(self, other, mergebool):
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

        if isinstance(
            other,
            (
                ak._v2.contents.regulararray.RegularArray,
                ak._v2.contents.listarray.ListArray,
                ak._v2.contents.listoffsetarray.ListOffsetArray,
            ),
        ):
            return self._content.mergeable(other.content, mergebool)

        else:
            return False

    def mergemany(self, others):
        if len(others) == 0:
            return self
        listarray = ak._v2.contents.listarray.ListArray(
            self.starts, self.stops, self._content, None, self._parameters, self._nplike
        )
        return listarray.mergemany(others)

    def fillna(self, value):
        return ListOffsetArray(
            self._offsets,
            self._content.fillna(value),
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        elif posaxis == depth + 1:
            offsets = self._compact_offsets64(True)
            if self._nplike.known_data:
                innerlength = offsets[offsets.length - 1]
            else:
                innerlength = ak._v2._typetracer.UnknownLength
            localindex = ak._v2.index.Index64.empty(innerlength, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_localindex",
                    localindex.dtype.type,
                    offsets.dtype.type,
                ](
                    localindex.to(self._nplike),
                    offsets.to(self._nplike),
                    offsets.length - 1,
                )
            )
            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                offsets,
                ak._v2.contents.NumpyArray(localindex),
                self._identifier,
                self._parameters,
                self._nplike,
            )
        else:
            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                self._offsets,
                self._content._localindex(posaxis, depth + 1),
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def numbers_to_type(self, name):
        return ak._v2.contents.listoffsetarray.ListOffsetArray(
            self._offsets,
            self._content.numbers_to_type(name),
            self._identifier,
            self._parameters,
            self._nplike,
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
                raise ValueError(
                    "array with strings can only be checked on uniqueness with axis=-1"
                )

            # FIXME: check validity error

            if isinstance(self._content, ak._v2.contents.NumpyArray):
                out, outoffsets = self._content._as_unique_strings(self._offsets)
                out2 = ak._v2.contents.listoffsetarray.ListOffsetArray(
                    outoffsets,
                    out,
                    None,
                    self._parameters,
                    self._nplike,
                )
                return out2.length == self.length

        if negaxis is None:
            return self._content._is_unique(negaxis, starts, parents, outlength)

        if not branch and (negaxis == depth):
            return self._content._is_unique(negaxis - 1, starts, parents, outlength)
        else:
            nextparents = ak._v2.index.Index64.empty(
                self._offsets[-1] - self._offsets[0], self._nplike
            )

            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.to(self._nplike),
                    self._offsets.to(self._nplike),
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
                raise np.AxisError("array with strings can only be sorted with axis=-1")

            # FIXME: check validity error

            if isinstance(self._content, ak._v2.contents.NumpyArray):
                out, nextoffsets = self._content._as_unique_strings(self._offsets)
                return ak._v2.contents.ListOffsetArray(
                    nextoffsets,
                    out,
                    None,
                    self._parameters,
                    self._nplike,
                )

        if not branch and (negaxis == depth):
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise np.AxisError("array with strings can only be sorted with axis=-1")

            assert self._offsets.length - 1 == parents.length

            nextlen = self._offsets[-1] - self._offsets[0]
            maxcount = ak._v2.index.Index64.empty(1, self._nplike)
            offsetscopy = ak._v2.index.Index64.empty(self._offsets.length, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64",
                    maxcount.dtype.type,
                    offsetscopy.dtype.type,
                    self._offsets.dtype.type,
                ](
                    maxcount.to(self._nplike),
                    offsetscopy.to(self._nplike),
                    self._offsets.to(self._nplike),
                    self._offsets.length - 1,
                )
            )

            distincts_length = outlength * maxcount[0]
            nextcarry = ak._v2.index.Index64.empty(nextlen, self._nplike)
            nextparents = ak._v2.index.Index64.empty(nextlen, self._nplike)
            maxnextparents = ak._v2.index.Index64.empty(1, self._nplike)
            distincts = ak._v2.index.Index64.empty(distincts_length, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_preparenext_64",
                    nextcarry.dtype.type,
                    nextparents.dtype.type,
                    maxnextparents.dtype.type,
                    distincts.dtype.type,
                    self._offsets.dtype.type,
                    offsetscopy.dtype.type,
                    parents.dtype.type,
                ](
                    nextcarry.to(self._nplike),
                    nextparents.to(self._nplike),
                    nextlen,
                    maxnextparents.to(self._nplike),
                    distincts.to(self._nplike),
                    distincts_length,
                    offsetscopy.to(self._nplike),
                    self._offsets.to(self._nplike),
                    self._offsets.length - 1,
                    parents.to(self._nplike),
                    maxcount[0],
                )
            )

            nextstarts = ak._v2.index.Index64.empty(maxnextparents[0] + 1, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64",
                    nextstarts.dtype.type,
                    nextparents.dtype.type,
                ](
                    nextstarts.to(self._nplike),
                    nextparents.to(self._nplike),
                    nextlen,
                )
            )

            nextcontent = self._content._carry(nextcarry, False, NestedIndexError)
            outcontent = nextcontent._unique(
                negaxis - 1,
                nextstarts,
                nextparents,
                maxnextparents[0] + 1,
            )

            outcarry = ak._v2.index.Index64.empty(nextlen, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_local_preparenext_64",
                    outcarry.dtype.type,
                    nextcarry.dtype.type,
                ](
                    outcarry.to(self._nplike),
                    nextcarry.to(self._nplike),
                    nextlen,
                )
            )

            return ak._v2.contents.ListOffsetArray(
                outcontent._compact_offsets64(True),
                outcontent._content._carry(outcarry, False, NestedIndexError),
                None,
                self._parameters,
                self._nplike,
            )

        else:
            nextparents = ak._v2.index.Index64.empty(
                self._offsets[-1] - self._offsets[0], self._nplike
            )

            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.to(self._nplike),
                    self._offsets.to(self._nplike),
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
            return ak._v2.contents.ListOffsetArray(
                outoffsets,
                outcontent,
                None,
                self._parameters,
                self._nplike,
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
        if self._offsets.length - 1 == 0:
            return ak._v2.contents.NumpyArray(
                self.nplike.empty(0, np.int64), None, None, self._nplike
            )

        branch, depth = self.branch_depth

        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            if branch or (negaxis != depth):
                raise np.AxisError("array with strings can only be sorted with axis=-1")

            # FIXME: check validity error

            if isinstance(self._content, ak._v2.contents.NumpyArray):
                nextcarry = ak._v2.index.Index64.empty(
                    self._offsets.length - 1, self._nplike
                )

                self_starts, self_stops = self._offsets[:-1], self._offsets[1:]
                self._handle_error(
                    self._nplike[
                        "awkward_ListOffsetArray_argsort_strings",
                        nextcarry.dtype.type,
                        parents.dtype.type,
                        self._content.dtype.type,
                        self_starts.dtype.type,
                        self_stops.dtype.type,
                    ](
                        nextcarry.to(self._nplike),
                        parents.to(self._nplike),
                        parents.length,
                        self._content._data,
                        self_starts.to(self._nplike),
                        self_stops.to(self._nplike),
                        stable,
                        ascending,
                        False,
                    )
                )
                return ak._v2.contents.NumpyArray(nextcarry, None, None, self._nplike)

        if not branch and (negaxis == depth):
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise np.AxisError("array with strings can only be sorted with axis=-1")

            assert self._offsets.length - 1 == parents.length

            maxcount = ak._v2.index.Index64.empty(1, self._nplike)
            offsetscopy = ak._v2.index.Index64.empty(self._offsets.length, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64",
                    maxcount.dtype.type,
                    offsetscopy.dtype.type,
                    self._offsets.dtype.type,
                ](
                    maxcount.to(self._nplike),
                    offsetscopy.to(self._nplike),
                    self._offsets.to(self._nplike),
                    self._offsets.length - 1,
                )
            )

            maxcount = maxcount[0]
            nextlen = self._offsets[-1] - self._offsets[0]

            nextcarry = ak._v2.index.Index64.empty(nextlen, self._nplike)
            nextparents = ak._v2.index.Index64.empty(nextlen, self._nplike)
            maxnextparents = ak._v2.index.Index64.empty(1, self._nplike)
            distincts = ak._v2.index.Index64.empty(maxcount * outlength, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_preparenext_64",
                    nextcarry.dtype.type,
                    nextparents.dtype.type,
                    maxnextparents.dtype.type,
                    distincts.dtype.type,
                    self._offsets.dtype.type,
                    offsetscopy.dtype.type,
                    parents.dtype.type,
                ](
                    nextcarry.to(self._nplike),
                    nextparents.to(self._nplike),
                    nextlen,
                    maxnextparents.to(self._nplike),
                    distincts.to(self._nplike),
                    maxcount * outlength,
                    offsetscopy.to(self._nplike),
                    self._offsets.to(self._nplike),
                    self._offsets.length - 1,
                    parents.to(self._nplike),
                    maxcount,
                )
            )

            nextstarts_length = maxnextparents[0] + 1
            nextstarts = ak._v2.index.Index64.empty(
                nextstarts_length, self._nplike, np.int64
            )
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64",
                    nextstarts.dtype.type,
                    nextparents.dtype.type,
                ](
                    nextstarts.to(self._nplike),
                    nextparents.to(self._nplike),
                    nextlen,
                )
            )

            nummissing = ak._v2.index.Index64.empty(maxcount, self._nplike)
            missing = ak._v2.index.Index64.empty(self._offsets[-1], self._nplike)
            nextshifts = ak._v2.index.Index64.empty(nextlen, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64",
                    nummissing.dtype.type,
                    missing.dtype.type,
                    nextshifts.dtype.type,
                    self._offsets.dtype.type,
                    starts.dtype.type,
                    parents.dtype.type,
                    nextcarry.dtype.type,
                ](
                    nummissing.to(self._nplike),
                    missing.to(self._nplike),
                    nextshifts.to(self._nplike),
                    self._offsets.to(self._nplike),
                    self._offsets.length - 1,
                    starts.to(self._nplike),
                    parents.to(self._nplike),
                    maxcount,
                    nextlen,
                    nextcarry.to(self._nplike),
                )
            )

            nextcontent = self._content._carry(nextcarry, False, NestedIndexError)
            outcontent = nextcontent._argsort_next(
                negaxis - 1,
                nextstarts,
                nextshifts,
                nextparents,
                nextstarts_length,
                ascending,
                stable,
                kind,
                order,
            )

            outcarry = ak._v2.index.Index64.empty(nextlen, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_local_preparenext_64",
                    outcarry.dtype.type,
                    nextcarry.dtype.type,
                ](
                    outcarry.to(self._nplike),
                    nextcarry.to(self._nplike),
                    nextlen,
                )
            )

            out_offsets = self._compact_offsets64(True)
            out = outcontent._carry(outcarry, False, NestedIndexError)
            return ak._v2.contents.ListOffsetArray(
                out_offsets,
                out,
                None,
                self._parameters,
                self._nplike,
            )
        else:
            nextparents = ak._v2.index.Index64.empty(
                self._offsets[-1] - self._offsets[0], self._nplike
            )

            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.to(self._nplike),
                    self._offsets.to(self._nplike),
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
            return ak._v2.contents.ListOffsetArray(
                outoffsets,
                outcontent,
                None,
                self._parameters,
                self._nplike,
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
                raise np.AxisError("array with strings can only be sorted with axis=-1")

            # FIXME: check validity error

            if isinstance(self._content, ak._v2.contents.NumpyArray):
                nextcarry = ak._v2.index.Index64.empty(
                    self._offsets.length - 1, self._nplike
                )

                starts, stops = self._offsets[:-1], self._offsets[1:]
                self._handle_error(
                    self._nplike[
                        "awkward_ListOffsetArray_argsort_strings",
                        nextcarry.dtype.type,
                        parents.dtype.type,
                        self._content.dtype.type,
                        starts.dtype.type,
                        stops.dtype.type,
                    ](
                        nextcarry.to(self._nplike),
                        parents.to(self._nplike),
                        parents.length,
                        self._content._data,
                        starts.to(self._nplike),
                        stops.to(self._nplike),
                        stable,
                        ascending,
                        False,
                    )
                )
                return self._carry(nextcarry, False, NestedIndexError)

        if not branch and (negaxis == depth):
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise np.AxisError("array with strings can only be sorted with axis=-1")

            assert self._offsets.length - 1 == parents.length

            nextlen = self._offsets[-1] - self._offsets[0]
            maxcount = ak._v2.index.Index64.empty(1, self._nplike)
            offsetscopy = ak._v2.index.Index64.empty(self._offsets.length, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64",
                    maxcount.dtype.type,
                    offsetscopy.dtype.type,
                    self._offsets.dtype.type,
                ](
                    maxcount.to(self._nplike),
                    offsetscopy.to(self._nplike),
                    self._offsets.to(self._nplike),
                    self._offsets.length - 1,
                )
            )

            distincts_length = outlength * maxcount[0]
            nextcarry = ak._v2.index.Index64.empty(nextlen, self._nplike)
            nextparents = ak._v2.index.Index64.empty(nextlen, self._nplike)
            maxnextparents = ak._v2.index.Index64.empty(1, self._nplike)
            distincts = ak._v2.index.Index64.empty(distincts_length, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_preparenext_64",
                    nextcarry.dtype.type,
                    nextparents.dtype.type,
                    maxnextparents.dtype.type,
                    distincts.dtype.type,
                    self._offsets.dtype.type,
                    offsetscopy.dtype.type,
                    parents.dtype.type,
                ](
                    nextcarry.to(self._nplike),
                    nextparents.to(self._nplike),
                    nextlen,
                    maxnextparents.to(self._nplike),
                    distincts.to(self._nplike),
                    distincts_length,
                    offsetscopy.to(self._nplike),
                    self._offsets.to(self._nplike),
                    self._offsets.length - 1,
                    parents.to(self._nplike),
                    maxcount[0],
                )
            )

            nextstarts = ak._v2.index.Index64.empty(maxnextparents[0] + 1, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64",
                    nextstarts.dtype.type,
                    nextparents.dtype.type,
                ](
                    nextstarts.to(self._nplike),
                    nextparents.to(self._nplike),
                    nextlen,
                )
            )

            nextcontent = self._content._carry(nextcarry, False, NestedIndexError)
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

            outcarry = ak._v2.index.Index64.empty(nextlen, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_local_preparenext_64",
                    outcarry.dtype.type,
                    nextcarry.dtype.type,
                ](
                    outcarry.to(self._nplike),
                    nextcarry.to(self._nplike),
                    nextlen,
                )
            )

            return ak._v2.contents.ListOffsetArray(
                self._compact_offsets64(True),
                outcontent._carry(outcarry, False, NestedIndexError),
                None,
                self._parameters,
                self._nplike,
            )
        else:
            nextparents = ak._v2.index.Index64.empty(
                self._offsets[-1] - self._offsets[0], self._nplike
            )

            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.to(self._nplike),
                    self._offsets.to(self._nplike),
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
            return ak._v2.contents.ListOffsetArray(
                outoffsets,
                outcontent,
                None,
                self._parameters,
                self._nplike,
            )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        elif posaxis == depth + 1:
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise ValueError(
                    "ak.combinations does not compute combinations of the characters of a string; please split it into lists"
                )

            starts = self.starts
            stops = self.stops

            totallen = ak._v2.index.Index64.empty(1, self._nplike, dtype=np.int64)
            offsets = ak._v2.index.Index64.empty(
                self.length + 1, self._nplike, dtype=np.int64
            )
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_combinations_length",
                    totallen.to(self._nplike).dtype.type,
                    offsets.to(self._nplike).dtype.type,
                    starts.to(self._nplike).dtype.type,
                    stops.to(self._nplike).dtype.type,
                ](
                    totallen.to(self._nplike),
                    offsets.to(self._nplike),
                    n,
                    replacement,
                    starts.to(self._nplike),
                    stops.to(self._nplike),
                    self.length,
                )
            )

            tocarryraw = self._nplike.empty(n, dtype=np.intp)
            tocarry = []

            for i in range(n):
                ptr = ak._v2.index.Index64.empty(
                    totallen[0], self._nplike, dtype=np.int64
                )
                tocarry.append(ptr)
                if self._nplike.known_data:
                    tocarryraw[i] = ptr.ptr

            toindex = ak._v2.index.Index64.empty(n, self._nplike, dtype=np.int64)
            fromindex = ak._v2.index.Index64.empty(n, self._nplike, dtype=np.int64)
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_combinations",
                    np.int64,
                    toindex.to(self._nplike).dtype.type,
                    fromindex.to(self._nplike).dtype.type,
                    starts.to(self._nplike).dtype.type,
                    stops.to(self._nplike).dtype.type,
                ](
                    tocarryraw,
                    toindex.to(self._nplike),
                    fromindex.to(self._nplike),
                    n,
                    replacement,
                    starts.to(self._nplike),
                    stops.to(self._nplike),
                    self.length,
                )
            )
            contents = []

            for ptr in tocarry:
                contents.append(self._content._carry(ptr, True, NestedIndexError))

            recordarray = ak._v2.contents.recordarray.RecordArray(
                contents, recordlookup, None, None, parameters, self._nplike
            )
            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                offsets, recordarray, self._identifier, self._parameters, self._nplike
            )
        else:
            compact = self.toListOffsetArray64(True)
            next = compact._content._combinations(
                n, replacement, recordlookup, parameters, posaxis, depth + 1
            )
            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                compact.offsets, next, self._identifier, self._parameters, self._nplike
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
        if self._offsets.nplike.known_data and self._offsets[0] != 0:
            next = self.toListOffsetArray64(True)
            return next._reduce_next(
                reducer,
                negaxis,
                starts,
                shifts,
                parents,
                outlength,
                mask,
                keepdims,
            )

        branch, depth = self.branch_depth
        globalstarts_length = self._offsets.length - 1
        parents_length = parents.length
        nextlen = self._offsets[-1] - self._offsets[0]

        if not branch and negaxis == depth:
            maxcount = ak._v2.index.Index64.empty(1, self._nplike)
            offsetscopy = ak._v2.index.Index64.empty(self.offsets.length, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64",
                    maxcount.dtype.type,
                    offsetscopy.dtype.type,
                    self._offsets.dtype.type,
                ](
                    maxcount.to(self._nplike),
                    offsetscopy.to(self._nplike),
                    self._offsets.to(self._nplike),
                    globalstarts_length,
                )
            )

            distincts_length = outlength * maxcount[0]
            np_nextcarry = self._nplike.empty(nextlen, dtype=np.int64)
            np_nextparents = self._nplike.empty(nextlen, dtype=np.int64)
            maxnextparents = ak._v2.index.Index64.empty(1, self._nplike)
            distincts = ak._v2.index.Index64.empty(distincts_length, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_preparenext_64",
                    np_nextcarry.dtype.type,
                    np_nextparents.dtype.type,
                    maxnextparents.dtype.type,
                    distincts.dtype.type,
                    self._offsets.dtype.type,
                    offsetscopy.dtype.type,
                    parents.dtype.type,
                ](
                    np_nextcarry,
                    np_nextparents,
                    nextlen,
                    maxnextparents.to(self._nplike),
                    distincts.to(self._nplike),
                    distincts_length,
                    offsetscopy.to(self._nplike),
                    self._offsets.to(self._nplike),
                    globalstarts_length,
                    parents.to(self._nplike),
                    maxcount[0],
                )
            )

            reorder = self._nplike.argsort(np_nextparents)
            nextcarry = ak._v2.index.Index64(np_nextcarry[reorder])
            nextparents = ak._v2.index.Index64(np_nextparents[reorder])

            nextstarts = ak._v2.index.Index64.empty(maxnextparents[0] + 1, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64",
                    nextstarts.dtype.type,
                    nextparents.dtype.type,
                ](
                    nextstarts.to(self._nplike),
                    nextparents.to(self._nplike),
                    nextlen,
                )
            )

            gaps = ak._v2.index.Index64.empty(outlength, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_findgaps_64",
                    gaps.dtype.type,
                    parents.dtype.type,
                ](
                    gaps.to(self._nplike),
                    parents.to(self._nplike),
                    parents_length,
                )
            )

            outstarts = ak._v2.index.Index64.empty(outlength, self._nplike)
            outstops = ak._v2.index.Index64.empty(outlength, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",
                    outstarts.dtype.type,
                    outstops.dtype.type,
                    distincts.dtype.type,
                    gaps.dtype.type,
                ](
                    outstarts.to(self._nplike),
                    outstops.to(self._nplike),
                    distincts.to(self._nplike),
                    distincts_length,
                    gaps.to(self._nplike),
                    outlength,
                )
            )

            if reducer.needs_position:
                nextshifts = ak._v2.index.Index64.empty(nextlen, self._nplike)
                nummissing = ak._v2.index.Index64.empty(maxcount[0], self._nplike)
                missing = ak._v2.index.Index64.empty(self._offsets[-1], self._nplike)
                self._handle_error(
                    self._nplike[
                        "awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64",
                        nummissing.dtype.type,
                        missing.dtype.type,
                        nextshifts.dtype.type,
                        self._offsets.dtype.type,
                        starts.dtype.type,
                        parents.dtype.type,
                        nextcarry.dtype.type,
                    ](
                        nummissing.to(self._nplike),
                        missing.to(self._nplike),
                        nextshifts.to(self._nplike),
                        self._offsets.to(self._nplike),
                        globalstarts_length,
                        starts.to(self._nplike),
                        parents.to(self._nplike),
                        maxcount[0],
                        nextlen,
                        nextcarry.to(self._nplike),
                    )
                )
            else:
                nextshifts = None

            nextcontent = self._content._carry(nextcarry, False, NestedIndexError)
            outcontent = nextcontent._reduce_next(
                reducer,
                negaxis - 1,
                nextstarts,
                nextshifts,
                nextparents,
                maxnextparents[0] + 1,
                mask,
                False,
            )

            out = ak._v2.contents.ListArray(
                outstarts,
                outstops,
                outcontent,
                None,
                None,
                self._nplike,
            )

            if keepdims:
                out = ak._v2.contents.RegularArray(
                    out,
                    1,
                    self.length,
                    None,
                    None,
                    self._nplike,
                ).toListOffsetArray64(False)

            return out

        else:
            nextparents = ak._v2.index.Index64.empty(nextlen, self._nplike)

            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.to(self._nplike),
                    self._offsets.to(self._nplike),
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
            )

            outoffsets = ak._v2.index.Index64.empty(outlength + 1, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListOffsetArray_reduce_local_outoffsets_64",
                    outoffsets.dtype.type,
                    parents.dtype.type,
                ](
                    outoffsets.to(self._nplike),
                    parents.to(self._nplike),
                    parents.length,
                    outlength,
                )
            )

            represents_regular = getattr(self, "_represents_regular", False)

            if keepdims and (
                not represents_regular or self._content.dimension_optiontype
            ):
                if isinstance(outcontent, ak._v2.contents.RegularArray):
                    outcontent = outcontent.toListOffsetArray64(False)

            return ak._v2.contents.ListOffsetArray(
                outoffsets,
                outcontent,
                None,
                None,
                self._nplike,
            )

    def _validityerror(self, path):
        if self.offsets.length < 1:
            return f'at {path} ("{type(self)}"): len(offsets) < 1'
        error = self._nplike[
            "awkward_ListArray_validity", self.starts.dtype.type, self.stops.dtype.type
        ](
            self.starts.to(self._nplike),
            self.stops.to(self._nplike),
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
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                return ""
            else:
                return self._content.validityerror(path + ".content")

    def _nbytes_part(self):
        result = self.offsets._nbytes_part() + self.content._nbytes_part()
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def _rpad(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.rpad_axis0(target, clip)
        if posaxis == depth + 1:
            if not clip:
                tolength = ak._v2.index.Index64.empty(1, self._nplike)
                offsets_ = ak._v2.index.Index64.empty(
                    self._offsets.length, self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_ListOffsetArray_rpad_length_axis1",
                        offsets_.dtype.type,
                        self._offsets.dtype.type,
                        tolength.dtype.type,
                    ](
                        offsets_.to(self._nplike),
                        self._offsets.to(self._nplike),
                        self._offsets.length - 1,
                        target,
                        tolength.to(self._nplike),
                    )
                )

                outindex = ak._v2.index.Index64.empty(tolength[0], self._nplike)
                self._handle_error(
                    self._nplike[
                        "awkward_ListOffsetArray_rpad_axis1",
                        outindex.dtype.type,
                        self._offsets.dtype.type,
                    ](
                        outindex.to(self._nplike),
                        self._offsets.to(self._nplike),
                        self._offsets.length - 1,
                        target,
                    )
                )
                next = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                    outindex,
                    self._content,
                    self._identifier,
                    self._parameters,
                    self._nplike,
                )
                return ak._v2.contents.listoffsetarray.ListOffsetArray(
                    offsets_,
                    next.simplify_optiontype(),
                    self._identifier,
                    self._parameters,
                    self._nplike,
                )
            else:
                starts_ = ak._v2.index.Index64.empty(
                    self._offsets.length - 1, self._nplike
                )
                stops_ = ak._v2.index.Index64.empty(
                    self._offsets.length - 1, self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_index_rpad_and_clip_axis1",
                        starts_.dtype.type,
                        stops_.dtype.type,
                    ](
                        starts_.to(self._nplike),
                        stops_.to(self._nplike),
                        target,
                        starts_.length,
                    )
                )

                outindex = ak._v2.index.Index64.empty(
                    target * (self._offsets.length - 1), self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_ListOffsetArray_rpad_and_clip_axis1",
                        outindex.dtype.type,
                        self._offsets.dtype.type,
                    ](
                        outindex.to(self._nplike),
                        self._offsets.to(self._nplike),
                        self._offsets.length - 1,
                        target,
                    )
                )
                next = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                    outindex,
                    self._content,
                    self._identifier,
                    self._parameters,
                    self._nplike,
                )
                return ak._v2.contents.regulararray.RegularArray(
                    next.simplify_optiontype(),
                    target,
                    self.length,
                    None,
                    self._parameters,
                    self._nplike,
                )
        else:
            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                self._offsets,
                self._content._rpad(target, posaxis, depth + 1, clip),
                None,
                self._parameters,
                self._nplike,
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

        npoffsets = self._offsets.to(numpy)
        akcontent = self._content[npoffsets[0] : npoffsets[length]]
        if len(npoffsets) > length + 1:
            npoffsets = npoffsets[: length + 1]

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
                next = ak._v2.contents.ListArray(
                    ak._v2.index.Index(new_starts),
                    ak._v2.index.Index(new_stops),
                    self._content,
                    None,
                    self._parameters,
                    self._nplike,
                )
                return next.toListOffsetArray64(True)._to_arrow(
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
            assert isinstance(akcontent, ak._v2.contents.NumpyArray)

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
                ak._v2._connect.pyarrow.to_awkwardarrow_type(
                    string_type, options["extensionarray"], mask_node, self
                ),
                length,
                [
                    ak._v2._connect.pyarrow.to_validbits(validbytes),
                    pyarrow.py_buffer(npoffsets),
                    pyarrow.py_buffer(akcontent.to(numpy)),
                ],
            )

        else:
            paarray = akcontent._to_arrow(
                pyarrow, None, None, akcontent.length, options
            )

            content_type = pyarrow.list_(paarray.type).value_field.with_nullable(
                akcontent.is_OptionType
            )

            if issubclass(npoffsets.dtype.type, np.int32):
                list_type = pyarrow.list_(content_type)
            else:
                list_type = pyarrow.large_list(content_type)

            return pyarrow.Array.from_buffers(
                ak._v2._connect.pyarrow.to_awkwardarrow_type(
                    list_type, options["extensionarray"], mask_node, self
                ),
                length,
                [
                    ak._v2._connect.pyarrow.to_validbits(validbytes),
                    pyarrow.py_buffer(npoffsets),
                ],
                children=[paarray],
                null_count=ak._v2._connect.pyarrow.to_null_count(
                    validbytes, options["count_nulls"]
                ),
            )

    def _to_numpy(self, allow_missing):
        array_param = self.parameter("__array__")
        if array_param == "bytestring" or array_param == "string":
            return self._nplike.array(self.to_list())

        return ak._v2.operations.convert.to_numpy(self.toRegularArray(), allow_missing)

    def _completely_flatten(self, nplike, options):
        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            return [ak._v2.operations.convert.to_numpy(self)]
        else:
            flat = self._content[self._offsets[0] : self._offsets[-1]]
            return flat._completely_flatten(nplike, options)

    def _recursively_apply(
        self, action, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                return ListOffsetArray(
                    self._offsets,
                    self._content._recursively_apply(
                        action,
                        depth + 1,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    ),
                    self._identifier,
                    self._parameters if options["keep_parameters"] else None,
                    self._nplike,
                )

        else:

            def continuation():
                self._content._recursively_apply(
                    action,
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
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise AssertionError(result)

    def packed(self):
        next = self.toListOffsetArray64(True)
        content = next._content.packed()
        if content.length != next._offsets[-1]:
            content = content[: next._offsets[-1]]
        return ListOffsetArray(
            next._offsets, content, next._identifier, next._parameters, self._nplike
        )

    def _to_list(self, behavior):
        if self.parameter("__array__") == "bytestring":
            content = ak._v2._util.tobytes(self._content.data)
            starts, stops = self.starts, self.stops
            out = [None] * starts.length
            for i in range(starts.length):
                out[i] = content[starts[i] : stops[i]]
            return out

        elif self.parameter("__array__") == "string":
            content = ak._v2._util.tobytes(self._content.data)
            starts, stops = self.starts, self.stops
            out = [None] * starts.length
            for i in range(starts.length):
                out[i] = content[starts[i] : stops[i]].decode(errors="surrogateescape")
            return out

        else:
            out = self._to_list_custom(behavior)
            if out is not None:
                return out

            content = self._content._to_list(behavior)
            starts, stops = self.starts, self.stops
            out = [None] * starts.length

            for i in range(starts.length):
                out[i] = content[starts[i] : stops[i]]
            return out
