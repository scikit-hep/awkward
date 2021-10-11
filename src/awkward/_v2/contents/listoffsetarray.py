# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.index import Index
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.listoffsetform import ListOffsetForm

np = ak.nplike.NumpyMetadata.instance()


class ListOffsetArray(Content):
    def __init__(self, offsets, content, identifier=None, parameters=None):
        if not isinstance(offsets, Index) and offsets.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        ):
            raise TypeError(
                "{0} 'offsets' must be an Index with dtype in (int32, uint32, int64), "
                "not {1}".format(type(self).__name__, repr(offsets))
            )
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not len(offsets) >= 1:
            raise ValueError(
                "{0} len(offsets) ({1}) must be >= 1".format(
                    type(self).__name__, len(offsets)
                )
            )

        self._offsets = offsets
        self._content = content
        self._init(identifier, parameters)

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

    @property
    def nplike(self):
        return self._offsets.nplike

    @property
    def nonvirtual_nplike(self):
        return self._offsets.nplike

    Form = ListOffsetForm

    @property
    def form(self):
        return self.Form(
            self._offsets.form,
            self._content.form,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return ListOffsetArray(
            ak._v2.index.Index(self._offsets.to(tt)),
            self._content.typetracer,
            self._typetracer_identifier(),
            self._parameters,
        )

    def __len__(self):
        return len(self._offsets) - 1

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ListOffsetArray len="]
        out.append(repr(str(len(self))))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._offsets._repr(indent + "    ", "<offsets>", "</offsets>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</ListOffsetArray>")
        out.append(post)
        return "".join(out)

    def toListOffsetArray64(self, start_at_zero=False):
        if issubclass(self._offsets.dtype.type, np.int64):
            if not start_at_zero or self._offsets[0] == 0:
                return self

            if start_at_zero:
                offsets = ak._v2.index.Index64(
                    self._offsets.to(self.nplike) - self._offsets[0]
                )
                content = self._content[self._offsets[0] :]
            else:
                offsets, content = self._offsets, self._content

            return ListOffsetArray(offsets, content, self._identifier, self._parameters)

        else:
            offsets = self._compact_offsets64(start_at_zero)
            return self._broadcast_tooffsets64(offsets)

    def toRegularArray(self):
        nplike = self.nplike

        start, stop = self._offsets[0], self._offsets[len(self._offsets) - 1]
        content = self._content._getitem_range(slice(start, stop))
        size = ak._v2.index.Index64.empty(1, nplike)
        self._handle_error(
            nplike[
                "awkward_ListOffsetArray_toRegularArray",
                size.dtype.type,
                self._offsets.dtype.type,
            ](
                size.to(nplike),
                self._offsets.to(nplike),
                len(self._offsets),
            )
        )

        return ak._v2.contents.RegularArray(
            content,
            size[0],
            len(self._offsets),
            self._identifier,
            self._parameters,
        )

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)) and self.nplike.known_shape:
            raise NestedIndexError(self, where)
        start, stop = self._offsets[where], self._offsets[where + 1]
        return self._content._getitem_range(slice(start, stop))

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        offsets = self._offsets[start : stop + 1]
        if len(offsets) == 0:
            offsets = Index(self.nplike.array([0], dtype=self._offsets.dtype))
        return ListOffsetArray(
            offsets,
            self._content,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return ListOffsetArray(
            self._offsets,
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return ListOffsetArray(
            self._offsets,
            self._content._getitem_fields(where, only_fields),
            self._fields_identifier(where),
            None,
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
        )

    def _compact_offsets64(self, start_at_zero):
        offsets_len = len(self._offsets) - 1
        out = ak._v2.index.Index64.empty(offsets_len + 1, self.nplike)
        self._handle_error(
            self.nplike[
                "awkward_ListOffsetArray_compact_offsets",
                out.dtype.type,
                self._offsets.dtype.type,
            ](out.to(self.nplike), self._offsets.to(self.nplike), offsets_len)
        )
        if isinstance(out.data, ak._v2._typetracer.TypeTracerArray):
            out.data.fill_other = len(self._content)
        return out

    def _broadcast_tooffsets64(self, offsets):
        nplike = self.nplike
        if len(offsets) == 0 or offsets[0] != 0:
            raise AssertionError(
                "broadcast_tooffsets64 can only be used with offsets that start at 0, not {0}".format(
                    "(empty)" if len(offsets) == 0 else str(offsets[0])
                )
            )

        if len(offsets) - 1 != len(self):
            raise AssertionError(
                "cannot broadcast {0} of length {1} to length {2}".format(
                    type(self).__name__, len(self), len(offsets) - 1
                )
            )

        if self._identifier is not None:
            identifier = self._identifier[slice(0, len(offsets) - 1)]
        else:
            identifier = self._identifier

        starts, stops = self.starts, self.stops

        nextcarry = ak._v2.index.Index64.empty(offsets[-1], nplike)
        self._handle_error(
            nplike[
                "awkward_ListArray_broadcast_tooffsets",
                nextcarry.dtype.type,
                offsets.dtype.type,
                starts.dtype.type,
                stops.dtype.type,
            ](
                nextcarry.to(nplike),
                offsets.to(nplike),
                len(offsets),
                starts.to(nplike),
                stops.to(nplike),
                len(self._content),
            )
        )

        nextcontent = self._content._carry(nextcarry, True, NestedIndexError)

        return ListOffsetArray(offsets, nextcontent, identifier, self._parameters)

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        out = ak._v2.contents.listarray.ListArray(
            self.starts, self.stops, self._content, self._identifier, self._parameters
        )
        return out._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

    def _getitem_next(self, head, tail, advanced):
        nplike = self.nplike

        if head == ():
            return self

        elif isinstance(head, int):
            assert advanced is None
            lenstarts = len(self._offsets) - 1
            starts, stops = self.starts, self.stops
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            nextcarry = ak._v2.index.Index64.empty(lenstarts, nplike)

            self._handle_error(
                nplike[
                    "awkward_ListArray_getitem_next_at",
                    nextcarry.dtype.type,
                    starts.dtype.type,
                    stops.dtype.type,
                ](
                    nextcarry.to(nplike),
                    starts.to(nplike),
                    stops.to(nplike),
                    lenstarts,
                    head,
                )
            )
            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
            return nextcontent._getitem_next(nexthead, nexttail, advanced)

        elif isinstance(head, slice):
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            lenstarts = len(self._offsets) - 1
            start, stop, step = head.start, head.stop, head.step

            step = 1 if step is None else step
            start = ak._util.kSliceNone if start is None else start
            stop = ak._util.kSliceNone if stop is None else stop

            carrylength = ak._v2.index.Index64.empty(1, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListArray_getitem_next_range_carrylength",
                    carrylength.dtype.type,
                    self.starts.dtype.type,
                    self.stops.dtype.type,
                ](
                    carrylength.to(nplike),
                    self.starts.to(nplike),
                    self.stops.to(nplike),
                    lenstarts,
                    start,
                    stop,
                    step,
                )
            )

            if self._starts.dtype == "int64":
                nextoffsets = ak._v2.index.Index64.empty(lenstarts + 1, nplike)
            elif self._starts.dtype == "int32":
                nextoffsets = ak._v2.index.Index32.empty(lenstarts + 1, nplike)
            elif self._starts.dtype == "uint32":
                nextoffsets = ak._v2.index.IndexU32.empty(lenstarts + 1, nplike)
            nextcarry = ak._v2.index.Index64.empty(carrylength[0], nplike)

            self._handle_error(
                nplike[
                    "awkward_ListArray_getitem_next_range",
                    nextoffsets.dtype.type,
                    nextcarry.dtype.type,
                    self.starts.dtype.type,
                    self.stops.dtype.type,
                ](
                    nextoffsets.to(nplike),
                    nextcarry.to(nplike),
                    self.starts.to(nplike),
                    self.stops.to(nplike),
                    lenstarts,
                    start,
                    stop,
                    step,
                )
            )

            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)

            if advanced is None or len(advanced) == 0:
                return ak._v2.contents.listoffsetarray.ListOffsetArray(
                    nextoffsets,
                    nextcontent._getitem_next(nexthead, nexttail, advanced),
                    self._identifier,
                    self._parameters,
                )

            else:
                total = ak._v2.index.Index64.empty(1, nplike)
                self._handle_error(
                    nplike[
                        "awkward_ListArray_getitem_next_range_counts",
                        total.dtype.type,
                        nextoffsets.dtype.type,
                    ](
                        total.to(nplike),
                        nextoffsets.to(nplike),
                        lenstarts,
                    )
                )

                nextadvanced = ak._v2.index.Index64.empty(total[0], nplike)
                self._handle_error(
                    nplike[
                        "awkward_ListArray_getitem_next_range_spreadadvanced",
                        nextadvanced.dtype.type,
                        advanced.dtype.type,
                        nextoffsets.dtype.type,
                    ](
                        nextadvanced.to(nplike),
                        advanced.to(nplike),
                        nextoffsets.to(nplike),
                        lenstarts,
                    )
                )

                return ak._v2.contents.listoffsetarray.ListOffsetArray(
                    nextoffsets,
                    nextcontent._getitem_next(nexthead, nexttail, nextadvanced),
                    self._identifier,
                    self._parameters,
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
            flathead = nplike.asarray(head.data.reshape(-1))
            lenstarts = len(self.starts)
            regular_flathead = ak._v2.index.Index64(flathead)
            if advanced is None or len(advanced) == 0:
                nextcarry = ak._v2.index.Index64.empty(
                    lenstarts * len(flathead), nplike
                )
                nextadvanced = ak._v2.index.Index64.empty(
                    lenstarts * len(flathead), nplike
                )
                self._handle_error(
                    nplike[
                        "awkward_ListArray_getitem_next_array",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.to(nplike),
                        nextadvanced.to(nplike),
                        self.starts.to(nplike),
                        self.stops.to(nplike),
                        regular_flathead.to(nplike),
                        lenstarts,
                        len(regular_flathead),
                        len(self._content),
                    ),
                    head,
                )
                nextcontent = self._content._carry(nextcarry, True, NestedIndexError)

                out = nextcontent._getitem_next(nexthead, nexttail, nextadvanced)
                if advanced is None:
                    return ak._v2._slicing.getitem_next_array_wrap(
                        out, head.metadata.get("shape", (len(head),))
                    )
                else:
                    return out

            else:
                nextcarry = ak._v2.index.Index64.empty(len(self), nplike)
                nextadvanced = ak._v2.index.Index64.empty(len(self), nplike)
                self._handle_error(
                    nplike[
                        "awkward_ListArray_getitem_next_array_advanced",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        self.starts.dtype.type,
                        self.stops.dtype.type,
                        regular_flathead.dtype.type,
                        advanced.dtype.type,
                    ](
                        nextcarry.to(nplike),
                        nextadvanced.to(nplike),
                        self.starts.to(nplike),
                        self.stops.to(nplike),
                        regular_flathead.to(nplike),
                        advanced.to(nplike),
                        lenstarts,
                        len(regular_flathead),
                        len(self._content),
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
            )
            return listarray._getitem_next(head, tail, advanced)

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def _localindex(self, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        elif posaxis == depth + 1:
            offsets = self._compact_offsets64(True)
            innerlength = offsets[len(offsets) - 1]
            localindex = ak._v2.index.Index64.empty(innerlength, self.nplike)
            self._handle_error(
                self.nplike[
                    "awkward_ListArray_localindex",
                    localindex.dtype.type,
                    offsets.dtype.type,
                ](localindex.to(self.nplike), offsets.to(self.nplike), len(offsets) - 1)
            )
            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                offsets,
                ak._v2.contents.NumpyArray(localindex),
                self._identifier,
                self._parameters,
            )
        else:
            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                self._offsets,
                self._content._localindex(posaxis, depth + 1),
                self._identifier,
                self._parameters,
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
        if len(self._offsets) - 1 == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))

        nplike = self.nplike

        branch, depth = self.branch_depth

        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            if branch or (negaxis != depth):
                raise ValueError("array with strings can only be sorted with axis=-1")

            # FIXME: check validity error

            if isinstance(self._content, ak._v2.contents.NumpyArray):
                nextcarry = ak._v2.index.Index64.empty(len(self._offsets) - 1, nplike)

                self_starts, self_stops = self._offsets[:-1], self._offsets[1:]
                self._handle_error(
                    nplike[
                        "awkward_ListOffsetArray_argsort_strings",
                        nextcarry.dtype.type,
                        parents.dtype.type,
                        self._content.dtype.type,
                        self_starts.dtype.type,
                        self_stops.dtype.type,
                    ](
                        nextcarry.to(nplike),
                        parents.to(nplike),
                        len(parents),
                        self._content._data,
                        self_starts.to(nplike),
                        self_stops.to(nplike),
                        stable,
                        ascending,
                        False,
                    )
                )
                return ak._v2.contents.NumpyArray(nextcarry)

        if not branch and (negaxis == depth):
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise ValueError("array with strings can only be sorted with axis=-1")
            if len(self._offsets) - 1 != len(parents):
                raise ValueError(
                    "length of self._offsets_ - 1 is not equal to parents length"
                )

            maxcount = ak._v2.index.Index64.empty(1, nplike)
            offsetscopy = ak._v2.index.Index64.empty(len(self._offsets), nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64",
                    maxcount.dtype.type,
                    offsetscopy.dtype.type,
                    self._offsets.dtype.type,
                ](
                    maxcount.to(nplike),
                    offsetscopy.to(nplike),
                    self._offsets.to(nplike),
                    len(self._offsets) - 1,
                )
            )

            maxcount = maxcount[0]
            nextlen = self._offsets[-1] - self._offsets[0]

            nextcarry = ak._v2.index.Index64.empty(nextlen, nplike)
            nextparents = ak._v2.index.Index64.empty(nextlen, nplike)
            maxnextparents = ak._v2.index.Index64.empty(1, nplike)
            distincts = ak._v2.index.Index64.empty(maxcount * outlength, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_preparenext_64",
                    nextcarry.dtype.type,
                    nextparents.dtype.type,
                    maxnextparents.dtype.type,
                    distincts.dtype.type,
                    self._offsets.dtype.type,
                    offsetscopy.dtype.type,
                    parents.dtype.type,
                ](
                    nextcarry.to(nplike),
                    nextparents.to(nplike),
                    nextlen,
                    maxnextparents.to(nplike),
                    distincts.to(nplike),
                    maxcount * outlength,
                    offsetscopy.to(nplike),
                    self._offsets.to(nplike),
                    len(self._offsets) - 1,
                    parents.to(nplike),
                    maxcount,
                )
            )

            nextstarts_length = maxnextparents[0] + 1
            nextstarts = ak._v2.index.Index64.empty(nextstarts_length, nplike, np.int64)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64",
                    nextstarts.dtype.type,
                    nextparents.dtype.type,
                ](
                    nextstarts.to(nplike),
                    nextparents.to(nplike),
                    nextlen,
                )
            )

            nummissing = ak._v2.index.Index64.empty(maxcount, nplike)
            missing = ak._v2.index.Index64.empty(self._offsets[-1], nplike)
            nextshifts = ak._v2.index.Index64.empty(nextlen, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64",
                    nummissing.dtype.type,
                    missing.dtype.type,
                    nextshifts.dtype.type,
                    self._offsets.dtype.type,
                    starts.dtype.type,
                    parents.dtype.type,
                    nextcarry.dtype.type,
                ](
                    nummissing.to(nplike),
                    missing.to(nplike),
                    nextshifts.to(nplike),
                    self._offsets.to(nplike),
                    len(self._offsets) - 1,
                    starts.to(nplike),
                    parents.to(nplike),
                    maxcount,
                    nextlen,
                    nextcarry.to(nplike),
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

            outcarry = ak._v2.index.Index64.empty(nextlen, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_local_preparenext_64",
                    outcarry.dtype.type,
                    nextcarry.dtype.type,
                ](
                    outcarry.to(nplike),
                    nextcarry.to(nplike),
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
            )
        else:
            nextparents = ak._v2.index.Index64.empty(
                self._offsets[-1] - self._offsets[0], nplike
            )

            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.to(nplike),
                    self._offsets.to(nplike),
                    len(self._offsets) - 1,
                )
            )

            trimmed = self._content[self._offsets[0] : self._offsets[-1]]
            outcontent = trimmed._argsort_next(
                negaxis,
                self._offsets[:-1],
                shifts,
                nextparents,
                len(self._offsets) - 1,
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
            )

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        nplike = self.nplike

        branch, depth = self.branch_depth

        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            if branch or (negaxis != depth):
                raise ValueError("array with strings can only be sorted with axis=-1")

            # FIXME: check validity error

            if isinstance(self._content, ak._v2.contents.NumpyArray):
                nextcarry = ak._v2.index.Index64.empty(len(self._offsets) - 1, nplike)

                starts, stops = self._offsets[:-1], self._offsets[1:]
                self._handle_error(
                    nplike[
                        "awkward_ListOffsetArray_argsort_strings",
                        nextcarry.dtype.type,
                        parents.dtype.type,
                        self._content.dtype.type,
                        starts.dtype.type,
                        stops.dtype.type,
                    ](
                        nextcarry.to(nplike),
                        parents.to(nplike),
                        len(parents),
                        self._content._data,
                        starts.to(nplike),
                        stops.to(nplike),
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
                raise ValueError("array with strings can only be sorted with axis=-1")
            if len(self._offsets) - 1 != len(parents):
                raise ValueError(
                    "length of self._offsets_ - 1 is not equal to parents length"
                )

            nextlen = self._offsets[-1] - self._offsets[0]
            maxcount = ak._v2.index.Index64.empty(1, nplike)
            offsetscopy = ak._v2.index.Index64.empty(len(self._offsets), nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64",
                    maxcount.dtype.type,
                    offsetscopy.dtype.type,
                    self._offsets.dtype.type,
                ](
                    maxcount.to(nplike),
                    offsetscopy.to(nplike),
                    self._offsets.to(nplike),
                    len(self._offsets) - 1,
                )
            )

            distincts_length = outlength * maxcount[0]
            nextcarry = ak._v2.index.Index64.empty(nextlen, nplike)
            nextparents = ak._v2.index.Index64.empty(nextlen, nplike)
            maxnextparents = ak._v2.index.Index64.empty(1, nplike)
            distincts = ak._v2.index.Index64.empty(distincts_length, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_preparenext_64",
                    nextcarry.dtype.type,
                    nextparents.dtype.type,
                    maxnextparents.dtype.type,
                    distincts.dtype.type,
                    self._offsets.dtype.type,
                    offsetscopy.dtype.type,
                    parents.dtype.type,
                ](
                    nextcarry.to(nplike),
                    nextparents.to(nplike),
                    nextlen,
                    maxnextparents.to(nplike),
                    distincts.to(nplike),
                    distincts_length,
                    offsetscopy.to(nplike),
                    self._offsets.to(nplike),
                    len(self._offsets) - 1,
                    parents.to(nplike),
                    maxcount[0],
                )
            )

            nextstarts = ak._v2.index.Index64.empty(maxnextparents[0] + 1, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64",
                    nextstarts.dtype.type,
                    nextparents.dtype.type,
                ](
                    nextstarts.to(nplike),
                    nextparents.to(nplike),
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

            outcarry = ak._v2.index.Index64.empty(nextlen, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_local_preparenext_64",
                    outcarry.dtype.type,
                    nextcarry.dtype.type,
                ](
                    outcarry.to(nplike),
                    nextcarry.to(nplike),
                    nextlen,
                )
            )

            return ak._v2.contents.ListOffsetArray(
                self._compact_offsets64(True),
                outcontent._carry(outcarry, False, NestedIndexError),
                None,
                self._parameters,
            )
        else:
            nextparents = ak._v2.index.Index64.empty(
                self._offsets[-1] - self._offsets[0], nplike
            )

            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.to(nplike),
                    self._offsets.to(nplike),
                    len(self._offsets) - 1,
                )
            )

            trimmed = self._content[self._offsets[0] : self._offsets[-1]]
            outcontent = trimmed._sort_next(
                negaxis,
                self._offsets[:-1],
                nextparents,
                len(self._offsets) - 1,
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
            )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
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

            nplike = self.nplike
            totallen = ak._v2.index.Index64.empty(1, nplike, dtype=np.int64)
            offsets = ak._v2.index.Index64.empty(len(self) + 1, nplike, dtype=np.int64)
            self._handle_error(
                nplike[
                    "awkward_ListArray_combinations_length",
                    totallen.to(nplike).dtype.type,
                    offsets.to(nplike).dtype.type,
                    starts.to(nplike).dtype.type,
                    stops.to(nplike).dtype.type,
                ](
                    totallen.to(nplike),
                    offsets.to(nplike),
                    n,
                    replacement,
                    starts.to(nplike),
                    stops.to(nplike),
                    len(self),
                )
            )

            tocarryraw = nplike.empty(n, dtype=np.intp)
            tocarry = []

            for i in range(n):
                ptr = ak._v2.index.Index64.empty(totallen[0], nplike, dtype=np.int64)
                tocarry.append(ptr)
                tocarryraw[i] = ptr.ptr

            toindex = ak._v2.index.Index64.empty(n, nplike, dtype=np.int64)
            fromindex = ak._v2.index.Index64.empty(n, nplike, dtype=np.int64)
            self._handle_error(
                nplike[
                    "awkward_ListArray_combinations",
                    np.int64,
                    toindex.to(nplike).dtype.type,
                    fromindex.to(nplike).dtype.type,
                    starts.to(nplike).dtype.type,
                    stops.to(nplike).dtype.type,
                ](
                    tocarryraw,
                    toindex.to(nplike),
                    fromindex.to(nplike),
                    n,
                    replacement,
                    starts.to(nplike),
                    stops.to(nplike),
                    len(self),
                )
            )
            contents = []

            for ptr in tocarry:
                contents.append(self._content._carry(ptr, True, NestedIndexError))

            recordarray = ak._v2.contents.recordarray.RecordArray(
                contents, recordlookup, parameters=parameters
            )
            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                offsets, recordarray, self._identifier, self._parameters
            )
        else:
            compact = self.toListOffsetArray64(True)
            next = compact._content._combinations(
                n, replacement, recordlookup, parameters, posaxis, depth + 1
            )
            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                compact.offsets, next, self._identifier, self._parameters
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
        if self.offsets[0] != 0:
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

        nplike = self.nplike

        branch, depth = self.branch_depth
        globalstarts_length = len(self._offsets) - 1
        parents_length = len(parents)
        nextlen = self._offsets[-1] - self._offsets[0]

        if not branch and negaxis == depth:
            maxcount = ak._v2.index.Index64.empty(1, nplike)
            offsetscopy = ak._v2.index.Index64.empty(len(self.offsets), nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64",
                    maxcount.dtype.type,
                    offsetscopy.dtype.type,
                    self._offsets.dtype.type,
                ](
                    maxcount.to(nplike),
                    offsetscopy.to(nplike),
                    self._offsets.to(nplike),
                    globalstarts_length,
                )
            )

            distincts_length = outlength * maxcount[0]
            nextcarry = ak._v2.index.Index64.empty(nextlen, nplike)
            nextparents = ak._v2.index.Index64.empty(nextlen, nplike)
            maxnextparents = ak._v2.index.Index64.empty(1, nplike)
            distincts = ak._v2.index.Index64.empty(distincts_length, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_preparenext_64",
                    nextcarry.dtype.type,
                    nextparents.dtype.type,
                    maxnextparents.dtype.type,
                    distincts.dtype.type,
                    self._offsets.dtype.type,
                    offsetscopy.dtype.type,
                    parents.dtype.type,
                ](
                    nextcarry.to(nplike),
                    nextparents.to(nplike),
                    nextlen,
                    maxnextparents.to(nplike),
                    distincts.to(nplike),
                    distincts_length,
                    offsetscopy.to(nplike),
                    self._offsets.to(nplike),
                    globalstarts_length,
                    parents.to(nplike),
                    maxcount[0],
                )
            )

            nextstarts = ak._v2.index.Index64.empty(maxnextparents[0] + 1, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64",
                    nextstarts.dtype.type,
                    nextparents.dtype.type,
                ](
                    nextstarts.to(nplike),
                    nextparents.to(nplike),
                    nextlen,
                )
            )

            gaps = ak._v2.index.Index64.empty(outlength, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_findgaps_64",
                    gaps.dtype.type,
                    parents.dtype.type,
                ](
                    gaps.to(nplike),
                    parents.to(nplike),
                    parents_length,
                )
            )

            outstarts = ak._v2.index.Index64.empty(outlength, nplike)
            outstops = ak._v2.index.Index64.empty(outlength, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",
                    outstarts.dtype.type,
                    outstops.dtype.type,
                    distincts.dtype.type,
                    gaps.dtype.type,
                ](
                    outstarts.to(nplike),
                    outstops.to(nplike),
                    distincts.to(nplike),
                    distincts_length,
                    gaps.to(nplike),
                    outlength,
                )
            )

            if reducer.needs_position:
                nextshifts = ak._v2.index.Index64.empty(nextlen, nplike)
                nummissing = ak._v2.index.Index64.empty(maxcount[0], nplike)
                missing = ak._v2.index.Index64.empty(self._offsets[-1], nplike)
                self._handle_error(
                    nplike[
                        "awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64",
                        nummissing.dtype.type,
                        missing.dtype.type,
                        nextshifts.dtype.type,
                        self._offsets.dtype.type,
                        starts.dtype.type,
                        parents.dtype.type,
                        nextcarry.dtype.type,
                    ](
                        nummissing.to(nplike),
                        missing.to(nplike),
                        nextshifts.to(nplike),
                        self._offsets.to(nplike),
                        globalstarts_length,
                        starts.to(nplike),
                        parents.to(nplike),
                        maxcount[0],
                        nextlen,
                        nextcarry.to(nplike),
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
            )

            if keepdims:
                out = ak._v2.contents.RegularArray(
                    out,
                    1,
                    len(self),
                    None,
                    None,
                ).toListOffsetArray64(False)

            return out

        else:
            nextparents = ak._v2.index.Index64.empty(nextlen, nplike)

            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.to(nplike),
                    self._offsets.to(nplike),
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
            outoffsets = ak._v2.index.Index64.empty(outlength + 1, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListOffsetArray_reduce_local_outoffsets_64",
                    outoffsets.dtype.type,
                    parents.dtype.type,
                ](
                    outoffsets.to(nplike),
                    parents.to(nplike),
                    len(parents),
                    outlength,
                )
            )

            if (
                keepdims and self._content.dimension_optiontype
            ):  # FIXME not self._represents_regular
                if isinstance(outcontent, ak._v2.contents.RegularArray):
                    outcontent = outcontent.toListOffsetArray64(False)

            return ak._v2.contents.ListOffsetArray(
                outoffsets,
                outcontent,
                None,
                None,
            )

    def _validityerror(self, path):
        if len(self.offsets) < 1:
            return 'at {0} ("{1}"): len(offsets) < 1'.format(path, type(self))
        error = self.nplike[
            "awkward_ListArray_validity", self.starts.dtype.type, self.stops.dtype.type
        ](
            self.starts.to(self.nplike),
            self.stops.to(self.nplike),
            len(self.starts),
            len(self.content),
        )
        if error.str is not None:
            if error.filename is None:
                filename = ""
            else:
                filename = " (in compiled code: " + error.filename.decode(
                    errors="surrogateescape"
                ).lstrip("\n").lstrip("(")
            message = error.str.decode(errors="surrogateescape")
            return 'at {0} ("{1}"): {2} at i={3}{4}'.format(
                path, type(self), message, error.id, filename
            )
        else:
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                return ""
            else:
                return self.content.validityerror(path + ".content")
