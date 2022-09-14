# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy

import awkward as ak
from awkward._v2.index import Index
from awkward._v2.contents.content import Content, unset
from awkward._v2.contents.listoffsetarray import ListOffsetArray
from awkward._v2.forms.listform import ListForm

np = ak.nplike.NumpyMetadata.instance()


class ListArray(Content):
    is_ListType = True

    def copy(
        self,
        starts=unset,
        stops=unset,
        content=unset,
        identifier=unset,
        parameters=unset,
        nplike=unset,
    ):
        return ListArray(
            self._starts if starts is unset else starts,
            self._stops if stops is unset else stops,
            self._content if content is unset else content,
            self._identifier if identifier is unset else identifier,
            self._parameters if parameters is unset else parameters,
            self._nplike if nplike is unset else nplike,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            starts=copy.deepcopy(self._starts, memo),
            stops=copy.deepcopy(self._stops, memo),
            content=copy.deepcopy(self._content, memo),
            identifier=copy.deepcopy(self._identifier, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    def __init__(
        self, starts, stops, content, identifier=None, parameters=None, nplike=None
    ):
        if not isinstance(starts, Index) and starts.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        ):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'starts' must be an Index with dtype in (int32, uint32, int64), "
                    "not {}".format(type(self).__name__, repr(starts))
                )
            )
        if not (isinstance(stops, Index) and starts.dtype == stops.dtype):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'stops' must be an Index with the same dtype as 'starts' ({}), "
                    "not {}".format(
                        type(self).__name__, repr(starts.dtype), repr(stops)
                    )
                )
            )
        if not isinstance(content, Content):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'content' must be a Content subtype, not {}".format(
                        type(self).__name__, repr(content)
                    )
                )
            )
        if (
            starts.nplike.known_shape
            and stops.nplike.known_shape
            and starts.length > stops.length
        ):
            raise ak._v2._util.error(
                ValueError(
                    "{} len(starts) ({}) must be <= len(stops) ({})".format(
                        type(self).__name__, starts.length, stops.length
                    )
                )
            )
        if nplike is None:
            nplike = content.nplike
        if nplike is None:
            nplike = starts.nplike

        self._starts = starts
        self._stops = stops
        self._content = content
        self._init(identifier, parameters, nplike)

    @property
    def starts(self):
        return self._starts

    @property
    def stops(self):
        return self._stops

    @property
    def content(self):
        return self._content

    Form = ListForm

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.Form(
            self._starts.form,
            self._stops.form,
            self._content._form_with_key(getkey),
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, nplike):
        assert isinstance(form, self.Form)
        key1 = getkey(self, form, "starts")
        key2 = getkey(self, form, "stops")
        container[key1] = ak._v2._util.little_endian(self._starts.raw(nplike))
        container[key2] = ak._v2._util.little_endian(self._stops.raw(nplike))
        self._content._to_buffers(form.content, getkey, container, nplike)

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return ListArray(
            ak._v2.index.Index(self._starts.raw(tt)),
            ak._v2.index.Index(self._stops.raw(tt)),
            self._content.typetracer,
            self._typetracer_identifier(),
            self._parameters,
            tt,
        )

    @property
    def length(self):
        return self._starts.length

    def _forget_length(self):
        return ListArray(
            self._starts.forget_length(),
            self._stops,
            self._content,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ListArray len="]
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._starts._repr(indent + "    ", "<starts>", "</starts>\n"))
        out.append(self._stops._repr(indent + "    ", "<stops>", "</stops>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</ListArray>")
        out.append(post)
        return "".join(out)

    def merge_parameters(self, parameters):
        return ListArray(
            self._starts,
            self._stops,
            self._content,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
            self._nplike,
        )

    def toListOffsetArray64(self, start_at_zero=False):
        starts = self._starts.data
        stops = self._stops.data

        if not self._nplike.known_data:
            offsets = self._nplike.index_nplike.empty(
                starts.shape[0] + 1, dtype=starts.dtype
            )
            return ListOffsetArray(
                ak._v2.index.Index(offsets, nplike=self._nplike),
                self._content,
                self._identifier,
                self._parameters,
                self._nplike,
            )

        elif self._nplike.index_nplike.array_equal(starts[1:], stops[:-1]):
            offsets = self._nplike.index_nplike.empty(
                starts.shape[0] + 1, dtype=starts.dtype
            )
            if offsets.shape[0] == 1:
                offsets[0] = 0
            else:
                offsets[:-1] = starts
                offsets[-1] = stops[-1]
            return ListOffsetArray(
                ak._v2.index.Index(offsets, nplike=self._nplike),
                self._content,
                self._identifier,
                self._parameters,
                self._nplike,
            ).toListOffsetArray64(start_at_zero=start_at_zero)

        else:
            offsets = self._compact_offsets64(start_at_zero)
            return self._broadcast_tooffsets64(offsets)

    def toRegularArray(self):
        offsets = self._compact_offsets64(True)
        return self._broadcast_tooffsets64(offsets).toRegularArray()

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if not self._nplike.known_data:
            return self._content._getitem_range(slice(0, 0))

        if where < 0:
            where += self.length
        if not (0 <= where < self.length) and self._nplike.known_shape:
            raise ak._v2._util.indexerror(self, where)
        start, stop = self._starts[where], self._stops[where]
        return self._content._getitem_range(slice(start, stop))

    def _getitem_range(self, where):
        if not self._nplike.known_shape:
            return self

        start, stop, step = where.indices(self.length)
        assert step == 1
        return ListArray(
            self._starts[start:stop],
            self._stops[start:stop],
            self._content,
            self._range_identifier(start, stop),
            self._parameters,
            self._nplike,
        )

    def _getitem_field(self, where, only_fields=()):
        return ListArray(
            self._starts,
            self._stops,
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
            self._nplike,
        )

    def _getitem_fields(self, where, only_fields=()):
        return ListArray(
            self._starts,
            self._stops,
            self._content._getitem_fields(where, only_fields),
            self._fields_identifier(where),
            None,
            self._nplike,
        )

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextstarts = self._starts[carry.data]
            nextstops = self._stops[: self._starts.length][carry.data]
        except IndexError as err:
            raise ak._v2._util.indexerror(self, carry.data, str(err)) from err

        return ListArray(
            nextstarts,
            nextstops,
            self._content,
            self._carry_identifier(carry),
            self._parameters,
            self._nplike,
        )

    def _compact_offsets64(self, start_at_zero):
        starts_len = self._starts.length
        out = ak._v2.index.Index64.empty(starts_len + 1, self._nplike)
        assert (
            out.nplike is self._nplike
            and self._starts.nplike is self._nplike
            and self._stops.nplike is self._nplike
        )
        self._handle_error(
            self._nplike[
                "awkward_ListArray_compact_offsets",
                out.dtype.type,
                self._starts.dtype.type,
                self._stops.dtype.type,
            ](
                out.data,
                self._starts.data,
                self._stops.data,
                starts_len,
            )
        )
        return out

    def _broadcast_tooffsets64(self, offsets):
        return ListOffsetArray._broadcast_tooffsets64(self, offsets)

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        slicestarts = slicestarts._to_nplike(self.nplike)
        slicestops = slicestops._to_nplike(self.nplike)
        if self._nplike.known_shape and slicestarts.length != self.length:
            raise ak._v2._util.indexerror(
                self,
                ak._v2.contents.ListArray(
                    slicestarts, slicestops, slicecontent, None, None, self._nplike
                ),
                "cannot fit jagged slice with length {} into {} of size {}".format(
                    slicestarts.length, type(self).__name__, self.length
                ),
            )

        if isinstance(slicecontent, ak._v2.contents.listoffsetarray.ListOffsetArray):
            outoffsets = ak._v2.index.Index64.empty(
                slicestarts.length + 1, self._nplike
            )
            assert (
                outoffsets.nplike is self._nplike
                and slicestarts.nplike is self._nplike
                and slicestops.nplike is self._nplike
                and self._starts.nplike is self._nplike
                and self._stops.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_jagged_descend",
                    outoffsets.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    outoffsets.data,
                    slicestarts.data,
                    slicestops.data,
                    slicestarts.length,
                    self._starts.data,
                    self._stops.data,
                ),
                slicer=ListArray(slicestarts, slicestops, slicecontent),
            )

            as_list_offset_array = self.toListOffsetArray64(False)
            next_content = as_list_offset_array._content[
                as_list_offset_array.offsets[0] : as_list_offset_array.offsets[-1]
            ]

            sliceoffsets = ak._v2.index.Index64(slicecontent._offsets)

            outcontent = next_content._getitem_next_jagged(
                sliceoffsets[:-1], sliceoffsets[1:], slicecontent._content, tail
            )

            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                outoffsets, outcontent, None, self._parameters, self._nplike
            )

        elif isinstance(slicecontent, ak._v2.contents.numpyarray.NumpyArray):
            carrylen = ak._v2.index.Index64.empty(1, self._nplike)
            assert (
                carrylen.nplike is self._nplike
                and slicestarts.nplike is self._nplike
                and slicestops.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_jagged_carrylen",
                    carrylen.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                ](
                    carrylen.data,
                    slicestarts.data,
                    slicestops.data,
                    slicestarts.length,
                ),
                slicer=ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
            )
            sliceindex = ak._v2.index.Index64(slicecontent._data)
            outoffsets = ak._v2.index.Index64.empty(
                slicestarts.length + 1, self._nplike
            )
            nextcarry = ak._v2.index.Index64.empty(carrylen[0], self._nplike)

            assert (
                outoffsets.nplike is self._nplike
                and nextcarry.nplike is self._nplike
                and slicestarts.nplike is self._nplike
                and slicestops.nplike is self._nplike
                and sliceindex.nplike is self._nplike
                and self._starts.nplike is self._nplike
                and self._stops.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_jagged_apply",
                    outoffsets.dtype.type,
                    nextcarry.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    sliceindex.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    outoffsets.data,
                    nextcarry.data,
                    slicestarts.data,
                    slicestops.data,
                    slicestarts.length,
                    sliceindex.data,
                    sliceindex.length,
                    self._starts.data,
                    self._stops.data,
                    self._content.length,
                ),
                slicer=ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
            )
            nextcontent = self._content._carry(nextcarry, True)
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            outcontent = nextcontent._getitem_next(nexthead, nexttail, None)

            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                outoffsets, outcontent, None, None, self._nplike
            )

        elif isinstance(
            slicecontent, ak._v2.contents.indexedoptionarray.IndexedOptionArray
        ):
            if self._nplike.known_shape and self._starts.length < slicestarts.length:
                raise ak._v2._util.indexerror(
                    self,
                    ak._v2.contents.ListArray(
                        slicestarts, slicestops, slicecontent, None, None, self._nplike
                    ),
                    "jagged slice length differs from array length",
                )

            missing = ak._v2.index.Index64(slicecontent._index)
            numvalid = ak._v2.index.Index64.empty(1, self._nplike)
            assert (
                numvalid.nplike is self._nplike
                and slicestarts.nplike is self._nplike
                and slicestops.nplike is self._nplike
                and missing.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_jagged_numvalid",
                    numvalid.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    missing.dtype.type,
                ](
                    numvalid.data,
                    slicestarts.data,
                    slicestops.data,
                    slicestarts.length,
                    missing.data,
                    missing.length,
                ),
                slicer=ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
            )

            nextcarry = ak._v2.index.Index64.empty(numvalid[0], self._nplike)

            smalloffsets = ak._v2.index.Index64.empty(
                slicestarts.length + 1, self._nplike
            )
            largeoffsets = ak._v2.index.Index64.empty(
                slicestarts.length + 1, self._nplike
            )

            assert (
                nextcarry.nplike is self._nplike
                and smalloffsets.nplike is self._nplike
                and largeoffsets.nplike is self._nplike
                and slicestarts.nplike is self._nplike
                and slicestops.nplike is self._nplike
                and missing.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_jagged_shrink",
                    nextcarry.dtype.type,
                    smalloffsets.dtype.type,
                    largeoffsets.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    missing.dtype.type,
                ](
                    nextcarry.data,
                    smalloffsets.data,
                    largeoffsets.data,
                    slicestarts.data,
                    slicestops.data,
                    slicestarts.length,
                    missing.data,
                ),
                slicer=ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
            )

            if isinstance(
                slicecontent._content,
                ak._v2.contents.listoffsetarray.ListOffsetArray,
            ):

                # Generate ranges between starts and stops
                as_list_offset_array = self.toListOffsetArray64(True)
                nextcontent = as_list_offset_array._content._carry(nextcarry, True)
                next = ak._v2.contents.listoffsetarray.ListOffsetArray(
                    smalloffsets, nextcontent, None, self._parameters, self._nplike
                )
                out = next._getitem_next_jagged(
                    smalloffsets[:-1], smalloffsets[1:], slicecontent._content, tail
                )

            else:
                out = self._getitem_next_jagged(
                    smalloffsets[:-1], smalloffsets[1:], slicecontent._content, tail
                )

            if isinstance(out, ak._v2.contents.listoffsetarray.ListOffsetArray):
                content = out._content
                if largeoffsets.nplike.known_data:
                    missing_trim = missing[0 : largeoffsets[-1]]
                else:
                    missing_trim = missing
                indexedoptionarray = (
                    ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                        missing_trim, content, None, self._parameters, self._nplike
                    )
                )
                if isinstance(self._nplike, ak._v2._typetracer.TypeTracer):
                    indexedoptionarray = indexedoptionarray.typetracer
                return ak._v2.contents.listoffsetarray.ListOffsetArray(
                    largeoffsets,
                    indexedoptionarray.simplify_optiontype(),
                    None,
                    self._parameters,
                    self._nplike,
                )
            else:
                raise ak._v2._util.error(
                    AssertionError(
                        "expected ListOffsetArray from ListArray._getitem_next_jagged, got {}".format(
                            type(out).__name__
                        )
                    )
                )

        elif isinstance(slicecontent, ak._v2.contents.emptyarray.EmptyArray):
            return self

        else:
            raise ak._v2._util.error(
                AssertionError(
                    "expected Index/IndexedOptionArray/ListOffsetArray in ListArray._getitem_next_jagged, got {}".format(
                        type(slicecontent).__name__
                    )
                )
            )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, int):
            assert advanced is None
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            lenstarts = self._starts.length
            nextcarry = ak._v2.index.Index64.empty(lenstarts, self._nplike)
            assert (
                nextcarry.nplike is self._nplike
                and self._starts.nplike is self._nplike
                and self._stops.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_next_at",
                    nextcarry.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    nextcarry.data,
                    self._starts.data,
                    self._stops.data,
                    lenstarts,
                    head,
                ),
                slicer=head,
            )
            nextcontent = self._content._carry(nextcarry, True)
            return nextcontent._getitem_next(nexthead, nexttail, advanced)

        elif isinstance(head, slice):
            lenstarts = self._starts.length

            nexthead, nexttail = ak._v2._slicing.headtail(tail)

            start, stop, step = head.start, head.stop, head.step

            step = 1 if step is None else step
            start = ak._util.kSliceNone if start is None else start
            stop = ak._util.kSliceNone if stop is None else stop

            if self._nplike.known_shape:
                carrylength = ak._v2.index.Index64.empty(1, self._nplike)
                assert (
                    carrylength.nplike is self._nplike
                    and self._starts.nplike is self._nplike
                    and self._stops.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_getitem_next_range_carrylength",
                        carrylength.dtype.type,
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                    ](
                        carrylength.data,
                        self._starts.data,
                        self._stops.data,
                        lenstarts,
                        start,
                        stop,
                        step,
                    ),
                    slicer=head,
                )
                nextcarry = ak._v2.index.Index64.empty(carrylength[0], self._nplike)
            else:
                nextcarry = ak._v2.index.Index64.empty(
                    ak._v2._typetracer.UnknownLength, self._nplike
                )

            if self._starts.dtype == "int64":
                nextoffsets = ak._v2.index.Index64.empty(lenstarts + 1, self._nplike)
            elif self._starts.dtype == "int32":
                nextoffsets = ak._v2.index.Index32.empty(lenstarts + 1, self._nplike)
            elif self._starts.dtype == "uint32":
                nextoffsets = ak._v2.index.IndexU32.empty(lenstarts + 1, self._nplike)

            assert (
                nextoffsets.nplike is self._nplike
                and nextcarry.nplike is self._nplike
                and self._starts.nplike is self._nplike
                and self._stops.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_next_range",
                    nextoffsets.dtype.type,
                    nextcarry.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    nextoffsets.data,
                    nextcarry.data,
                    self._starts.data,
                    self._stops.data,
                    lenstarts,
                    start,
                    stop,
                    step,
                ),
                slicer=head,
            )

            nextcontent = self._content._carry(nextcarry, True)

            if advanced is None or advanced.length == 0:
                return ak._v2.contents.listoffsetarray.ListOffsetArray(
                    nextoffsets,
                    nextcontent._getitem_next(nexthead, nexttail, advanced),
                    self._identifier,
                    self._parameters,
                    self._nplike,
                )
            else:
                if self._nplike.known_shape:
                    total = ak._v2.index.Index64.empty(1, self._nplike)
                    assert (
                        total.nplike is self._nplike
                        and nextoffsets.nplike is self._nplike
                    )
                    self._handle_error(
                        self._nplike[
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
                    nextadvanced = ak._v2.index.Index64.empty(total[0], self._nplike)
                else:
                    nextadvanced = ak._v2.index.Index64.empty(
                        ak._v2._typetracer.UnknownLength, self._nplike
                    )
                advanced = advanced._to_nplike(self.nplike)
                assert (
                    nextadvanced.nplike is self._nplike
                    and advanced.nplike is self._nplike
                    and nextoffsets.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
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
            lenstarts = self._starts.length

            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            flathead = self._nplike.index_nplike.asarray(head.data.reshape(-1))
            regular_flathead = ak._v2.index.Index64(flathead, nplike=self.nplike)
            if advanced is None or advanced.length == 0:
                nextcarry = ak._v2.index.Index64.empty(
                    lenstarts * flathead.shape[0], self._nplike
                )
                nextadvanced = ak._v2.index.Index64.empty(
                    lenstarts * flathead.shape[0], self._nplike
                )
                assert (
                    nextcarry.nplike is self._nplike
                    and nextadvanced.nplike is self._nplike
                    and self._starts.nplike is self._nplike
                    and self._stops.nplike is self._nplike
                    and regular_flathead.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_getitem_next_array",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.data,
                        nextadvanced.data,
                        self._starts.data,
                        self._stops.data,
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
                    return ak._v2._slicing.getitem_next_array_wrap(
                        out, head.metadata.get("shape", (head.length,)), self.length
                    )
                else:
                    return out

            else:
                nextcarry = ak._v2.index.Index64.empty(lenstarts, self._nplike)
                nextadvanced = ak._v2.index.Index64.empty(lenstarts, self._nplike)
                advanced = advanced._to_nplike(self.nplike)
                assert (
                    nextcarry.nplike is self._nplike
                    and nextadvanced.nplike is self._nplike
                    and self._starts.nplike is self._nplike
                    and self._stops.nplike is self._nplike
                    and regular_flathead.nplike is self._nplike
                    and advanced.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_getitem_next_array_advanced",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                        regular_flathead.dtype.type,
                        advanced.dtype.type,
                    ](
                        nextcarry.data,
                        nextadvanced.data,
                        self._starts.data,
                        self._stops.data,
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

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            headlength = head.length
            head = head._to_nplike(self.nplike)
            if advanced is not None:
                raise ak._v2._util.indexerror(
                    self,
                    head,
                    "cannot mix jagged slice with NumPy-style advanced indexing",
                )
            length = self.length
            singleoffsets = ak._v2.index.Index64(head.offsets.data)
            multistarts = ak._v2.index.Index64.empty(head.length * length, self._nplike)
            multistops = ak._v2.index.Index64.empty(head.length * length, self._nplike)
            nextcarry = ak._v2.index.Index64.empty(head.length * length, self._nplike)

            assert (
                multistarts.nplike is self._nplike
                and multistops.nplike is self._nplike
                and singleoffsets.nplike is self._nplike
                and nextcarry.nplike is self._nplike
                and self._starts.nplike is self._nplike
                and self._stops.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_jagged_expand",
                    multistarts.dtype.type,
                    multistops.dtype.type,
                    singleoffsets.dtype.type,
                    nextcarry.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    multistarts.data,
                    multistops.data,
                    singleoffsets.data,
                    nextcarry.data,
                    self._starts.data,
                    self._stops.data,
                    head.length,
                    length,
                ),
                slicer=head,
            )
            carried = self._content._carry(nextcarry, True)
            down = carried._getitem_next_jagged(
                multistarts, multistops, head._content, tail
            )

            return ak._v2.contents.regulararray.RegularArray(
                down, headlength, 1, None, self._parameters, self._nplike
            )

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise ak._v2._util.error(AssertionError(repr(head)))

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            out = self.length
            if ak._v2._util.isint(out):
                return np.int64(out)
            else:
                return out
        elif posaxis == depth + 1:
            tonum = ak._v2.index.Index64.empty(self.length, self._nplike)
            assert (
                tonum.nplike is self._nplike
                and self._starts.nplike is self._nplike
                and self._stops.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_num",
                    tonum.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    tonum.data,
                    self._starts.data,
                    self._stops.data,
                    self.length,
                )
            )
            return ak._v2.contents.numpyarray.NumpyArray(
                tonum, None, None, self._nplike
            )
        else:
            return self.toListOffsetArray64(True).num(posaxis, depth)

    def _offsets_and_flattened(self, axis, depth):
        return self.toListOffsetArray64(True)._offsets_and_flattened(axis, depth)

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

        head, tail = self._merging_strategy(others)

        total_length = 0
        for array in head:
            total_length += array.length

        contents = []

        parameters = self._parameters
        for array in head:
            parameters = ak._v2._util.merge_parameters(
                parameters, array._parameters, True
            )

            if isinstance(
                array,
                (
                    ak._v2.contents.listarray.ListArray,
                    ak._v2.contents.listoffsetarray.ListOffsetArray,
                    ak._v2.contents.regulararray.RegularArray,
                ),
            ):
                contents.append(array.content)

            elif isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                pass
            else:
                raise ak._v2._util.error(
                    ValueError(
                        "cannot merge "
                        + type(self).__name__
                        + " with "
                        + type(array).__name__
                        + "."
                    )
                )

        tail_contents = contents[1:]
        nextcontent = contents[0].mergemany(tail_contents)

        nextstarts = ak._v2.index.Index64.empty(total_length, self._nplike)
        nextstops = ak._v2.index.Index64.empty(total_length, self._nplike)

        contentlength_so_far = 0
        length_so_far = 0

        for array in head:
            if isinstance(
                array,
                (
                    ak._v2.contents.listarray.ListArray,
                    ak._v2.contents.listoffsetarray.ListOffsetArray,
                ),
            ):
                array_starts = ak._v2.index.Index(array.starts)
                array_stops = ak._v2.index.Index(array.stops)

                assert (
                    nextstarts.nplike is self._nplike
                    and nextstops.nplike is self._nplike
                    and array_starts.nplike is self._nplike
                    and array_stops.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_fill",
                        nextstarts.dtype.type,
                        nextstops.dtype.type,
                        array_starts.dtype.type,
                        array_stops.dtype.type,
                    ](
                        nextstarts.data,
                        length_so_far,
                        nextstops.data,
                        length_so_far,
                        array_starts.data,
                        array_stops.data,
                        array.length,
                        contentlength_so_far,
                    )
                )
                contentlength_so_far += array.content.length
                length_so_far += array.length

            elif isinstance(array, ak._v2.contents.regulararray.RegularArray):
                listoffsetarray = array.toListOffsetArray64(True)

                array_starts = ak._v2.index.Index64(listoffsetarray.starts)
                array_stops = ak._v2.index.Index64(listoffsetarray.stops)

                assert (
                    nextstarts.nplike is self._nplike
                    and nextstops.nplike is self._nplike
                    and array_starts.nplike is self._nplike
                    and array_stops.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_fill",
                        nextstarts.dtype.type,
                        nextstops.dtype.type,
                        array_starts.dtype.type,
                        array_stops.dtype.type,
                    ](
                        nextstarts.data,
                        length_so_far,
                        nextstops.data,
                        length_so_far,
                        array_starts.data,
                        array_stops.data,
                        listoffsetarray.length,
                        contentlength_so_far,
                    )
                )
                contentlength_so_far += array.content.length
                length_so_far += listoffsetarray.length

            elif isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                pass

        next = ak._v2.contents.listarray.ListArray(
            nextstarts, nextstops, nextcontent, None, parameters, self._nplike
        )

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed.mergemany(tail[1:])

    def fill_none(self, value):
        return ListArray(
            self._starts,
            self._stops,
            self._content.fill_none(value),
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def _local_index(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._local_index_axis0()
        elif posaxis == depth + 1:
            offsets = self._compact_offsets64(True)
            if self._nplike.known_data:
                innerlength = offsets[offsets.length - 1]
            else:
                innerlength = ak._v2._typetracer.UnknownLength
            localindex = ak._v2.index.Index64.empty(innerlength, self._nplike)
            assert localindex.nplike is self._nplike and offsets.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_localindex",
                    localindex.dtype.type,
                    offsets.dtype.type,
                ](
                    localindex.data,
                    offsets.data,
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
            return ak._v2.contents.listarray.ListArray(
                self._starts,
                self._stops,
                self._content._local_index(posaxis, depth + 1),
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def numbers_to_type(self, name):
        return ak._v2.contents.listarray.ListArray(
            self._starts,
            self._stops,
            self._content.numbers_to_type(name),
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._starts.length == 0:
            return True

        return self.toListOffsetArray64(True)._is_unique(
            negaxis, starts, parents, outlength
        )

    def _unique(self, negaxis, starts, parents, outlength):
        if self._starts.length == 0:
            return self
        return self.toListOffsetArray64(True)._unique(
            negaxis, starts, parents, outlength
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
        next = self.toListOffsetArray64(True)
        out = next._argsort_next(
            negaxis,
            starts,
            shifts,
            parents,
            outlength,
            ascending,
            stable,
            kind,
            order,
        )
        return out

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        return self.toListOffsetArray64(True)._sort_next(
            negaxis,
            starts,
            parents,
            outlength,
            ascending,
            stable,
            kind,
            order,
        )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        return ListOffsetArray._combinations(
            self, n, replacement, recordlookup, parameters, axis, depth
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
        return self.toListOffsetArray64(True)._reduce_next(
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

    def _validity_error(self, path):
        if self._nplike.known_shape and self.stops.length < self.starts.length:
            return f'at {path} ("{type(self)}"): len(stops) < len(starts)'
        assert self.starts.nplike is self._nplike and self.stops.nplike is self._nplike
        error = self._nplike[
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
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                return ""
            else:
                return self._content.validity_error(path + ".content")

    def _nbytes_part(self):
        result = (
            self.starts._nbytes_part()
            + self.stops._nbytes_part()
            + self.content._nbytes_part()
        )
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def _pad_none(self, target, axis, depth, clip):
        if not clip:
            posaxis = self.axis_wrap_if_negative(axis)
            if posaxis == depth:
                return self.pad_none_axis0(target, clip)
            elif posaxis == depth + 1:
                min_ = ak._v2.index.Index64.empty(1, self._nplike)
                assert (
                    min_.nplike is self._nplike
                    and self._starts.nplike is self._nplike
                    and self._stops.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_min_range",
                        min_.dtype.type,
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                    ](
                        min_.data,
                        self._starts.data,
                        self._stops.data,
                        self._starts.length,
                    )
                )
                # TODO: Replace the kernel call with below code once typtracer supports '-'
                # min_ = self._nplike.min(self._stops.data - self._starts.data)
                if target < min_[0]:
                    return self
                else:
                    tolength = ak._v2.index.Index64.empty(1, self._nplike)
                    assert (
                        tolength.nplike is self._nplike
                        and self._starts.nplike is self._nplike
                        and self._stops.nplike is self._nplike
                    )
                    self._handle_error(
                        self._nplike[
                            "awkward_ListArray_rpad_and_clip_length_axis1",
                            tolength.dtype.type,
                            self._starts.dtype.type,
                            self._stops.dtype.type,
                        ](
                            tolength.data,
                            self._starts.data,
                            self._stops.data,
                            target,
                            self._starts.length,
                        )
                    )

                    index = ak._v2.index.Index64.empty(tolength[0], self._nplike)
                    starts_ = ak._v2.index.Index64.empty(
                        self._starts.length, self._nplike
                    )
                    stops_ = ak._v2.index.Index64.empty(
                        self._stops.length, self._nplike
                    )
                    assert (
                        index.nplike is self._nplike
                        and self._starts.nplike is self._nplike
                        and self._stops.nplike is self._nplike
                        and starts_.nplike is self._nplike
                        and stops_.nplike is self._nplike
                    )
                    self._handle_error(
                        self._nplike[
                            "awkward_ListArray_rpad_axis1",
                            index.dtype.type,
                            self._starts.dtype.type,
                            self._stops.dtype.type,
                            starts_.dtype.type,
                            stops_.dtype.type,
                        ](
                            index.data,
                            self._starts.data,
                            self._stops.data,
                            starts_.data,
                            stops_.data,
                            target,
                            self._starts.length,
                        )
                    )
                    next = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                        index,
                        self._content,
                        None,
                        None,
                        self._nplike,
                    )

                    return ak._v2.contents.listarray.ListArray(
                        starts_,
                        stops_,
                        next.simplify_optiontype(),
                        None,
                        self._parameters,
                        self._nplike,
                    )
            else:
                return ak._v2.contents.listarray.ListArray(
                    self._starts,
                    self._stops,
                    self._content._pad_none(target, posaxis, depth + 1, clip),
                    None,
                    self._parameters,
                    self._nplike,
                )
        else:
            return self.toListOffsetArray64(True)._pad_none(
                target, axis, depth, clip=True
            )

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        return self.toListOffsetArray64(False)._to_arrow(
            pyarrow, mask_node, validbytes, length, options
        )

    def _to_numpy(self, allow_missing):
        return ak._v2.operations.to_numpy(self.toRegularArray(), allow_missing)

    def _completely_flatten(self, nplike, options):
        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            return [ak._v2.operations.to_numpy(self)]
        else:
            next = self.toListOffsetArray64(False)
            flat = next.content[next.offsets[0] : next.offsets[-1]]
            return flat._completely_flatten(nplike, options)

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if (
            self._nplike.known_shape
            and self._nplike.known_data
            and self._starts.length != 0
        ):
            startsmin = self._starts.data.min()
            starts = ak._v2.index.Index(
                self._starts.data - startsmin, nplike=self._nplike
            )
            stops = ak._v2.index.Index(
                self._stops.data - startsmin, nplike=self._nplike
            )
            content = self._content[startsmin : self._stops.data.max()]
        else:
            starts, stops, content = self._starts, self._stops, self._content

        if options["return_array"]:

            def continuation():
                return ListArray(
                    starts,
                    stops,
                    content._recursively_apply(
                        action,
                        behavior,
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
        return self.toListOffsetArray64(True).packed()

    def _to_list(self, behavior, json_conversions):
        return ListOffsetArray._to_list(self, behavior, json_conversions)

    def _to_nplike(self, nplike):
        starts = self._starts._to_nplike(nplike)
        stops = self._stops._to_nplike(nplike)
        content = self._content._to_nplike(nplike)
        return ListArray(
            starts,
            stops,
            content,
            identifier=self._identifier,
            parameters=self._parameters,
            nplike=nplike,
        )

    def _layout_equal(self, other, index_dtype=True, numpyarray=True):
        return (
            self.starts.layout_equal(other.starts, index_dtype, numpyarray)
            and self.stops.layout_equal(other.stops, index_dtype, numpyarray)
            and self.content.layout_equal(other.content, index_dtype, numpyarray)
        )
