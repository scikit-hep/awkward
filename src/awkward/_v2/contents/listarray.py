# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import copy

import awkward as ak
from awkward._v2.index import Index
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.contents.listoffsetarray import ListOffsetArray
from awkward._v2.forms.listform import ListForm
from awkward._v2.forms.form import _parameters_equal

np = ak.nplike.NumpyMetadata.instance()


class ListArray(Content):
    is_ListType = True

    def __init__(self, starts, stops, content, identifier=None, parameters=None):
        if not isinstance(starts, Index) and starts.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        ):
            raise TypeError(
                "{0} 'starts' must be an Index with dtype in (int32, uint32, int64), "
                "not {1}".format(type(self).__name__, repr(starts))
            )
        if not (isinstance(stops, Index) and starts.dtype == stops.dtype):
            raise TypeError(
                "{0} 'stops' must be an Index with the same dtype as 'starts' ({1}), "
                "not {2}".format(type(self).__name__, repr(starts.dtype), repr(stops))
            )
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not len(starts) <= len(stops):
            raise ValueError(
                "{0} len(starts) ({1}) must be <= len(stops) ({2})".format(
                    type(self).__name__, len(starts), len(stops)
                )
            )

        self._starts = starts
        self._stops = stops
        self._content = content
        self._init(identifier, parameters)

    @property
    def starts(self):
        return self._starts

    @property
    def stops(self):
        return self._stops

    @property
    def content(self):
        return self._content

    @property
    def nplike(self):
        return self._starts.nplike

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
        container[key1] = ak._v2._util.little_endian(self._starts.to(nplike))
        container[key2] = ak._v2._util.little_endian(self._stops.to(nplike))
        self._content._to_buffers(form.content, getkey, container, nplike)

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return ListArray(
            ak._v2.index.Index(self._starts.to(tt)),
            ak._v2.index.Index(self._stops.to(tt)),
            self._content.typetracer,
            self._typetracer_identifier(),
            self._parameters,
        )

    def __len__(self):
        return len(self._starts)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ListArray len="]
        out.append(repr(str(len(self))))
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
        )

    def toListOffsetArray64(self, start_at_zero=False):
        offsets = self._compact_offsets64(start_at_zero)
        return self._broadcast_tooffsets64(offsets)

    def toRegularArray(self):
        offsets = self._compact_offsets64(True)
        return self._broadcast_tooffsets64(offsets).toRegularArray()

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)) and self.nplike.known_shape:
            raise NestedIndexError(self, where)
        start, stop = self._starts[where], self._stops[where]
        return self._content._getitem_range(slice(start, stop))

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        return ListArray(
            self._starts[start:stop],
            self._stops[start:stop],
            self._content,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return ListArray(
            self._starts,
            self._stops,
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return ListArray(
            self._starts,
            self._stops,
            self._content._getitem_fields(where, only_fields),
            self._fields_identifier(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextstarts = self._starts[carry.data]
            nextstops = self._stops[: len(self._starts)][carry.data]
        except IndexError as err:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, str(err))
            else:
                raise exception(str(err))

        return ListArray(
            nextstarts,
            nextstops,
            self._content,
            self._carry_identifier(carry, exception),
            self._parameters,
        )

    def _compact_offsets64(self, start_at_zero):
        starts_len = len(self._starts)
        out = ak._v2.index.Index64.empty(starts_len + 1, self.nplike)
        self._handle_error(
            self.nplike[
                "awkward_ListArray_compact_offsets",
                out.dtype.type,
                self._starts.dtype.type,
                self._stops.dtype.type,
            ](
                out.to(self.nplike),
                self._starts.to(self.nplike),
                self._stops.to(self.nplike),
                starts_len,
            )
        )
        if isinstance(out.data, ak._v2._typetracer.TypeTracerArray):
            out.data.fill_other = len(self._content)
        return out

    def _broadcast_tooffsets64(self, offsets):
        return ListOffsetArray._broadcast_tooffsets64(self, offsets)

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        nplike = self.nplike
        if len(slicestarts) != len(self) and nplike.known_shape:
            raise NestedIndexError(
                self,
                ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
                "cannot fit jagged slice with length {0} into {1} of size {2}".format(
                    len(slicestarts), type(self).__name__, len(self)
                ),
            )

        if isinstance(slicecontent, ak._v2.contents.listoffsetarray.ListOffsetArray):
            outoffsets = ak._v2.index.Index64.empty(len(slicestarts) + 1, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListArray_getitem_jagged_descend",
                    outoffsets.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    outoffsets.to(nplike),
                    slicestarts.to(nplike),
                    slicestops.to(nplike),
                    len(slicestarts),
                    self._starts.to(nplike),
                    self._stops.to(nplike),
                )
            )

            asListOffsetArray64 = self.toListOffsetArray64(True)
            next_content = asListOffsetArray64._content

            sliceoffsets = ak._v2.index.Index64(slicecontent._offsets)

            outcontent = next_content._getitem_next_jagged(
                sliceoffsets[:-1], sliceoffsets[1:], slicecontent._content, tail
            )

            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                outoffsets, outcontent, None, self._parameters
            )

        elif isinstance(slicecontent, ak._v2.contents.numpyarray.NumpyArray):
            carrylen = ak._v2.index.Index64.empty(1, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListArray_getitem_jagged_carrylen",
                    carrylen.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                ](
                    carrylen.to(nplike),
                    slicestarts.to(nplike),
                    slicestops.to(nplike),
                    len(slicestarts),
                )
            )
            sliceindex = ak._v2.index.Index64(slicecontent._data)
            outoffsets = ak._v2.index.Index64.zeros(len(slicestarts) + 1, nplike)
            nextcarry = ak._v2.index.Index64.zeros(carrylen[0], nplike)

            self._handle_error(
                nplike[
                    "awkward_ListArray_getitem_jagged_apply",
                    outoffsets.dtype.type,
                    nextcarry.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    sliceindex.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    outoffsets.to(nplike),
                    nextcarry.to(nplike),
                    slicestarts.to(nplike),
                    slicestops.to(nplike),
                    len(slicestarts),
                    sliceindex.to(nplike),
                    len(sliceindex),
                    self._starts.to(nplike),
                    self._stops.to(nplike),
                    len(self._content),
                )
            )

            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            outcontent = nextcontent._getitem_next(nexthead, nexttail, None)

            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                outoffsets, outcontent
            )

        elif isinstance(
            slicecontent, ak._v2.contents.indexedoptionarray.IndexedOptionArray
        ):
            if len(self._starts) < len(slicestarts) and nplike.known_shape:
                raise NestedIndexError(
                    self,
                    ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
                    "jagged slice length differs from array length",
                )

            missing = ak._v2.index.Index64(slicecontent._index)
            numvalid = ak._v2.index.Index64.empty(1, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListArray_getitem_jagged_numvalid",
                    numvalid.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    missing.dtype.type,
                ](
                    numvalid.to(nplike),
                    slicestarts.to(nplike),
                    slicestops.to(nplike),
                    len(slicestarts),
                    missing.to(nplike),
                    len(missing),
                )
            )

            nextcarry = ak._v2.index.Index64.empty(numvalid[0], nplike)
            if isinstance(nextcarry.data, ak._v2._typetracer.TypeTracerArray):
                nextcarry.data.shape = ak._v2._typetracer.Interval(
                    0, len(slicecontent._index)
                )

            smalloffsets = ak._v2.index.Index64.empty(len(slicestarts) + 1, nplike)
            largeoffsets = ak._v2.index.Index64.empty(len(slicestarts) + 1, nplike)

            self._handle_error(
                nplike[
                    "awkward_ListArray_getitem_jagged_shrink",
                    nextcarry.dtype.type,
                    smalloffsets.dtype.type,
                    largeoffsets.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    missing.dtype.type,
                ](
                    nextcarry.to(nplike),
                    smalloffsets.to(nplike),
                    largeoffsets.to(nplike),
                    slicestarts.to(nplike),
                    slicestops.to(nplike),
                    len(slicestarts),
                    missing.to(nplike),
                )
            )

            if isinstance(
                slicecontent._content,
                ak._v2.contents.listoffsetarray.ListOffsetArray,
            ):
                nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
                next = ak._v2.contents.listoffsetarray.ListOffsetArray(
                    smalloffsets, nextcontent, None, self._parameters
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
                missing_trim = ak._v2.index.Index64(missing[0 : largeoffsets[-1]])
                indexedoptionarray = (
                    ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                        missing_trim, content, None, self._parameters
                    )
                )
                if isinstance(self.nplike, ak._v2._typetracer.TypeTracer):
                    indexedoptionarray = indexedoptionarray.typetracer
                return ak._v2.contents.listoffsetarray.ListOffsetArray(
                    largeoffsets,
                    indexedoptionarray.simplify_optiontype(),
                    None,
                    self._parameters,
                )
            else:
                raise AssertionError(
                    "expected ListOffsetArray from ListArray._getitem_next_jagged, got {0}".format(
                        type(out).__name__
                    )
                )

        elif isinstance(slicecontent, ak._v2.contents.emptyarray.EmptyArray):
            return self

        else:
            raise AssertionError(
                "expected Index/IndexedOptionArray/ListOffsetArray in ListArray._getitem_next_jagged, got {0}".format(
                    type(slicecontent).__name__
                )
            )

    def _getitem_next(self, head, tail, advanced):
        nplike = self.nplike

        if head == ():
            return self

        elif isinstance(head, int):
            assert advanced is None
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            lenstarts = len(self._starts)
            nextcarry = ak._v2.index.Index64.empty(lenstarts, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListArray_getitem_next_at",
                    nextcarry.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    nextcarry.to(nplike),
                    self._starts.to(nplike),
                    self._stops.to(nplike),
                    lenstarts,
                    head,
                )
            )
            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
            return nextcontent._getitem_next(nexthead, nexttail, advanced)

        elif isinstance(head, slice):
            lenstarts = len(self._starts)

            nexthead, nexttail = ak._v2._slicing.headtail(tail)

            start, stop, step = head.start, head.stop, head.step

            step = 1 if step is None else step
            start = ak._util.kSliceNone if start is None else start
            stop = ak._util.kSliceNone if stop is None else stop

            carrylength = ak._v2.index.Index64.empty(1, nplike)
            self._handle_error(
                nplike[
                    "awkward_ListArray_getitem_next_range_carrylength",
                    carrylength.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    carrylength.to(nplike),
                    self._starts.to(nplike),
                    self._stops.to(nplike),
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
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    nextoffsets.to(nplike),
                    nextcarry.to(nplike),
                    self._starts.to(nplike),
                    self._stops.to(nplike),
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
            lenstarts = len(self._starts)

            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            flathead = nplike.asarray(head.data.reshape(-1))
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
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.to(nplike),
                        nextadvanced.to(nplike),
                        self._starts.to(nplike),
                        self._stops.to(nplike),
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
                nextcarry = ak._v2.index.Index64.empty(lenstarts, nplike)
                nextadvanced = ak._v2.index.Index64.empty(lenstarts, nplike)
                self._handle_error(
                    nplike[
                        "awkward_ListArray_getitem_next_array_advanced",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                        regular_flathead.dtype.type,
                        advanced.dtype.type,
                    ](
                        nextcarry.to(nplike),
                        nextadvanced.to(nplike),
                        self._starts.to(nplike),
                        self._stops.to(nplike),
                        regular_flathead.to(nplike),
                        advanced.to(nplike),
                        lenstarts,
                        len(regular_flathead),
                        len(self._content),
                    )
                )
                nextcontent = self._content._carry(nextcarry, True, NestedIndexError)

                return nextcontent._getitem_next(nexthead, nexttail, nextadvanced)

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            if advanced is not None:
                raise NestedIndexError(
                    self,
                    head,
                    "cannot mix jagged slice with NumPy-style advanced indexing",
                )
            length = len(self)
            singleoffsets = ak._v2.index.Index64(head.offsets.data)
            multistarts = ak._v2.index.Index64.empty(len(head) * length, nplike)
            multistops = ak._v2.index.Index64.empty(len(head) * length, nplike)
            nextcarry = ak._v2.index.Index64.empty(len(head) * length, nplike)

            self._handle_error(
                nplike[
                    "awkward_ListArray_getitem_jagged_expand",
                    multistarts.dtype.type,
                    multistops.dtype.type,
                    singleoffsets.dtype.type,
                    nextcarry.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    multistarts.to(nplike),
                    multistops.to(nplike),
                    singleoffsets.to(nplike),
                    nextcarry.to(nplike),
                    self._starts.to(nplike),
                    self._stops.to(nplike),
                    len(head),
                    length,
                ),
            )
            carried = self._content._carry(nextcarry, True, NestedIndexError)
            down = carried._getitem_next_jagged(
                multistarts, multistops, head._content, tail
            )

            return ak._v2.contents.regulararray.RegularArray(
                down, len(head), 1, None, self._parameters
            )

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            out = ak._v2.index.Index64.empty(1, self.nplike)
            out[0] = len(self)
            return ak._v2.contents.numpyarray.NumpyArray(out)[0]
        elif posaxis == depth + 1:
            tonum = ak._v2.index.Index64.empty(len(self), self.nplike)
            self._handle_error(
                self.nplike[
                    "awkward_ListArray_num",
                    tonum.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    tonum.to(self.nplike),
                    self._starts.to(self.nplike),
                    self._stops.to(self.nplike),
                    len(self),
                )
            )
            return ak._v2.contents.numpyarray.NumpyArray(tonum)
        else:
            return self.toListOffsetArray64(True).num(posaxis, depth)

    def _offsets_and_flattened(self, axis, depth):
        return self.toListOffsetArray64(True)._offsets_and_flattened(axis, depth)

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
            self.mergeable(other.content, mergebool)

        if isinstance(
            other,
            (
                ak._v2.contents.regulararray.RegularArray,
                ak._v2.contents.listarray.ListArray,
                ak._v2.contents.listoffsetarray.ListOffsetArray,
            ),
        ):
            self._content.mergeable(other.content, mergebool)

        else:
            return False

    def mergemany(self, others):
        if len(others) == 0:
            return self

        head, tail = self._merging_strategy(others)

        total_length = 0
        for array in head:
            total_length += len(array)

        contents = []

        for array in head:
            parameters = ak._v2._util.merge_parameters(
                self._parameters, array._parameters
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
                raise ValueError(
                    "cannot merge "
                    + type(self).__name__
                    + " with "
                    + type(array).__name__
                    + "."
                )

        tail_contents = contents[1:]
        nextcontent = contents[0].mergemany(tail_contents)

        nextstarts = ak._v2.index.Index64.empty(total_length, self.nplike)
        nextstops = ak._v2.index.Index64.empty(total_length, self.nplike)

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

                self._handle_error(
                    self.nplike[
                        "awkward_ListArray_fill",
                        nextstarts.dtype.type,
                        nextstops.dtype.type,
                        array_starts.dtype.type,
                        array_stops.dtype.type,
                    ](
                        nextstarts.to(self.nplike),
                        length_so_far,
                        nextstops.to(self.nplike),
                        length_so_far,
                        array_starts.to(self.nplike),
                        array_stops.to(self.nplike),
                        len(array),
                        contentlength_so_far,
                    )
                )
                contentlength_so_far += len(array.content)
                length_so_far += len(array)

            elif isinstance(array, ak._v2.contents.regulararray.RegularArray):
                listoffsetarray = array.toListOffsetArray64(True)

                array_starts = ak._v2.index.Index64(listoffsetarray.starts)
                array_stops = ak._v2.index.Index64(listoffsetarray.stops)

                self._handle_error(
                    self.nplike[
                        "awkward_ListArray_fill",
                        nextstarts.dtype.type,
                        nextstops.dtype.type,
                        array_starts.dtype.type,
                        array_stops.dtype.type,
                    ](
                        nextstarts.to(self.nplike),
                        length_so_far,
                        nextstops.to(self.nplike),
                        length_so_far,
                        array_starts.to(self.nplike),
                        array_stops.to(self.nplike),
                        len(listoffsetarray),
                        contentlength_so_far,
                    )
                )
                contentlength_so_far += len(array.content)
                length_so_far += len(listoffsetarray)

            elif isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                pass

        next = ak._v2.contents.listarray.ListArray(
            nextstarts, nextstops, nextcontent, None, parameters
        )

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed.mergemany(tail[1:])

    def fillna(self, value):
        return ListArray(
            self._starts,
            self._stops,
            self._content.fillna(value),
            self._identifier,
            self._parameters,
        )

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
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
            return ak._v2.contents.listarray.ListArray(
                self._starts,
                self._stops,
                self._content._localindex(posaxis, depth + 1),
                self._identifier,
                self._parameters,
            )

    def numbers_to_type(self, name):
        return ak._v2.contents.listarray.ListArray(
            self._starts,
            self._stops,
            self._content.numbers_to_type(name),
            self._identifier,
            self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if len(self._starts) == 0:
            return True

        return self.toListOffsetArray64(True)._is_unique(
            negaxis, starts, parents, outlength
        )

    def _unique(self, negaxis, starts, parents, outlength):
        if len(self._starts) == 0:
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
        if len(self._starts) == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))

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
        )

    def _validityerror(self, path):
        if len(self.stops) < len(self.starts):
            return 'at {0} ("{1}"): len(stops) < len(starts)'.format(path, type(self))
        error = self.nplike[
            "awkward_ListArray_validity", self.starts.dtype.type, self.stops.dtype.type
        ](
            self.starts.to(self.nplike),
            self.stops.to(self.nplike),
            len(self.starts),
            len(self._content),
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
                return self._content.validityerror(path + ".content")

    def _nbytes_part(self):
        result = (
            self.starts._nbytes_part()
            + self.stops._nbytes_part()
            + self.content._nbytes_part()
        )
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def _rpad(self, target, axis, depth, clip):
        if not clip:
            posaxis = self.axis_wrap_if_negative(axis)
            if posaxis == depth:
                return self.rpad_axis0(target, clip)
            elif posaxis == depth + 1:
                min_ = ak._v2.index.Index64.empty(1, self.nplike)
                self._handle_error(
                    self.nplike[
                        "awkward_ListArray_min_range",
                        min_.dtype.type,
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                    ](
                        min_.to(self.nplike),
                        self._starts.to(self.nplike),
                        self._stops.to(self.nplike),
                        len(self._starts),
                    )
                )
                # TODO: Replace the kernel call with below code once typtracer supports '-'
                # min_ = self.nplike.min(self._stops.data - self._starts.data)
                if target < min_[0]:
                    return self
                else:
                    tolength = ak._v2.index.Index64.zeros(1, self.nplike)
                    self._handle_error(
                        self.nplike[
                            "awkward_ListArray_rpad_and_clip_length_axis1",
                            tolength.dtype.type,
                            self._starts.dtype.type,
                            self._stops.dtype.type,
                        ](
                            tolength.to(self.nplike),
                            self._starts.to(self.nplike),
                            self._stops.to(self.nplike),
                            target,
                            len(self._starts),
                        )
                    )

                    index = ak._v2.index.Index64.empty(tolength[0], self.nplike)
                    starts_ = ak._v2.index.Index64.empty(len(self._starts), self.nplike)
                    stops_ = ak._v2.index.Index64.empty(len(self._stops), self.nplike)
                    self._handle_error(
                        self.nplike[
                            "awkward_ListArray_rpad_axis1",
                            index.dtype.type,
                            self._starts.dtype.type,
                            self._stops.dtype.type,
                            starts_.dtype.type,
                            stops_.dtype.type,
                        ](
                            index.to(self.nplike),
                            self._starts.to(self.nplike),
                            self._stops.to(self.nplike),
                            starts_.to(self.nplike),
                            stops_.to(self.nplike),
                            target,
                            len(self._starts),
                        )
                    )
                    next = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                        index,
                        self._content,
                        None,
                        None,
                    )

                    return ak._v2.contents.listarray.ListArray(
                        starts_,
                        stops_,
                        next.simplify_optiontype(),
                        None,
                        self._parameters,
                    )
            else:
                return ak._v2.contents.listarray.ListArray(
                    self._starts,
                    self._stops,
                    self._content._rpad(target, posaxis, depth + 1),
                    None,
                    self._parameters,
                )
        else:
            return self.toListOffsetArray64(True)._rpad(target, axis, depth, clip=True)

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        return self.toListOffsetArray64(False)._to_arrow(
            pyarrow, mask_node, validbytes, length, options
        )

    def _to_numpy(self, allow_missing):
        return ak._v2.operations.convert.to_numpy(self.toRegularArray(), allow_missing)

    def _completely_flatten(self, nplike, options):
        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            return [ak._v2.operations.convert.to_numpy(self)]
        else:
            next = self.toListOffsetArray64(False)
            flat = next.content[next.offsets[0] : next.offsets[-1]]
            return flat._completely_flatten(nplike, options)

    def _recursively_apply(
        self, action, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                return ListArray(
                    self._starts,
                    self._stops,
                    self._content._recursively_apply(
                        action,
                        depth + 1,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    ),
                    self._identifier,
                    self._parameters if options["keep_parameters"] else None,
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
        return self.toListOffsetArray64(True).packed()

    def _to_list(self, behavior):
        return ListOffsetArray._to_list(self, behavior)
