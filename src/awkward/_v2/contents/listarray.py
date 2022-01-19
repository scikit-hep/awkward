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

    def __init__(
        self, starts, stops, content, identifier=None, parameters=None, nplike=None
    ):
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
        if starts.length > stops.length:
            raise ValueError(
                "{0} len(starts) ({1}) must be <= len(stops) ({2})".format(
                    type(self).__name__, starts.length, stops.length
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
            ak._v2._typetracer.TypeTracer.instance(),
        )

    @property
    def length(self):
        return self._starts.length

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
            raise NestedIndexError(self, where)
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

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextstarts = self._starts[carry.data]
            nextstops = self._stops[: self._starts.length][carry.data]
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
            self._nplike,
        )

    def _compact_offsets64(self, start_at_zero):
        starts_len = self._starts.length
        out = ak._v2.index.Index64.empty(starts_len + 1, self._nplike)
        self._handle_error(
            self._nplike[
                "awkward_ListArray_compact_offsets",
                out.dtype.type,
                self._starts.dtype.type,
                self._stops.dtype.type,
            ](
                out.to(self._nplike),
                self._starts.to(self._nplike),
                self._stops.to(self._nplike),
                starts_len,
            )
        )
        return out

    def _broadcast_tooffsets64(self, offsets):
        return ListOffsetArray._broadcast_tooffsets64(self, offsets)

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        if slicestarts.length != self.length and self._nplike.known_shape:
            raise NestedIndexError(
                self,
                ak._v2.contents.ListArray(
                    slicestarts, slicestops, slicecontent, None, None, self._nplike
                ),
                "cannot fit jagged slice with length {0} into {1} of size {2}".format(
                    slicestarts.length, type(self).__name__, self.length
                ),
            )

        if isinstance(slicecontent, ak._v2.contents.listoffsetarray.ListOffsetArray):
            outoffsets = ak._v2.index.Index64.empty(
                slicestarts.length + 1, self._nplike
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
                    outoffsets.to(self._nplike),
                    slicestarts.to(self._nplike),
                    slicestops.to(self._nplike),
                    slicestarts.length,
                    self._starts.to(self._nplike),
                    self._stops.to(self._nplike),
                )
            )

            asListOffsetArray64 = self.toListOffsetArray64(True)
            next_content = asListOffsetArray64._content

            sliceoffsets = ak._v2.index.Index64(slicecontent._offsets)

            outcontent = next_content._getitem_next_jagged(
                sliceoffsets[:-1], sliceoffsets[1:], slicecontent._content, tail
            )

            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                outoffsets, outcontent, None, self._parameters, self._nplike
            )

        elif isinstance(slicecontent, ak._v2.contents.numpyarray.NumpyArray):
            carrylen = ak._v2.index.Index64.empty(1, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_jagged_carrylen",
                    carrylen.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                ](
                    carrylen.to(self._nplike),
                    slicestarts.to(self._nplike),
                    slicestops.to(self._nplike),
                    slicestarts.length,
                )
            )
            sliceindex = ak._v2.index.Index64(slicecontent._data)
            outoffsets = ak._v2.index.Index64.empty(
                slicestarts.length + 1, self._nplike
            )
            nextcarry = ak._v2.index.Index64.empty(carrylen[0], self._nplike)

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
                    outoffsets.to(self._nplike),
                    nextcarry.to(self._nplike),
                    slicestarts.to(self._nplike),
                    slicestops.to(self._nplike),
                    slicestarts.length,
                    sliceindex.to(self._nplike),
                    sliceindex.length,
                    self._starts.to(self._nplike),
                    self._stops.to(self._nplike),
                    self._content.length,
                )
            )

            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            outcontent = nextcontent._getitem_next(nexthead, nexttail, None)

            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                outoffsets, outcontent, None, None, self._nplike
            )

        elif isinstance(
            slicecontent, ak._v2.contents.indexedoptionarray.IndexedOptionArray
        ):
            if self._starts.length < slicestarts.length and self._nplike.known_shape:
                raise NestedIndexError(
                    self,
                    ak._v2.contents.ListArray(
                        slicestarts, slicestops, slicecontent, None, None, self._nplike
                    ),
                    "jagged slice length differs from array length",
                )

            missing = ak._v2.index.Index64(slicecontent._index)
            numvalid = ak._v2.index.Index64.empty(1, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_jagged_numvalid",
                    numvalid.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    missing.dtype.type,
                ](
                    numvalid.to(self._nplike),
                    slicestarts.to(self._nplike),
                    slicestops.to(self._nplike),
                    slicestarts.length,
                    missing.to(self._nplike),
                    missing.length,
                )
            )

            nextcarry = ak._v2.index.Index64.empty(numvalid[0], self._nplike)

            smalloffsets = ak._v2.index.Index64.empty(
                slicestarts.length + 1, self._nplike
            )
            largeoffsets = ak._v2.index.Index64.empty(
                slicestarts.length + 1, self._nplike
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
                    nextcarry.to(self._nplike),
                    smalloffsets.to(self._nplike),
                    largeoffsets.to(self._nplike),
                    slicestarts.to(self._nplike),
                    slicestops.to(self._nplike),
                    slicestarts.length,
                    missing.to(self._nplike),
                )
            )

            if isinstance(
                slicecontent._content,
                ak._v2.contents.listoffsetarray.ListOffsetArray,
            ):
                nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
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
        if head == ():
            return self

        elif isinstance(head, int):
            assert advanced is None
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            lenstarts = self._starts.length
            nextcarry = ak._v2.index.Index64.empty(lenstarts, self._nplike)
            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_next_at",
                    nextcarry.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    nextcarry.to(self._nplike),
                    self._starts.to(self._nplike),
                    self._stops.to(self._nplike),
                    lenstarts,
                    head,
                )
            )
            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
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
                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_getitem_next_range_carrylength",
                        carrylength.dtype.type,
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                    ](
                        carrylength.to(self._nplike),
                        self._starts.to(self._nplike),
                        self._stops.to(self._nplike),
                        lenstarts,
                        start,
                        stop,
                        step,
                    )
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

            self._handle_error(
                self._nplike[
                    "awkward_ListArray_getitem_next_range",
                    nextoffsets.dtype.type,
                    nextcarry.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    nextoffsets.to(self._nplike),
                    nextcarry.to(self._nplike),
                    self._starts.to(self._nplike),
                    self._stops.to(self._nplike),
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
                if self._nplike.known_shape:
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
                else:
                    nextadvanced = ak._v2.index.Index64.empty(
                        ak._v2._typetracer.UnknownLength, self._nplike
                    )

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
            lenstarts = self._starts.length

            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            flathead = self._nplike.asarray(head.data.reshape(-1))
            regular_flathead = ak._v2.index.Index64(flathead)
            if advanced is None or advanced.length == 0:
                nextcarry = ak._v2.index.Index64.empty(
                    lenstarts * flathead.shape[0], self._nplike
                )
                nextadvanced = ak._v2.index.Index64.empty(
                    lenstarts * flathead.shape[0], self._nplike
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
                        nextcarry.to(self._nplike),
                        nextadvanced.to(self._nplike),
                        self._starts.to(self._nplike),
                        self._stops.to(self._nplike),
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
                nextcarry = ak._v2.index.Index64.empty(lenstarts, self._nplike)
                nextadvanced = ak._v2.index.Index64.empty(lenstarts, self._nplike)
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
                        nextcarry.to(self._nplike),
                        nextadvanced.to(self._nplike),
                        self._starts.to(self._nplike),
                        self._stops.to(self._nplike),
                        regular_flathead.to(self._nplike),
                        advanced.to(self._nplike),
                        lenstarts,
                        regular_flathead.length,
                        self._content.length,
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
            length = self.length
            singleoffsets = ak._v2.index.Index64(head.offsets.data)
            multistarts = ak._v2.index.Index64.empty(head.length * length, self._nplike)
            multistops = ak._v2.index.Index64.empty(head.length * length, self._nplike)
            nextcarry = ak._v2.index.Index64.empty(head.length * length, self._nplike)

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
                    multistarts.to(self._nplike),
                    multistops.to(self._nplike),
                    singleoffsets.to(self._nplike),
                    nextcarry.to(self._nplike),
                    self._starts.to(self._nplike),
                    self._stops.to(self._nplike),
                    head.length,
                    length,
                ),
            )
            carried = self._content._carry(nextcarry, True, NestedIndexError)
            down = carried._getitem_next_jagged(
                multistarts, multistops, head._content, tail
            )

            return ak._v2.contents.regulararray.RegularArray(
                down, head.length, 1, None, self._parameters, self._nplike
            )

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
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    tonum.to(self._nplike),
                    self._starts.to(self._nplike),
                    self._stops.to(self._nplike),
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

                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_fill",
                        nextstarts.dtype.type,
                        nextstops.dtype.type,
                        array_starts.dtype.type,
                        array_stops.dtype.type,
                    ](
                        nextstarts.to(self._nplike),
                        length_so_far,
                        nextstops.to(self._nplike),
                        length_so_far,
                        array_starts.to(self._nplike),
                        array_stops.to(self._nplike),
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

                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_fill",
                        nextstarts.dtype.type,
                        nextstops.dtype.type,
                        array_starts.dtype.type,
                        array_stops.dtype.type,
                    ](
                        nextstarts.to(self._nplike),
                        length_so_far,
                        nextstops.to(self._nplike),
                        length_so_far,
                        array_starts.to(self._nplike),
                        array_stops.to(self._nplike),
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

    def fillna(self, value):
        return ListArray(
            self._starts,
            self._stops,
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
            return ak._v2.contents.listarray.ListArray(
                self._starts,
                self._stops,
                self._content._localindex(posaxis, depth + 1),
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
        out = self.toListOffsetArray64(True)
        return out._unique(negaxis, starts, parents, outlength)

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
        if self._starts.length == 0:
            return ak._v2.contents.NumpyArray(
                self._nplike.empty(0, np.int64), None, None, self._nplike
            )

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
        if self.stops.length < self.starts.length:
            return 'at {0} ("{1}"): len(stops) < len(starts)'.format(path, type(self))
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
                min_ = ak._v2.index.Index64.empty(1, self._nplike)
                self._handle_error(
                    self._nplike[
                        "awkward_ListArray_min_range",
                        min_.dtype.type,
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                    ](
                        min_.to(self._nplike),
                        self._starts.to(self._nplike),
                        self._stops.to(self._nplike),
                        self._starts.length,
                    )
                )
                # TODO: Replace the kernel call with below code once typtracer supports '-'
                # min_ = self._nplike.min(self._stops.data - self._starts.data)
                if target < min_[0]:
                    return self
                else:
                    tolength = ak._v2.index.Index64.empty(1, self._nplike)
                    self._handle_error(
                        self._nplike[
                            "awkward_ListArray_rpad_and_clip_length_axis1",
                            tolength.dtype.type,
                            self._starts.dtype.type,
                            self._stops.dtype.type,
                        ](
                            tolength.to(self._nplike),
                            self._starts.to(self._nplike),
                            self._stops.to(self._nplike),
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
                    self._handle_error(
                        self._nplike[
                            "awkward_ListArray_rpad_axis1",
                            index.dtype.type,
                            self._starts.dtype.type,
                            self._stops.dtype.type,
                            starts_.dtype.type,
                            stops_.dtype.type,
                        ](
                            index.to(self._nplike),
                            self._starts.to(self._nplike),
                            self._stops.to(self._nplike),
                            starts_.to(self._nplike),
                            stops_.to(self._nplike),
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
                    self._content._rpad(target, posaxis, depth + 1),
                    None,
                    self._parameters,
                    self._nplike,
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
        return self.toListOffsetArray64(True).packed()

    def _to_list(self, behavior):
        return ListOffsetArray._to_list(self, behavior)
