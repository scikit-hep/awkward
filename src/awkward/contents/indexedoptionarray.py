# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy

import awkward as ak
from awkward._util import unset
from awkward.contents.content import Content
from awkward.forms.indexedoptionform import IndexedOptionForm
from awkward.index import Index
from awkward.typing import Final, Self

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


class IndexedOptionArray(Content):
    is_option = True
    is_indexed = True

    def __init__(self, index, content, *, parameters=None):
        if not (
            isinstance(index, Index)
            and index.dtype
            in (
                np.dtype(np.int32),
                np.dtype(np.int64),
            )
        ):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'index' must be an Index with dtype in (int32, uint32, int64), "
                    "not {}".format(type(self).__name__, repr(index))
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
        is_cat = parameters is not None and parameters.get("__array__") == "categorical"
        if (content.is_union and not is_cat) or content.is_indexed or content.is_option:
            raise ak._errors.wrap_error(
                TypeError(
                    "{0} cannot contain a union-type (unless categorical), option-type, or indexed 'content' ({1}); try {0}.simplified instead".format(
                        type(self).__name__, type(content).__name__
                    )
                )
            )

        assert index.nplike is content.backend.index_nplike

        self._index = index
        self._content = content
        self._init(parameters, content.backend)

    @property
    def index(self):
        return self._index

    @property
    def content(self):
        return self._content

    form_cls: Final = IndexedOptionForm

    def copy(self, index=unset, content=unset, *, parameters=unset):
        return IndexedOptionArray(
            self._index if index is unset else index,
            self._content if content is unset else content,
            parameters=self._parameters if parameters is unset else parameters,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            index=copy.deepcopy(self._index, memo),
            content=copy.deepcopy(self._content, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(cls, index, content, *, parameters=None):
        is_cat = parameters is not None and parameters.get("__array__") == "categorical"

        if content.is_union and not is_cat:
            return content._union_of_optionarrays(index, parameters)

        elif content.is_indexed or content.is_option:
            backend = content.backend
            if content.is_indexed:
                inner = content.index
            else:
                inner = content.to_IndexedOptionArray64().index
            result = ak.index.Index64.empty(index.length, nplike=backend.index_nplike)

            Content._selfless_handle_error(
                backend[
                    "awkward_IndexedArray_simplify",
                    result.dtype.type,
                    index.dtype.type,
                    inner.dtype.type,
                ](
                    result.data,
                    index.data,
                    index.length,
                    inner.data,
                    inner.length,
                )
            )
            return IndexedOptionArray(
                result,
                content.content,
                parameters=ak._util.merge_parameters(content._parameters, parameters),
            )

        else:
            return cls(index, content, parameters=parameters)

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.form_cls(
            self._index.form,
            self._content._form_with_key(getkey),
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, backend):
        assert isinstance(form, self.form_cls)
        key = getkey(self, form, "index")
        container[key] = ak._util.little_endian(self._index.raw(backend.index_nplike))
        self._content._to_buffers(form.content, getkey, container, backend)

    def _to_typetracer(self, forget_length: bool) -> Self:
        index = self._index.to_nplike(ak._typetracer.TypeTracer.instance())
        return IndexedOptionArray(
            index.forget_length() if forget_length else index,
            self._content._to_typetracer(False),
            parameters=self._parameters,
        )

    def _touch_data(self, recursive):
        if not self._backend.index_nplike.known_data:
            self._index.data.touch_data()
        if recursive:
            self._content._touch_data(recursive)

    def _touch_shape(self, recursive):
        if not self._backend.index_nplike.known_shape:
            self._index.data.touch_shape()
        if recursive:
            self._content._touch_shape(recursive)

    @property
    def length(self):
        return self._index.length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<IndexedOptionArray len="]
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._index._repr(indent + "    ", "<index>", "</index>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</IndexedOptionArray>")
        out.append(post)
        return "".join(out)

    def to_IndexedOptionArray64(self):
        if self._index.dtype == np.dtype(np.int64):
            return self
        else:
            return IndexedOptionArray(
                self._index.astype(np.int64), self._content, parameters=self._parameters
            )

    def to_ByteMaskedArray(self, valid_when):
        mask = ak.index.Index8(self.mask_as_bool(valid_when))

        carry = self._index.data
        too_negative = carry < -1
        if self._backend.nplike.any(too_negative, prefer=False):
            carry = carry.copy()
            carry[too_negative] = -1
        carry = ak.index.Index(carry)

        if self._content.length == 0:
            content = self._content.form.length_one_array(
                backend=self._backend, highlevel=False
            )._carry(carry, False)
        else:
            content = self._content._carry(carry, False)

        return ak.contents.ByteMaskedArray(
            mask, content, valid_when, parameters=self._parameters
        )

    def to_BitMaskedArray(self, valid_when, lsb_order):
        return self.to_ByteMaskedArray(valid_when).to_BitMaskedArray(
            valid_when, lsb_order
        )

    def mask_as_bool(self, valid_when=True):
        if valid_when:
            return self._index.raw(self._backend.index_nplike) >= 0
        else:
            return self._index.raw(self._backend.index_nplike) < 0

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if not self._backend.nplike.known_data:
            self._touch_data(recursive=False)
            return ak._typetracer.MaybeNone(self._content._getitem_at(where))

        if where < 0:
            where += self.length
        if self._backend.nplike.known_shape and not 0 <= where < self.length:
            raise ak._errors.index_error(self, where)
        if self._index[where] < 0:
            return None
        else:
            return self._content._getitem_at(self._index[where])

    def _getitem_range(self, where):
        if not self._backend.nplike.known_shape:
            self._touch_shape(recursive=False)
            return self

        start, stop, step = where.indices(self.length)
        assert step == 1
        return IndexedOptionArray(
            self._index[start:stop], self._content, parameters=self._parameters
        )

    def _getitem_field(self, where, only_fields=()):
        return IndexedOptionArray.simplified(
            self._index,
            self._content._getitem_field(where, only_fields),
            parameters=None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return IndexedOptionArray.simplified(
            self._index,
            self._content._getitem_fields(where, only_fields),
            parameters=None,
        )

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak.index.Index)

        try:
            nextindex = self._index[carry.data]
        except IndexError as err:
            raise ak._errors.index_error(self, carry.data, str(err)) from err

        return IndexedOptionArray(nextindex, self._content, parameters=self._parameters)

    def _nextcarry_outindex(self, backend):
        numnull = ak.index.Index64.empty(1, nplike=backend.index_nplike)

        assert (
            numnull.nplike is self._backend.index_nplike
            and self._index.nplike is self._backend.index_nplike
        )
        self._handle_error(
            backend[
                "awkward_IndexedArray_numnull",
                numnull.dtype.type,
                self._index.dtype.type,
            ](
                numnull.data,
                self._index.data,
                self._index.length,
            )
        )
        nextcarry = ak.index.Index64.empty(
            self._index.length - numnull[0], backend.index_nplike
        )
        outindex = ak.index.Index.empty(
            self._index.length,
            backend.index_nplike,
            dtype=self._index.dtype,
        )

        assert (
            nextcarry.nplike is self._backend.index_nplike
            and outindex.nplike is self._backend.index_nplike
            and self._index.nplike is self._backend.index_nplike
        )
        self._handle_error(
            backend[
                "awkward_IndexedArray_getitem_nextcarry_outindex",
                nextcarry.dtype.type,
                outindex.dtype.type,
                self._index.dtype.type,
            ](
                nextcarry.data,
                outindex.data,
                self._index.data,
                self._index.length,
                self._content.length,
            )
        )

        return numnull[0], nextcarry, outindex

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        slicestarts = slicestarts.to_nplike(self._backend.index_nplike)
        slicestops = slicestops.to_nplike(self._backend.index_nplike)

        if self._backend.nplike.known_shape and slicestarts.length != self.length:
            raise ak._errors.index_error(
                self,
                ak.contents.ListArray(
                    slicestarts, slicestops, slicecontent, parameters=None
                ),
                "cannot fit jagged slice with length {} into {} of size {}".format(
                    slicestarts.length, type(self).__name__, self.length
                ),
            )

        numnull, nextcarry, outindex = self._nextcarry_outindex(self._backend)

        reducedstarts = ak.index.Index64.empty(
            self.length - numnull, nplike=self._backend.index_nplike
        )
        reducedstops = ak.index.Index64.empty(
            self.length - numnull, nplike=self._backend.index_nplike
        )
        assert (
            outindex.nplike is self._backend.index_nplike
            and slicestarts.nplike is self._backend.index_nplike
            and slicestops.nplike is self._backend.index_nplike
            and reducedstarts.nplike is self._backend.index_nplike
            and reducedstops.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_MaskedArray_getitem_next_jagged_project",
                outindex.dtype.type,
                slicestarts.dtype.type,
                slicestops.dtype.type,
                reducedstarts.dtype.type,
                reducedstops.dtype.type,
            ](
                outindex.data,
                slicestarts.data,
                slicestops.data,
                reducedstarts.data,
                reducedstops.data,
                self.length,
            ),
            slicer=ak.contents.ListArray(slicestarts, slicestops, slicecontent),
        )
        next = self._content._carry(nextcarry, True)
        out = next._getitem_next_jagged(reducedstarts, reducedstops, slicecontent, tail)

        return ak.contents.IndexedOptionArray.simplified(
            outindex, out, parameters=self._parameters
        )

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        return self._getitem_next_jagged_generic(
            slicestarts, slicestops, slicecontent, tail
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(
            head, (int, slice, ak.index.Index64, ak.contents.ListOffsetArray)
        ):
            nexthead, nexttail = ak._slicing.headtail(tail)

            numnull, nextcarry, outindex = self._nextcarry_outindex(self._backend)

            next = self._content._carry(nextcarry, True)
            out = next._getitem_next(head, tail, advanced)
            return IndexedOptionArray.simplified(
                outindex, out, parameters=self._parameters
            )

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list) and isinstance(head[0], str):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise ak._errors.wrap_error(AssertionError(repr(head)))

    def project(self, mask=None):
        if mask is not None:
            if self._backend.nplike.known_shape and self._index.length != mask.length:
                raise ak._errors.wrap_error(
                    ValueError(
                        "mask length ({}) is not equal to {} length ({})".format(
                            mask.length(), type(self).__name__, self._index.length
                        )
                    )
                )
            nextindex = ak.index.Index64.empty(
                self._index.length, self._backend.index_nplike
            )
            assert (
                nextindex.nplike is self._backend.index_nplike
                and mask.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_IndexedArray_overlay_mask",
                    nextindex.dtype.type,
                    mask.dtype.type,
                    self._index.dtype.type,
                ](
                    nextindex.data,
                    mask.data,
                    self._index.data,
                    self._index.length,
                )
            )
            next = ak.contents.IndexedOptionArray(
                nextindex, self._content, parameters=self._parameters
            )
            return next.project()
        else:
            numnull = ak.index.Index64.empty(1, self._backend.index_nplike)

            assert (
                numnull.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_IndexedArray_numnull",
                    numnull.dtype.type,
                    self._index.dtype.type,
                ](
                    numnull.data,
                    self._index.data,
                    self._index.length,
                )
            )

            nextcarry = ak.index.Index64.empty(
                self.length - numnull[0], self._backend.index_nplike
            )

            assert (
                nextcarry.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_IndexedArray_flatten_nextcarry",
                    nextcarry.dtype.type,
                    self._index.dtype.type,
                ](
                    nextcarry.data,
                    self._index.data,
                    self._index.length,
                    self._content.length,
                )
            )

            return self._content._carry(nextcarry, False)

    def _offsets_and_flattened(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            raise ak._errors.wrap_error(np.AxisError("axis=0 not allowed for flatten"))
        else:
            numnull, nextcarry, outindex = self._nextcarry_outindex(self._backend)
            next = self._content._carry(nextcarry, False)

            offsets, flattened = next._offsets_and_flattened(axis, depth)

            if offsets.length == 0:
                return (
                    offsets,
                    ak.contents.IndexedOptionArray(
                        outindex, flattened, parameters=self._parameters
                    ),
                )

            else:
                outoffsets = ak.index.Index64.empty(
                    offsets.length + numnull,
                    self._backend.index_nplike,
                    dtype=np.int64,
                )

                assert (
                    outoffsets.nplike is self._backend.index_nplike
                    and outindex.nplike is self._backend.index_nplike
                    and offsets.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_IndexedArray_flatten_none2empty",
                        outoffsets.dtype.type,
                        outindex.dtype.type,
                        offsets.dtype.type,
                    ](
                        outoffsets.data,
                        outindex.data,
                        outindex.length,
                        offsets.data,
                        offsets.length,
                    )
                )
                return (outoffsets, flattened)

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
            return self._content._mergeable(other.content, mergebool)

        else:
            return self._content._mergeable(other, mergebool)

    def _merging_strategy(self, others):
        if len(others) == 0:
            raise ak._errors.wrap_error(
                ValueError(
                    "to merge this array with 'others', at least one other must be provided"
                )
            )

        head = [self]
        tail = []

        i = 0
        while i < len(others):
            other = others[i]
            if isinstance(other, ak.contents.UnionArray):
                break
            else:
                head.append(other)
            i = i + 1

        while i < len(others):
            tail.append(others[i])
            i = i + 1

        if any(
            isinstance(x.backend.nplike, ak._typetracer.TypeTracer) for x in head + tail
        ):
            head = [
                x
                if isinstance(x.backend.nplike, ak._typetracer.TypeTracer)
                else x.to_typetracer()
                for x in head
            ]
            tail = [
                x
                if isinstance(x.backend.nplike, ak._typetracer.TypeTracer)
                else x.to_typetracer()
                for x in tail
            ]

        return (head, tail)

    def _reverse_merge(self, other):
        theirlength = other.length
        mylength = self.length
        index = ak.index.Index64.empty(
            (theirlength + mylength), self._backend.index_nplike
        )

        content = other._mergemany([self._content])

        assert index.nplike is self._backend.index_nplike
        self._handle_error(
            self._backend["awkward_IndexedArray_fill_count", index.dtype.type](
                index.data,
                0,
                theirlength,
                0,
            )
        )
        reinterpreted_index = ak.index.Index(
            self._backend.index_nplike.asarray(self.index.data),
            nplike=self._backend.index_nplike,
        )

        assert (
            index.nplike is self._backend.index_nplike
            and reinterpreted_index.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_IndexedArray_fill",
                index.dtype.type,
                reinterpreted_index.dtype.type,
            ](
                index.data,
                theirlength,
                reinterpreted_index.data,
                mylength,
                theirlength,
            )
        )
        parameters = ak._util.merge_parameters(self._parameters, other._parameters)

        return ak.contents.IndexedOptionArray.simplified(
            index, content, parameters=parameters
        )

    def _mergemany(self, others):
        if len(others) == 0:
            return self

        head, tail = self._merging_strategy(others)

        total_length = 0
        for array in head:
            total_length += array.length

        contents = []
        contentlength_so_far = 0
        length_so_far = 0
        nextindex = ak.index.Index64.empty(total_length, self._backend.index_nplike)
        parameters = self._parameters

        for array in head:
            parameters = ak._util.merge_parameters(parameters, array._parameters, True)

            if isinstance(
                array,
                (
                    ak.contents.ByteMaskedArray,
                    ak.contents.BitMaskedArray,
                    ak.contents.UnmaskedArray,
                ),
            ):
                array = array.to_IndexedOptionArray64()

            if isinstance(array, ak.contents.IndexedOptionArray):
                contents.append(array.content)
                array_index = array.index
                assert (
                    nextindex.nplike is self._backend.index_nplike
                    and array_index.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_IndexedArray_fill",
                        nextindex.dtype.type,
                        array_index.dtype.type,
                    ](
                        nextindex.data,
                        length_so_far,
                        array_index.data,
                        array.length,
                        contentlength_so_far,
                    )
                )
                contentlength_so_far += array.content.length
                length_so_far += array.length

            elif isinstance(array, ak.contents.EmptyArray):
                pass
            else:
                contents.append(array)
                assert nextindex.nplike is self._backend.index_nplike
                self._handle_error(
                    self._backend[
                        "awkward_IndexedArray_fill_count",
                        nextindex.dtype.type,
                    ](
                        nextindex.data,
                        length_so_far,
                        array.length,
                        contentlength_so_far,
                    )
                )
                contentlength_so_far += array.length
                length_so_far += array.length

        tail_contents = contents[1:]
        nextcontent = contents[0]._mergemany(tail_contents)
        next = ak.contents.IndexedOptionArray(
            nextindex, nextcontent, parameters=parameters
        )

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed._mergemany(tail[1:])

    def _fill_none(self, value: Content) -> Content:
        if value.backend.nplike.known_shape and value.length != 1:
            raise ak._errors.wrap_error(
                ValueError(f"fill_none value length ({value.length}) is not equal to 1")
            )

        contents = [self._content, value]
        tags = ak.index.Index8(self.mask_as_bool(valid_when=False))
        index = ak.index.Index64.empty(tags.length, self._backend.index_nplike)

        assert (
            index.nplike is self._backend.index_nplike
            and self._index.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_UnionArray_fillna", index.dtype.type, self._index.dtype.type
            ](index.data, self._index.data, tags.length)
        )
        return ak.contents.UnionArray.simplified(
            tags,
            index,
            contents,
            parameters=self._parameters,
            merge=True,
            mergebool=True,
        )

    def _local_index(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        else:
            _, nextcarry, outindex = self._nextcarry_outindex(self._backend)

            next = self._content._carry(nextcarry, False)
            out = next._local_index(axis, depth)
            out2 = ak.contents.IndexedOptionArray(
                outindex, out, parameters=self._parameters
            )
            return out2

    def _is_subrange_equal(self, starts, stops, length, sorted=True):
        nextstarts = ak.index.Index64.empty(length, self._backend.index_nplike)
        nextstops = ak.index.Index64.empty(length, self._backend.index_nplike)

        subranges_length = ak.index.Index64.empty(1, self._backend.index_nplike)
        assert (
            self._index.nplike is self._backend.index_nplike
            and starts.nplike is self._backend.index_nplike
            and stops.nplike is self._backend.index_nplike
            and nextstarts.nplike is self._backend.index_nplike
            and nextstops.nplike is self._backend.index_nplike
            and subranges_length.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_IndexedArray_ranges_next_64",
                self._index.dtype.type,
                starts.dtype.type,
                stops.dtype.type,
                nextstarts.dtype.type,
                nextstops.dtype.type,
                subranges_length.dtype.type,
            ](
                self._index.data,
                starts.data,
                stops.data,
                length,
                nextstarts.data,
                nextstops.data,
                subranges_length.data,
            )
        )

        nextcarry = ak.index.Index64.empty(
            subranges_length[0], self._backend.index_nplike
        )
        assert (
            self._index.nplike is self._backend.index_nplike
            and starts.nplike is self._backend.index_nplike
            and stops.nplike is self._backend.index_nplike
            and nextcarry.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_IndexedArray_ranges_carry_next_64",
                self._index.dtype.type,
                starts.dtype.type,
                stops.dtype.type,
                nextcarry.dtype.type,
            ](
                self._index.data,
                starts.data,
                stops.data,
                length,
                nextcarry.data,
            )
        )

        next = self._content._carry(nextcarry, False)
        if nextstarts.length > 1:
            return next._is_subrange_equal(nextstarts, nextstops, nextstarts.length)
        else:
            return next._subranges_equal(
                nextstarts, nextstops, nextstarts.length, False
            )

    def _numbers_to_type(self, name):
        return ak.contents.IndexedOptionArray(
            self._index,
            self._content._numbers_to_type(name),
            parameters=self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._index.length == 0:
            return True

        projected = self.project()
        return projected._is_unique(negaxis, starts, parents, outlength)

    def _unique(self, negaxis, starts, parents, outlength):
        branch, depth = self.branch_depth

        inject_nones = (
            True if not branch and (negaxis is not None and negaxis != depth) else False
        )

        index_length = self._index.length

        next, nextparents, numnull, outindex = self._rearrange_prepare_next(parents)

        out = next._unique(
            negaxis,
            starts,
            nextparents,
            outlength,
        )

        if branch or (negaxis is not None and negaxis != depth):
            nextoutindex = ak.index.Index64.empty(
                parents.length, self._backend.index_nplike
            )
            assert (
                nextoutindex.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
                and nextparents.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_IndexedArray_local_preparenext",
                    nextoutindex.dtype.type,
                    starts.dtype.type,
                    parents.dtype.type,
                    nextparents.dtype.type,
                ](
                    nextoutindex.data,
                    starts.data,
                    parents.data,
                    parents.length,
                    nextparents.data,
                    nextparents.length,
                )
            )

            return ak.contents.IndexedOptionArray.simplified(
                nextoutindex, out, parameters=self._parameters
            )

        if isinstance(out, ak.contents.ListOffsetArray):
            newnulls = ak.index.Index64.empty(
                self._index.length, self._backend.index_nplike
            )
            len_newnulls = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert (
                newnulls.nplike is self._backend.index_nplike
                and len_newnulls.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_IndexedArray_numnull_parents",
                    newnulls.dtype.type,
                    len_newnulls.dtype.type,
                    self._index.dtype.type,
                ](
                    newnulls.data,
                    len_newnulls.data,
                    self._index.data,
                    index_length,
                )
            )

            newindex = ak.index.Index64.empty(
                out._offsets[-1] + len_newnulls[0], self._backend.index_nplike
            )
            newoffsets = ak.index.Index64.empty(
                out._offsets.length, self._backend.index_nplike
            )
            assert (
                newindex.nplike is self._backend.index_nplike
                and newoffsets.nplike is self._backend.index_nplike
                and out._offsets.nplike is self._backend.index_nplike
                and newnulls.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_IndexedArray_unique_next_index_and_offsets_64",
                    newindex.dtype.type,
                    newoffsets.dtype.type,
                    out._offsets.dtype.type,
                    newnulls.dtype.type,
                ](
                    newindex.data,
                    newoffsets.data,
                    out._offsets.data,
                    newnulls.data,
                    starts.length,
                )
            )

            out = ak.contents.IndexedOptionArray.simplified(
                newindex[: newoffsets[-1]], out._content, parameters=self._parameters
            )
            return ak.contents.ListOffsetArray(
                newoffsets, out, parameters=self._parameters
            )

        if isinstance(out, ak.contents.NumpyArray):
            # We do not pass None values to the content, so it will have computed
            # the unique _non null_ values. We therefore need to account for None
            # values in the result. We do this by creating an IndexedOptionArray
            # and tacking the index -1 onto the end
            nextoutindex = ak.index.Index64.empty(
                out.length + 1, self._backend.index_nplike
            )
            assert nextoutindex.nplike is self._backend.index_nplike
            self._handle_error(
                self._backend[
                    "awkward_IndexedArray_numnull_unique_64",
                    nextoutindex.dtype.type,
                ](
                    nextoutindex.data,
                    out.length,
                )
            )

            return ak.contents.IndexedOptionArray.simplified(
                nextoutindex, out, parameters=self._parameters
            )

        if inject_nones:
            out = ak.contents.RegularArray(
                out, out.length, 0, parameters=self._parameters
            )

        return out

    def _rearrange_nextshifts(self, nextparents, shifts):
        nextshifts = ak.index.Index64.empty(
            nextparents.length, self._backend.index_nplike
        )
        assert nextshifts.nplike is self._backend.index_nplike

        if shifts is None:
            assert (
                nextshifts.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64",
                    nextshifts.dtype.type,
                    self._index.dtype.type,
                ](
                    nextshifts.data,
                    self._index.data,
                    self._index.length,
                )
            )
        else:
            assert (
                nextshifts.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
                and shifts.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
                    nextshifts.dtype.type,
                    self._index.dtype.type,
                    shifts.dtype.type,
                ](
                    nextshifts.data,
                    self._index.data,
                    self._index.length,
                    shifts.data,
                )
            )
        return nextshifts

    def _rearrange_prepare_next(self, parents):
        assert (
            self._index.nplike is self._backend.index_nplike
            and parents.nplike is self._backend.index_nplike
        )
        index_length = self._index.length
        numnull = ak.index.Index64.empty(1, self._backend.index_nplike)
        assert numnull.nplike is self._backend.index_nplike

        self._handle_error(
            self._backend[
                "awkward_IndexedArray_numnull",
                numnull.dtype.type,
                self._index.dtype.type,
            ](
                numnull.data,
                self._index.data,
                index_length,
            )
        )
        next_length = index_length - numnull[0]
        nextparents = ak.index.Index64.empty(next_length, self._backend.index_nplike)
        nextcarry = ak.index.Index64.empty(next_length, self._backend.index_nplike)
        outindex = ak.index.Index64.empty(index_length, self._backend.index_nplike)
        assert (
            nextcarry.nplike is self._backend.index_nplike
            and nextparents.nplike is self._backend.index_nplike
            and outindex.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_IndexedArray_reduce_next_64",
                nextcarry.dtype.type,
                nextparents.dtype.type,
                outindex.dtype.type,
                self._index.dtype.type,
                parents.dtype.type,
            ](
                nextcarry.data,
                nextparents.data,
                outindex.data,
                self._index.data,
                parents.data,
                index_length,
            )
        )
        next = self._content._carry(nextcarry, False)
        return next, nextparents, numnull, outindex

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
        assert (
            starts.nplike is self._backend.index_nplike
            and parents.nplike is self._backend.index_nplike
            and self._index.nplike is self._backend.index_nplike
        )

        branch, depth = self.branch_depth

        next, nextparents, numnull, outindex = self._rearrange_prepare_next(parents)

        if (not branch) and negaxis == depth:
            nextshifts = self._rearrange_nextshifts(nextparents, shifts)
        else:
            nextshifts = None

        out = next._argsort_next(
            negaxis,
            starts,
            nextshifts,
            nextparents,
            outlength,
            ascending,
            stable,
            kind,
            order,
        )

        # `next._argsort_next` is given the non-None values. We choose to
        # sort None values to the end of the list, meaning we need to grow `out`
        # to account for these None values. First, we locate these nones within
        # their sublists
        nulls_merged = False
        nulls_index = ak.index.Index64.empty(numnull[0], self._backend.index_nplike)
        assert nulls_index.nplike is self._backend.index_nplike
        self._handle_error(
            self._backend[
                "awkward_IndexedArray_index_of_nulls",
                nulls_index.dtype.type,
                self._index.dtype.type,
                parents.dtype.type,
                starts.dtype.type,
            ](
                nulls_index.data,
                self._index.data,
                self._index.length,
                parents.data,
                starts.data,
            )
        )
        # If we wrap a NumpyArray (i.e., axis=-1), then we want `argmax` to return
        # the indices of each `None` value, rather than `None` itself.
        # We can test for this condition by seeing whether the NumpyArray of indices
        # is mergeable with our content (`out = next._argsort_next result`).
        # If so, try to concatenate them at the end of `out`.`
        nulls_index_content = ak.contents.NumpyArray(
            nulls_index, parameters=None, backend=self._backend
        )
        if out._mergeable(nulls_index_content, True):
            out = out._mergemany([nulls_index_content])
            nulls_merged = True

        nextoutindex = ak.index.Index64.empty(
            parents.length, self._backend.index_nplike
        )
        assert (
            nextoutindex.nplike is self._backend.index_nplike
            and starts.nplike is self._backend.index_nplike
            and parents.nplike is self._backend.index_nplike
            and nextparents.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_IndexedArray_local_preparenext",
                nextoutindex.dtype.type,
                starts.dtype.type,
                parents.dtype.type,
                nextparents.dtype.type,
            ](
                nextoutindex.data,
                starts.data,
                parents.data,
                parents.length,
                nextparents.data,
                nextparents.length,
            )
        )

        if nulls_merged:
            # awkward_IndexedArray_local_preparenext uses -1 to
            # indicate `None` values. Given that this code-path runs
            # only when the `None` value indices are explicitly stored in out,
            # we need to mapping the -1 values to their corresponding indices
            # in `out`
            assert nextoutindex.nplike is self._backend.index_nplike
            self._handle_error(
                self._backend["awkward_Index_nones_as_index", nextoutindex.dtype.type](
                    nextoutindex.data,
                    nextoutindex.length,
                )
            )

        out = ak.contents.IndexedOptionArray.simplified(
            nextoutindex, out, parameters=self._parameters
        )

        inject_nones = (
            True if (numnull[0] > 0 and not branch and negaxis != depth) else False
        )

        # If we want the None's at this depth to be injected
        # into the dense ([x y z None None]) rearranger result.
        # Here, we index the dense content with an index
        # that maps the values to the correct locations
        if inject_nones:
            return ak.contents.IndexedOptionArray.simplified(
                outindex, out, parameters=self._parameters
            )
        # Otherwise, if we are rearranging (e.g sorting) the contents of this layout,
        # then we do NOT want to return an optional layout,
        # OR we are branching
        else:
            return out

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        assert (
            starts.nplike is self._backend.index_nplike
            and parents.nplike is self._backend.index_nplike
        )
        branch, depth = self.branch_depth

        next, nextparents, numnull, outindex = self._rearrange_prepare_next(parents)

        out = next._sort_next(
            negaxis,
            starts,
            nextparents,
            outlength,
            ascending,
            stable,
            kind,
            order,
        )

        nextoutindex = ak.index.Index64.empty(
            parents.length, self._backend.index_nplike
        )
        assert nextoutindex.nplike is self._backend.index_nplike

        self._handle_error(
            self._backend[
                "awkward_IndexedArray_local_preparenext",
                nextoutindex.dtype.type,
                starts.dtype.type,
                parents.dtype.type,
                nextparents.dtype.type,
            ](
                nextoutindex.data,
                starts.data,
                parents.data,
                parents.length,
                nextparents.data,
                nextparents.length,
            )
        )
        out = ak.contents.IndexedOptionArray.simplified(
            nextoutindex, out, parameters=self._parameters
        )

        inject_nones = True if not branch and negaxis != depth else False

        # If we want the None's at this depth to be injected
        # into the dense ([x y z None None]) rearranger result.
        # Here, we index the dense content with an index
        # that maps the values to the correct locations
        if inject_nones:
            return ak.contents.IndexedOptionArray.simplified(
                outindex, out, parameters=self._parameters
            )
        # Otherwise, if we are rearranging (e.g sorting) the contents of this layout,
        # then we do NOT want to return an optional layout
        # OR we are branching
        else:
            return out

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
        branch, depth = self.branch_depth

        next, nextparents, numnull, outindex = self._rearrange_prepare_next(parents)

        if reducer.needs_position and (not branch and negaxis == depth):
            nextshifts = self._rearrange_nextshifts(nextparents, shifts)
        else:
            nextshifts = None

        out = next._reduce_next(
            reducer,
            negaxis,
            starts,
            nextshifts,
            nextparents,
            outlength,
            mask,
            keepdims,
            behavior,
        )

        # If we are reducing the contents of this layout,
        # then we do NOT want to return an optional layout
        if not branch and negaxis == depth:
            return out
        else:
            if out.is_list:
                out_content = out.content[out.starts[0] :]
            elif out.is_regular:
                out_content = out.content
            else:
                raise ak._errors.wrap_error(
                    AssertionError(
                        "reduce_next with unbranching depth > negaxis is only "
                        "expected to return RegularArray or ListOffsetArray or "
                        "IndexedOptionArray; "
                        "instead, it returned " + out
                    )
                )

            if starts.nplike.known_data and starts.length > 0 and starts[0] != 0:
                raise ak._errors.wrap_error(
                    AssertionError(
                        "reduce_next with unbranching depth > negaxis expects a "
                        "ListOffsetArray whose offsets start at zero ({})".format(
                            starts[0]
                        )
                    )
                )
            # In this branch, we're above the axis at which the reduction takes place.
            # `next._reduce_next` is therefore expected to return a list/regular layout
            # node. As detailed in `RegularArray._reduce_next`, `_reduce_next` wraps the
            # reduction in a list-type of length `outlength` before returning to the caller,
            # which effectively means that the reduction of *this* layout corresponds to the
            # child of the returned `next._reduce_next(...)`, i.e. `out.content`. So, we unpack
            # the returned list type and wrap its child by a new `IndexedOptionArray`, before
            # re-wrapping the result to have the length and starts requested by the caller.
            outoffsets = ak.index.Index64.empty(
                starts.length + 1, self._backend.index_nplike
            )
            assert (
                outoffsets.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_IndexedArray_reduce_next_fix_offsets_64",
                    outoffsets.dtype.type,
                    starts.dtype.type,
                ](
                    outoffsets.data,
                    starts.data,
                    starts.length,
                    outindex.length,
                )
            )

            # Apply `outindex` to appropriate content
            inner = ak.contents.IndexedOptionArray.simplified(
                outindex, out_content, parameters=self._parameters
            )

            # Re-wrap content
            return ak.contents.ListOffsetArray(
                outoffsets, inner, parameters=self._parameters
            )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            _, nextcarry, outindex = self._nextcarry_outindex(self._backend)
            next = self._content._carry(nextcarry, True)
            out = next._combinations(
                n, replacement, recordlookup, parameters, axis, depth
            )
            return IndexedOptionArray.simplified(outindex, out, parameters=parameters)

    def _validity_error(self, path):
        assert self.index.nplike is self._backend.index_nplike
        error = self._backend["awkward_IndexedArray_validity", self.index.dtype.type](
            self.index.data, self.index.length, self._content.length, True
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
        return self.index._nbytes_part() + self.content._nbytes_part()

    def _pad_none(self, target, axis, depth, clip):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._pad_none_axis0(target, clip)
        elif posaxis is not None and posaxis + 1 == depth + 1:
            mask = ak.index.Index8(self.mask_as_bool(valid_when=False))
            index = ak.index.Index64.empty(mask.length, self._backend.index_nplike)
            assert (
                index.nplike is self._backend.index_nplike
                and mask.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1",
                    index.dtype.type,
                    mask.dtype.type,
                ](index.data, mask.data, mask.length)
            )
            next = self.project()._pad_none(target, axis, depth, clip)
            return ak.contents.IndexedOptionArray.simplified(
                index, next, parameters=self._parameters
            )
        else:
            return ak.contents.IndexedOptionArray(
                self._index,
                self._content._pad_none(target, axis, depth, clip),
                parameters=self._parameters,
            )

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        index = numpy.array(self._index, copy=True)
        this_validbytes = self.mask_as_bool(valid_when=True)
        index[~this_validbytes] = 0

        if self.parameter("__array__") == "categorical":
            # The new IndexedArray will have this parameter, but the rest
            # will be in the AwkwardArrowType.mask_parameters.
            next_parameters = {"__array__": "categorical"}
        else:
            next_parameters = None

        next = ak.contents.IndexedArray(
            ak.index.Index(index), self._content, parameters=next_parameters
        )
        return next._to_arrow(
            pyarrow,
            self,
            ak._connect.pyarrow.and_validbytes(validbytes, this_validbytes),
            length,
            options,
        )

    def _to_numpy(self, allow_missing):
        content = self.project()._to_numpy(allow_missing)

        shape = list(content.shape)
        shape[0] = self.length
        data = numpy.empty(shape, dtype=content.dtype)
        mask0 = numpy.asarray(self.mask_as_bool(valid_when=False)).view(np.bool_)
        if mask0.any():
            if allow_missing:
                mask = numpy.broadcast_to(
                    mask0.reshape((shape[0],) + (1,) * (len(shape) - 1)), shape
                )
                if isinstance(content, numpy.ma.MaskedArray):
                    mask1 = numpy.ma.getmaskarray(content)
                    mask = mask.copy()
                    mask[~mask0] |= mask1

                data[~mask0] = content

                if issubclass(content.dtype.type, (bool, np.bool_)):
                    data[mask0] = False
                elif issubclass(content.dtype.type, np.floating):
                    data[mask0] = np.nan
                elif issubclass(content.dtype.type, np.complexfloating):
                    data[mask0] = np.nan + np.nan * 1j
                elif issubclass(content.dtype.type, np.integer):
                    data[mask0] = np.iinfo(content.dtype).max
                elif issubclass(content.dtype.type, (np.datetime64, np.timedelta64)):
                    data[mask0] = numpy.array([np.iinfo(np.int64).max], content.dtype)[
                        0
                    ]
                else:
                    raise ak._errors.wrap_error(
                        AssertionError(f"unrecognized dtype: {content.dtype}")
                    )

                return numpy.ma.MaskedArray(data, mask)
            else:
                raise ak._errors.wrap_error(
                    ValueError(
                        "ak.to_numpy cannot convert 'None' values to "
                        "np.ma.MaskedArray unless the "
                        "'allow_missing' parameter is set to True"
                    )
                )
        else:
            if allow_missing:
                return numpy.ma.MaskedArray(content)
            else:
                return content

    def _completely_flatten(self, backend, options):
        branch, depth = self.branch_depth
        if branch or options["drop_nones"] or depth > 1:
            return self.project()._completely_flatten(backend, options)
        else:
            return [self]

    def _drop_none(self):
        return self.project()

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if (
            self._backend.nplike.known_shape
            and self._backend.nplike.known_data
            and self._index.length != 0
        ):
            npindex = self._index.data
            npselect = npindex >= 0
            if self._backend.index_nplike.any(npselect):
                indexmin = npindex[npselect].min()
                index = ak.index.Index(
                    npindex - indexmin, nplike=self._backend.index_nplike
                )
                content = self._content[indexmin : npindex.max() + 1]
            else:
                index, content = self._index, self._content
        else:
            if (
                not self._backend.nplike.known_shape
                or not self._backend.nplike.known_data
            ):
                self._touch_data(recursive=False)
            index, content = self._index, self._content

        if options["return_array"]:
            if options["return_simplified"]:
                make = IndexedOptionArray.simplified
            else:
                make = IndexedOptionArray

            def continuation():
                return make(
                    index,
                    content._recursively_apply(
                        action,
                        behavior,
                        depth,
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
        original_index = self._index.raw(self._backend.nplike)
        is_none = original_index < 0
        num_none = self._backend.index_nplike.count_nonzero(is_none)
        if self.parameter("__array__") == "categorical" or self._content.length <= (
            len(original_index) - num_none
        ):
            return ak.contents.IndexedOptionArray(
                self._index, self._content.to_packed(), parameters=self._parameters
            )

        else:
            new_index = self._backend.index_nplike.empty(
                len(original_index), dtype=original_index.dtype
            )
            new_index[is_none] = -1
            new_index[~is_none] = self._backend.index_nplike.arange(
                len(original_index) - num_none,
                dtype=original_index.dtype,
            )
            return ak.contents.IndexedOptionArray(
                ak.index.Index(new_index, nplike=self._backend.index_nplike),
                self.project().to_packed(),
                parameters=self._parameters,
            )

    def _to_list(self, behavior, json_conversions):
        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        index = self._index.raw(numpy)
        not_missing = index >= 0

        nextcontent = self._content._carry(ak.index.Index(index[not_missing]), False)
        out = nextcontent._to_list(behavior, json_conversions)

        for i, isvalid in enumerate(not_missing):
            if not isvalid:
                out.insert(i, None)

        return out

    def to_backend(self, backend: ak._backends.Backend) -> Self:
        content = self._content.to_backend(backend)
        index = self._index.to_nplike(backend.index_nplike)
        return IndexedOptionArray(index, content, parameters=self._parameters)

    def _is_equal_to(self, other, index_dtype, numpyarray):
        return self.index.is_equal_to(
            other.index, index_dtype, numpyarray
        ) and self.content.is_equal_to(other.content, index_dtype, numpyarray)
