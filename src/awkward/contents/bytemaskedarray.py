# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy
import json
import math

import awkward as ak
from awkward._util import unset
from awkward.contents.content import Content
from awkward.forms.bytemaskedform import ByteMaskedForm
from awkward.index import Index
from awkward.typing import Final, Self

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


class ByteMaskedArray(Content):
    is_option = True

    def __init__(self, mask, content, valid_when, *, parameters=None):
        if not (isinstance(mask, Index) and mask.dtype == np.dtype(np.int8)):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'mask' must be an Index with dtype=int8, not {}".format(
                        type(self).__name__, repr(mask)
                    )
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
        if content.is_union or content.is_indexed or content.is_option:
            raise ak._errors.wrap_error(
                TypeError(
                    "{0} cannot contain a union-type, option-type, or indexed 'content' ({1}); try {0}.simplified instead".format(
                        type(self).__name__, type(content).__name__
                    )
                )
            )
        if not isinstance(valid_when, bool):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'valid_when' must be boolean, not {}".format(
                        type(self).__name__, repr(valid_when)
                    )
                )
            )
        if (
            mask.nplike.known_shape
            and content.backend.nplike.known_shape
            and mask.length > content.length
        ):
            raise ak._errors.wrap_error(
                ValueError(
                    "{} len(mask) ({}) must be <= len(content) ({})".format(
                        type(self).__name__, mask.length, content.length
                    )
                )
            )

        assert mask.nplike is content.backend.index_nplike

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._init(parameters, content.backend)

    @property
    def mask(self):
        return self._mask

    @property
    def content(self):
        return self._content

    @property
    def valid_when(self):
        return self._valid_when

    form_cls: Final = ByteMaskedForm

    def copy(self, mask=unset, content=unset, valid_when=unset, *, parameters=unset):
        return ByteMaskedArray(
            self._mask if mask is unset else mask,
            self._content if content is unset else content,
            self._valid_when if valid_when is unset else valid_when,
            parameters=self._parameters if parameters is unset else parameters,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            mask=copy.deepcopy(self._mask, memo),
            content=copy.deepcopy(self._content, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(
        cls,
        mask,
        content,
        valid_when,
        *,
        parameters=None,
    ):
        if content.is_union or content.is_indexed or content.is_option:
            backend = content.backend
            index = ak.index.Index64.empty(mask.length, nplike=backend.index_nplike)
            Content._selfless_handle_error(
                backend[
                    "awkward_ByteMaskedArray_toIndexedOptionArray",
                    index.dtype.type,
                    mask.dtype.type,
                ](
                    index.data,
                    mask.data,
                    mask.length,
                    valid_when,
                ),
            )
            if content.is_union:
                return content._union_of_optionarrays(index, parameters)
            else:
                return ak.contents.IndexedOptionArray.simplified(
                    index, content, parameters=parameters
                )
        else:
            return cls(mask, content, valid_when, parameters=parameters)

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.form_cls(
            self._mask.form,
            self._content._form_with_key(getkey),
            self._valid_when,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, backend):
        assert isinstance(form, self.form_cls)
        key = getkey(self, form, "mask")
        container[key] = ak._util.little_endian(self._mask.raw(backend.index_nplike))
        self._content._to_buffers(form.content, getkey, container, backend)

    def _to_typetracer(self, forget_length: bool) -> Self:
        tt = ak._typetracer.TypeTracer.instance()
        mask = self._mask.to_nplike(tt)
        return ByteMaskedArray(
            mask.forget_length() if forget_length else mask,
            self._content._to_typetracer(False),
            self._valid_when,
            parameters=self._parameters,
        )

    def _touch_data(self, recursive):
        if not self._backend.index_nplike.known_data:
            self._mask.data.touch_data()
        if recursive:
            self._content._touch_data(recursive)

    def _touch_shape(self, recursive):
        if not self._backend.index_nplike.known_shape:
            self._mask.data.touch_shape()
        if recursive:
            self._content._touch_shape(recursive)

    @property
    def length(self):
        return self._mask.length

    def _forget_length(self):
        return ByteMaskedArray(
            self._mask.forget_length(),
            self._content,
            self._valid_when,
            parameters=self._parameters,
        )

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ByteMaskedArray valid_when="]
        out.append(repr(json.dumps(self._valid_when)))
        out.append(" len=")
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._mask._repr(indent + "    ", "<mask>", "</mask>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</ByteMaskedArray>")
        out.append(post)
        return "".join(out)

    def to_IndexedOptionArray64(self):
        index = ak.index.Index64.empty(
            self._mask.length, nplike=self._backend.index_nplike
        )
        assert (
            index.nplike is self._backend.index_nplike
            and self._mask.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_ByteMaskedArray_toIndexedOptionArray",
                index.dtype.type,
                self._mask.dtype.type,
            ](
                index.data,
                self._mask.data,
                self._mask.length,
                self._valid_when,
            ),
        )
        return ak.contents.IndexedOptionArray(
            index, self._content, parameters=self._parameters
        )

    def to_ByteMaskedArray(self, valid_when):
        if valid_when == self._valid_when:
            return self
        else:
            return ByteMaskedArray(
                ak.index.Index8(
                    self._backend.index_nplike.logical_not(
                        self._mask.data.astype(np.bool_)
                    ).astype(dtype=np.int8)
                ),
                self._content,
                valid_when,
                parameters=self._parameters,
            )

    def to_BitMaskedArray(self, valid_when, lsb_order):
        if not self._backend.nplike.known_data:
            self._touch_data(recursive=False)
            if self._backend.nplike.known_shape:
                excess_length = int(math.ceil(self.length / 8.0))
            else:
                excess_length = ak._typetracer.UnknownLength
            return ak.contents.BitMaskedArray(
                ak.index.IndexU8(
                    self._backend.nplike.empty(excess_length, dtype=np.uint8)
                ),
                self._content,
                valid_when,
                self.length,
                lsb_order,
                parameters=self._parameters,
            )

        else:
            import awkward._connect.pyarrow

            bytearray = self.mask_as_bool(valid_when).view(np.uint8)
            bitarray = awkward._connect.pyarrow.packbits(bytearray, lsb_order)

            return ak.contents.BitMaskedArray(
                ak.index.IndexU8(bitarray),
                self._content,
                valid_when,
                self.length,
                lsb_order,
                parameters=self._parameters,
            )

    def mask_as_bool(self, valid_when=None):
        if valid_when is None:
            valid_when = self._valid_when

        if valid_when == self._valid_when:
            return self._mask.raw(self._backend.index_nplike) != 0
        else:
            return self._mask.raw(self._backend.index_nplike) != 1

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
        if self._mask[where] == self._valid_when:
            return self._content._getitem_at(where)
        else:
            return None

    def _getitem_range(self, where):
        if not self._backend.nplike.known_shape:
            self._touch_shape(recursive=False)
            return self

        start, stop, step = where.indices(self.length)
        assert step == 1
        return ByteMaskedArray(
            self._mask[start:stop],
            self._content._getitem_range(slice(start, stop)),
            self._valid_when,
            parameters=self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return ByteMaskedArray.simplified(
            self._mask,
            self._content._getitem_field(where, only_fields),
            self._valid_when,
            parameters=None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return ByteMaskedArray.simplified(
            self._mask,
            self._content._getitem_fields(where, only_fields),
            self._valid_when,
            parameters=None,
        )

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak.index.Index)

        try:
            nextmask = self._mask[carry.data]
        except IndexError as err:
            raise ak._errors.index_error(self, carry.data, str(err)) from err

        return ByteMaskedArray.simplified(
            nextmask,
            self._content._carry(carry, allow_lazy),
            self._valid_when,
            parameters=self._parameters,
        )

    def _nextcarry_outindex(
        self, backend: ak._backends.Backend
    ) -> tuple[int, ak.index.Index64, ak.index.Index64]:
        numnull = ak.index.Index64.empty(1, nplike=backend.index_nplike)

        assert (
            numnull.nplike is self._backend.index_nplike
            and self._mask.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_ByteMaskedArray_numnull",
                numnull.dtype.type,
                self._mask.dtype.type,
            ](
                numnull.data,
                self._mask.data,
                self._mask.length,
                self._valid_when,
            )
        )
        nextcarry = ak.index.Index64.empty(
            self.length - numnull[0], nplike=backend.index_nplike
        )
        outindex = ak.index.Index64.empty(self.length, nplike=backend.index_nplike)
        assert (
            nextcarry.nplike is self._backend.index_nplike
            and outindex.nplike is self._backend.index_nplike
            and self._mask.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_ByteMaskedArray_getitem_nextcarry_outindex",
                nextcarry.dtype.type,
                outindex.dtype.type,
                self._mask.dtype.type,
            ](
                nextcarry.data,
                outindex.data,
                self._mask.raw(self._backend.nplike),
                self._mask.length,
                self._valid_when,
            )
        )
        return numnull[0], nextcarry, outindex

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        if (
            slicestarts.nplike.known_shape
            and self._backend.nplike.known_shape
            and slicestarts.length != self.length
        ):
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
            and reducedstarts.nplike is self._backend.nplike
            and reducedstops.nplike is self._backend.nplike
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
            _, nextcarry, outindex = self._nextcarry_outindex(self._backend)

            next = self._content._carry(nextcarry, True)
            out = next._getitem_next(head, tail, advanced)
            return ak.contents.IndexedOptionArray.simplified(
                outindex, out, parameters=self._parameters
            )

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
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
        mask_length = self._mask.length
        numnull = ak.index.Index64.zeros(1, nplike=self._backend.index_nplike)

        if mask is not None:
            if self._backend.nplike.known_shape and mask_length != mask.length:
                raise ak._errors.wrap_error(
                    ValueError(
                        "mask length ({}) is not equal to {} length ({})".format(
                            mask.length, type(self).__name__, mask_length
                        )
                    )
                )

            nextmask = ak.index.Index8.empty(
                mask_length, nplike=self._backend.index_nplike
            )
            assert (
                nextmask.nplike is self._backend.index_nplike
                and mask.nplike is self._backend.index_nplike
                and self._mask.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ByteMaskedArray_overlay_mask",
                    nextmask.dtype.type,
                    mask.dtype.type,
                    self._mask.dtype.type,
                ](
                    nextmask.data,
                    mask.data,
                    self._mask.data,
                    mask_length,
                    self._valid_when,
                )
            )
            valid_when = False
            next = ByteMaskedArray(
                nextmask, self._content, valid_when, parameters=self._parameters
            )
            return next.project()

        else:
            assert (
                numnull.nplike is self._backend.index_nplike
                and self._mask.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ByteMaskedArray_numnull",
                    numnull.dtype.type,
                    self._mask.dtype.type,
                ](
                    numnull.data,
                    self._mask.data,
                    mask_length,
                    self._valid_when,
                )
            )
            nextcarry = ak.index.Index64.empty(
                mask_length - numnull[0], nplike=self._backend.index_nplike
            )
            assert (
                nextcarry.nplike is self._backend.index_nplike
                and self._mask.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ByteMaskedArray_getitem_nextcarry",
                    nextcarry.dtype.type,
                    self._mask.dtype.type,
                ](
                    nextcarry.data,
                    self._mask.data,
                    mask_length,
                    self._valid_when,
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
                    nplike=self._backend.index_nplike,
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

    def _reverse_merge(self, other):
        return self.to_IndexedOptionArray64()._reverse_merge(other)

    def _mergemany(self, others):
        if len(others) == 0:
            return self

        if all(
            isinstance(x, ByteMaskedArray) and x._valid_when == self._valid_when
            for x in others
        ):
            parameters = self._parameters
            masks = [self._mask.data[: self.length]]
            tail_contents = []
            length = 0
            for x in others:
                parameters = ak._util.merge_parameters(parameters, x._parameters, True)
                masks.append(x._mask.data[: x.length])
                tail_contents.append(x._content[: x.length])
                length += x.length

            return ByteMaskedArray(
                ak.index.Index8(self._backend.nplike.concatenate(masks)),
                self._content[: self.length]._mergemany(tail_contents),
                self._valid_when,
                parameters=parameters,
            )

        else:
            return self.to_IndexedOptionArray64()._mergemany(others)

    def _fill_none(self, value: Content) -> Content:
        return self.to_IndexedOptionArray64()._fill_none(value)

    def _local_index(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        else:
            _, nextcarry, outindex = self._nextcarry_outindex(self._backend)

            next = self._content._carry(nextcarry, False)
            out = next._local_index(axis, depth)
            return ak.contents.IndexedOptionArray.simplified(
                outindex, out, parameters=self._parameters
            )

    def _numbers_to_type(self, name):
        return ak.contents.ByteMaskedArray(
            self._mask,
            self._content._numbers_to_type(name),
            self._valid_when,
            parameters=self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._mask.length == 0:
            return True
        return self.to_IndexedOptionArray64()._is_unique(
            negaxis, starts, parents, outlength
        )

    def _unique(self, negaxis, starts, parents, outlength):
        if self._mask.length == 0:
            return self
        return self.to_IndexedOptionArray64()._unique(
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
        return self.to_IndexedOptionArray64()._argsort_next(
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

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        return self.to_IndexedOptionArray64()._sort_next(
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
        if n < 1:
            raise ak._errors.wrap_error(
                ValueError("in combinations, 'n' must be at least 1")
            )
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            _, nextcarry, outindex = self._nextcarry_outindex(self._backend)

            next = self._content._carry(nextcarry, True)
            out = next._combinations(
                n, replacement, recordlookup, parameters, axis, depth
            )
            return ak.contents.IndexedOptionArray.simplified(
                outindex, out, parameters=parameters
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
        mask_length = self._mask.length

        numnull = ak.index.Index64.empty(1, nplike=self._backend.index_nplike)
        assert (
            numnull.nplike is self._backend.index_nplike
            and self._mask.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_ByteMaskedArray_numnull",
                numnull.dtype.type,
                self._mask.dtype.type,
            ](
                numnull.data,
                self._mask.data,
                mask_length,
                self._valid_when,
            )
        )

        next_length = mask_length - numnull[0]
        nextcarry = ak.index.Index64.empty(
            next_length, nplike=self._backend.index_nplike
        )
        nextparents = ak.index.Index64.empty(
            next_length, nplike=self._backend.index_nplike
        )
        outindex = ak.index.Index64.empty(
            mask_length, nplike=self._backend.index_nplike
        )
        assert (
            nextcarry.nplike is self._backend.index_nplike
            and nextparents.nplike is self._backend.index_nplike
            and outindex.nplike is self._backend.index_nplike
            and self._mask.nplike is self._backend.index_nplike
            and parents.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
                "awkward_ByteMaskedArray_reduce_next_64",
                nextcarry.dtype.type,
                nextparents.dtype.type,
                outindex.dtype.type,
                self._mask.dtype.type,
                parents.dtype.type,
            ](
                nextcarry.data,
                nextparents.data,
                outindex.data,
                self._mask.data,
                parents.data,
                mask_length,
                self._valid_when,
            )
        )

        branch, depth = self.branch_depth

        if reducer.needs_position and (not branch and negaxis == depth):
            nextshifts = ak.index.Index64.empty(
                next_length, nplike=self._backend.index_nplike
            )
            if shifts is None:
                assert (
                    nextshifts.nplike is self._backend.index_nplike
                    and self._mask.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64",
                        nextshifts.dtype.type,
                        self._mask.dtype.type,
                    ](
                        nextshifts.data,
                        self._mask.data,
                        mask_length,
                        self._valid_when,
                    )
                )
            else:
                assert (
                    nextshifts.nplike is self._backend.index_nplike
                    and self._mask.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
                        nextshifts.dtype.type,
                        self._mask.dtype.type,
                        shifts.dtype.type,
                    ](
                        nextshifts.data,
                        self._mask.data,
                        mask_length,
                        self._valid_when,
                        shifts.data,
                    )
                )
        else:
            nextshifts = None

        next = self._content._carry(nextcarry, False)

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

        if not branch and negaxis == depth:
            return out
        else:
            if out.is_list:
                out_content = out.content[out.starts[0] :]
            elif out.is_regular:
                out_content = out.content
            else:
                raise ak._errors.wrap_error(
                    ValueError(
                        "reduce_next with unbranching depth > negaxis is only "
                        "expected to return RegularArray or ListOffsetArray64; "
                        "instead, it returned " + out
                    )
                )

            outoffsets = ak.index.Index64.empty(
                starts.length + 1, nplike=self._backend.index_nplike
            )
            assert outoffsets.nplike is self._backend.nplike
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

            tmp = ak.contents.IndexedOptionArray.simplified(
                outindex, out_content, parameters=None
            )
            return ak.contents.ListOffsetArray(outoffsets, tmp, parameters=None)

    def _validity_error(self, path):
        if self._backend.nplike.known_shape and self._content.length < self.mask.length:
            return f'at {path} ("{type(self)}"): len(content) < len(mask)'
        else:
            return self._content._validity_error(path + ".content")

    def _nbytes_part(self):
        return self.mask._nbytes_part() + self.content._nbytes_part()

    def _pad_none(self, target, axis, depth, clip):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._pad_none_axis0(target, clip)
        elif posaxis is not None and posaxis + 1 == depth + 1:
            mask = ak.index.Index8(self.mask_as_bool(valid_when=False))
            index = ak.index.Index64.empty(
                mask.length, nplike=self._backend.index_nplike
            )
            assert (
                index.nplike is self._backend.index_nplike
                and self._mask.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1",
                    index.dtype.type,
                    self._mask.dtype.type,
                ](
                    index.data,
                    self._mask.data,
                    self._mask.length,
                )
            )
            next = self.project()._pad_none(target, axis, depth, clip)
            return ak.contents.IndexedOptionArray.simplified(
                index, next, parameters=self._parameters
            )
        else:
            return ak.contents.ByteMaskedArray(
                self._mask,
                self._content._pad_none(target, axis, depth, clip),
                self._valid_when,
                parameters=self._parameters,
            )

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        this_validbytes = self.mask_as_bool(valid_when=True)

        return self._content._to_arrow(
            pyarrow,
            self,
            ak._connect.pyarrow.and_validbytes(validbytes, this_validbytes),
            length,
            options,
        )

    def _to_numpy(self, allow_missing):
        return self.to_IndexedOptionArray64()._to_numpy(allow_missing)

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
        if self._backend.nplike.known_shape:
            content = self._content[0 : self._mask.length]
        else:
            content = self._content

        if options["return_array"]:
            if options["return_simplified"]:
                make = ByteMaskedArray.simplified
            else:
                make = ByteMaskedArray

            def continuation():
                return make(
                    self._mask,
                    content._recursively_apply(
                        action,
                        behavior,
                        depth,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    ),
                    self._valid_when,
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
        if self._content.is_record:
            next = self.to_IndexedOptionArray64()
            content = next._content.to_packed()
            if content.length > self._mask.length:
                content = content[: self._mask.length]

            return ak.contents.IndexedOptionArray(
                next._index, content, parameters=next._parameters
            )

        else:
            content = self._content.to_packed()
            if content.length > self._mask.length:
                content = content[: self._mask.length]

            return ByteMaskedArray(
                self._mask, content, self._valid_when, parameters=self._parameters
            )

    def _to_list(self, behavior, json_conversions):
        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        mask = self.mask_as_bool(valid_when=True)
        out = self._content._getitem_range(slice(0, len(mask)))._to_list(
            behavior, json_conversions
        )

        for i, isvalid in enumerate(mask):
            if not isvalid:
                out[i] = None

        return out

    def to_backend(self, backend: ak._backends.Backend) -> Self:
        content = self._content.to_backend(backend)
        mask = self._mask.to_nplike(backend.index_nplike)
        return ByteMaskedArray(
            mask, content, valid_when=self._valid_when, parameters=self._parameters
        )

    def _is_equal_to(self, other, index_dtype, numpyarray):
        return (
            self.valid_when == other.valid_when
            and self.mask.is_equal_to(other.mask, index_dtype, numpyarray)
            and self.content.is_equal_to(other.content, index_dtype, numpyarray)
        )
