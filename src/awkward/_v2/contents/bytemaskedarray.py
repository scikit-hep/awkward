# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json
import copy
import math

import awkward as ak
from awkward._v2.index import Index
from awkward._v2.contents.content import Content, unset
from awkward._v2.forms.bytemaskedform import ByteMaskedForm

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class ByteMaskedArray(Content):
    is_OptionType = True

    def copy(
        self,
        mask=unset,
        content=unset,
        valid_when=unset,
        identifier=unset,
        parameters=unset,
        nplike=unset,
    ):
        return ByteMaskedArray(
            self._mask if mask is unset else mask,
            self._content if content is unset else content,
            self._valid_when if valid_when is unset else valid_when,
            self._identifier if identifier is unset else identifier,
            self._parameters if parameters is unset else parameters,
            self._nplike if nplike is unset else nplike,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            mask=copy.deepcopy(self._mask, memo),
            content=copy.deepcopy(self._content, memo),
            identifier=copy.deepcopy(self._identifier, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    def __init__(
        self, mask, content, valid_when, identifier=None, parameters=None, nplike=None
    ):
        if not (isinstance(mask, Index) and mask.dtype == np.dtype(np.int8)):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'mask' must be an Index with dtype=int8, not {}".format(
                        type(self).__name__, repr(mask)
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
        if not isinstance(valid_when, bool):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'valid_when' must be boolean, not {}".format(
                        type(self).__name__, repr(valid_when)
                    )
                )
            )
        if (
            mask.nplike.known_shape
            and content.nplike.known_shape
            and mask.length > content.length
        ):
            raise ak._v2._util.error(
                ValueError(
                    "{} len(mask) ({}) must be <= len(content) ({})".format(
                        type(self).__name__, mask.length, content.length
                    )
                )
            )
        if nplike is None:
            nplike = content.nplike
        if nplike is None:
            nplike = mask.nplike

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._init(identifier, parameters, nplike)

    @property
    def mask(self):
        return self._mask

    @property
    def content(self):
        return self._content

    @property
    def valid_when(self):
        return self._valid_when

    Form = ByteMaskedForm

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.Form(
            self._mask.form,
            self._content._form_with_key(getkey),
            self._valid_when,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, nplike):
        assert isinstance(form, self.Form)
        key = getkey(self, form, "mask")
        container[key] = ak._v2._util.little_endian(self._mask.raw(nplike))
        self._content._to_buffers(form.content, getkey, container, nplike)

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return ByteMaskedArray(
            ak._v2.index.Index(self._mask.raw(tt)),
            self._content.typetracer,
            self._valid_when,
            self._typetracer_identifier(),
            self._parameters,
            tt,
        )

    @property
    def length(self):
        return self._mask.length

    def _forget_length(self):
        return ByteMaskedArray(
            self._mask.forget_length(),
            self._content,
            self._valid_when,
            self._identifier,
            self._parameters,
            self._nplike,
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

    def merge_parameters(self, parameters):
        return ByteMaskedArray(
            self._mask,
            self._content,
            self._valid_when,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
            self._nplike,
        )

    def toIndexedOptionArray64(self):
        index = ak._v2.index.Index64.empty(self._mask.length, self._nplike)
        assert index.nplike is self._nplike and self._mask.nplike is self._nplike
        self._handle_error(
            self._nplike[
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

        return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            index, self._content, self._identifier, self._parameters, self._nplike
        )

    def toByteMaskedArray(self, valid_when):
        if valid_when == self._valid_when:
            return self
        else:
            return ByteMaskedArray(
                ak._v2.Index8(~self._mask.data),
                self._content,
                valid_when,
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def toBitMaskedArray(self, valid_when, lsb_order):
        if not self._nplike.known_data:
            if self._nplike.known_shape:
                excess_length = int(math.ceil(self.length / 8.0))
            else:
                excess_length = ak._v2._typetracer.UnknownLength
            return ak._v2.contents.BitMaskedArray(
                ak._v2.index.IndexU8(self._nplike.empty(excess_length, dtype=np.uint8)),
                self._content,
                valid_when,
                self.length,
                lsb_order,
                self._identifier,
                self._parameters,
                self._nplike,
            )

        else:
            import awkward._v2._connect.pyarrow

            bytearray = self.mask_as_bool(valid_when, self._nplike).view(np.uint8)
            bitarray = awkward._v2._connect.pyarrow.packbits(bytearray, lsb_order)

            return ak._v2.contents.BitMaskedArray(
                ak._v2.index.IndexU8(bitarray),
                self._content,
                valid_when,
                self.length,
                lsb_order,
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def mask_as_bool(self, valid_when=None, nplike=None):
        if valid_when is None:
            valid_when = self._valid_when
        if nplike is None:
            nplike = self._nplike

        if valid_when == self._valid_when:
            return self._mask.raw(nplike) != 0
        else:
            return self._mask.raw(nplike) != 1

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if not self._nplike.known_data:
            return ak._v2._typetracer.MaybeNone(self._content._getitem_at(where))

        if where < 0:
            where += self.length
        if self._nplike.known_shape and not 0 <= where < self.length:
            raise ak._v2._util.indexerror(self, where)
        if self._mask[where] == self._valid_when:
            return self._content._getitem_at(where)
        else:
            return None

    def _getitem_range(self, where):
        if not self._nplike.known_shape:
            return self

        start, stop, step = where.indices(self.length)
        assert step == 1
        return ByteMaskedArray(
            self._mask[start:stop],
            self._content._getitem_range(slice(start, stop)),
            self._valid_when,
            self._range_identifier(start, stop),
            self._parameters,
            self._nplike,
        )

    def _getitem_field(self, where, only_fields=()):
        return ByteMaskedArray(
            self._mask,
            self._content._getitem_field(where, only_fields),
            self._valid_when,
            self._field_identifier(where),
            None,
            self._nplike,
        ).simplify_optiontype()

    def _getitem_fields(self, where, only_fields=()):
        return ByteMaskedArray(
            self._mask,
            self._content._getitem_fields(where, only_fields),
            self._valid_when,
            self._fields_identifier(where),
            None,
            self._nplike,
        ).simplify_optiontype()

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextmask = self._mask[carry.data]
        except IndexError as err:
            raise ak._v2._util.indexerror(self, carry.data, str(err)) from err

        return ByteMaskedArray(
            nextmask,
            self._content._carry(carry, allow_lazy),
            self._valid_when,
            self._carry_identifier(carry),
            self._parameters,
            self._nplike,
        )

    def _nextcarry_outindex(self, numnull):
        assert numnull.nplike is self._nplike and self._mask.nplike is self._nplike
        self._handle_error(
            self._nplike[
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
        nextcarry = ak._v2.index.Index64.empty(self.length - numnull[0], self._nplike)
        outindex = ak._v2.index.Index64.empty(self.length, self._nplike)
        assert (
            nextcarry.nplike is self._nplike
            and outindex.nplike is self._nplike
            and self._mask.nplike is self._nplike
        )
        self._handle_error(
            self._nplike[
                "awkward_ByteMaskedArray_getitem_nextcarry_outindex",
                nextcarry.dtype.type,
                outindex.dtype.type,
                self._mask.dtype.type,
            ](
                nextcarry.data,
                outindex.data,
                self._mask.raw(self._nplike),
                self._mask.length,
                self._valid_when,
            )
        )
        return nextcarry, outindex

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        if (
            slicestarts.nplike.known_shape
            and self._nplike.known_shape
            and slicestarts.length != self.length
        ):
            raise ak._v2._util.indexerror(
                self,
                ak._v2.contents.ListArray(
                    slicestarts, slicestops, slicecontent, None, None, self._nplike
                ),
                "cannot fit jagged slice with length {} into {} of size {}".format(
                    slicestarts.length, type(self).__name__, self.length
                ),
            )

        numnull = ak._v2.index.Index64.empty(1, self._nplike)
        nextcarry, outindex = self._nextcarry_outindex(numnull)

        reducedstarts = ak._v2.index.Index64.empty(
            self.length - numnull[0], self._nplike
        )
        reducedstops = ak._v2.index.Index64.empty(
            self.length - numnull[0], self._nplike
        )

        assert (
            outindex.nplike is self._nplike
            and slicestarts.nplike is self._nplike
            and slicestops.nplike is self._nplike
            and reducedstarts.nplike is self._nplike
            and reducedstops.nplike is self._nplike
        )
        self._handle_error(
            self._nplike[
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
            slicer=ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
        )

        next = self._content._carry(nextcarry, True)
        out = next._getitem_next_jagged(reducedstarts, reducedstops, slicecontent, tail)

        out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            outindex, out, self._identifier, self._parameters, self._nplike
        )
        return out2.simplify_optiontype()

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        return self._getitem_next_jagged_generic(
            slicestarts, slicestops, slicecontent, tail
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(
            head, (int, slice, ak._v2.index.Index64, ak._v2.contents.ListOffsetArray)
        ):
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            numnull = ak._v2.index.Index64.empty(1, self._nplike)
            nextcarry, outindex = self._nextcarry_outindex(numnull)

            next = self._content._carry(nextcarry, True)
            out = next._getitem_next(head, tail, advanced)
            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex,
                out,
                self._identifier,
                self._parameters,
                self._nplike,
            )
            return out2.simplify_optiontype()

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise ak._v2._util.error(AssertionError(repr(head)))

    def project(self, mask=None):
        mask_length = self._mask.length
        numnull = ak._v2.index.Index64.zeros(1, self._nplike)

        if mask is not None:
            if self._nplike.known_shape and mask_length != mask.length:
                raise ak._v2._util.error(
                    ValueError(
                        "mask length ({}) is not equal to {} length ({})".format(
                            mask.length, type(self).__name__, mask_length
                        )
                    )
                )

            nextmask = ak._v2.index.Index8.empty(mask_length, self._nplike)
            assert (
                nextmask.nplike is self._nplike
                and mask.nplike is self._nplike
                and self._mask.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
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
                nextmask,
                self._content,
                valid_when,
                self._identifier,
                self._parameters,
                self._nplike,
            )
            return next.project()

        else:
            assert numnull.nplike is self._nplike and self._mask.nplike is self._nplike
            self._handle_error(
                self.nplike[
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
            nextcarry = ak._v2.index.Index64.empty(
                mask_length - numnull[0], self._nplike
            )
            assert (
                nextcarry.nplike is self._nplike and self._mask.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
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

    def simplify_optiontype(self):
        if isinstance(
            self._content,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):
            return self.toIndexedOptionArray64().simplify_optiontype()
        else:
            return self

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            out = self.length
            if ak._v2._util.isint(out):
                return np.int64(out)
            else:
                return out
        else:
            numnull = ak._v2.index.Index64.empty(1, self._nplike, dtype=np.int64)
            nextcarry, outindex = self._nextcarry_outindex(numnull)

            next = self._content._carry(nextcarry, False)
            out = next.num(posaxis, depth)

            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex, out, None, self.parameters, self._nplike
            )
            return out2.simplify_optiontype()

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            raise ak._v2._util.error(np.AxisError("axis=0 not allowed for flatten"))
        else:
            numnull = ak._v2.index.Index64.empty(1, self._nplike)
            nextcarry, outindex = self._nextcarry_outindex(numnull)

            next = self._content._carry(nextcarry, False)

            offsets, flattened = next._offsets_and_flattened(posaxis, depth)

            if offsets.length == 0:
                return (
                    offsets,
                    ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                        outindex, flattened, None, self._parameters, self._nplike
                    ),
                )

            else:
                outoffsets = ak._v2.index.Index64.empty(
                    offsets.length + numnull[0], self._nplike, dtype=np.int64
                )

                assert (
                    outoffsets.nplike is self._nplike
                    and outindex.nplike is self._nplike
                    and offsets.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
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
            return self._content.mergeable(other.content, mergebool)

        else:
            return self._content.mergeable(other, mergebool)

    def _reverse_merge(self, other):
        return self.toIndexedOptionArray64()._reverse_merge(other)

    def mergemany(self, others):
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
                parameters = ak._v2._util.merge_parameters(
                    parameters, x._parameters, True
                )
                masks.append(x._mask.data[: x.length])
                tail_contents.append(x._content[: x.length])
                length += x.length

            return ByteMaskedArray(
                ak._v2.index.Index8(self._nplike.concatenate(masks)),
                self._content[: self.length].mergemany(tail_contents),
                self._valid_when,
                None,
                parameters,
                self._nplike,
            )

        else:
            return self.toIndexedOptionArray64().mergemany(others)

    def fill_none(self, value):
        return self.toIndexedOptionArray64().fill_none(value)

    def _local_index(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._local_index_axis0()
        else:
            numnull = ak._v2.index.Index64.empty(1, self._nplike)
            nextcarry, outindex = self._nextcarry_outindex(numnull)

            next = self._content._carry(nextcarry, False)
            out = next._local_index(posaxis, depth)
            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex,
                out,
                self._identifier,
                self._parameters,
                self._nplike,
            )
            return out2.simplify_optiontype()

    def numbers_to_type(self, name):
        return ak._v2.contents.bytemaskedarray.ByteMaskedArray(
            self._mask,
            self._content.numbers_to_type(name),
            self._valid_when,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._mask.length == 0:
            return True
        return self.toIndexedOptionArray64()._is_unique(
            negaxis, starts, parents, outlength
        )

    def _unique(self, negaxis, starts, parents, outlength):
        if self._mask.length == 0:
            return self
        return self.toIndexedOptionArray64()._unique(
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
        return self.toIndexedOptionArray64()._argsort_next(
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
        return self.toIndexedOptionArray64()._sort_next(
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
            raise ak._v2._util.error(
                ValueError("in combinations, 'n' must be at least 1")
            )
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            numnull = ak._v2.index.Index64.empty(1, self._nplike, dtype=np.int64)
            nextcarry, outindex = self._nextcarry_outindex(numnull)

            next = self._content._carry(nextcarry, True)
            out = next._combinations(
                n, replacement, recordlookup, parameters, posaxis, depth
            )
            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex, out, None, parameters, self._nplike
            )
            return out2.simplify_optiontype()

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

        numnull = ak._v2.index.Index64.empty(1, self._nplike)
        assert numnull.nplike is self._nplike and self._mask.nplike is self._nplike
        self._handle_error(
            self._nplike[
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
        nextcarry = ak._v2.index.Index64.empty(next_length, self._nplike)
        nextparents = ak._v2.index.Index64.empty(next_length, self._nplike)
        outindex = ak._v2.index.Index64.empty(mask_length, self._nplike)
        assert (
            nextcarry.nplike is self._nplike
            and nextparents.nplike is self._nplike
            and outindex.nplike is self._nplike
            and self._mask.nplike is self._nplike
            and parents.nplike is self._nplike
        )
        self._handle_error(
            self._nplike[
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
            nextshifts = ak._v2.index.Index64.empty(next_length, self._nplike)
            if shifts is None:
                assert (
                    nextshifts.nplike is self._nplike
                    and self._mask.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
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
                    nextshifts.nplike is self._nplike
                    and self._mask.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
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
            if out.is_ListType:
                out_content = out.content[out.starts[0] :]
            elif out.is_RegularType:
                out_content = out.content
            else:
                raise ak._v2._util.error(
                    ValueError(
                        "reduce_next with unbranching depth > negaxis is only "
                        "expected to return RegularArray or ListOffsetArray64; "
                        "instead, it returned " + out
                    )
                )

            outoffsets = ak._v2.index.Index64.empty(starts.length + 1, self._nplike)
            assert outoffsets.nplike is self._nplike
            self._handle_error(
                self._nplike[
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

            tmp = ak._v2.contents.IndexedOptionArray(
                outindex,
                out_content,
                None,
                None,
                self._nplike,
            ).simplify_optiontype()

            return ak._v2.contents.ListOffsetArray(
                outoffsets,
                tmp,
                None,
                None,
                self._nplike,
            )

    def _validity_error(self, path):
        if self._nplike.known_shape and self._content.length < self.mask.length:
            return f'at {path} ("{type(self)}"): len(content) < len(mask)'
        elif isinstance(
            self._content,
            (
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):
            return "{0} contains \"{1}\", the operation that made it might have forgotten to call 'simplify_optiontype()'"
        else:
            return self._content.validity_error(path + ".content")

    def _nbytes_part(self):
        result = self.mask._nbytes_part() + self.content._nbytes_part()
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def _pad_none(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.pad_none_axis0(target, clip)
        elif posaxis == depth + 1:
            mask = ak._v2.index.Index8(self.mask_as_bool(valid_when=False))
            index = ak._v2.index.Index64.empty(mask.length, self._nplike)
            assert index.nplike is self._nplike and self._mask.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1",
                    index.dtype.type,
                    self._mask.dtype.type,
                ](
                    index.data,
                    self._mask.data,
                    self._mask.length,
                )
            )
            next = self.project()._pad_none(target, posaxis, depth, clip)
            return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                index,
                next,
                None,
                self._parameters,
                self._nplike,
            ).simplify_optiontype()
        else:
            return ak._v2.contents.bytemaskedarray.ByteMaskedArray(
                self._mask,
                self._content._pad_none(target, posaxis, depth, clip),
                self._valid_when,
                None,
                self._parameters,
                self._nplike,
            )

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        this_validbytes = self.mask_as_bool(valid_when=True)

        return self._content._to_arrow(
            pyarrow,
            self,
            ak._v2._connect.pyarrow.and_validbytes(validbytes, this_validbytes),
            length,
            options,
        )

    def _to_numpy(self, allow_missing):
        return self.toIndexedOptionArray64()._to_numpy(allow_missing)

    def _completely_flatten(self, nplike, options):
        return self.project()._completely_flatten(nplike, options)

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if self._nplike.known_shape:
            content = self._content[0 : self._mask.length]
        else:
            content = self._content

        if options["return_array"]:

            def continuation():
                return ByteMaskedArray(
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
                    self._identifier,
                    self._parameters if options["keep_parameters"] else None,
                    self._nplike,
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
        if self._content.is_RecordType:
            next = self.toIndexedOptionArray64()
            content = next._content.packed()
            if content.length > self._mask.length:
                content = content[: self._mask.length]

            return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                next._index,
                content,
                next._identifier,
                next._parameters,
                self._nplike,
            )

        else:
            content = self._content.packed()
            if content.length > self._mask.length:
                content = content[: self._mask.length]

            return ByteMaskedArray(
                self._mask,
                content,
                self._valid_when,
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def _to_list(self, behavior, json_conversions):
        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        mask = self.mask_as_bool(valid_when=True, nplike=self.nplike)
        out = self._content._getitem_range(slice(0, len(mask)))._to_list(
            behavior, json_conversions
        )

        for i, isvalid in enumerate(mask):
            if not isvalid:
                out[i] = None

        return out

    def _to_nplike(self, nplike):
        content = self._content._to_nplike(nplike)
        mask = self._mask._to_nplike(nplike)
        return ByteMaskedArray(
            mask,
            content,
            valid_when=self._valid_when,
            identifier=self._identifier,
            parameters=self._parameters,
            nplike=nplike,
        )

    def _layout_equal(self, other, index_dtype=True, numpyarray=True):
        return (
            self.valid_when == other.valid_when
            and self.mask.layout_equal(other.mask, index_dtype, numpyarray)
            and self.content.layout_equal(other.content, index_dtype, numpyarray)
        )
