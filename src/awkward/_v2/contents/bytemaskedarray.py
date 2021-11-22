# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json
import copy

import awkward as ak
from awkward._v2.index import Index
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.bytemaskedform import ByteMaskedForm
from awkward._v2.forms.form import _parameters_equal

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class ByteMaskedArray(Content):
    is_OptionType = True

    def __init__(self, mask, content, valid_when, identifier=None, parameters=None):
        if not (isinstance(mask, Index) and mask.dtype == np.dtype(np.int8)):
            raise TypeError(
                "{0} 'mask' must be an Index with dtype=int8, not {1}".format(
                    type(self).__name__, repr(mask)
                )
            )
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not isinstance(valid_when, bool):
            raise TypeError(
                "{0} 'valid_when' must be boolean, not {1}".format(
                    type(self).__name__, repr(valid_when)
                )
            )
        if not len(mask) <= len(content):
            raise ValueError(
                "{0} len(mask) ({1}) must be <= len(content) ({2})".format(
                    type(self).__name__, len(mask), len(content)
                )
            )

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._init(identifier, parameters)

    @property
    def mask(self):
        return self._mask

    @property
    def content(self):
        return self._content

    @property
    def valid_when(self):
        return self._valid_when

    @property
    def nplike(self):
        return self._mask.nplike

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
        container[key] = ak._v2._util.little_endian(self._mask.to(nplike))
        self._content._to_buffers(form.content, getkey, container, nplike)

    @property
    def typetracer(self):
        return ByteMaskedArray(
            ak._v2.index.Index(self._mask.to(ak._v2._typetracer.TypeTracer.instance())),
            self._content.typetracer,
            self._valid_when,
            self._typetracer_identifier(),
            self._parameters,
        )

    def __len__(self):
        return len(self._mask)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ByteMaskedArray valid_when="]
        out.append(repr(json.dumps(self._valid_when)))
        out.append(" len=")
        out.append(repr(str(len(self))))
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
        )

    def toIndexedOptionArray64(self):
        nplike = self.nplike
        index = ak._v2.index.Index64.empty(len(self._mask), nplike)
        self._handle_error(
            nplike[
                "awkward_ByteMaskedArray_toIndexedOptionArray",
                index.dtype.type,
                self._mask.dtype.type,
            ](
                index.to(nplike),
                self._mask.to(nplike),
                len(self._mask),
                self._valid_when,
            ),
        )

        return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            index, self._content, self._identifier, self._parameters
        )

    def mask_as_bool(self, valid_when=None, nplike=None):
        if valid_when is None:
            valid_when = self._valid_when
        if nplike is None:
            nplike = self._mask.nplike

        if valid_when == self._valid_when:
            return self._mask.to(nplike) != 0
        else:
            return self._mask.to(nplike) != 1

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)) and self.nplike.known_shape:
            raise NestedIndexError(self, where)
        if self._mask[where] == self._valid_when:
            return self._content._getitem_at(where)
        else:
            return None

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        return ByteMaskedArray(
            self._mask[start:stop],
            self._content._getitem_range(slice(start, stop)),
            self._valid_when,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return ByteMaskedArray(
            self._mask,
            self._content._getitem_field(where, only_fields),
            self._valid_when,
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return ByteMaskedArray(
            self._mask,
            self._content._getitem_fields(where, only_fields),
            self._valid_when,
            self._fields_identifier(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextmask = self._mask[carry.data]
        except IndexError as err:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, str(err))
            else:
                raise exception(str(err))

        return ByteMaskedArray(
            nextmask,
            self._content._carry(carry, allow_lazy, exception),
            self._valid_when,
            self._carry_identifier(carry, exception),
            self._parameters,
        )

    def _nextcarry_outindex(self, numnull):
        nplike = self.nplike
        self._handle_error(
            nplike[
                "awkward_ByteMaskedArray_numnull",
                numnull.dtype.type,
                self._mask.dtype.type,
            ](
                numnull.to(nplike),
                self._mask.to(nplike),
                len(self._mask),
                self._valid_when,
            )
        )
        nextcarry = ak._v2.index.Index64.empty(len(self) - numnull[0], nplike)
        outindex = ak._v2.index.Index64.empty(len(self), nplike)
        self._handle_error(
            nplike[
                "awkward_ByteMaskedArray_getitem_nextcarry_outindex",
                nextcarry.dtype.type,
                outindex.dtype.type,
                self._mask.dtype.type,
            ](
                nextcarry.to(nplike),
                outindex.to(nplike),
                self._mask.to(nplike),
                len(self._mask),
                self._valid_when,
            )
        )
        return nextcarry, outindex

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        nplike = self.nplike
        if len(slicestarts) != len(self) and nplike.known_shape:
            raise NestedIndexError(
                self,
                ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
                "cannot fit jagged slice with length {0} into {1} of size {2}".format(
                    len(slicestarts), type(self).__name__, len(self)
                ),
            )

        numnull = ak._v2.index.Index64.empty(1, nplike)
        nextcarry, outindex = self._nextcarry_outindex(numnull)

        reducedstarts = ak._v2.index.Index64.empty(len(self) - numnull[0], nplike)
        reducedstops = ak._v2.index.Index64.empty(len(self) - numnull[0], nplike)

        self._handle_error(
            nplike[
                "awkward_MaskedArray_getitem_next_jagged_project",
                outindex.dtype.type,
                slicestarts.dtype.type,
                slicestops.dtype.type,
                reducedstarts.dtype.type,
                reducedstops.dtype.type,
            ](
                outindex.to(nplike),
                slicestarts.to(nplike),
                slicestops.to(nplike),
                reducedstarts.to(nplike),
                reducedstops.to(nplike),
                len(self),
            )
        )

        next = self._content._carry(nextcarry, True, NestedIndexError)
        out = next._getitem_next_jagged(reducedstarts, reducedstops, slicecontent, tail)

        out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            outindex, out, self._identifier, self._parameters
        )
        return out2.simplify_optiontype()

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        return self._getitem_next_jagged_generic(
            slicestarts, slicestops, slicecontent, tail
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, (int, slice, ak._v2.index.Index64)):
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            numnull = ak._v2.index.Index64.empty(1, self.nplike)
            nextcarry, outindex = self._nextcarry_outindex(numnull)

            next = self._content._carry(nextcarry, True, NestedIndexError)
            out = next._getitem_next(head, tail, advanced)
            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex,
                out,
                self._identifier,
                self._parameters,
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

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            return self._getitem_next_jagged_generic(head, tail, advanced)

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def project(self, mask=None):
        mask_length = len(self._mask)
        numnull = ak._v2.index.Index64.zeros(1, self.nplike)

        if mask is not None:
            if mask_length != len(mask):
                raise ValueError(
                    "mask length ({0}) is not equal to {1} length ({2})".format(
                        len(mask), type(self).__name__, mask_length
                    )
                )

            nextmask = ak._v2.index.Index8.zeros(mask_length, self.nplike)
            self._handle_error(
                self.nplike[
                    "awkward_ByteMaskedArray_overlay_mask",
                    nextmask.dtype.type,
                    mask.dtype.type,
                    self._mask.dtype.type,
                ](
                    nextmask.to(self.nplike),
                    mask.to(self.nplike),
                    self._mask.to(self.nplike),
                    mask_length,
                    self._valid_when,
                )
            )
            valid_when = False
            next = ByteMaskedArray(
                nextmask, self._content, valid_when, self._identifier, self._parameters
            )
            return next.project()

        else:
            self._handle_error(
                self.nplike[
                    "awkward_ByteMaskedArray_numnull",
                    numnull.dtype.type,
                    self._mask.dtype.type,
                ](
                    numnull.to(self.nplike),
                    self._mask.to(self.nplike),
                    mask_length,
                    self._valid_when,
                )
            )
            nextcarry = ak._v2.index.Index64.zeros(
                mask_length - numnull[0], self.nplike
            )
            self._handle_error(
                self.nplike[
                    "awkward_ByteMaskedArray_getitem_nextcarry",
                    nextcarry.dtype.type,
                    self._mask.dtype.type,
                ](
                    nextcarry.to(self.nplike),
                    self._mask.to(self.nplike),
                    mask_length,
                    self._valid_when,
                )
            )

            return self._content._carry(nextcarry, False, NestedIndexError)

    def simplify_optiontype(self):
        if isinstance(
            self._content,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskeddarray.BitMaskedArray,
                ak._v2.contents.unmaskeddarray.UnmaskedArray,
            ),
        ):
            return self.toIndexedOptionArray64.simplify_optiontype
        else:
            return self

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            out = ak._v2.index.Index64.empty(1, self.nplike)
            out[0] = len(self)
            return ak._v2.contents.numpyarray.NumpyArray(out)[0]
        else:
            numnull = ak._v2.index.Index64.empty(1, self.nplike, dtype=np.int64)
            nextcarry, outindex = self._nextcarry_outindex(numnull)

            next = self._content._carry(nextcarry, False, NestedIndexError)
            out = next.num(posaxis, depth)

            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex, out, parameters=self.parameters
            )
            return out2.simplify_optiontype()

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            raise np.AxisError(self, "axis=0 not allowed for flatten")
        else:
            numnull = ak._v2.index.Index64.empty(1, self.nplike)
            nextcarry, outindex = self._nextcarry_outindex(numnull)

            next = self._content._carry(nextcarry, False, NestedIndexError)

            offsets, flattened = next._offsets_and_flattened(posaxis, depth)

            if len(offsets) == 0:
                return (
                    offsets,
                    ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                        outindex, flattened, None, self._parameters
                    ),
                )

            else:
                outoffsets = ak._v2.index.Index64.empty(
                    len(offsets) + numnull[0], self.nplike, dtype=np.int64
                )

                self._handle_error(
                    self.nplike[
                        "awkward_IndexedArray_flatten_none2empty",
                        outoffsets.dtype.type,
                        outindex.dtype.type,
                        offsets.dtype.type,
                    ](
                        outoffsets.to(self.nplike),
                        outindex.to(self.nplike),
                        len(outindex),
                        offsets.to(self.nplike),
                        len(offsets),
                    )
                )
                return (outoffsets, flattened)

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
            self._content.mergeable(other.content, mergebool)

        else:
            return self._content.mergeable(other, mergebool)

    def _reverse_merge(self, other):
        return self.toIndexedOptionArray64()._reverse_merge(other)

    def mergemany(self, others):
        if len(others) == 0:
            return self
        return self.toIndexedOptionArray64().mergemany(others)

    def fillna(self, value):
        return self.toIndexedOptionArray64().fillna(value)

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        else:
            numnull = ak._v2.index.Index64.empty(1, self.nplike)
            nextcarry, outindex = self._nextcarry_outindex(numnull)

            next = self._content._carry(nextcarry, False, NestedIndexError)
            out = next._localindex(posaxis, depth)
            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex,
                out,
                self._identifier,
                self._parameters,
            )
            return out2.simplify_optiontype()

    def numbers_to_type(self, name):
        return ak._v2.contents.bytemaskedarray.ByteMaskedArray(
            self._mask,
            self._content.numbers_to_type(name),
            self._valid_when,
            self._identifier,
            self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if len(self._mask) == 0:
            return True
        return self.toIndexedOptionArray64()._is_unique(
            negaxis, starts, parents, outlength
        )

    def _unique(self, negaxis, starts, parents, outlength):
        if len(self._mask) == 0:
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
        if len(self._mask) == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))

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
            raise ValueError("in combinations, 'n' must be at least 1")
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            numnull = ak._v2.index.Index64.empty(1, self.nplike, dtype=np.int64)
            nextcarry, outindex = self._nextcarry_outindex(numnull)

            next = self._content._carry(nextcarry, True, NestedIndexError)
            out = next._combinations(
                n, replacement, recordlookup, parameters, posaxis, depth
            )
            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex, out, parameters=parameters
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
    ):
        nplike = self.nplike

        mask_length = len(self._mask)

        numnull = ak._v2.index.Index64.zeros(1, nplike)
        self._handle_error(
            nplike[
                "awkward_ByteMaskedArray_numnull",
                numnull.dtype.type,
                self._mask.dtype.type,
            ](
                numnull.to(nplike),
                self._mask.to(nplike),
                mask_length,
                self._valid_when,
            )
        )

        next_length = mask_length - numnull[0]
        nextcarry = ak._v2.index.Index64.zeros(next_length, nplike)
        nextparents = ak._v2.index.Index64.zeros(next_length, nplike)
        outindex = ak._v2.index.Index64.zeros(mask_length, nplike)
        self._handle_error(
            nplike[
                "awkward_ByteMaskedArray_reduce_next_64",
                nextcarry.dtype.type,
                nextparents.dtype.type,
                outindex.dtype.type,
                self._mask.dtype.type,
                parents.dtype.type,
            ](
                nextcarry.to(nplike),
                nextparents.to(nplike),
                outindex.to(nplike),
                self._mask.to(nplike),
                parents.to(nplike),
                mask_length,
                self._valid_when,
            )
        )

        branch, depth = self.branch_depth

        if reducer.needs_position and (not branch and negaxis == depth):
            nextshifts = ak._v2.index.Index64.zeros(next_length, nplike)
            if shifts is None:
                self._handle_error(
                    nplike[
                        "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64",
                        nextshifts.dtype.type,
                        self._mask.dtype.type,
                    ](
                        nextshifts.to(nplike),
                        self._mask.to(nplike),
                        mask_length,
                        self._valid_when,
                    )
                )
            else:
                self._handle_error(
                    nplike[
                        "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
                        nextshifts.dtype.type,
                        self._mask.dtype.type,
                        shifts.dtype.type,
                    ](
                        nextshifts.to(nplike),
                        self._mask.to(nplike),
                        mask_length,
                        self._valid_when,
                        shifts.to(nplike),
                    )
                )
        else:
            nextshifts = None

        next = self._content._carry(nextcarry, False, NestedIndexError)
        if isinstance(next, ak._v2.contents.RegularArray):
            next = next.toListOffsetArray64(True)

        out = next._reduce_next(
            reducer,
            negaxis,
            starts,
            nextshifts,
            nextparents,
            outlength,
            mask,
            keepdims,
        )

        if not branch and negaxis == depth:
            return out
        else:
            if isinstance(out, ak._v2.contents.RegularArray):
                out = out.toListOffsetArray64(True)

            if isinstance(out, ak._v2.contents.ListOffsetArray):
                outoffsets = ak._v2.index.Index64.zeros(len(starts) + 1, nplike)
                self._handle_error(
                    nplike[
                        "awkward_IndexedArray_reduce_next_fix_offsets_64",
                        outoffsets.dtype.type,
                        starts.dtype.type,
                    ](
                        outoffsets.to(nplike),
                        starts.to(nplike),
                        len(starts),
                        len(outindex),
                    )
                )

                tmp = ak._v2.contents.IndexedOptionArray(
                    outindex,
                    out.content,
                    None,
                    None,
                ).simplify_optiontype()

                return ak._v2.contents.ListOffsetArray(
                    outoffsets,
                    tmp,
                    None,
                    None,
                )

            else:
                raise ValueError(
                    "reduce_next with unbranching depth > negaxis is only "
                    "expected to return RegularArray or ListOffsetArray64; "
                    "instead, it returned " + out
                )

    def _validityerror(self, path):
        if len(self._content) < len(self.mask):
            return 'at {0} ("{1}"): len(content) < len(mask)'.format(path, type(self))
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
            return self._content.validityerror(path + ".content")

    def _nbytes_part(self):
        result = self.mask._nbytes_part() + self.content._nbytes_part()
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def bytemask(self):
        if not self._valid_when:
            return self._mask
        else:
            out = ak._v2.index.Index64.empty(len(self), self.nplike)
            self._handle_error(
                self.nplike[
                    "awkward_ByteMaskedArray_mask",
                    out.dtype.type,
                    self._mask.dtype.type,
                ](
                    out.to(self.nplike),
                    self._mask.to(self.nplike),
                    len(self._mask),
                    self._valid_when,
                )
            )
            return out

    def _rpad(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.rpad_axis0(target, clip)
        elif posaxis == depth + 1:
            mask = self.bytemask()
            index = ak._v2.index.Index64.empty(len(mask), self.nplike)
            self._handle_error(
                self.nplike[
                    "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1",
                    index.dtype.type,
                    self._mask.dtype.type,
                ](
                    index.to(self.nplike),
                    self._mask.to(self.nplike),
                    len(self._mask),
                )
            )
            next = self.project()._rpad(target, posaxis, depth, clip)
            return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                index,
                next.simplify_optiontype(),
                None,
                self._parameters,
            )
        else:
            return ak._v2.contents.bytemaskedarray.ByteMaskedArray(
                self._mask,
                self._content._rpad(target, posaxis, depth, clip),
                self._valid_when,
                None,
                self._parameters,
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
        self, action, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                return ByteMaskedArray(
                    self._mask,
                    self._content._recursively_apply(
                        action,
                        depth,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    ),
                    self._valid_when,
                    self._identifier,
                    self._parameters if options["keep_parameters"] else None,
                )

        else:

            def continuation():
                self._content._recursively_apply(
                    action,
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
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise AssertionError(result)

    def packed(self):
        if self._content.is_RecordType:
            next = self.toIndexedOptionArray64()
            content = next._content.packed()
            if len(content) > len(self._mask):
                content = content[: len(self._mask)]

            return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                next._index,
                content,
                next._identifier,
                next._parameters,
            )

        else:
            content = self._content.packed()
            if len(content) > len(self._mask):
                content = content[: len(self._mask)]

            return ByteMaskedArray(
                self._mask,
                content,
                self._valid_when,
                self._identifier,
                self._parameters,
            )

    def _to_list(self, behavior):
        out = self._to_list_custom(behavior)
        if out is not None:
            return out

        mask = self.mask_as_bool(valid_when=True, nplike=numpy)
        content = self._content._to_list(behavior)
        out = [None] * len(self._mask)
        for i, isvalid in enumerate(mask):
            if isvalid:
                out[i] = content[i]
        return out
