# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

import awkward as ak
from awkward._v2.index import Index
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.bytemaskedform import ByteMaskedForm

np = ak.nplike.NumpyMetadata.instance()


class ByteMaskedArray(Content):
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

    @property
    def nonvirtual_nplike(self):
        return self._mask.nplike

    Form = ByteMaskedForm

    @property
    def form(self):
        return self.Form(
            self._mask.form,
            self._content.form,
            self._valid_when,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
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

    def mask_as_bool(self, valid_when=None):
        if valid_when is None:
            valid_when = self._valid_when
        if valid_when == self._valid_when:
            return self._mask.data != 0
        else:
            return self._mask.data != 1

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)):
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
        if len(slicestarts) != len(self):
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
        return out2._simplify_optiontype()

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
            return out2._simplify_optiontype()

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

    def _localindex(self, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        else:
            numnull = ak._v2.index.Index64.empty(1, self.nplike)
            nextcarry, outindex = self._nextcarry_outindex(numnull)

            next = self.content._carry(nextcarry, False, NestedIndexError)
            out = next._localindex(posaxis, depth)
            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex,
                out,
                self._identifier,
                self._parameters,
            )
            return out2._simplify_optiontype()

    def toIndexedOptionArray(self):
        nplike = self._mask.nplike
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
            )
        )
        return ak._v2.contents.IndexedOptionArray(
            index,
            self._content,
            self._identifier,
            self._parameters,
        )

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        return self.toIndexedOptionArray()._sort_next(
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
        posaxis = self._axis_wrap_if_negative(axis)
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
            return out2._simplify_optiontype()

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
        raise NotImplementedError

    ### FIXME: implementation should look like this:

    # int64_t numnull;
    # struct Error err1 = kernel::ByteMaskedArray_numnull(
    #   kernel::lib::cpu,   // DERIVE
    #   &numnull,
    #   mask_.data(),
    #   mask_.length(),
    #   valid_when_);
    # util::handle_error(err1, classname(), identities_.get());

    # Index64 nextparents(mask_.length() - numnull);
    # Index64 nextcarry(mask_.length() - numnull);
    # Index64 outindex(mask_.length());
    # struct Error err2 = kernel::ByteMaskedArray_reduce_next_64(
    #   kernel::lib::cpu,   // DERIVE
    #   nextcarry.data(),
    #   nextparents.data(),
    #   outindex.data(),
    #   mask_.data(),
    #   parents.data(),
    #   mask_.length(),
    #   valid_when_);
    # util::handle_error(err2, classname(), identities_.get());

    # std::pair<bool, int64_t> branchdepth = branch_depth();

    # bool make_shifts = (reducer.returns_positions()  &&
    #                     !branchdepth.first  && negaxis == branchdepth.second);

    # Index64 nextshifts(make_shifts ? mask_.length() - numnull : 0);
    # if (make_shifts) {
    #   if (shifts.length() == 0) {
    #     struct Error err3 =
    #         kernel::ByteMaskedArray_reduce_next_nonlocal_nextshifts_64(
    #       kernel::lib::cpu,   // DERIVE
    #       nextshifts.data(),
    #       mask_.data(),
    #       mask_.length(),
    #       valid_when_);
    #     util::handle_error(err3, classname(), identities_.get());
    #   }
    #   else {
    #     struct Error err3 =
    #         kernel::ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64(
    #       kernel::lib::cpu,   // DERIVE
    #       nextshifts.data(),
    #       mask_.data(),
    #       mask_.length(),
    #       valid_when_,
    #       shifts.data());
    #     util::handle_error(err3, classname(), identities_.get());
    #   }
    # }

    # ContentPtr next = content_.get()->carry(nextcarry, false);
    # if (RegularArray* raw = dynamic_cast<RegularArray*>(next.get())) {
    #   next = raw->toListOffsetArray64(true);
    # }

    # ContentPtr out = next.get()->reduce_next(reducer,
    #                                          negaxis,
    #                                          starts,
    #                                          nextshifts,
    #                                          nextparents,
    #                                          outlength,
    #                                          mask,
    #                                          keepdims);

    # if (!branchdepth.first  &&  negaxis == branchdepth.second) {
    #   return out;
    # }
    # else {
    #   if (RegularArray* raw =
    #       dynamic_cast<RegularArray*>(out.get())) {
    #     out = raw->toListOffsetArray64(true);
    #   }
    #   if (ListOffsetArray64* raw =
    #       dynamic_cast<ListOffsetArray64*>(out.get())) {
    #     Index64 outoffsets(starts.length() + 1);
    #     if (starts.length() > 0  &&  starts.getitem_at_nowrap(0) != 0) {
    #       throw std::runtime_error(
    #         std::string("reduce_next with unbranching depth > negaxis expects "
    #                     "a ListOffsetArray64 whose offsets start at zero")
    #         + FILENAME(__LINE__));
    #     }
    #     struct Error err4 = kernel::IndexedArray_reduce_next_fix_offsets_64(
    #       kernel::lib::cpu,   // DERIVE
    #       outoffsets.data(),
    #       starts.data(),
    #       starts.length(),
    #       outindex.length());
    #     util::handle_error(err4, classname(), identities_.get());

    #     return std::make_shared<ListOffsetArray64>(
    #       raw->identities(),
    #       raw->parameters(),
    #       outoffsets,
    #       IndexedOptionArray64(Identities::none(),
    #                            util::Parameters(),
    #                            outindex,
    #                            raw->content()).simplify_optiontype());
    #   }
    #   else {
    #     throw std::runtime_error(
    #       std::string("reduce_next with unbranching depth > negaxis is only "
    #                   "expected to return RegularArray or ListOffsetArray64; "
    #                   "instead, it returned ")
    #       + out.get()->classname() + FILENAME(__LINE__));
    #   }
    # }

    def _validityerror(self, path):
        if len(self.content) < len(self.mask):
            return 'at {0} ("{1}"): len(content) < len(mask)'.format(path, type(self))
        elif isinstance(
            self.content,
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
            return self.content.validityerror(path + ".content")

