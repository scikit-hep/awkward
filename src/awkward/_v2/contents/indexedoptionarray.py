# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.index import Index
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.indexedoptionform import IndexedOptionForm

np = ak.nplike.NumpyMetadata.instance()


class IndexedOptionArray(Content):
    def __init__(self, index, content, identifier=None, parameters=None):
        if not (
            isinstance(index, Index)
            and index.dtype
            in (
                np.dtype(np.int32),
                np.dtype(np.int64),
            )
        ):
            raise TypeError(
                "{0} 'index' must be an Index with dtype in (int32, uint32, int64), "
                "not {1}".format(type(self).__name__, repr(index))
            )
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )

        self._index = index
        self._content = content
        self._init(identifier, parameters)

    @property
    def index(self):
        return self._index

    @property
    def content(self):
        return self._content

    @property
    def nplike(self):
        return self._index.nplike

    @property
    def nonvirtual_nplike(self):
        return self._index.nplike

    Form = IndexedOptionForm

    @property
    def form(self):
        return self.Form(
            self._index.form,
            self._content.form,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return IndexedOptionArray(
            ak._v2.index.Index(self._index.to(tt)),
            self._content.typetracer,
            self._typetracer_identifier(),
            self._parameters,
        )

    def __len__(self):
        return len(self._index)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<IndexedOptionArray len="]
        out.append(repr(str(len(self))))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._index._repr(indent + "    ", "<index>", "</index>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</IndexedOptionArray>")
        out.append(post)
        return "".join(out)

    def toIndexedOptionArray64(self):
        if self._index.dtype == np.dtype(np.int64):
            return self
        else:
            return IndexedOptionArray(
                self._index.astype(np.int64),
                self._content,
                identifier=self._identifier,
                parameters=self._parameters,
            )

    def mask_as_bool(self, valid_when=True):
        if valid_when:
            return self._index.data >= 0
        else:
            return self._index.data < 0

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)) and self.nplike.known_shape:
            raise NestedIndexError(self, where)
        if self._index[where] < 0:
            return None
        else:
            return self._content._getitem_at(self._index[where])

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        return IndexedOptionArray(
            self._index[start:stop],
            self._content,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return IndexedOptionArray(
            self._index,
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return IndexedOptionArray(
            self._index,
            self._content._getitem_fields(where, only_fields),
            self._fields_identifier(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextindex = self._index[carry.data]
        except IndexError as err:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, str(err))
            else:
                raise exception(str(err))

        return IndexedOptionArray(
            nextindex,
            self._content,
            self._carry_identifier(carry, exception),
            self._parameters,
        )

    def _nextcarry_outindex(self, nplike):
        numnull = ak._v2.index.Index64.empty(1, nplike)

        self._handle_error(
            nplike[
                "awkward_IndexedArray_numnull",
                numnull.dtype.type,
                self._index.dtype.type,
            ](
                numnull.to(nplike),
                self._index.to(nplike),
                len(self._index),
            )
        )
        nextcarry = ak._v2.index.Index64.empty(len(self._index) - numnull[0], nplike)
        outindex = ak._v2.index.Index.empty(len(self._index), nplike, self._index.dtype)

        self._handle_error(
            nplike[
                "awkward_IndexedArray_getitem_nextcarry_outindex",
                nextcarry.dtype.type,
                outindex.dtype.type,
                self._index.dtype.type,
            ](
                nextcarry.to(nplike),
                outindex.to(nplike),
                self._index.to(nplike),
                len(self._index),
                len(self._content),
            )
        )

        return numnull[0], nextcarry, outindex

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

        numnull, nextcarry, outindex = self._nextcarry_outindex(nplike)

        reducedstarts = ak._v2.index.Index64.empty(len(self) - numnull, nplike)
        reducedstops = ak._v2.index.Index64.empty(len(self) - numnull, nplike)
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

            numnull, nextcarry, outindex = self._nextcarry_outindex(self.nplike)

            next = self._content._carry(nextcarry, True, NestedIndexError)
            out = next._getitem_next(head, tail, advanced)
            out2 = IndexedOptionArray(outindex, out, self._identifier, self._parameters)
            return out2._simplify_optiontype()

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list) and ak._util.isstr(head[0]):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            raise NotImplementedError

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def project(self):
        numnull = ak._v2.index.Index64.empty(1, self.nplike)

        self._handle_error(
            self.nplike[
                "awkward_IndexedArray_numnull",
                numnull.dtype.type,
                self._index.dtype.type,
            ](numnull.to(self.nplike), self._index.to(self.nplike), len(self._index))
        )

        nextcarry = ak._v2.index.Index64.empty(len(self) - numnull[0], self.nplike)

        self._handle_error(
            self.nplike[
                "awkward_IndexedArray_flatten_nextcarry",
                nextcarry.dtype.type,
                self._index.dtype.type,
            ](
                nextcarry.to(self.nplike),
                self._index.to(self.nplike),
                len(self._index),
                len(self._content),
            )
        )

        return self._content._carry(nextcarry, False, NestedIndexError)

    def _localindex(self, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        else:
            _, nextcarry, outindex = self._nextcarry_outindex(self.nplike)

            next = self._content._carry(nextcarry, False, NestedIndexError)
            out = next._localindex(posaxis, depth)
            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex,
                out,
                self._identifier,
                self._parameters,
            )
            return out2

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
        if len(self._index) == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))

        nplike = self.nplike
        branch, depth = self.branch_depth

        index_length = len(self._index)
        parents_length = len(parents)

        starts_length = len(starts)
        numnull = ak._v2.index.Index64.empty(1, nplike)
        self._handle_error(
            nplike[
                "awkward_IndexedArray_numnull",
                numnull.dtype.type,
                self._index.dtype.type,
            ](
                numnull.to(nplike),
                self._index.to(nplike),
                index_length,
            )
        )

        next_length = index_length - numnull[0]
        nextparents = ak._v2.index.Index64.empty(next_length, nplike)
        nextcarry = ak._v2.index.Index64.empty(next_length, nplike)
        outindex = ak._v2.index.Index64.empty(index_length, nplike)
        self._handle_error(
            nplike[
                "awkward_IndexedArray_reduce_next_64",
                nextcarry.dtype.type,
                nextparents.dtype.type,
                outindex.dtype.type,
                self._index.dtype.type,
                parents.dtype.type,
            ](
                nextcarry.to(nplike),
                nextparents.to(nplike),
                outindex.to(nplike),
                self._index.to(nplike),
                parents.to(nplike),
                index_length,
            )
        )

        next = self._content._carry(nextcarry, False, NestedIndexError)

        inject_nones = True if not branch and negaxis != depth else False

        if (not branch) and negaxis == depth:
            nextshifts = ak._v2.index.Index64.empty(
                next_length,
                nplike,
            )
            if shifts is None:
                self._handle_error(
                    nplike[
                        "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64",
                        nextshifts.dtype.type,
                        self._index.dtype.type,
                    ](
                        nextshifts.to(nplike),
                        self._index.to(nplike),
                        len(self._index),
                    )
                )
            else:
                self._handle_error(
                    nplike[
                        "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
                        nextshifts.dtype.type,
                        self._index.dtype.type,
                        shifts.dtype.type,
                    ](
                        nextshifts.to(nplike),
                        self._index.to(nplike),
                        len(self._index),
                        shifts.to(nplike),
                    )
                )
        else:
            nextshifts = None

        inject_nones = (
            True if (numnull[0] > 0 and not branch and negaxis != depth) else False
        )

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

        nulls_merged = False
        nulls_index = ak._v2.index.Index64.empty(numnull[0], nplike)
        self._handle_error(
            nplike[
                "awkward_IndexedArray_index_of_nulls",
                nulls_index.dtype.type,
                self._index.dtype.type,
                parents.dtype.type,
                starts.dtype.type,
            ](
                nulls_index.to(nplike),
                self._index.to(nplike),
                len(self._index),
                parents.to(nplike),
                starts.to(nplike),
            )
        )

        ind = ak._v2.contents.NumpyArray(nulls_index)

        # # FIXME: work around for if mergeable: test it on two 1D arrays
        if isinstance(out, ak._v2.contents.NumpyArray):
            full_index = ak._v2.index.Index64.empty(len(out) + len(ind), nplike)
            self._handle_error(
                nplike[
                    "awkward_NumpyArray_fill",
                    full_index.dtype.type,
                    out._data.dtype.type,
                ](
                    full_index.to(nplike),
                    0,
                    out._data,
                    len(out),
                )
            )
            self._handle_error(
                nplike[
                    "awkward_NumpyArray_fill",
                    full_index.dtype.type,
                    ind._data.dtype.type,
                ](
                    full_index.to(nplike),
                    len(out),
                    ind._data,
                    len(ind),
                )
            )
            nulls_merged = True
            out = ak._v2.contents.NumpyArray(full_index, nplike=self.nplike)
        #
        # # if self._mergeable(ind, True):
        # #     out.merge(ind)
        # #     nulls_merged = True

        nextoutindex = ak._v2.index.Index64.empty(parents_length, nplike)
        self._handle_error(
            nplike[
                "awkward_IndexedArray_local_preparenext_64",
                nextoutindex.dtype.type,
                starts.dtype.type,
                parents.dtype.type,
                nextparents.dtype.type,
            ](
                nextoutindex.to(nplike),
                starts.to(nplike),
                parents.to(nplike),
                parents_length,
                nextparents.to(nplike),
                next_length,
            )
        )

        if nulls_merged:
            self._handle_error(
                nplike["awkward_Index_nones_as_index", nextoutindex.dtype.type](
                    nextoutindex.to(nplike),
                    len(nextoutindex),
                )
            )

        out = ak._v2.contents.IndexedOptionArray(
            nextoutindex,
            out,
            None,
            self._parameters,
        )._simplify_optiontype()

        if inject_nones:
            out = ak._v2.contents.RegularArray(
                out,
                parents_length,
                0,
                None,
                self._parameters,
            )

        if (not branch) and negaxis == depth:
            return out
        else:
            if isinstance(out, ak._v2.contents.RegularArray):
                out = out.toListOffsetArray64(True)
            if isinstance(out, ak._v2.contents.ListOffsetArray):
                if len(starts) > 0 and starts[0] != 0:
                    raise RuntimeError(
                        "sort_next with unbranching depth > negaxis expects a "
                        "ListOffsetArray64 whose offsets start at zero"
                    )
                outoffsets = ak._v2.index.Index64.empty(len(starts) + 1, nplike)
                self._handle_error(
                    nplike[
                        "awkward_IndexedArray_reduce_next_fix_offsets_64",
                        outoffsets.dtype.type,
                        starts.dtype.type,
                    ](
                        outoffsets.to(nplike),
                        starts.to(nplike),
                        starts_length,
                        len(outindex),
                    )
                )

                tmp = ak._v2.contents.IndexedOptionArray(
                    outindex,
                    out._content,
                    None,
                    self._parameters,
                )._simplify_optiontype()

                if inject_nones:
                    return tmp

                return ak._v2.contents.ListOffsetArray(
                    outoffsets,
                    tmp,
                    None,
                    self._parameters,
                )

            if isinstance(out, ak._v2.contents.IndexedOptionArray):
                return out
            else:
                raise RuntimeError(
                    "argsort_next with unbranching depth > negaxis is only "
                    "expected to return RegularArray or ListOffsetArray or "
                    "IndexedOptionArray; "
                    "instead, it returned " + out
                )

        return out

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        nplike = self.nplike
        branch, depth = self.branch_depth

        index_length = len(self._index)
        parents_length = len(parents)

        starts_length = len(starts)
        numnull = ak._v2.index.Index64.empty(1, nplike)
        self._handle_error(
            nplike[
                "awkward_IndexedArray_numnull",
                numnull.dtype.type,
                self._index.dtype.type,
            ](
                numnull.to(nplike),
                self._index.to(nplike),
                index_length,
            )
        )

        next_length = index_length - numnull[0]
        nextparents = ak._v2.index.Index64.empty(next_length, nplike)
        nextcarry = ak._v2.index.Index64.empty(next_length, nplike)
        outindex = ak._v2.index.Index64.empty(index_length, nplike)
        self._handle_error(
            nplike[
                "awkward_IndexedArray_reduce_next_64",
                nextcarry.dtype.type,
                nextparents.dtype.type,
                outindex.dtype.type,
                self._index.dtype.type,
                parents.dtype.type,
            ](
                nextcarry.to(nplike),
                nextparents.to(nplike),
                outindex.to(nplike),
                self._index.to(nplike),
                parents.to(nplike),
                index_length,
            )
        )

        next = self._content._carry(nextcarry, False, NestedIndexError)

        inject_nones = True if not branch and negaxis != depth else False

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

        nextoutindex = ak._v2.index.Index64.empty(parents_length, nplike)
        self._handle_error(
            nplike[
                "awkward_IndexedArray_local_preparenext_64",
                nextoutindex.dtype.type,
                starts.dtype.type,
                parents.dtype.type,
                nextparents.dtype.type,
            ](
                nextoutindex.to(nplike),
                starts.to(nplike),
                parents.to(nplike),
                parents_length,
                nextparents.to(nplike),
                next_length,
            )
        )

        out = ak._v2.contents.IndexedOptionArray(
            nextoutindex,
            out,
            None,
            self._parameters,
        )._simplify_optiontype()

        if inject_nones:
            out = ak._v2.contents.RegularArray(
                out,
                parents_length,
                0,
                None,
                self._parameters,
            )

        if not branch and negaxis == depth:
            return out
        else:
            if isinstance(out, ak._v2.contents.RegularArray):
                out = out.toListOffsetArray64(True)
            if isinstance(out, ak._v2.contents.ListOffsetArray):
                if len(starts) > 0 and starts[0] != 0:
                    raise RuntimeError(
                        "sort_next with unbranching depth > negaxis expects a "
                        "ListOffsetArray64 whose offsets start at zero"
                    )
                outoffsets = ak._v2.index.Index64.empty(len(starts) + 1, nplike)
                self._handle_error(
                    nplike[
                        "awkward_IndexedArray_reduce_next_fix_offsets_64",
                        outoffsets.dtype.type,
                        starts.dtype.type,
                    ](
                        outoffsets.to(nplike),
                        starts.to(nplike),
                        starts_length,
                        len(outindex),
                    )
                )

                tmp = ak._v2.contents.IndexedOptionArray(
                    outindex,
                    out._content,
                    None,
                    self._parameters,
                )._simplify_optiontype()

                if inject_nones:
                    return tmp

                return ak._v2.contents.ListOffsetArray(
                    outoffsets,
                    tmp,
                    None,
                    self._parameters,
                )

            if isinstance(out, ak._v2.contents.IndexedOptionArray):
                return out
            else:
                raise RuntimeError(
                    "sort_next with unbranching depth > negaxis is only "
                    "expected to return RegularArray or ListOffsetArray or "
                    "IndexedOptionArray; "
                    "instead, it returned " + out
                )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            _, nextcarry, outindex = self._nextcarry_outindex(self.nplike)
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
        nplike = self.nplike
        branch, depth = self.branch_depth

        index_length = len(self._index)

        numnull = ak._v2.index.Index64.empty(1, nplike)
        self._handle_error(
            nplike[
                "awkward_IndexedArray_numnull",
                numnull.dtype.type,
                self._index.dtype.type,
            ](
                numnull.to(nplike),
                self._index.to(nplike),
                index_length,
            )
        )

        next_length = index_length - numnull[0]
        nextcarry = ak._v2.index.Index64.empty(next_length, nplike)
        nextparents = ak._v2.index.Index64.empty(next_length, nplike)
        outindex = ak._v2.index.Index64.empty(index_length, nplike)
        self._handle_error(
            nplike[
                "awkward_IndexedArray_reduce_next_64",
                nextcarry.dtype.type,
                nextparents.dtype.type,
                outindex.dtype.type,
                self._index.dtype.type,
                parents.dtype.type,
            ](
                nextcarry.to(nplike),
                nextparents.to(nplike),
                outindex.to(nplike),
                self._index.to(nplike),
                parents.to(nplike),
                index_length,
            )
        )

        if reducer.needs_position and (not branch and negaxis == depth):
            nextshifts = ak._v2.index.Index64.empty(next_length, nplike)
            if shifts is None:
                self._handle_error(
                    nplike[
                        "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64",
                        nextshifts.dtype.type,
                        self.index.dtype.type,
                    ](
                        nextshifts.to(nplike),
                        self.index.to(nplike),
                        len(self.index),
                    )
                )
            else:
                self._handle_error(
                    nplike[
                        "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
                        nextshifts.dtype.type,
                        self.index.dtype.type,
                        shifts.dtype.type,
                    ](
                        nextshifts.to(nplike),
                        self.index.to(nplike),
                        len(self.index),
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
                outoffsets = ak._v2.index.Index64.empty(len(starts) + 1, nplike)
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
                )._simplify_optiontype()

                return ak._v2.contents.ListOffsetArray(
                    outoffsets,
                    tmp,
                    None,
                    None,
                )

        return out

    def _validityerror(self, path):
        error = self.nplike["awkward_IndexedArray_validity", self.index.dtype.type](
            self.index.to(self.nplike), len(self.index), len(self.content), True
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
