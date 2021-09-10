# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.index import Index
from awkward._v2.contents.content import Content, NestedIndexError
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

    def __len__(self):
        return len(self._index)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<IndexedOptionArray len="]
        out.append(repr(str(len(self))))
        out.append(">\n")
        out.append(self._index._repr(indent + "    ", "<index>", "</index>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</IndexedOptionArray>")
        out.append(post)
        return "".join(out)

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)):
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

    def nextcarry_outindex(self, nplike):
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

    def _getitem_next(self, head, tail, advanced):
        nplike = self.nplike  # noqa: F841

        if head == ():
            return self

        elif isinstance(head, (int, slice, ak._v2.index.Index64)):
            nexthead, nexttail = self._headtail(tail)

            numnull, nextcarry, outindex = self.nextcarry_outindex(nplike)

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
            raise NotImplementedError

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

        nextcarry = ak._v2.index.Index64.empty(len(self) - numnull.value, self.nplike)

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
            _, nextcarry, outindex = self.nextcarry_outindex(self.nplike)

            next = self._content._carry(nextcarry, False, NestedIndexError)
            out = next._localindex(posaxis, depth)
            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex,
                out,
                self._identifier,
                self._parameters,
            )
            return out2

    def _sort_next(
        self, parent, negaxis, starts, parents, outlength, shifts, ascending, stable
    ):
        if len(self._index) == 0:
            return self

        nplike = self.nplike
        branch, depth = self.branch_depth

        index_length = len(self._index)
        parents_length = len(parents)

        starts_length = len(starts)
        numnull = ak._v2.index.Index64.zeros(1, nplike)
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
        nextparents = ak._v2.index.Index64.zeros(next_length, nplike)
        nextcarry = ak._v2.index.Index64.zeros(next_length, nplike)
        outindex = ak._v2.index.Index64.zeros(index_length, nplike)
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

        # make_shifts = True if not branch and negaxis == depth else False
        # shifts_length = index_length if make_shifts else 0
        # nextshifts = ak._v2.index.Index64.zeros(shifts_length, nplike)

        next = self._content._carry(nextcarry, False, NestedIndexError)

        # nextshifts = ak._v2.index.Index64.empty(next_length, nplike)
        # if make_shifts:
        #     if shifts is None:
        #         self._handle_error(
        #             nplike[
        #                 "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64",
        #                 nextshifts.dtype.type,
        #                 self._index.dtype.type,
        #             ](
        #                 nextshifts.to(nplike),
        #                 self._index.to(nplike),
        #                 index_length,
        #             )
        #         )
        #     else:
        #         self._handle_error(
        #             nplike[
        #                 "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
        #                 nextshifts.dtype.type,
        #                 self._index.dtype.type,
        #                 shifts.dtype.type,
        #             ](
        #                 nextshifts.to(nplike),
        #                 self._index.to(nplike),
        #                 index_length,
        #                 shifts.to(nplike),
        #             )
        #         )

        inject_nones = True if not branch and negaxis != depth else False

        out = next._sort_next(
            self, negaxis, starts, nextparents, outlength, shifts, ascending, stable
        )

        # # FIXME: parents to starts
        # offsets_length = ak._v2.index.Index64.empty(1, nplike)
        # self._handle_error(
        #     nplike[
        #         "awkward_sorting_ranges_length",
        #         offsets_length.dtype.type,
        #         parents.dtype.type,
        #     ](
        #         offsets_length.to(nplike),
        #         parents.to(nplike),
        #         len(parents),
        #     )
        # )
        # offsets = ak._v2.index.Index64.zeros(offsets_length, nplike)
        # self._handle_error(
        #     nplike[
        #         "awkward_sorting_ranges",
        #         offsets.dtype.type,
        #         nextparents.dtype.type,
        #     ](
        #         offsets.to(nplike),
        #         offsets_length[0],
        #         nextparents.to(nplike),
        #         len(parents),
        #     )
        # )

        nextoutindex = ak._v2.index.Index64.zeros(parents_length, nplike)
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
                outoffsets = ak._v2.index.Index64.zeros(len(starts) + 1, nplike)
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
