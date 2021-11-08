# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import copy

import awkward as ak
from awkward._v2.index import Index
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.indexedoptionform import IndexedOptionForm
from awkward._v2.forms.form import _parameters_equal

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class IndexedOptionArray(Content):
    is_OptionType = True
    is_IndexedType = True

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

    def merge_parameters(self, parameters):
        return IndexedOptionArray(
            self._index,
            self._content,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
        )

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

    def mask_as_bool(self, valid_when=True, nplike=None):
        if nplike is None:
            nplike = self._content.nplike

        if valid_when:
            return self._index.to(nplike) >= 0
        else:
            return self._index.to(nplike) < 0

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

            numnull, nextcarry, outindex = self._nextcarry_outindex(self.nplike)

            next = self._content._carry(nextcarry, True, NestedIndexError)
            out = next._getitem_next(head, tail, advanced)
            out2 = IndexedOptionArray(outindex, out, self._identifier, self._parameters)
            return out2.simplify_optiontype()

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

    def project(self, mask=None):
        if mask is not None:
            if len(self._index) != len(mask):
                raise ValueError(
                    "mask length ({0}) is not equal to {1} length ({2})".format(
                        mask.length(), type(self).__name__, len(self._index)
                    )
                )
            nextindex = ak._v2.index.Index64.empty(len(self._index), self.nplike)
            self._handle_error(
                self.nplike[
                    "awkward_IndexedArray_overlay_mask",
                    nextindex.dtype.type,
                    mask.dtype.type,
                    self._index.dtype.type,
                ](
                    nextindex.to(self.nplike),
                    mask.to(self.nplike),
                    self._index.to(self.nplike),
                    len(self._index),
                )
            )
            next = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                nextindex, self.content, self.identifier, self.parameters
            )
            return next.project()
        else:
            numnull = ak._v2.index.Index64.empty(1, self.nplike)

            self._handle_error(
                self.nplike[
                    "awkward_IndexedArray_numnull",
                    numnull.dtype.type,
                    self._index.dtype.type,
                ](
                    numnull.to(self.nplike),
                    self._index.to(self.nplike),
                    len(self._index),
                )
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

    def simplify_optiontype(self):
        if isinstance(
            self.content,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):

            if isinstance(
                self.content,
                (
                    ak._v2.contents.indexedarray.IndexedArray,
                    ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ),
            ):
                inner = self.content.index
                result = ak._v2.index.Index64.zeros(len(self.index), self.nplike)
            elif isinstance(
                self.content,
                (
                    ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                    ak._v2.contents.bitmaskedarray.BitMaskedArray,
                    ak._v2.contents.unmaskedarray.UnmaskedArray,
                ),
            ):
                rawcontent = self.content.toIndexedOptionArray64()
                inner = rawcontent.index
                result = ak._v2.index.Index64.empty(len(self.index), self.nplike)
            self._handle_error(
                self.nplike[
                    "awkward_IndexedArray_simplify",
                    result.dtype.type,
                    self._index.dtype.type,
                    inner.dtype.type,
                ](
                    result.to(self.nplike),
                    self._index.to(self.nplike),
                    len(self._index),
                    inner.to(self.nplike),
                    len(inner),
                )
            )
            return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                result, self.content.content, self.identifier, self.parameters
            )

        else:
            return self

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
            self.content.mergeable(other.content, mergebool)

        else:
            return self.content.mergeable(other, mergebool)

    def _merging_strategy(self, others):
        if len(others) == 0:
            raise ValueError(
                "to merge this array with 'others', at least one other must be provided"
            )

        head = [self]
        tail = []

        i = 0
        while i < len(others):
            other = others[i]
            if isinstance(other, ak._v2.contents.unionarray.UnionArray):
                break
            else:
                head.append(other)
            i = i + 1

        while i < len(others):
            tail.append(others[i])
            i = i + 1

        return (head, tail)

    def _reverse_merge(self, other):
        theirlength = len(other)
        mylength = len(self)
        index = ak._v2.index.Index64.empty((theirlength + mylength), self.nplike)

        content = other.merge(self.content)

        self._handle_error(
            self.nplike["awkward_IndexedArray_fill_count", index.dtype.type](
                index.to(self.nplike),
                0,
                theirlength,
                0,
            )
        )
        reinterpreted_index = ak._v2.index.Index(self.nplike.asarray(self.index.data))

        self._handle_error(
            self.nplike[
                "awkward_IndexedArray_fill",
                index.dtype.type,
                reinterpreted_index.dtype.type,
            ](
                index.to(self.nplike),
                theirlength,
                reinterpreted_index.to(self.nplike),
                mylength,
                theirlength,
            )
        )
        parameters = dict(self.parameters.items() & other.parameters.items())

        return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            index, content, None, parameters
        )

    def mergemany(self, others):
        if len(others) == 0:
            return self

        head, tail = self._merging_strategy(others)

        total_length = 0
        for array in head:
            total_length += len(array)

        contents = []
        contentlength_so_far = 0
        length_so_far = 0
        nextindex = ak._v2.index.Index64.empty(total_length, self.nplike)
        parameters = {}

        for array in head:
            parameters = dict(self.parameters.items() & array.parameters.items())

            if isinstance(
                array,
                (
                    ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                    ak._v2.contents.bitmaskedarray.BitMaskedArray,
                    ak._v2.contents.unmaskedarray.UnmaskedArray,
                ),
            ):
                array = array.toIndexedOptionArray64()

            if isinstance(array, ak._v2.contents.indexedoptionarray.IndexedOptionArray):
                contents.append(array.content)
                array_index = array.index
                self._handle_error(
                    self.nplike[
                        "awkward_IndexedArray_fill",
                        nextindex.dtype.type,
                        array_index.dtype.type,
                    ](
                        nextindex.to(self.nplike),
                        length_so_far,
                        array_index.to(self.nplike),
                        len(array),
                        contentlength_so_far,
                    )
                )
                contentlength_so_far += len(array.content)
                length_so_far += len(array)

            elif isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                pass
            else:
                contents.append(array)
                self._handle_error(
                    self.nplike[
                        "awkward_IndexedArray_fill_count",
                        nextindex.dtype.type,
                    ](
                        nextindex.to(self.nplike),
                        length_so_far,
                        len(array),
                        contentlength_so_far,
                    )
                )
                contentlength_so_far += len(array)
                length_so_far += len(array)

        tail_contents = contents[1:]
        nextcontent = contents[0].mergemany(tail_contents)
        next = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            nextindex, nextcontent, None, parameters
        )

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed.mergemany(tail[1:])

        raise NotImplementedError(
            "not implemented: " + type(self).__name__ + " ::mergemany"
        )

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
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
        ).simplify_optiontype()

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
                ).simplify_optiontype()

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
        ).simplify_optiontype()

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
                ).simplify_optiontype()

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
        posaxis = self.axis_wrap_if_negative(axis)
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
                ).simplify_optiontype()

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

    def bytemask(self):
        out = ak._v2.index.Index8.empty(len(self.index), self.nplike)
        self._handle_error(
            self.nplike[
                "awkward_IndexedArray_mask", out.dtype.type, self._index.dtype.type
            ](out.to(self.nplike), self._index.to(self.nplike), len(self._index))
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
                    mask.dtype.type,
                ](index.to(self.nplike), mask.to(self.nplike), len(mask))
            )
            # index = self.nplike.full(len(self.mask), -1)
            # index[self.mask  == 1] = self.nplike.arange(len(index[self.mask  == 1]))
            next = self.project()._rpad(target, posaxis, depth, clip)
            return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                index,
                next,
                None,
                self._parameters,
            ).simplify_optiontype()
        else:
            return self.project()._rpad(target, posaxis, depth, clip)

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

        next = ak._v2.contents.IndexedArray(
            ak._v2.index.Index(index), self._content, parameters=next_parameters
        )
        return next._to_arrow(
            pyarrow,
            self,
            ak._v2._connect.pyarrow.and_validbytes(validbytes, this_validbytes),
            length,
            options,
        )

    def _completely_flatten(self, nplike, options):
        return self.project()._completely_flatten(nplike, options)

    def _recursively_apply(
        self, action, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                return IndexedOptionArray(
                    self._index,
                    self._content._recursively_apply(
                        action,
                        depth,
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
