# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import copy

import awkward as ak
from awkward._v2.index import Index
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.indexedform import IndexedForm
from awkward._v2.forms.form import _parameters_equal

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class IndexedArray(Content):
    is_IndexedType = True

    def __init__(self, index, content, identifier=None, parameters=None):
        if not (
            isinstance(index, Index)
            and index.dtype
            in (
                np.dtype(np.int32),
                np.dtype(np.uint32),
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

    Form = IndexedForm

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.Form(
            self._index.form,
            self._content._form_with_key(getkey),
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, nplike):
        assert isinstance(form, self.Form)
        key = getkey(self, form, "index")
        container[key] = ak._v2._util.little_endian(self._index.to(nplike))
        self._content._to_buffers(form.content, getkey, container, nplike)

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return IndexedArray(
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
        out = [indent, pre, "<IndexedArray len="]
        out.append(repr(str(len(self))))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._index._repr(indent + "    ", "<index>", "</index>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</IndexedArray>")
        out.append(post)
        return "".join(out)

    def merge_parameters(self, parameters):
        return IndexedArray(
            self._index,
            self._content,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
        )

    def toIndexedOptionArray64(self):
        return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            self._index, self._content, self._identifier, self._parameters
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
        return self._content._getitem_at(self._index[where])

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        return IndexedArray(
            self._index[start:stop],
            self._content,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return IndexedArray(
            self._index,
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return IndexedArray(
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

        return IndexedArray(
            nextindex,
            self._content,
            self._carry_identifier(carry, exception),
            self._parameters,
        )

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

        nextcarry = ak._v2.index.Index64.zeros(len(self), nplike)
        self._handle_error(
            nplike[
                "awkward_IndexedArray_getitem_nextcarry",
                nextcarry.dtype.type,
                self._index.dtype.type,
            ](
                nextcarry.to(nplike),
                self._index.to(nplike),
                len(self._index),
                len(self._content),
            )
        )
        # an eager carry (allow_lazy = false) to avoid infinite loop (unproven)
        next = self._content._carry(nextcarry, False, NestedIndexError)
        return next._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        return self._getitem_next_jagged_generic(
            slicestarts, slicestops, slicecontent, tail
        )

    def _getitem_next(self, head, tail, advanced):
        nplike = self.nplike

        if head == ():
            return self

        elif isinstance(head, (int, slice, ak._v2.index.Index64)):
            nexthead, nexttail = ak._v2._slicing.headtail(tail)

            nextcarry = ak._v2.index.Index64.empty(len(self._index), nplike)
            self._handle_error(
                nplike[
                    "awkward_IndexedArray_getitem_nextcarry",
                    nextcarry.dtype.type,
                    self._index.dtype.type,
                ](
                    nextcarry.to(nplike),
                    self._index.to(nplike),
                    len(self._index),
                    len(self._content),
                )
            )

            next = self._content._carry(nextcarry, False, NestedIndexError)
            return next._getitem_next(head, tail, advanced)

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
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
                nextindex, self._content, self._identifier, self._parameters
            )
            return next.project()

        else:
            nextcarry = ak._v2.index.Index64.empty(len(self), self.nplike)
            self._handle_error(
                self.nplike[
                    "awkward_IndexedArray_getitem_nextcarry",
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
            self._content,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):

            if isinstance(
                self._content,
                (
                    ak._v2.contents.indexedarray.IndexedArray,
                    ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ),
            ):
                inner = self._content.index
                result = ak._v2.index.Index64.empty(len(self.index), self.nplike)
            elif isinstance(
                self._content,
                (
                    ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                    ak._v2.contents.bitmaskedarray.BitMaskedArray,
                    ak._v2.contents.unmaskedarray.UnmaskedArray,
                ),
            ):
                rawcontent = self._content.toIndexedOptionArray64()
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
            if isinstance(self._content, ak._v2.contents.indexedarray.IndexedArray):
                return IndexedArray(
                    result, self._content.content, self._identifier, self._parameters
                )

            if isinstance(
                self._content,
                (
                    ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                    ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                    ak._v2.contents.bitmaskedarray.BitMaskedArray,
                    ak._v2.contents.unmaskedarray.UnmaskedArray,
                ),
            ):
                return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                    result, self._content.content, self._identifier, self._parameters
                )

        else:
            return self

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            out = ak._v2.index.Index64.empty(1, self.nplike)
            out[0] = len(self)
            return ak._v2.contents.numpyarray.NumpyArray(out)[0]
        else:
            return self.project().num(posaxis, depth)

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            raise np.AxisError(self, "axis=0 not allowed for flatten")

        else:
            return self.project()._offsets_and_flattened(posaxis, depth)

    def mergeable(self, other, mergebool):
        if not _parameters_equal(self._parameters, other._parameters):
            return False

        if isinstance(
            other,
            (
                ak._v2.contents.emptyArray.EmptyArray,
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
            if isinstance(other, ak._v2.content.unionarray.UnionArray):
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

        content = other.merge(self._content)

        self._handle_error(
            self.nplike["awkward_IndexedArray_fill_to64_count", index.dtype.type](
                index.to(self.nplike),
                0,
                theirlength,
                0,
            )
        )
        reinterpreted_index = ak._v2.index.Index(
            np.asarray(index.data.view(self[0].dtype))
        )

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
        parameters = ak._v2._util.merge_parameters(self._parameters, other._parameters)

        return ak._v2.contents.indexedarray.IndexedArray(
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
            parameters = ak._v2._util.merge_parameters(
                self._parameters, array._parameters
            )

            if isinstance(
                array,
                (
                    ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                    ak._v2.contents.bitmaskedarray.BitMaskedArray,
                    ak._v2.contents.unmaskedarray.UnmaskedArray,
                ),
            ):
                array = array.toIndexedOptionArray64()

            if isinstance(array, ak._v2.contents.indexedarray.IndexedArray):
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
        next = ak._v2.contents.indexedarray.IndexedArray(
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

    def fillna(self, value):
        if len(value) != 1:
            raise ValueError(
                "fillna value length ({0}) is not equal to 1".format(len(value))
            )

        return IndexedArray(
            self._index, self._content.fillna(value), None, self._parameters
        )

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        else:
            return self.project()._localindex(posaxis, depth)

    def _unique_index(self, index, sorted=True):
        nplike = self.nplike

        next = ak._v2.index.Index64.empty(len(self), nplike)
        length = ak._v2.index.Index64.empty(1, nplike)

        if not sorted:
            next = self._index
            offsets = ak._v2.index.Index64.zeros(2, nplike)
            offsets[1] = len(next)
            self._handle_error(
                nplike[
                    "awkward_sort",
                    next.dtype.type,
                    next.dtype.type,
                    offsets.dtype.type,
                ](
                    next.to(nplike),
                    next.to(nplike),
                    offsets[1],
                    offsets.to(nplike),
                    2,
                    offsets[1],
                    True,
                    False,
                )
            )

            self._handle_error(
                nplike["awkward_unique", next.dtype.type, length.dtype.type](
                    next.to(nplike),
                    len(self._index),
                    length.to(nplike),
                )
            )

        else:
            self._handle_error(
                nplike[
                    "awkward_unique_copy",
                    self._index.dtype.type,
                    next.dtype.type,
                    length.dtype.type,
                ](
                    self._index.to(nplike),
                    next.to(nplike),
                    len(self._index),
                    length.to(nplike),
                )
            )

        return next[0 : length[0]]

    def numbers_to_type(self, name):
        return ak._v2.contents.indexedarray.IndexedArray(
            self._index,
            self._content.numbers_to_type(name),
            self._identifier,
            self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if len(self._index) == 0:
            return True

        nextindex = self._unique_index(self._index)

        next = self._content._carry(nextindex, False, NestedIndexError)
        return next._is_unique(negaxis, starts, parents, outlength)

    def _unique(self, negaxis, starts, parents, outlength):
        if len(self._index) == 0:
            return self

        nplike = self.nplike
        branch, depth = self.branch_depth

        index_length = len(self._index)
        parents_length = len(parents)
        next_length = index_length

        nextcarry = ak._v2.index.Index64.zeros(index_length, nplike)
        nextparents = ak._v2.index.Index64.zeros(index_length, nplike)
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
        next = self._content._carry(nextcarry, False, NestedIndexError)
        unique = next._unique(
            negaxis,
            starts,
            nextparents,
            outlength,
        )

        if branch or (negaxis is not None and negaxis != depth):
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
                unique,
                None,
                self._parameters,
            ).simplify_optiontype()

            return out

        if not branch and negaxis == depth:
            return unique
        else:

            if isinstance(unique, ak._v2.contents.RegularArray):
                unique = unique.toListOffsetArray64(True)

            elif isinstance(unique, ak._v2.contents.ListOffsetArray):
                if len(starts) > 0 and starts[0] != 0:
                    raise AssertionError(
                        "reduce_next with unbranching depth > negaxis expects a "
                        "ListOffsetArray64 whose offsets start at zero ({0})".format(
                            starts[0]
                        )
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
                        len(starts),
                        len(self._index),
                    )
                )

                tmp = ak._v2.contents.IndexedArray(
                    outindex,
                    unique._content,
                    None,
                    None,
                )

                return ak._v2.contents.ListOffsetArray(
                    outoffsets,
                    tmp,
                    None,
                    None,
                )

            elif isinstance(unique, ak._v2.contents.NumpyArray):
                nextoutindex = ak._v2.index.Index64.empty(len(unique), nplike)
                # FIXME: move to kernel
                for i in range(len(unique)):
                    nextoutindex[i] = i

                out = ak._v2.contents.IndexedOptionArray(
                    nextoutindex,
                    unique,
                    None,
                    self._parameters,
                ).simplify_optiontype()

                return out

        raise NotImplementedError

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

        next = self._content._carry(self._index, False, NestedIndexError)
        return next._argsort_next(
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
        self,
        negaxis,
        starts,
        parents,
        outlength,
        ascending,
        stable,
        kind,
        order,
    ):
        next = self._content._carry(self._index, False, NestedIndexError)
        return next._sort_next(
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
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            return self.project()._combinations(
                n, replacement, recordlookup, parameters, posaxis, depth
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
        nplike = self.nplike
        branch, depth = self.branch_depth

        index_length = len(self._index)

        nextcarry = ak._v2.index.Index64.zeros(index_length, nplike)
        nextparents = ak._v2.index.Index64.zeros(index_length, nplike)
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
        next = self._content._carry(nextcarry, False, NestedIndexError)
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
        )

        if not branch and negaxis == depth:
            return out
        else:
            if isinstance(out, ak._v2.contents.RegularArray):
                out = out.toListOffsetArray64(True)

            elif isinstance(out, ak._v2.contents.ListOffsetArray):
                if len(starts) > 0 and starts[0] != 0:
                    raise AssertionError(
                        "reduce_next with unbranching depth > negaxis expects a "
                        "ListOffsetArray64 whose offsets start at zero ({0})".format(
                            starts[0]
                        )
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
                        len(starts),
                        len(self._index),
                    )
                )

                tmp = ak._v2.contents.IndexedArray(
                    outindex,
                    out._content,
                    None,
                    None,
                )

                return ak._v2.contents.ListOffsetArray(
                    outoffsets,
                    tmp,
                    None,
                    None,
                )

            else:
                raise AssertionError(
                    "reduce_next with unbranching depth > negaxis is only "
                    "expected to return RegularArray or ListOffsetArray64; "
                    "instead, it returned " + out.classname
                )

    def _validityerror(self, path):
        error = self.nplike["awkward_IndexedArray_validity", self.index.dtype.type](
            self.index.to(self.nplike), len(self.index), len(self._content), False
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
        result = self.index._nbytes_part() + self.content._nbytes_part()
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def _rpad(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.rpad_axis0(target, clip)
        elif posaxis == depth + 1:
            return self.project()._rpad(target, posaxis, depth, clip)
        else:
            return ak._v2.contents.indexedarray.IndexedArray(
                self._index,
                self._content._rpad(target, posaxis, depth, clip),
                None,
                self._parameters,
            )

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        if (
            not options["categorical_as_dictionary"]
            and self.parameter("__array__") == "categorical"
        ):
            next_parameters = dict(self._parameters)
            del next_parameters["__array__"]
            next = IndexedArray(
                self._index, self._content, self._identifier, next_parameters
            )
            return next._to_arrow(pyarrow, mask_node, validbytes, length, options)

        index = self._index.to(numpy)

        if self.parameter("__array__") == "categorical":
            dictionary = self._content._to_arrow(
                pyarrow, None, None, len(self._content), options
            )
            out = pyarrow.DictionaryArray.from_arrays(
                index,
                dictionary,
                None if validbytes is None else ~validbytes,
            )
            if options["extensionarray"]:
                return ak._v2._connect.pyarrow.AwkwardArrowArray.from_storage(
                    ak._v2._connect.pyarrow.to_awkwardarrow_type(
                        out.type, options["extensionarray"], mask_node, self
                    ),
                    out,
                )
            else:
                return out

        else:
            if len(self._content) == 0:
                # IndexedOptionArray._to_arrow replaces -1 in the index with 0. So behind
                # every masked value is self._content[0], unless len(self._content) == 0.
                # In that case, don't call self._content[index]; it's empty anyway.
                next = self._content
            else:
                next = self._content._carry(
                    ak._v2.index.Index(index), False, IndexError
                )

            return next.merge_parameters(self._parameters)._to_arrow(
                pyarrow, mask_node, validbytes, length, options
            )

    def _to_numpy(self, allow_missing):
        return self.project()._to_numpy(allow_missing)

    def _completely_flatten(self, nplike, options):
        return self.project()._completely_flatten(nplike, options)

    def _recursively_apply(
        self, action, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                return IndexedArray(
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

    def packed(self):
        return self.project().packed()

    def _to_list(self, behavior):
        out = self._to_list_custom(behavior)
        if out is not None:
            return out

        index = self._index.to(numpy)
        content = self._content._to_list(behavior)
        out = [None] * len(index)
        for i, ind in enumerate(index):
            out[i] = content[ind]
        return out
