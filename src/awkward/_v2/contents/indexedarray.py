# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy

import awkward as ak
from awkward._v2.index import Index
from awkward._v2.contents.content import Content, unset
from awkward._v2.forms.indexedform import IndexedForm

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class IndexedArray(Content):
    is_IndexedType = True

    def copy(
        self,
        index=unset,
        content=unset,
        identifier=unset,
        parameters=unset,
        nplike=unset,
    ):
        return IndexedArray(
            self._index if index is unset else index,
            self._content if content is unset else content,
            self._identifier if identifier is unset else identifier,
            self._parameters if parameters is unset else parameters,
            self._nplike if nplike is unset else nplike,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            index=copy.deepcopy(self._index, memo),
            content=copy.deepcopy(self._content, memo),
            identifier=copy.deepcopy(self._identifier, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    def __init__(self, index, content, identifier=None, parameters=None, nplike=None):
        if not (
            isinstance(index, Index)
            and index.dtype
            in (
                np.dtype(np.int32),
                np.dtype(np.uint32),
                np.dtype(np.int64),
            )
        ):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'index' must be an Index with dtype in (int32, uint32, int64), "
                    "not {}".format(type(self).__name__, repr(index))
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
        if nplike is None:
            nplike = content.nplike
        if nplike is None:
            nplike = index.nplike

        self._index = index
        self._content = content
        self._init(identifier, parameters, nplike)

    @property
    def index(self):
        return self._index

    @property
    def content(self):
        return self._content

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
        container[key] = ak._v2._util.little_endian(self._index.raw(nplike))
        self._content._to_buffers(form.content, getkey, container, nplike)

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return IndexedArray(
            ak._v2.index.Index(self._index.raw(tt)),
            self._content.typetracer,
            self._typetracer_identifier(),
            self._parameters,
            tt,
        )

    @property
    def length(self):
        return self._index.length

    def _forget_length(self):
        return IndexedArray(
            self._index.forget_length(),
            self._content,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<IndexedArray len="]
        out.append(repr(str(self.length)))
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
            self._nplike,
        )

    def toIndexedOptionArray64(self):
        return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            self._index, self._content, self._identifier, self._parameters, self._nplike
        )

    def mask_as_bool(self, valid_when=True):
        if valid_when:
            return self._index.data >= 0
        else:
            return self._index.data < 0

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if not self._nplike.known_data:
            return self._content._getitem_at(where)

        if where < 0:
            where += self.length
        if self._nplike.known_shape and not 0 <= where < self.length:
            raise ak._v2._util.indexerror(self, where)
        return self._content._getitem_at(self._index[where])

    def _getitem_range(self, where):
        if not self._nplike.known_shape:
            return self

        start, stop, step = where.indices(self.length)
        assert step == 1
        return IndexedArray(
            self._index[start:stop],
            self._content,
            self._range_identifier(start, stop),
            self._parameters,
            self._nplike,
        )

    def _getitem_field(self, where, only_fields=()):
        return IndexedArray(
            self._index,
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
            self._nplike,
        )

    def _getitem_fields(self, where, only_fields=()):
        return IndexedArray(
            self._index,
            self._content._getitem_fields(where, only_fields),
            self._fields_identifier(where),
            None,
            self._nplike,
        )

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextindex = self._index[carry.data]
        except IndexError as err:
            raise ak._v2._util.indexerror(self, carry.data, str(err)) from err

        return IndexedArray(
            nextindex,
            self._content,
            self._carry_identifier(carry),
            self._parameters,
            self._nplike,
        )

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        if self._nplike.known_shape and slicestarts.length != self.length:
            raise ak._v2._util.indexerror(
                self,
                ak._v2.contents.ListArray(
                    slicestarts, slicestops, slicecontent, None, None, self._nplike
                ),
                "cannot fit jagged slice with length {} into {} of size {}".format(
                    slicestarts.length, type(self).__name__, self.length
                ),
            )

        nextcarry = ak._v2.index.Index64.empty(self.length, self._nplike)
        assert nextcarry.nplike is self._nplike and self._index.nplike is self._nplike
        self._handle_error(
            self._nplike[
                "awkward_IndexedArray_getitem_nextcarry",
                nextcarry.dtype.type,
                self._index.dtype.type,
            ](
                nextcarry.data,
                self._index.data,
                self._index.length,
                self._content.length,
            ),
            slicer=ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
        )
        # an eager carry (allow_lazy = false) to avoid infinite loop (unproven)
        next = self._content._carry(nextcarry, False)
        return next._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

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

            nextcarry = ak._v2.index.Index64.empty(self._index.length, self._nplike)
            assert (
                nextcarry.nplike is self._nplike and self._index.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_IndexedArray_getitem_nextcarry",
                    nextcarry.dtype.type,
                    self._index.dtype.type,
                ](
                    nextcarry.data,
                    self._index.data,
                    self._index.length,
                    self._content.length,
                ),
                slicer=head,
            )

            next = self._content._carry(nextcarry, False)
            return next._getitem_next(head, tail, advanced)

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
        if mask is not None:
            if self._nplike.known_shape and self._index.length != mask.length:
                raise ak._v2._util.error(
                    ValueError(
                        "mask length ({}) is not equal to {} length ({})".format(
                            mask.length(), type(self).__name__, self._index.length
                        )
                    )
                )
            nextindex = ak._v2.index.Index64.empty(self._index.length, self._nplike)
            assert (
                nextindex.nplike is self._nplike
                and mask.nplike is self._nplike
                and self._index.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
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
            next = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                nextindex,
                self._content,
                self._identifier,
                self._parameters,
                self._nplike,
            )
            return next.project()

        else:
            nextcarry = ak._v2.index.Index64.empty(self.length, self._nplike)
            assert (
                nextcarry.nplike is self._nplike and self._index.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_IndexedArray_getitem_nextcarry",
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
                result = ak._v2.index.Index64.empty(self.index.length, self._nplike)
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
                result = ak._v2.index.Index64.empty(self.index.length, self._nplike)

            assert (
                result.nplike is self._nplike
                and self._index.nplike is self._nplike
                and inner.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_IndexedArray_simplify",
                    result.dtype.type,
                    self._index.dtype.type,
                    inner.dtype.type,
                ](
                    result.data,
                    self._index.data,
                    self._index.length,
                    inner.data,
                    inner.length,
                )
            )
            if isinstance(self._content, ak._v2.contents.indexedarray.IndexedArray):
                return IndexedArray(
                    result,
                    self._content.content,
                    self._identifier,
                    self._parameters,
                    self._nplike,
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
                    result,
                    self._content.content,
                    self._identifier,
                    self._parameters,
                    self._nplike,
                )

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
            return self.project().num(posaxis, depth)

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            raise ak._v2._util.error(np.AxisError("axis=0 not allowed for flatten"))

        else:
            return self.project()._offsets_and_flattened(posaxis, depth)

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

    def _merging_strategy(self, others):
        if len(others) == 0:
            raise ak._v2._util.error(
                ValueError(
                    "to merge this array with 'others', at least one other must be provided"
                )
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

        if any(
            isinstance(x.nplike, ak._v2._typetracer.TypeTracer) for x in head + tail
        ):
            head = [
                x
                if isinstance(x.nplike, ak._v2._typetracer.TypeTracer)
                else x.typetracer
                for x in head
            ]
            tail = [
                x
                if isinstance(x.nplike, ak._v2._typetracer.TypeTracer)
                else x.typetracer
                for x in tail
            ]

        return (head, tail)

    def _reverse_merge(self, other):
        theirlength = other.length
        mylength = self.length
        index = ak._v2.index.Index64.empty((theirlength + mylength), self._nplike)

        content = other.merge(self._content)

        assert index.nplike is self._nplike
        self._handle_error(
            self._nplike["awkward_IndexedArray_fill_to64_count", index.dtype.type](
                index.data,
                0,
                theirlength,
                0,
            )
        )
        reinterpreted_index = ak._v2.index.Index(
            np.asarray(index.data.view(self[0].dtype))
        )

        assert (
            index.nplike is self._nplike and reinterpreted_index.nplike is self._nplike
        )
        self._handle_error(
            self._nplike[
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
        parameters = ak._v2._util.merge_parameters(self._parameters, other._parameters)

        return ak._v2.contents.indexedarray.IndexedArray(
            index, content, None, parameters, self._nplike
        )

    def mergemany(self, others):
        if len(others) == 0:
            return self

        head, tail = self._merging_strategy(others)

        total_length = 0
        for array in head:
            total_length += array.length

        contents = []
        contentlength_so_far = 0
        length_so_far = 0
        nextindex = ak._v2.index.Index64.empty(total_length, self._nplike)
        parameters = self._parameters

        parameters = self._parameters
        for array in head:
            parameters = ak._v2._util.merge_parameters(
                parameters, array._parameters, True
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
                assert (
                    nextindex.nplike is self._nplike
                    and array_index.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
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

            elif isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                pass
            else:
                contents.append(array)
                assert nextindex.nplike is self._nplike
                self._handle_error(
                    self._nplike[
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
        nextcontent = contents[0].mergemany(tail_contents)
        next = ak._v2.contents.indexedarray.IndexedArray(
            nextindex, nextcontent, None, parameters, self._nplike
        )

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed.mergemany(tail[1:])

        raise ak._v2._util.error(
            NotImplementedError(
                "not implemented: " + type(self).__name__ + " ::mergemany"
            )
        )

    def fill_none(self, value):
        if value.nplike.known_shape and value.length != 1:
            raise ak._v2._util.error(
                ValueError(f"fill_none value length ({value.length}) is not equal to 1")
            )
        return IndexedArray(
            self._index,
            self._content.fill_none(value),
            None,
            self._parameters,
            self._nplike,
        )

    def _local_index(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._local_index_axis0()
        else:
            return self.project()._local_index(posaxis, depth)

    def _unique_index(self, index, sorted=True):
        next = ak._v2.index.Index64.zeros(self.length, self._nplike)
        length = ak._v2.index.Index64.zeros(1, self._nplike)

        if not sorted:
            next = self._index
            offsets = ak._v2.index.Index64.empty(2, self._nplike)
            offsets[0] = 0
            offsets[1] = next.length
            assert next.nplike is self._nplike and offsets.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_sort",
                    next.dtype.type,
                    next.dtype.type,
                    offsets.dtype.type,
                ](
                    next.data,
                    next.data,
                    offsets[1],
                    offsets.data,
                    2,
                    offsets[1],
                    True,
                    False,
                )
            )

            assert next.nplike is self._nplike and length.nplike is self._nplike
            self._handle_error(
                self._nplike["awkward_unique", next.dtype.type, length.dtype.type](
                    next.data,
                    self._index.length,
                    length.data,
                )
            )

        else:
            assert (
                self._index.nplike is self._nplike
                and next.nplike is self._nplike
                and length.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_unique_copy",
                    self._index.dtype.type,
                    next.dtype.type,
                    length.dtype.type,
                ](
                    self._index.data,
                    next.data,
                    self._index.length,
                    length.data,
                )
            )

        return next[0 : length[0]]

    def numbers_to_type(self, name):
        return ak._v2.contents.indexedarray.IndexedArray(
            self._index,
            self._content.numbers_to_type(name),
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._index.length == 0:
            return True

        nextindex = self._unique_index(self._index)

        if len(nextindex) != len(self._index):
            return False

        next = self._content._carry(nextindex, False)
        return next._is_unique(negaxis, starts, parents, outlength)

    def _unique(self, negaxis, starts, parents, outlength):
        if self._index.length == 0:
            return self

        branch, depth = self.branch_depth

        index_length = self._index.length
        parents_length = parents.length
        next_length = index_length

        nextcarry = ak._v2.index.Index64.empty(index_length, self._nplike)
        nextparents = ak._v2.index.Index64.empty(index_length, self._nplike)
        outindex = ak._v2.index.Index64.empty(index_length, self._nplike)
        assert (
            nextcarry.nplike is self._nplike
            and nextparents.nplike is self._nplike
            and outindex.nplike is self._nplike
            and self._index.nplike is self._nplike
            and parents.nplike is self._nplike
        )
        self._handle_error(
            self._nplike[
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
        unique = next._unique(
            negaxis,
            starts,
            nextparents,
            outlength,
        )

        if branch or (negaxis is not None and negaxis != depth):
            nextoutindex = ak._v2.index.Index64.empty(parents_length, self._nplike)
            assert (
                nextoutindex.nplike is self._nplike
                and starts.nplike is self._nplike
                and parents.nplike is self._nplike
                and nextparents.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_IndexedArray_local_preparenext_64",
                    nextoutindex.dtype.type,
                    starts.dtype.type,
                    parents.dtype.type,
                    nextparents.dtype.type,
                ](
                    nextoutindex.data,
                    starts.data,
                    parents.data,
                    parents_length,
                    nextparents.data,
                    next_length,
                )
            )

            return ak._v2.contents.IndexedOptionArray(
                nextoutindex,
                unique,
                None,
                self._parameters,
                self._nplike,
            ).simplify_optiontype()

        if not branch and negaxis == depth:
            return unique
        else:
            if isinstance(unique, ak._v2.contents.RegularArray):
                unique = unique.toListOffsetArray64(True)

            if isinstance(unique, ak._v2.contents.ListOffsetArray):
                if starts.nplike.known_data and starts.length > 0 and starts[0] != 0:
                    raise ak._v2._util.error(
                        AssertionError(
                            "reduce_next with unbranching depth > negaxis expects a "
                            "ListOffsetArray64 whose offsets start at zero ({})".format(
                                starts[0]
                            )
                        )
                    )

                outoffsets = ak._v2.index.Index64.empty(starts.length + 1, self._nplike)
                assert (
                    outoffsets.nplike is self._nplike and starts.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_IndexedArray_reduce_next_fix_offsets_64",
                        outoffsets.dtype.type,
                        starts.dtype.type,
                    ](
                        outoffsets.data,
                        starts.data,
                        starts.length,
                        self._index.length,
                    )
                )

                tmp = ak._v2.contents.IndexedArray(
                    outindex,
                    unique._content,
                    None,
                    None,
                    self._nplike,
                )

                return ak._v2.contents.ListOffsetArray(
                    outoffsets,
                    tmp,
                    None,
                    None,
                    self._nplike,
                )

            elif isinstance(unique, ak._v2.contents.NumpyArray):
                nextoutindex = ak._v2.index.Index64(
                    self._nplike.index_nplike.arange(unique.length, dtype=np.int64),
                    nplike=self.nplike,
                )
                out = ak._v2.contents.IndexedOptionArray(
                    nextoutindex,
                    unique,
                    None,
                    self._parameters,
                    self._nplike,
                ).simplify_optiontype()

                return out

        raise ak._v2._util.error(NotImplementedError)

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
        next = self._content._carry(self._index, False)
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
        next = self._content._carry(self._index, False)
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
        behavior,
    ):
        next = self._content._carry(self._index, False)
        return next._reduce_next(
            reducer,
            negaxis,
            starts,
            shifts,
            parents,
            outlength,
            mask,
            keepdims,
            behavior,
        )

    def _validity_error(self, path):
        error = self._nplike["awkward_IndexedArray_validity", self.index.dtype.type](
            self.index.data, self.index.length, self._content.length, False
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
        result = self.index._nbytes_part() + self.content._nbytes_part()
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def _pad_none(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.pad_none_axis0(target, clip)
        elif posaxis == depth + 1:
            return self.project()._pad_none(target, posaxis, depth, clip)
        else:
            return ak._v2.contents.indexedarray.IndexedArray(
                self._index,
                self._content._pad_none(target, posaxis, depth, clip),
                None,
                self._parameters,
                self._nplike,
            )

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        if (
            not options["categorical_as_dictionary"]
            and self.parameter("__array__") == "categorical"
        ):
            next_parameters = dict(self._parameters)
            del next_parameters["__array__"]
            next = IndexedArray(
                self._index,
                self._content,
                self._identifier,
                next_parameters,
                self._nplike,
            )
            return next._to_arrow(pyarrow, mask_node, validbytes, length, options)

        index = self._index.raw(numpy)

        if self.parameter("__array__") == "categorical":
            dictionary = self._content._to_arrow(
                pyarrow, None, None, self._content.length, options
            )
            out = pyarrow.DictionaryArray.from_arrays(
                index,
                dictionary,
                None if validbytes is None else ~validbytes,
            )
            if options["extensionarray"]:
                return ak._v2._connect.pyarrow.AwkwardArrowArray.from_storage(
                    ak._v2._connect.pyarrow.to_awkwardarrow_type(
                        out.type,
                        options["extensionarray"],
                        options["record_is_scalar"],
                        mask_node,
                        self,
                    ),
                    out,
                )
            else:
                return out

        else:
            if self._content.length == 0:
                # IndexedOptionArray._to_arrow replaces -1 in the index with 0. So behind
                # every masked value is self._content[0], unless self._content.length == 0.
                # In that case, don't call self._content[index]; it's empty anyway.
                next = self._content
            else:
                next = self._content._carry(ak._v2.index.Index(index), False)

            return next.merge_parameters(self._parameters)._to_arrow(
                pyarrow, mask_node, validbytes, length, options
            )

    def _to_numpy(self, allow_missing):
        return self.project()._to_numpy(allow_missing)

    def _completely_flatten(self, nplike, options):
        return self.project()._completely_flatten(nplike, options)

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if (
            self._nplike.known_shape
            and self._nplike.known_data
            and self._index.length != 0
        ):
            npindex = self._index.data
            indexmin = npindex.min()
            index = ak._v2.index.Index(npindex - indexmin, nplike=self._nplike)
            content = self._content[indexmin : npindex.max() + 1]
        else:
            index, content = self._index, self._content

        if options["return_array"]:

            def continuation():
                return IndexedArray(
                    index,
                    content._recursively_apply(
                        action,
                        behavior,
                        depth,
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
        if self.parameter("__array__") == "categorical":
            return IndexedArray(
                self._index,
                self._content.packed(),
                identifier=self._identifier,
                parameters=self._parameters,
                nplike=self._nplike,
            )
        else:
            return self.project().packed()

    def _to_list(self, behavior, json_conversions):
        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        index = self._index.raw(numpy)
        nextcontent = self._content._carry(ak._v2.index.Index(index), False)
        return nextcontent._to_list(behavior, json_conversions)

    def _to_nplike(self, nplike):
        index = self._index._to_nplike(nplike)
        content = self._content._to_nplike(nplike)
        return IndexedArray(
            index,
            content,
            identifier=self._identifier,
            parameters=self._parameters,
            nplike=nplike,
        )

    def _layout_equal(self, other, index_dtype=True, numpyarray=True):
        return self.index.layout_equal(
            other.index, index_dtype, numpyarray
        ) and self.content.layout_equal(other.content, index_dtype, numpyarray)
