# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

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

    def __init__(self, index, content, identifier=None, parameters=None, nplike=None):
        if not (
            isinstance(index, Index)
            and index.dtype
            in (
                np.dtype(np.int32),
                np.dtype(np.int64),
            )
        ):
            raise TypeError(
                "{} 'index' must be an Index with dtype in (int32, uint32, int64), "
                "not {}".format(type(self).__name__, repr(index))
            )
        if not isinstance(content, Content):
            raise TypeError(
                "{} 'content' must be a Content subtype, not {}".format(
                    type(self).__name__, repr(content)
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

    Form = IndexedOptionForm

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
        return IndexedOptionArray(
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
        return IndexedOptionArray(
            self._index.forget_length(),
            self._content,
            self._identifier,
            self._parameters,
            self._nplike,
        )

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

    def merge_parameters(self, parameters):
        return IndexedOptionArray(
            self._index,
            self._content,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
            self._nplike,
        )

    def toIndexedOptionArray64(self):
        if self._index.dtype == np.dtype(np.int64):
            return self
        else:
            return IndexedOptionArray(
                self._index.astype(np.int64),
                self._content,
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def mask_as_bool(self, valid_when=True, nplike=None):
        if nplike is None:
            nplike = self._nplike

        if valid_when:
            return self._index.raw(nplike) >= 0
        else:
            return self._index.raw(nplike) < 0

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if not self._nplike.known_data:
            return ak._v2._typetracer.MaybeNone(self._content._getitem_at(where))

        if where < 0:
            where += self.length
        if self._nplike.known_shape and not 0 <= where < self.length:
            raise NestedIndexError(self, where)
        if self._index[where] < 0:
            return None
        else:
            return self._content._getitem_at(self._index[where])

    def _getitem_range(self, where):
        if not self._nplike.known_shape:
            return self

        start, stop, step = where.indices(self.length)
        assert step == 1
        return IndexedOptionArray(
            self._index[start:stop],
            self._content,
            self._range_identifier(start, stop),
            self._parameters,
            self._nplike,
        )

    def _getitem_field(self, where, only_fields=()):
        return IndexedOptionArray(
            self._index,
            self._content._getitem_field(where, only_fields),
            self._field_identifier(where),
            None,
            self._nplike,
        )

    def _getitem_fields(self, where, only_fields=()):
        return IndexedOptionArray(
            self._index,
            self._content._getitem_fields(where, only_fields),
            self._fields_identifier(where),
            None,
            self._nplike,
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
            self._nplike,
        )

    def _nextcarry_outindex(self, nplike):
        numnull = ak._v2.index.Index64.empty(1, nplike)

        assert numnull.nplike is self._nplike and self._index.nplike is self._nplike
        self._handle_error(
            nplike[
                "awkward_IndexedArray_numnull",
                numnull.dtype.type,
                self._index.dtype.type,
            ](
                numnull.data,
                self._index.data,
                self._index.length,
            )
        )
        nextcarry = ak._v2.index.Index64.empty(self._index.length - numnull[0], nplike)
        outindex = ak._v2.index.Index.empty(
            self._index.length, nplike, self._index.dtype
        )

        assert (
            nextcarry.nplike is self._nplike
            and outindex.nplike is self._nplike
            and self._index.nplike is self._nplike
        )
        self._handle_error(
            nplike[
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
        slicestarts = slicestarts._to_nplike(self.nplike)
        slicestops = slicestops._to_nplike(self.nplike)

        if self._nplike.known_shape and slicestarts.length != self.length:
            raise NestedIndexError(
                self,
                ak._v2.contents.ListArray(
                    slicestarts, slicestops, slicecontent, None, None, self._nplike
                ),
                "cannot fit jagged slice with length {} into {} of size {}".format(
                    slicestarts.length, type(self).__name__, self.length
                ),
            )

        numnull, nextcarry, outindex = self._nextcarry_outindex(self._nplike)

        reducedstarts = ak._v2.index.Index64.empty(self.length - numnull, self._nplike)
        reducedstops = ak._v2.index.Index64.empty(self.length - numnull, self._nplike)
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
            )
        )
        next = self._content._carry(nextcarry, True, NestedIndexError)
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

        elif isinstance(head, (int, slice, ak._v2.index.Index64)):
            nexthead, nexttail = ak._v2._slicing.headtail(tail)

            numnull, nextcarry, outindex = self._nextcarry_outindex(self._nplike)

            next = self._content._carry(nextcarry, True, NestedIndexError)
            out = next._getitem_next(head, tail, advanced)
            out2 = IndexedOptionArray(
                outindex, out, self._identifier, self._parameters, self._nplike
            )
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
            if self._nplike.known_shape and self._index.length != mask.length:
                raise ValueError(
                    "mask length ({}) is not equal to {} length ({})".format(
                        mask.length(), type(self).__name__, self._index.length
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
            numnull = ak._v2.index.Index64.empty(1, self._nplike)

            assert numnull.nplike is self._nplike and self._index.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_IndexedArray_numnull",
                    numnull.dtype.type,
                    self._index.dtype.type,
                ](
                    numnull.data,
                    self._index.data,
                    self._index.length,
                )
            )

            nextcarry = ak._v2.index.Index64.empty(
                self.length - numnull[0], self._nplike
            )

            assert (
                nextcarry.nplike is self._nplike and self._index.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
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
            out = ak._v2.index.Index64.empty(1, self._nplike)
            out[0] = self.length
            return ak._v2.contents.numpyarray.NumpyArray(out, None, None, self._nplike)[
                0
            ]
        _, nextcarry, outindex = self._nextcarry_outindex(self._nplike)
        next = self._content._carry(nextcarry, False, NestedIndexError)
        out = next.num(posaxis, depth)
        out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            outindex, out, None, self.parameters, self._nplike
        )
        return out2.simplify_optiontype()

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            raise np.AxisError(self, "axis=0 not allowed for flatten")
        else:
            numnull, nextcarry, outindex = self._nextcarry_outindex(self._nplike)
            next = self._content._carry(nextcarry, False, NestedIndexError)

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
                    offsets.length + numnull, self._nplike, dtype=np.int64
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
            return self._content.mergeable(other.content, mergebool)

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
            self._nplike["awkward_IndexedArray_fill_count", index.dtype.type](
                index.data,
                0,
                theirlength,
                0,
            )
        )
        reinterpreted_index = ak._v2.index.Index(self._nplike.asarray(self.index.data))

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

        return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
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

        for array in head:
            parameters = ak._v2._util.merge_parameters(
                self._parameters, array._parameters, True
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

            if isinstance(array, ak._v2.contents.indexedoptionarray.IndexedOptionArray):
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
        next = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            nextindex, nextcontent, None, parameters, self._nplike
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
        if value.nplike.known_shape and value.length != 1:
            raise ValueError(f"fillna value length ({value.length}) is not equal to 1")

        contents = [self._content, value]
        tags = self.bytemask()
        index = ak._v2.index.Index64.empty(tags.length, self._nplike)

        assert index.nplike is self._nplike and self._index.nplike is self._nplike
        self._handle_error(
            self._nplike[
                "awkward_UnionArray_fillna", index.dtype.type, self._index.dtype.type
            ](index.data, self._index.data, tags.length)
        )
        out = ak._v2.contents.unionarray.UnionArray(
            tags, index, contents, None, self._parameters, self._nplike
        )
        return out.simplify_uniontype(True, True)

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        else:
            _, nextcarry, outindex = self._nextcarry_outindex(self._nplike)

            next = self._content._carry(nextcarry, False, NestedIndexError)
            out = next._localindex(posaxis, depth)
            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex,
                out,
                self._identifier,
                self._parameters,
                self._nplike,
            )
            return out2

    def _is_subrange_equal(self, starts, stops, length, sorted=True):
        nextstarts = ak._v2.index.Index64.empty(length, self._nplike)
        nextstops = ak._v2.index.Index64.empty(length, self._nplike)

        subranges_length = ak._v2.index.Index64.empty(1, self._nplike)
        assert (
            self._index.nplike is self._nplike
            and starts.nplike is self._nplike
            and stops.nplike is self._nplike
            and nextstarts.nplike is self._nplike
            and nextstops.nplike is self._nplike
            and subranges_length.nplike is self._nplike
        )
        self._handle_error(
            self._nplike[
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

        nextcarry = ak._v2.index.Index64.empty(subranges_length[0], self._nplike)
        assert (
            self._index.nplike is self._nplike
            and starts.nplike is self._nplike
            and stops.nplike is self._nplike
            and nextcarry.nplike is self._nplike
        )
        self._handle_error(
            self._nplike[
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

        next = self._content._carry(nextcarry, False, NestedIndexError)
        if nextstarts.length > 1:
            return next._is_subrange_equal(nextstarts, nextstops, nextstarts.length)
        else:
            return next._subranges_equal(
                nextstarts, nextstops, nextstarts.length, False
            )

    def numbers_to_type(self, name):
        return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            self._index,
            self._content.numbers_to_type(name),
            self._identifier,
            self._parameters,
            self._nplike,
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
        parents_length = parents.length

        numnull = ak._v2.index.Index64.empty(1, self._nplike)
        assert numnull.nplike is self._nplike and self._index.nplike is self._nplike
        self._handle_error(
            self._nplike[
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
        nextparents = ak._v2.index.Index64.empty(next_length, self._nplike)
        nextcarry = ak._v2.index.Index64.empty(next_length, self._nplike)
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
        next = self._content._carry(nextcarry, False, NestedIndexError)
        out = next._unique(
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
                out,
                None,
                self._parameters,
                self._nplike,
            ).simplify_optiontype()

        if isinstance(out, ak._v2.contents.ListOffsetArray):
            newnulls = ak._v2.index.Index64.empty(self._index.length, self._nplike)
            len_newnulls = ak._v2.index.Index64.empty(1, self._nplike)
            assert (
                newnulls.nplike is self._nplike
                and len_newnulls.nplike is self._nplike
                and self._index.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
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

            newindex = ak._v2.index.Index64.empty(
                out._offsets[-1] + len_newnulls[0], self._nplike
            )
            newoffsets = ak._v2.index.Index64.empty(out._offsets.length, self._nplike)
            assert (
                newindex.nplike is self._nplike
                and newoffsets.nplike is self._nplike
                and out._offsets.nplike is self._nplike
                and newnulls.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
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

            out = ak._v2.contents.IndexedOptionArray(
                newindex[: newoffsets[-1]],
                out._content,
                None,
                self._parameters,
                self._nplike,
            ).simplify_optiontype()

            return ak._v2.contents.ListOffsetArray(
                newoffsets,
                out,
                None,
                self._parameters,
                self._nplike,
            )

        if isinstance(out, ak._v2.contents.NumpyArray):
            nextoutindex = ak._v2.index.Index64.empty(out.length + 1, self._nplike)
            assert nextoutindex.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_IndexedArray_numnull_unique_64",
                    nextoutindex.dtype.type,
                ](
                    nextoutindex.data,
                    out.length,
                )
            )

            return ak._v2.contents.IndexedOptionArray(
                nextoutindex,
                out,
                None,
                self._parameters,
                self._nplike,
            ).simplify_optiontype()

        if inject_nones:
            out = ak._v2.contents.RegularArray(
                out,
                nextoutindex.length,
                0,
                None,
                self._parameters,
                self._nplike,
            )

        return out

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
        if self._index.length == 0:
            return ak._v2.contents.NumpyArray(
                self._nplike.empty(0, np.int64), None, None, self._nplike
            )

        branch, depth = self.branch_depth

        index_length = self._index.length
        parents_length = parents.length

        starts_length = starts.length
        numnull = ak._v2.index.Index64.empty(1, self._nplike)
        assert numnull.nplike is self._nplike and self._index.nplike is self._nplike
        self._handle_error(
            self._nplike[
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
        nextparents = ak._v2.index.Index64.empty(next_length, self._nplike)
        nextcarry = ak._v2.index.Index64.empty(next_length, self._nplike)
        outindex = ak._v2.index.Index64.empty(index_length, self._nplike)
        assert (
            nextcarry.nplike is self._nplike
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

        next = self._content._carry(nextcarry, False, NestedIndexError)

        inject_nones = True if not branch and negaxis != depth else False

        if (not branch) and negaxis == depth:
            nextshifts = ak._v2.index.Index64.empty(
                next_length,
                self._nplike,
            )
            if shifts is None:
                assert (
                    nextshifts.nplike is self._nplike
                    and self._index.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
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
                    nextshifts.nplike is self._nplike
                    and self._index.nplike is self._nplike
                    and shifts.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
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
        nulls_index = ak._v2.index.Index64.empty(numnull[0], self._nplike)
        assert (
            nulls_index.nplike is self._nplike
            and self._index.nplike is self._nplike
            and parents.nplike is self._nplike
            and starts.nplike is self._nplike
        )
        self._handle_error(
            self._nplike[
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

        ind = ak._v2.contents.NumpyArray(nulls_index, None, None, self._nplike)

        if self.mergeable(ind, True):
            out = out.merge(ind)
            nulls_merged = True

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

        if nulls_merged:
            assert nextoutindex.nplike is self._nplike
            self._handle_error(
                self._nplike["awkward_Index_nones_as_index", nextoutindex.dtype.type](
                    nextoutindex.data,
                    nextoutindex.length,
                )
            )

        out = ak._v2.contents.IndexedOptionArray(
            nextoutindex,
            out,
            None,
            self._parameters,
            self._nplike,
        ).simplify_optiontype()

        if inject_nones:
            out = ak._v2.contents.RegularArray(
                out,
                parents_length,
                0,
                None,
                self._parameters,
                self._nplike,
            )

        if (not branch) and negaxis == depth:
            return out
        else:
            if isinstance(out, ak._v2.contents.RegularArray):
                out = out.toListOffsetArray64(True)
            if isinstance(out, ak._v2.contents.ListOffsetArray):
                if starts.nplike.known_data and starts.length > 0 and starts[0] != 0:
                    raise AssertionError(
                        "sort_next with unbranching depth > negaxis expects a "
                        "ListOffsetArray64 whose offsets start at zero"
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
                        starts_length,
                        outindex.length,
                    )
                )

                tmp = ak._v2.contents.IndexedOptionArray(
                    outindex,
                    out._content,
                    None,
                    self._parameters,
                    self._nplike,
                ).simplify_optiontype()

                if inject_nones:
                    return tmp

                return ak._v2.contents.ListOffsetArray(
                    outoffsets,
                    tmp,
                    None,
                    self._parameters,
                    self._nplike,
                )

            if isinstance(out, ak._v2.contents.IndexedOptionArray):
                return out
            else:
                raise AssertionError(
                    "argsort_next with unbranching depth > negaxis is only "
                    "expected to return RegularArray or ListOffsetArray or "
                    "IndexedOptionArray; "
                    "instead, it returned " + out
                )

        return out

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        branch, depth = self.branch_depth

        index_length = self._index.length
        parents_length = parents.length

        starts_length = starts.length
        numnull = ak._v2.index.Index64.empty(1, self._nplike)
        assert numnull.nplike is self._nplike and self._index.nplike is self._nplike
        self._handle_error(
            self._nplike[
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
        nextparents = ak._v2.index.Index64.empty(next_length, self._nplike)
        nextcarry = ak._v2.index.Index64.empty(next_length, self._nplike)
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

        out = ak._v2.contents.IndexedOptionArray(
            nextoutindex,
            out,
            None,
            self._parameters,
            self._nplike,
        ).simplify_optiontype()

        if inject_nones:
            out = ak._v2.contents.RegularArray(
                out,
                parents_length if self._nplike.known_shape else 1,
                0,
                None,
                self._parameters,
                self._nplike,
            )

        if not branch and negaxis == depth:
            return out
        else:
            if isinstance(out, ak._v2.contents.RegularArray):
                out = out.toListOffsetArray64(True)
            if isinstance(out, ak._v2.contents.ListOffsetArray):
                if starts.nplike.known_data and starts.length > 0 and starts[0] != 0:
                    raise AssertionError(
                        "sort_next with unbranching depth > negaxis expects a "
                        "ListOffsetArray64 whose offsets start at zero"
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
                        starts_length,
                        outindex.length,
                    )
                )

                tmp = ak._v2.contents.IndexedOptionArray(
                    outindex,
                    out._content,
                    None,
                    self._parameters,
                    self._nplike,
                ).simplify_optiontype()

                if inject_nones:
                    return tmp

                return ak._v2.contents.ListOffsetArray(
                    outoffsets,
                    tmp,
                    None,
                    self._parameters,
                    self._nplike,
                )

            if isinstance(out, ak._v2.contents.IndexedOptionArray):
                return out
            else:
                raise AssertionError(
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
            _, nextcarry, outindex = self._nextcarry_outindex(self._nplike)
            next = self._content._carry(nextcarry, True, NestedIndexError)
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
    ):
        branch, depth = self.branch_depth

        index_length = self._index.length

        numnull = ak._v2.index.Index64.empty(1, self._nplike)
        assert numnull.nplike is self._nplike and self._index.nplike is self._nplike
        self._handle_error(
            self._nplike[
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
        nextcarry = ak._v2.index.Index64.empty(next_length, self._nplike)
        nextparents = ak._v2.index.Index64.empty(next_length, self._nplike)
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

        if reducer.needs_position and (not branch and negaxis == depth):
            nextshifts = ak._v2.index.Index64.empty(next_length, self._nplike)
            if shifts is None:
                assert (
                    nextshifts.nplike is self._nplike
                    and self.index.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64",
                        nextshifts.dtype.type,
                        self.index.dtype.type,
                    ](
                        nextshifts.data,
                        self.index.data,
                        self.index.length,
                    )
                )
            else:
                assert (
                    nextshifts.nplike is self._nplike
                    and self.index.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
                        nextshifts.dtype.type,
                        self.index.dtype.type,
                        shifts.dtype.type,
                    ](
                        nextshifts.data,
                        self.index.data,
                        self.index.length,
                        shifts.data,
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
                    out.content,
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

        return out

    def _validityerror(self, path):
        assert self.index.nplike is self._nplike
        error = self._nplike["awkward_IndexedArray_validity", self.index.dtype.type](
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

    def bytemask(self):
        out = ak._v2.index.Index8.empty(self.index.length, self._nplike)
        assert out.nplike is self._nplike and self._index.nplike is self._nplike
        self._handle_error(
            self._nplike[
                "awkward_IndexedArray_mask", out.dtype.type, self._index.dtype.type
            ](out.data, self._index.data, self._index.length)
        )
        return out

    def _rpad(self, target, axis, depth, clip):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self.rpad_axis0(target, clip)
        elif posaxis == depth + 1:
            mask = self.bytemask()
            index = ak._v2.index.Index64.empty(mask.length, self._nplike)
            assert index.nplike is self._nplike and mask.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1",
                    index.dtype.type,
                    mask.dtype.type,
                ](index.data, mask.data, mask.length)
            )
            next = self.project()._rpad(target, posaxis, depth, clip)
            return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                index,
                next,
                None,
                self._parameters,
                self._nplike,
            ).simplify_optiontype()
        else:
            return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                self._index,
                self._content._rpad(target, posaxis, depth, clip),
                None,
                self._parameters,
                self._nplike,
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

        next = ak._v2.contents.IndexedArray(
            ak._v2.index.Index(index),
            self._content,
            None,
            next_parameters,
            self._nplike,
        )
        return next._to_arrow(
            pyarrow,
            self,
            ak._v2._connect.pyarrow.and_validbytes(validbytes, this_validbytes),
            length,
            options,
        )

    def _to_numpy(self, allow_missing):
        content = ak._v2.operations.convert.to_numpy(
            self.project(), allow_missing=allow_missing
        )

        shape = list(content.shape)
        shape[0] = self.length
        data = numpy.empty(shape, dtype=content.dtype)
        mask0 = numpy.asarray(self.bytemask()).view(np.bool_)
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
                return numpy.ma.MaskedArray(data, mask)
            else:
                raise ValueError(
                    "ak.to_numpy cannot convert 'None' values to "
                    "np.ma.MaskedArray unless the "
                    "'allow_missing' parameter is set to True"
                )
        else:
            if allow_missing:
                return numpy.ma.MaskedArray(content)
            else:
                return content

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
                    self._nplike,
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
        original_index = self._index.raw(self._nplike)

        is_none = original_index < 0
        num_none = self._nplike.count_nonzero(is_none)
        if self._content.length > len(original_index) - num_none:
            new_index = self._nplike.empty(
                len(original_index), dtype=original_index.dtype
            )
            new_index[is_none] = -1
            new_index[~is_none] = self._nplike.arange(
                len(original_index) - num_none, dtype=original_index.dtype
            )
            return ak._v2.contents.IndexedOptionArray(
                ak._v2.index.Index(new_index),
                self.project().packed(),
                self._identifier,
                self._parameters,
                self._nplike,
            )

        else:
            return ak._v2.contents.IndexedOptionArray(
                self._index,
                self._content.packed(),
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def _to_list(self, behavior):
        out = self._to_list_custom(behavior)
        if out is not None:
            return out

        index = self._index.raw(numpy)
        content = self._content._to_list(behavior)
        out = [None] * len(index)
        for i, ind in enumerate(index):
            if ind >= 0:
                out[i] = content[ind]
        return out

    def _to_nplike(self, nplike):
        index = self._index._to_nplike(nplike)
        content = self._content._to_nplike(nplike)
        return IndexedOptionArray(
            index, content, self.identifier, self.parameters, nplike=nplike
        )

    def _to_json(
        self,
        nan_string,
        infinity_string,
        minus_infinity_string,
        complex_real_string,
        complex_imag_string,
    ):
        out = self._to_json_custom()
        if out is not None:
            return out

        index = self._index.raw(numpy)
        content = self._content._to_json(
            nan_string,
            infinity_string,
            minus_infinity_string,
            complex_real_string,
            complex_imag_string,
        )
        out = [None] * len(index)
        for i, ind in enumerate(index):
            if ind >= 0:
                out[i] = content[ind]
        return out
