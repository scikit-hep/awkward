# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy

import awkward as ak
from awkward._v2.index import Index
from awkward._v2.contents.content import Content, unset
from awkward._v2.forms.indexedoptionform import IndexedOptionForm

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class IndexedOptionArray(Content):
    is_OptionType = True
    is_IndexedType = True

    def copy(
        self,
        index=unset,
        content=unset,
        identifier=unset,
        parameters=unset,
        nplike=unset,
    ):
        return IndexedOptionArray(
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

    def toByteMaskedArray(self, valid_when):
        mask = ak._v2.index.Index8(self.mask_as_bool(valid_when, self._nplike))

        carry = self._index.data
        too_negative = carry < -1
        if self._nplike.any(too_negative, prefer=False):
            carry = carry.copy()
            carry[too_negative] = -1
        carry = ak._v2.index.Index(carry)

        if self._content.length == 0:
            content = self._content.dummy()._carry(carry, False)
        else:
            content = self._content._carry(carry, False)

        return ak._v2.contents.ByteMaskedArray(
            mask,
            content,
            valid_when,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def toBitMaskedArray(self, valid_when, lsb_order):
        return self.toByteMaskedArray(valid_when).toBitMaskedArray(
            valid_when, lsb_order
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
            raise ak._v2._util.indexerror(self, where)
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

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextindex = self._index[carry.data]
        except IndexError as err:
            raise ak._v2._util.indexerror(self, carry.data, str(err)) from err

        return IndexedOptionArray(
            nextindex,
            self._content,
            self._carry_identifier(carry),
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
            raise ak._v2._util.indexerror(
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

            numnull, nextcarry, outindex = self._nextcarry_outindex(self._nplike)

            next = self._content._carry(nextcarry, True)
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
        _, nextcarry, outindex = self._nextcarry_outindex(self._nplike)
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
            numnull, nextcarry, outindex = self._nextcarry_outindex(self._nplike)
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
            self._nplike["awkward_IndexedArray_fill_count", index.dtype.type](
                index.data,
                0,
                theirlength,
                0,
            )
        )
        reinterpreted_index = ak._v2.index.Index(
            self._nplike.index_nplike.asarray(self.index.data), nplike=self.nplike
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

        contents = [self._content, value]
        tags = ak._v2.index.Index8(self.mask_as_bool(valid_when=False))
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

    def _local_index(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._local_index_axis0()
        else:
            _, nextcarry, outindex = self._nextcarry_outindex(self._nplike)

            next = self._content._carry(nextcarry, False)
            out = next._local_index(posaxis, depth)
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

        next = self._content._carry(nextcarry, False)
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

        next, nextparents, numnull, outindex = self._rearrange_prepare_next(parents)

        out = next._unique(
            negaxis,
            starts,
            nextparents,
            outlength,
        )

        if branch or (negaxis is not None and negaxis != depth):
            nextoutindex = ak._v2.index.Index64.empty(parents.length, self._nplike)
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
                    parents.length,
                    nextparents.data,
                    nextparents.length,
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
            # We do not pass None values to the content, so it will have computed
            # the unique _non null_ values. We therefore need to account for None
            # values in the result. We do this by creating an IndexedOptionArray
            # and tacking the index -1 onto the end
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
                out.length,
                0,
                None,
                self._parameters,
                self._nplike,
            )

        return out

    def _rearrange_nextshifts(self, nextparents, shifts):
        nextshifts = ak._v2.index.Index64.empty(
            nextparents.length,
            self._nplike,
        )
        assert nextshifts.nplike is self._nplike

        if shifts is None:
            assert (
                nextshifts.nplike is self._nplike and self._index.nplike is self._nplike
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
        return nextshifts

    def _rearrange_prepare_next(self, parents):
        assert self._index.nplike is self._nplike and parents.nplike is self._nplike
        index_length = self._index.length
        numnull = ak._v2.index.Index64.empty(1, self._nplike)
        assert numnull.nplike is self._nplike

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
        return next, nextparents, numnull, outindex

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
        assert (
            starts.nplike is self._nplike
            and parents.nplike is self._nplike
            and self._index.nplike is self._nplike
        )

        branch, depth = self.branch_depth

        next, nextparents, numnull, outindex = self._rearrange_prepare_next(parents)

        if (not branch) and negaxis == depth:
            nextshifts = self._rearrange_nextshifts(nextparents, shifts)
        else:
            nextshifts = None

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

        # `next._argsort_next` is given the non-None values. We choose to
        # sort None values to the end of the list, meaning we need to grow `out`
        # to account for these None values. First, we locate these nones within
        # their sublists
        nulls_merged = False
        nulls_index = ak._v2.index.Index64.empty(numnull[0], self._nplike)
        assert nulls_index.nplike is self._nplike
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
        # If we wrap a NumpyArray (i.e., axis=-1), then we want `argmax` to return
        # the indices of each `None` value, rather than `None` itself.
        # We can test for this condition by seeing whether the NumpyArray of indices
        # is mergeable with our content (`out = next._argsort_next result`).
        # If so, try to concatenate them at the end of `out`.`
        nulls_index_content = ak._v2.contents.NumpyArray(
            nulls_index, None, None, self._nplike
        )
        if out.mergeable(nulls_index_content, True):
            out = out.merge(nulls_index_content)
            nulls_merged = True

        nextoutindex = ak._v2.index.Index64.empty(parents.length, self._nplike)
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
                parents.length,
                nextparents.data,
                nextparents.length,
            )
        )

        if nulls_merged:
            # awkward_IndexedArray_local_preparenext_64 uses -1 to
            # indicate `None` values. Given that this code-path runs
            # only when the `None` value indices are explicitly stored in out,
            # we need to mapping the -1 values to their corresponding indices
            # in `out`
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

        inject_nones = (
            True if (numnull[0] > 0 and not branch and negaxis != depth) else False
        )

        # If we want the None's at this depth to be injected
        # into the dense ([x y z None None]) rearranger result.
        # Here, we index the dense content with an index
        # that maps the values to the correct locations
        if inject_nones:
            return ak._v2.contents.IndexedOptionArray(
                outindex,
                out,
                None,
                self._parameters,
                self._nplike,
            ).simplify_optiontype()
        # Otherwise, if we are rearranging (e.g sorting) the contents of this layout,
        # then we do NOT want to return an optional layout,
        # OR we are branching
        else:
            return out

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        assert starts.nplike is self._nplike and parents.nplike is self._nplike
        branch, depth = self.branch_depth

        next, nextparents, numnull, outindex = self._rearrange_prepare_next(parents)

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

        nextoutindex = ak._v2.index.Index64.empty(parents.length, self._nplike)
        assert nextoutindex.nplike is self._nplike

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
                parents.length,
                nextparents.data,
                nextparents.length,
            )
        )
        out = ak._v2.contents.IndexedOptionArray(
            nextoutindex,
            out,
            None,
            self._parameters,
            self._nplike,
        ).simplify_optiontype()

        inject_nones = True if not branch and negaxis != depth else False

        # If we want the None's at this depth to be injected
        # into the dense ([x y z None None]) rearranger result.
        # Here, we index the dense content with an index
        # that maps the values to the correct locations
        if inject_nones:
            return ak._v2.contents.IndexedOptionArray(
                outindex,
                out,
                None,
                self._parameters,
                self._nplike,
            ).simplify_optiontype()
        # Otherwise, if we are rearranging (e.g sorting) the contents of this layout,
        # then we do NOT want to return an optional layout
        # OR we are branching
        else:
            return out

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
        branch, depth = self.branch_depth

        next, nextparents, numnull, outindex = self._rearrange_prepare_next(parents)

        if reducer.needs_position and (not branch and negaxis == depth):
            nextshifts = self._rearrange_nextshifts(nextparents, shifts)
        else:
            nextshifts = None

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
            behavior,
        )

        # If we are reducing the contents of this layout,
        # then we do NOT want to return an optional layout
        if not branch and negaxis == depth:
            return out
        else:
            if isinstance(out, ak._v2.contents.RegularArray):
                out = out.toListOffsetArray64(True)

            # If the result of `_reduce_next` is a list, and we're not applying at this
            # depth, then it will have offsets given by the boundaries in parents.
            # This means that we need to look at the _contents_ to which the `outindex`
            # belongs to add the option type
            if isinstance(out, ak._v2.contents.ListOffsetArray):
                if starts.nplike.known_data and starts.length > 0 and starts[0] != 0:
                    raise ak._v2._util.error(
                        AssertionError(
                            "reduce_next with unbranching depth > negaxis expects a "
                            "ListOffsetArray whose offsets start at zero ({})".format(
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
                        outindex.length,
                    )
                )

                # Apply `outindex` to appropriate content
                inner = ak._v2.contents.IndexedOptionArray(
                    outindex,
                    out._content,
                    None,
                    self._parameters,
                    self._nplike,
                ).simplify_optiontype()

                # Re-wrap content
                return ak._v2.contents.ListOffsetArray(
                    outoffsets,
                    inner,
                    None,
                    self._parameters,
                    self._nplike,
                )

            else:
                raise ak._v2._util.error(
                    AssertionError(
                        "reduce_next with unbranching depth > negaxis is only "
                        "expected to return RegularArray or ListOffsetArray or "
                        "IndexedOptionArray; "
                        "instead, it returned " + out
                    )
                )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            _, nextcarry, outindex = self._nextcarry_outindex(self._nplike)
            next = self._content._carry(nextcarry, True)
            out = next._combinations(
                n, replacement, recordlookup, parameters, posaxis, depth
            )
            out2 = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                outindex, out, None, parameters, self._nplike
            )
            return out2.simplify_optiontype()

    def _validity_error(self, path):
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
            mask = ak._v2.index.Index8(self.mask_as_bool(valid_when=False))
            index = ak._v2.index.Index64.empty(mask.length, self._nplike)
            assert index.nplike is self._nplike and mask.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1",
                    index.dtype.type,
                    mask.dtype.type,
                ](index.data, mask.data, mask.length)
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
            return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                self._index,
                self._content._pad_none(target, posaxis, depth, clip),
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
        content = ak._v2.operations.to_numpy(
            self.project(), allow_missing=allow_missing
        )

        shape = list(content.shape)
        shape[0] = self.length
        data = numpy.empty(shape, dtype=content.dtype)
        mask0 = numpy.asarray(self.mask_as_bool(valid_when=False)).view(np.bool_)
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

                if issubclass(content.dtype.type, (bool, np.bool_)):
                    data[mask0] = False
                elif issubclass(content.dtype.type, np.floating):
                    data[mask0] = np.nan
                elif issubclass(content.dtype.type, np.complexfloating):
                    data[mask0] = np.nan + np.nan * 1j
                elif issubclass(content.dtype.type, np.integer):
                    data[mask0] = np.iinfo(content.dtype).max
                elif issubclass(content.dtype.type, (np.datetime64, np.timedelta64)):
                    data[mask0] = numpy.array([np.iinfo(np.int64).max], content.dtype)[
                        0
                    ]
                else:
                    raise ak._v2._util.error(
                        AssertionError(f"unrecognized dtype: {content.dtype}")
                    )

                return numpy.ma.MaskedArray(data, mask)
            else:
                raise ak._v2._util.error(
                    ValueError(
                        "ak.to_numpy cannot convert 'None' values to "
                        "np.ma.MaskedArray unless the "
                        "'allow_missing' parameter is set to True"
                    )
                )
        else:
            if allow_missing:
                return numpy.ma.MaskedArray(content)
            else:
                return content

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
            npselect = npindex >= 0
            if self._nplike.index_nplike.any(npselect):
                indexmin = npindex[npselect].min()
                index = ak._v2.index.Index(npindex - indexmin, nplike=self._nplike)
                content = self._content[indexmin : npindex.max() + 1]
            else:
                index, content = self._index, self._content
        else:
            index, content = self._index, self._content

        if options["return_array"]:

            def continuation():
                return IndexedOptionArray(
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
        original_index = self._index.raw(self._nplike)
        is_none = original_index < 0
        num_none = self._nplike.index_nplike.count_nonzero(is_none)
        if self.parameter("__array__") == "categorical" or self._content.length <= (
            len(original_index) - num_none
        ):
            return ak._v2.contents.IndexedOptionArray(
                self._index,
                self._content.packed(),
                self._identifier,
                self._parameters,
                self._nplike,
            )

        else:
            new_index = self._nplike.index_nplike.empty(
                len(original_index), dtype=original_index.dtype
            )
            new_index[is_none] = -1
            new_index[~is_none] = self._nplike.index_nplike.arange(
                len(original_index) - num_none,
                dtype=original_index.dtype,
            )
            return ak._v2.contents.IndexedOptionArray(
                ak._v2.index.Index(new_index, nplike=self.nplike),
                self.project().packed(),
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def _to_list(self, behavior, json_conversions):
        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        index = self._index.raw(numpy)
        not_missing = index >= 0

        nextcontent = self._content._carry(
            ak._v2.index.Index(index[not_missing]), False
        )
        out = nextcontent._to_list(behavior, json_conversions)

        for i, isvalid in enumerate(not_missing):
            if not isvalid:
                out.insert(i, None)

        return out

    def _to_nplike(self, nplike):
        index = self._index._to_nplike(nplike)
        content = self._content._to_nplike(nplike)
        return IndexedOptionArray(
            index, content, self.identifier, self.parameters, nplike=nplike
        )

    def _layout_equal(self, other, index_dtype=True, numpyarray=True):
        return self.index.layout_equal(
            other.index, index_dtype, numpyarray
        ) and self.content.layout_equal(other.content, index_dtype, numpyarray)
