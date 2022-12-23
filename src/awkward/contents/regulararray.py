# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy

import awkward as ak
from awkward._util import unset
from awkward.contents.content import Content
from awkward.forms.regularform import RegularForm
from awkward.typing import Final, Self

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


class RegularArray(Content):
    is_list = True
    is_regular = True

    def __init__(self, content, size, zeros_length=0, *, parameters=None):
        if not isinstance(content, Content):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'content' must be a Content subtype, not {}".format(
                        type(self).__name__, repr(content)
                    )
                )
            )
        if not isinstance(size, ak._typetracer.UnknownLengthType):
            if not (ak._util.is_integer(size) and size >= 0):
                raise ak._errors.wrap_error(
                    TypeError(
                        "{} 'size' must be a non-negative integer, not {}".format(
                            type(self).__name__, size
                        )
                    )
                )
            else:
                size = int(size)
        if not isinstance(zeros_length, ak._typetracer.UnknownLengthType):
            if not (ak._util.is_integer(zeros_length) and zeros_length >= 0):
                raise ak._errors.wrap_error(
                    TypeError(
                        "{} 'zeros_length' must be a non-negative integer, not {}".format(
                            type(self).__name__, zeros_length
                        )
                    )
                )

        if parameters is not None and parameters.get("__array__") == "string":
            if not content.is_numpy or not content.parameter("__array__") == "char":
                raise ak._errors.wrap_error(
                    ValueError(
                        "{} is a string, so its 'content' must be uint8 NumpyArray of char, not {}".format(
                            type(self).__name__, repr(content)
                        )
                    )
                )
        if parameters is not None and parameters.get("__array__") == "bytestring":
            if not content.is_numpy or not content.parameter("__array__") == "byte":
                raise ak._errors.wrap_error(
                    ValueError(
                        "{} is a bytestring, so its 'content' must be uint8 NumpyArray of byte, not {}".format(
                            type(self).__name__, repr(content)
                        )
                    )
                )

        self._content = content
        self._size = size
        if size != 0:
            self._length = content.length // size  # floor division
        else:
            self._length = zeros_length
        self._init(parameters, content.backend)

    @property
    def content(self):
        return self._content

    @property
    def size(self):
        return self._size

    form_cls: Final = RegularForm

    def copy(self, content=unset, size=unset, zeros_length=unset, *, parameters=unset):
        return RegularArray(
            self._content if content is unset else content,
            self._size if size is unset else size,
            self._length if zeros_length is unset else zeros_length,
            parameters=self._parameters if parameters is unset else parameters,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            content=copy.deepcopy(self._content, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(cls, content, size, zeros_length=0, *, parameters=None):
        return cls(content, size, zeros_length, parameters=parameters)

    @property
    def offsets(self):
        return self._compact_offsets64(True)

    @property
    def starts(self):
        return self._compact_offsets64(True)[:-1]

    @property
    def stops(self):
        return self._compact_offsets64(True)[1:]

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.form_cls(
            self._content._form_with_key(getkey),
            self._size,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, backend):
        assert isinstance(form, self.form_cls)
        self._content._to_buffers(form.content, getkey, container, backend)

    def _to_typetracer(self, forget_length: bool) -> Self:
        return RegularArray(
            self._content._to_typetracer(forget_length),
            self._size,
            ak._typetracer.UnknownLength if forget_length else self._length,
            parameters=self._parameters,
        )

    def _touch_data(self, recursive):
        if recursive:
            self._content._touch_data(recursive)

    def _touch_shape(self, recursive):
        if recursive:
            self._content._touch_shape(recursive)

    @property
    def length(self):
        return self._length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<RegularArray size="]
        out.append(repr(str(self._size)))
        out.append(" len=")
        out.append(repr(str(self._length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</RegularArray>")
        out.append(post)
        return "".join(out)

    def to_ListOffsetArray64(self, start_at_zero=False):
        offsets = self._compact_offsets64(start_at_zero)
        return self._broadcast_tooffsets64(offsets)

    def to_RegularArray(self):
        return self

    def maybe_to_NumpyArray(self) -> ak.contents.NumpyArray | None:
        content = self._content[: self._length * self._size].maybe_to_NumpyArray()

        if content is not None:
            shape = (self._length, self._size) + content.data.shape[1:]
            return ak.contents.NumpyArray(
                content.data.reshape(shape),
                parameters=ak._util.merge_parameters(
                    self._parameters, content.parameters
                ),
                backend=content.backend,
            )
        else:
            return None

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if self._backend.nplike.known_shape and where < 0:
            where += self._length

        if where < 0 or where >= self._length:
            raise ak._errors.index_error(self, where)
        start, stop = (where) * self._size, (where + 1) * self._size
        return self._content._getitem_range(slice(start, stop))

    def _getitem_range(self, where):
        if not self._backend.nplike.known_shape:
            self._touch_shape(recursive=False)
            return self

        start, stop, step = where.indices(self._length)
        assert step == 1
        zeros_length = stop - start
        substart, substop = start * self._size, stop * self._size
        return RegularArray(
            self._content._getitem_range(slice(substart, substop)),
            self._size,
            zeros_length,
            parameters=self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return RegularArray(
            self._content._getitem_field(where, only_fields),
            self._size,
            self._length,
            parameters=None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return RegularArray(
            self._content._getitem_fields(where, only_fields),
            self._size,
            self._length,
            parameters=None,
        )

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak.index.Index)

        where = carry.data

        copied = allow_lazy == "copied"
        if not issubclass(where.dtype.type, np.int64):
            where = where.astype(np.int64)
            copied = True

        negative = where < 0
        if self._backend.index_nplike.any(negative, prefer=False):
            if not copied:
                where = where.copy()
                copied = True
            where[negative] += self._length

        if self._backend.index_nplike.any(where >= self._length, prefer=False):
            raise ak._errors.index_error(self, where)

        nextcarry = ak.index.Index64.empty(
            where.shape[0] * self._size, self._backend.index_nplike
        )

        assert nextcarry.nplike is self._backend.index_nplike
        self._handle_error(
            self._backend[
                "awkward_RegularArray_getitem_carry",
                nextcarry.dtype.type,
                where.dtype.type,
            ](
                nextcarry.data,
                where,
                where.shape[0],
                self._size,
            ),
            slicer=carry,
        )

        return RegularArray(
            self._content._carry(nextcarry, allow_lazy),
            self._size,
            where.shape[0],
            parameters=self._parameters,
        )

    def _compact_offsets64(self, start_at_zero):
        out = ak.index.Index64.empty((self._length + 1), self._backend.index_nplike)
        assert out.nplike is self._backend.index_nplike
        self._handle_error(
            self._backend["awkward_RegularArray_compact_offsets", out.dtype.type](
                out.data,
                self._length,
                self._size,
            )
        )
        return out

    def _broadcast_tooffsets64(self, offsets):
        if offsets.nplike.known_data and (offsets.length == 0 or offsets[0] != 0):
            raise ak._errors.wrap_error(
                AssertionError(
                    "broadcast_tooffsets64 can only be used with offsets that start at 0, not {}".format(
                        "(empty)" if offsets.length == 0 else str(offsets[0])
                    )
                )
            )

        if offsets.nplike.known_shape and offsets.length - 1 != self._length:
            raise ak._errors.wrap_error(
                AssertionError(
                    "cannot broadcast RegularArray of length {} to length {}".format(
                        self._length, offsets.length - 1
                    )
                )
            )

        if self._size == 1:
            carrylen = offsets[-1]
            nextcarry = ak.index.Index64.empty(carrylen, self._backend.index_nplike)
            assert (
                nextcarry.nplike is self._backend.index_nplike
                and offsets.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_RegularArray_broadcast_tooffsets_size1",
                    nextcarry.dtype.type,
                    offsets.dtype.type,
                ](
                    nextcarry.data,
                    offsets.data,
                    offsets.length,
                )
            )
            nextcontent = self._content._carry(nextcarry, True)
            return ak.contents.ListOffsetArray(
                offsets, nextcontent, parameters=self._parameters
            )

        else:
            assert offsets.nplike is self._backend.index_nplike
            self._handle_error(
                self._backend[
                    "awkward_RegularArray_broadcast_tooffsets", offsets.dtype.type
                ](
                    offsets.data,
                    offsets.length,
                    self._size,
                )
            )
            return ak.contents.ListOffsetArray(
                offsets, self._content, parameters=self._parameters
            )

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        out = self.to_ListOffsetArray64(True)
        return out._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, int):
            nexthead, nexttail = ak._slicing.headtail(tail)
            nextcarry = ak.index.Index64.empty(self._length, self._backend.index_nplike)
            assert nextcarry.nplike is self._backend.index_nplike
            self._handle_error(
                self._backend[
                    "awkward_RegularArray_getitem_next_at", nextcarry.dtype.type
                ](
                    nextcarry.data,
                    head,
                    self._length,
                    self._size,
                ),
                slicer=head,
            )
            nextcontent = self._content._carry(nextcarry, True)
            return nextcontent._getitem_next(nexthead, nexttail, advanced)

        elif isinstance(head, slice):
            nexthead, nexttail = ak._slicing.headtail(tail)
            start, stop, step = head.indices(self._size)

            nextsize = 0
            if step > 0 and stop - start > 0:
                diff = stop - start
                nextsize = diff // step
                if diff % step != 0:
                    nextsize += 1
            elif step < 0 and stop - start < 0:
                diff = start - stop
                nextsize = diff // (step * -1)
                if diff % step != 0:
                    nextsize += 1

            nextcarry = ak.index.Index64.empty(
                self._length * nextsize, self._backend.index_nplike
            )
            assert nextcarry.nplike is self._backend.index_nplike
            self._handle_error(
                self._backend[
                    "awkward_RegularArray_getitem_next_range",
                    nextcarry.dtype.type,
                ](
                    nextcarry.data,
                    start,
                    step,
                    self._length,
                    self._size,
                    nextsize,
                ),
                slicer=head,
            )

            nextcontent = self._content._carry(nextcarry, True)

            if advanced is None or advanced.length == 0:
                return RegularArray(
                    nextcontent._getitem_next(nexthead, nexttail, advanced),
                    nextsize,
                    self._length,
                    parameters=self._parameters,
                )
            else:
                nextadvanced = ak.index.Index64.empty(
                    self._length * nextsize, self._backend.index_nplike
                )
                advanced = advanced.to_nplike(self._backend.index_nplike)
                assert (
                    nextadvanced.nplike is self._backend.index_nplike
                    and advanced.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_RegularArray_getitem_next_range_spreadadvanced",
                        nextadvanced.dtype.type,
                        advanced.dtype.type,
                    ](
                        nextadvanced.data,
                        advanced.data,
                        self._length,
                        nextsize,
                    ),
                    slicer=head,
                )
                return RegularArray(
                    nextcontent._getitem_next(nexthead, nexttail, nextadvanced),
                    nextsize,
                    self._length,
                    parameters=self._parameters,
                )

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak.index.Index64):
            head = head.to_nplike(self._backend.index_nplike)
            nexthead, nexttail = ak._slicing.headtail(tail)
            flathead = self._backend.index_nplike.asarray(head.data.reshape(-1))

            regular_flathead = ak.index.Index64.empty(
                flathead.shape[0], self._backend.index_nplike
            )
            assert regular_flathead.nplike is self._backend.index_nplike
            self._handle_error(
                self._backend[
                    "awkward_RegularArray_getitem_next_array_regularize",
                    regular_flathead.dtype.type,
                    flathead.dtype.type,
                ](
                    regular_flathead.data,
                    flathead,
                    flathead.shape[0],
                    self._size,
                ),
                slicer=head,
            )

            if advanced is None or advanced.length == 0:
                nextcarry = ak.index.Index64.empty(
                    self._length * flathead.shape[0], self._backend.index_nplike
                )
                nextadvanced = ak.index.Index64.empty(
                    self._length * flathead.shape[0], self._backend.index_nplike
                )
                assert (
                    nextcarry.nplike is self._backend.index_nplike
                    and nextadvanced.nplike is self._backend.index_nplike
                    and regular_flathead.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_RegularArray_getitem_next_array",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.data,
                        nextadvanced.data,
                        regular_flathead.data,
                        self._length,
                        regular_flathead.length,
                        self._size,
                    ),
                    slicer=head,
                )
                nextcontent = self._content._carry(nextcarry, True)

                out = nextcontent._getitem_next(nexthead, nexttail, nextadvanced)
                if advanced is None:
                    return ak._slicing.getitem_next_array_wrap(
                        out, head.metadata.get("shape", (head.length,)), self._length
                    )
                else:
                    return out

            elif self._size == 0:
                nextcarry = ak.index.Index64.empty(0, nplike=self._backend.index_nplike)
                nextadvanced = ak.index.Index64.empty(
                    0, nplike=self._backend.index_nplike
                )
                nextcontent = self._content._carry(nextcarry, True)
                return nextcontent._getitem_next(nexthead, nexttail, nextadvanced)

            else:
                nextcarry = ak.index.Index64.empty(
                    self._length, self._backend.index_nplike
                )
                nextadvanced = ak.index.Index64.empty(
                    self._length, self._backend.index_nplike
                )
                advanced = advanced.to_nplike(self._backend.index_nplike)
                assert (
                    nextcarry.nplike is self._backend.index_nplike
                    and nextadvanced.nplike is self._backend.index_nplike
                    and advanced.nplike is self._backend.index_nplike
                    and regular_flathead.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_RegularArray_getitem_next_array_advanced",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        advanced.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.data,
                        nextadvanced.data,
                        advanced.data,
                        regular_flathead.data,
                        self._length,
                        regular_flathead.length,
                        self._size,
                    ),
                    slicer=head,
                )
                nextcontent = self._content._carry(nextcarry, True)
                return nextcontent._getitem_next(nexthead, nexttail, nextadvanced)

        elif isinstance(head, ak.contents.ListOffsetArray):
            headlength = head.length
            head = head.to_backend(self._backend)

            if advanced is not None:
                raise ak._errors.index_error(
                    self,
                    head,
                    "cannot mix jagged slice with NumPy-style advanced indexing",
                )

            if self._backend.nplike.known_shape and head.length != self._size:
                raise ak._errors.index_error(
                    self,
                    head,
                    "cannot fit jagged slice with length {} into {} of size {}".format(
                        head.length, type(self).__name__, self._size
                    ),
                )

            regularlength = self._length
            singleoffsets = head._offsets
            multistarts = ak.index.Index64.empty(
                head.length * regularlength, self._backend.index_nplike
            )
            multistops = ak.index.Index64.empty(
                head.length * regularlength, self._backend.index_nplike
            )

            assert singleoffsets.nplike is self._backend.index_nplike
            self._handle_error(
                self._backend[
                    "awkward_RegularArray_getitem_jagged_expand",
                    multistarts.dtype.type,
                    multistops.dtype.type,
                    singleoffsets.dtype.type,
                ](
                    multistarts.data,
                    multistops.data,
                    singleoffsets.data,
                    head.length,
                    regularlength,
                ),
                slicer=head,
            )
            down = self._content._getitem_next_jagged(
                multistarts, multistops, head._content, tail
            )

            return RegularArray(
                down, headlength, self._length, parameters=self._parameters
            )

        elif isinstance(head, ak.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise ak._errors.wrap_error(AssertionError(repr(head)))

    def _offsets_and_flattened(self, axis, depth):
        return self.to_ListOffsetArray64(True)._offsets_and_flattened(axis, depth)

    def _mergeable_next(self, other, mergebool):
        if isinstance(
            other,
            (
                ak.contents.IndexedArray,
                ak.contents.IndexedOptionArray,
                ak.contents.ByteMaskedArray,
                ak.contents.BitMaskedArray,
                ak.contents.UnmaskedArray,
            ),
        ):
            return self._mergeable(other.content, mergebool)

        elif isinstance(
            other,
            (
                ak.contents.RegularArray,
                ak.contents.ListArray,
                ak.contents.ListOffsetArray,
            ),
        ):
            return self._content._mergeable(other.content, mergebool)

        # For n-dimensional NumpyArrays, let's now convert them to RegularArray
        # We could add a special case that tries to first convert self to NumpyArray
        # and merge conventionally, but it's not worth it at this stage.
        elif isinstance(other, ak.contents.NumpyArray) and other.purelist_depth > 1:
            return self._content._mergeable(other.to_RegularArray().content, mergebool)
        else:
            return False

    def _mergemany(self, others):
        if len(others) == 0:
            return self

        if any(x.is_option for x in others):
            return ak.contents.UnmaskedArray(self)._mergemany(others)

        # Regularize NumpyArray into RegularArray (or NumpyArray if 1D)
        others = [
            o.to_RegularArray() if isinstance(o, ak.contents.NumpyArray) else o
            for o in others
        ]

        if all(x.is_regular and x.size == self.size for x in others):
            parameters = self._parameters
            tail_contents = []
            zeros_length = self._length
            for x in others:
                parameters = ak._util.merge_parameters(parameters, x._parameters, True)
                tail_contents.append(x._content[: x._length * x._size])
                zeros_length += x._length

            return RegularArray(
                self._content[: self._length * self._size]._mergemany(tail_contents),
                self._size,
                zeros_length,
                parameters=parameters,
            )

        else:
            return self.to_ListOffsetArray64(True)._mergemany(others)

    def _fill_none(self, value: Content) -> Content:
        return RegularArray(
            self._content._fill_none(value),
            self._size,
            self._length,
            parameters=self._parameters,
        )

    def _local_index(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        elif posaxis is not None and posaxis + 1 == depth + 1:
            localindex = ak.index.Index64.empty(
                self._length * self._size, nplike=self._backend.index_nplike
            )
            self._handle_error(
                self._backend["awkward_RegularArray_localindex", np.int64](
                    localindex.data,
                    self._size,
                    self._length,
                )
            )
            return ak.contents.RegularArray(
                ak.contents.NumpyArray(localindex), self._size, self._length
            )
        else:
            return ak.contents.RegularArray(
                self._content._local_index(axis, depth + 1), self._size, self._length
            )

    def _numbers_to_type(self, name):
        return ak.contents.RegularArray(
            self._content._numbers_to_type(name),
            self._size,
            self._length,
            parameters=self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._length == 0:
            return True

        return self.to_ListOffsetArray64(True)._is_unique(
            negaxis,
            starts,
            parents,
            outlength,
        )

    def _unique(self, negaxis, starts, parents, outlength):
        if self._length == 0:
            return self
        out = self.to_ListOffsetArray64(True)._unique(
            negaxis,
            starts,
            parents,
            outlength,
        )

        if isinstance(out, ak.contents.RegularArray):
            if isinstance(out._content, ak.contents.ListOffsetArray):
                return ak.contents.RegularArray(
                    out._content.to_RegularArray(),
                    out._size,
                    out._length,
                    parameters=out._parameters,
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
        next = self.to_ListOffsetArray64(True)
        out = next._argsort_next(
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

        if isinstance(out, ak.contents.RegularArray):
            if isinstance(out._content, ak.contents.ListOffsetArray):
                return ak.contents.RegularArray(
                    out._content.to_RegularArray(),
                    out._size,
                    out._length,
                    parameters=out._parameters,
                )

        return out

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        out = self.to_ListOffsetArray64(True)._sort_next(
            negaxis,
            starts,
            parents,
            outlength,
            ascending,
            stable,
            kind,
            order,
        )

        # FIXME
        # if isinstance(out, ak.contents.RegularArray):
        #     if isinstance(out._content, ak.contents.ListOffsetArray):
        #         return ak.contents.RegularArray(
        #             out._content.to_RegularArray(),
        #             out._size,
        #             out._length,
        #             None,
        #             out._parameters,
        #             self._backend.nplike,
        #         )

        return out

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        elif posaxis is not None and posaxis + 1 == depth + 1:
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise ak._errors.wrap_error(
                    ValueError(
                        "ak.combinations does not compute combinations of the characters of a string; please split it into lists"
                    )
                )

            size = self._size
            if replacement:
                size = size + (n - 1)
            thisn = n
            if thisn > size:
                combinationslen = 0
            elif thisn == size:
                combinationslen = 1
            else:
                if thisn * 2 > size:
                    thisn = size - thisn
                combinationslen = size
                for j in range(2, thisn + 1):
                    combinationslen = combinationslen * (size - j + 1)
                    combinationslen = combinationslen // j

            totallen = combinationslen * self._length
            tocarryraw = self._backend.index_nplike.empty(n, dtype=np.intp)
            tocarry = []
            for i in range(n):
                ptr = ak.index.Index64.empty(
                    totallen,
                    nplike=self._backend.index_nplike,
                    dtype=np.int64,
                )
                tocarry.append(ptr)
                if self._backend.nplike.known_data:
                    tocarryraw[i] = ptr.ptr

            toindex = ak.index.Index64.empty(
                n, self._backend.index_nplike, dtype=np.int64
            )
            fromindex = ak.index.Index64.empty(
                n, self._backend.index_nplike, dtype=np.int64
            )

            if self._size != 0:
                assert (
                    toindex.nplike is self._backend.index_nplike
                    and fromindex.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
                        "awkward_RegularArray_combinations_64",
                        np.int64,
                        toindex.data.dtype.type,
                        fromindex.data.dtype.type,
                    ](
                        tocarryraw,
                        toindex.data,
                        fromindex.data,
                        n,
                        replacement,
                        self._size,
                        self._length,
                    )
                )

            contents = []
            length = None
            for ptr in tocarry:
                contents.append(self._content._carry(ptr, True))
                length = contents[-1].length
            assert length is not None
            recordarray = ak.contents.RecordArray(
                contents,
                recordlookup,
                length,
                parameters=parameters,
                backend=self._backend,
            )
            return ak.contents.RegularArray(
                recordarray, combinationslen, self._length, parameters=self._parameters
            )
        else:
            next = self._content._getitem_range(
                slice(0, self._length * self._size)
            )._combinations(n, replacement, recordlookup, parameters, axis, depth + 1)
            return ak.contents.RegularArray(
                next, self._size, self._length, parameters=self._parameters
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
        branch, depth = self.branch_depth
        nextlen = len(self) * self._size

        if not branch and negaxis == depth:
            if self.size == 0:
                nextstarts = ak.index.Index64(
                    self._backend.index_nplike.full(0, len(self)),
                    nplike=self._backend.index_nplike,
                )
            else:
                nextstarts = ak.index.Index64(
                    self._backend.index_nplike.arange(0, nextlen, self.size),
                    nplike=self._backend.index_nplike,
                )
                assert nextstarts.length == len(self)

            nextcarry = ak.index.Index64.empty(
                nextlen, nplike=self._backend.index_nplike
            )
            nextparents = ak.index.Index64.empty(
                nextlen, nplike=self._backend.index_nplike
            )
            assert (
                parents.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
                and nextparents.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_RegularArray_reduce_nonlocal_preparenext",
                    nextcarry.dtype.type,
                    nextparents.dtype.type,
                    parents.dtype.type,
                ](
                    nextcarry.data,
                    nextparents.data,
                    parents.data,
                    self._size,
                    len(self),
                )
            )

            if reducer.needs_position:
                # Regular arrays have the same length rows, so there can be no "missing" values
                # unlike ragged list types
                nextshifts = ak.index.Index64.zeros(
                    nextcarry.length, nplike=self._backend.index_nplike
                )
            else:
                nextshifts = None

            nextcontent = self._content._carry(nextcarry, False)
            outcontent = nextcontent._reduce_next(
                reducer,
                negaxis - 1,
                nextstarts,
                nextshifts,
                nextparents,
                # We want a result of length
                outlength * self.size,
                mask,
                False,
                behavior,
            )

            out = ak.contents.RegularArray(
                outcontent, self._size, outlength, parameters=None
            )

            if keepdims:
                out = ak.contents.RegularArray(out, 1, self.length, parameters=None)
            return out
        else:
            nextparents = ak.index.Index64.empty(nextlen, self._backend.index_nplike)

            assert nextparents.nplike is self._backend.index_nplike
            self._handle_error(
                self._backend[
                    "awkward_RegularArray_reduce_local_nextparents",
                    nextparents.dtype.type,
                ](
                    nextparents.data,
                    self._size,
                    len(self),
                )
            )

            if self._size > 0:
                nextstarts = ak.index.Index64(
                    self._backend.index_nplike.arange(0, len(nextparents), self._size),
                    nplike=self._backend.index_nplike,
                )
            else:
                assert len(nextparents) == 0
                nextstarts = ak.index.Index64(
                    self._backend.index_nplike.zeros(0),
                    nplike=self._backend.index_nplike,
                )

            outcontent = self._content._reduce_next(
                reducer,
                negaxis,
                nextstarts,
                shifts,
                nextparents,
                len(self),
                mask,
                keepdims,
                behavior,
            )

            # Understanding this logic is nontrivial without examples. So, let's start with a
            # somewhat simple example of `5 * 4 * 3 * int64`. This is composed of the following layouts
            #                                     ^........^ NumpyArray(5*4*3)
            #                                 ^............^ RegularArray(..., size=3)
            #                             ^................^ RegularArray(..., size=4)
            # If the reduction axis is 2 (or -1), then the resulting shape is `5 * 4 * int64`.
            # This corresponds to `RegularArray(NumpyArray(5*4), size=4)`, i.e. the `size=3` layout
            # *disappears*, and is replaced by the bare reduction `NumpyArray`. Therefore, if
            # this layout is _directly_ above the reduction, it won't survive the reduction.
            #
            # The `_reduce_next` mechanism returns its first result (by recursion) from the
            # leaf `NumpyArray`. The parent `RegularArray` takes the `negaxis != depth` branch,
            # and asks the `NumpyArray` to perform the reduction. The `_reduce_next` is such that the
            # callee must wrap the reduction result into a list whose length is given by the *caller*,
            # i.e. the parent. Therefore, the `RegularArray(..., size=3)` layout reduces its content with
            # `outlength=len(self)=5*4`. The parent of this `RegularArray` (with `size=4`)
            # will reduce its contents with `outlength=5`. It is *this* reduction that must be returned as a
            # `RegularArray(..., size=4)`. Clearly, the `RegularArray(..., size=3)` layout has disappeared
            # and been replaced by a `NumpyArray` of the appropriate size for the `RegularArray(..., size=4)`
            # to wrap. Hence, we only want to interpret the content as a `RegularArray(..., size=self._size)`
            # iff. we are not the direct parent of the reduced layout.

            # At `depth == negaxis+1`, we are above the dimension being reduced. All implemented
            # _dimensional_ types should return `RegularArray` for `keepdims=True`, so we don't need
            # to convert `outcontent` to a `RegularArray`
            if keepdims and depth == negaxis + 1:
                assert outcontent.is_regular
            # At `depth >= negaxis + 2`, we are wrapping at _least_ one other list type. This list-type
            # may return a ListOffsetArray, which we need to convert to a `RegularArray`. We know that
            # the result can be reinterpreted as a `RegularArray`, because it's not the immediate parent
            # of the reduction, so it must exist in the type.
            elif depth >= negaxis + 2:
                if outcontent.is_list:
                    # Let's only deal with ListOffsetArray
                    outcontent = outcontent.to_ListOffsetArray64(False)
                    # Fast-path to convert data that we know should be regular (avoid a kernel call)
                    start, stop = (
                        outcontent.offsets[0],
                        outcontent.offsets[outcontent.offsets.length - 1],
                    )

                    trimmed = outcontent.content._getitem_range(slice(start, stop))
                    assert len(trimmed) == self._size * len(outcontent)

                    outcontent = ak.contents.RegularArray(
                        trimmed,
                        size=self._size,
                        zeros_length=len(self),
                    )
                else:
                    assert outcontent.is_regular

            outoffsets = ak.index.Index64.empty(
                outlength + 1, self._backend.index_nplike
            )
            assert (
                outoffsets.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_outoffsets_64",
                    outoffsets.dtype.type,
                    parents.dtype.type,
                ](
                    outoffsets.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )

            return ak.contents.ListOffsetArray(outoffsets, outcontent, parameters=None)

    def _validity_error(self, path):
        if self.size < 0:
            return f'at {path} ("{type(self)}"): size < 0'

        return self._content._validity_error(path + ".content")

    def _nbytes_part(self):
        return self.content._nbytes_part()

    def _pad_none(self, target, axis, depth, clip):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._pad_none_axis0(target, clip)

        elif posaxis is not None and posaxis + 1 == depth + 1:
            if not clip:
                if target < self._size:
                    return self
                else:
                    return self._pad_none(target, axis, depth, True)

            else:
                index = ak.index.Index64.empty(
                    self._length * target, self._backend.index_nplike
                )
                assert index.nplike is self._backend.index_nplike
                self._handle_error(
                    self._backend[
                        "awkward_RegularArray_rpad_and_clip_axis1", index.dtype.type
                    ](index.data, target, self._size, self._length)
                )
                next = ak.contents.IndexedOptionArray.simplified(
                    index, self._content, parameters=self._parameters
                )
                return ak.contents.RegularArray(
                    next,
                    target,
                    self._length,
                    parameters=self._parameters,
                )

        else:
            return ak.contents.RegularArray(
                self._content._pad_none(target, axis, depth + 1, clip),
                self._size,
                self._length,
                parameters=self._parameters,
            )

    def _to_numpy(self, allow_missing):
        array_param = self.parameter("__array__")
        if array_param in {"bytestring", "string"}:
            return self._backend.nplike.array(self.to_list())

        out = self._content.to_numpy(allow_missing)
        shape = (self._length, self._size) + out.shape[1:]
        return out[: self._length * self._size].reshape(shape)

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        if self.parameter("__array__") == "string":
            return self.to_ListOffsetArray64(False)._to_arrow(
                pyarrow, mask_node, validbytes, length, options
            )

        is_bytestring = self.parameter("__array__") == "bytestring"

        akcontent = self._content[: self._length * self._size]

        if is_bytestring:
            assert isinstance(akcontent, ak.contents.NumpyArray)

            return pyarrow.Array.from_buffers(
                ak._connect.pyarrow.to_awkwardarrow_type(
                    pyarrow.binary(self._size),
                    options["extensionarray"],
                    options["record_is_scalar"],
                    mask_node,
                    self,
                ),
                self._length,
                [
                    ak._connect.pyarrow.to_validbits(validbytes),
                    pyarrow.py_buffer(akcontent._raw(numpy)),
                ],
            )

        else:
            paarray = akcontent._to_arrow(
                pyarrow, None, None, self._length * self._size, options
            )

            content_type = pyarrow.list_(paarray.type).value_field.with_nullable(
                akcontent.is_option
            )

            return pyarrow.Array.from_buffers(
                ak._connect.pyarrow.to_awkwardarrow_type(
                    pyarrow.list_(content_type, self._size),
                    options["extensionarray"],
                    options["record_is_scalar"],
                    mask_node,
                    self,
                ),
                self._length,
                [
                    ak._connect.pyarrow.to_validbits(validbytes),
                ],
                children=[paarray],
                null_count=ak._connect.pyarrow.to_null_count(
                    validbytes, options["count_nulls"]
                ),
            )

    def _completely_flatten(self, backend, options):
        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            return [self]
        else:
            flat = self._content[: self._length * self._size]
            return flat._completely_flatten(backend, options)

    def _drop_none(self):
        return self.toListOffsetArray64()._drop_none()

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if self._backend.nplike.known_shape:
            content = self._content[: self._length * self._size]
        else:
            self._touch_data(recursive=False)
            content = self._content

        if options["return_array"]:

            def continuation():
                return RegularArray(
                    content._recursively_apply(
                        action,
                        behavior,
                        depth + 1,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    ),
                    self._size,
                    self._length,
                    parameters=self._parameters if options["keep_parameters"] else None,
                )

        else:

            def continuation():
                content._recursively_apply(
                    action,
                    behavior,
                    depth + 1,
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
            backend=self._backend,
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise ak._errors.wrap_error(AssertionError(result))

    def to_packed(self) -> Self:
        length = self._length * self._size
        if self._content.length == length:
            content = self._content.to_packed()
        else:
            content = self._content[:length].to_packed()

        return RegularArray(
            content, self._size, self._length, parameters=self._parameters
        )

    def _to_list(self, behavior, json_conversions):
        if self.parameter("__array__") == "bytestring":
            convert_bytes = (
                None if json_conversions is None else json_conversions["convert_bytes"]
            )
            content = ak._util.tobytes(self._content.data)
            length, size = self._length, self._size
            out = [None] * length
            if convert_bytes is None:
                for i in range(length):
                    out[i] = content[(i) * size : (i + 1) * size]
            else:
                for i in range(length):
                    out[i] = convert_bytes(content[(i) * size : (i + 1) * size])
            return out

        elif self.parameter("__array__") == "string":
            data = self._content.data
            if hasattr(data, "tobytes"):

                def tostring(x):
                    return x.tobytes().decode(errors="surrogateescape")

            else:

                def tostring(x):
                    return x.tostring().decode(errors="surrogateescape")

            length, size = self._length, self._size
            out = [None] * length
            for i in range(length):
                out[i] = tostring(data[(i) * size : (i + 1) * size])
            return out

        else:
            out = self._to_list_custom(behavior, json_conversions)
            if out is not None:
                return out

            content = self._content._to_list(behavior, json_conversions)
            length, size = self._length, self._size
            out = [None] * length
            for i in range(length):
                out[i] = content[(i) * size : (i + 1) * size]
            return out

    def to_backend(self, backend: ak._backends.Backend) -> Self:
        content = self._content.to_backend(backend)
        return RegularArray(
            content, self._size, zeros_length=self._length, parameters=self._parameters
        )

    def _is_equal_to(self, other, index_dtype, numpyarray):
        return self._size == other.size and self._content.is_equal_to(
            self._content, index_dtype, numpyarray
        )
