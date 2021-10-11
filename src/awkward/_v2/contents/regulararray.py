# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import numpy as np

import awkward as ak
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.regularform import RegularForm


class RegularArray(Content):
    def __init__(self, content, size, zeros_length=0, identifier=None, parameters=None):
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not (ak._util.isint(size) and size >= 0):
            raise TypeError(
                "{0} 'size' must be a non-negative integer, not {1}".format(
                    type(self).__name__, size
                )
            )
        if not (ak._util.isint(zeros_length) and zeros_length >= 0):
            raise TypeError(
                "{0} 'zeros_length' must be a non-negative integer, not {1}".format(
                    type(self).__name__, zeros_length
                )
            )

        self._content = content
        self._size = int(size)
        if size != 0:
            self._length = len(content) // size  # floor division
        else:
            self._length = zeros_length
        self._init(identifier, parameters)

    @property
    def size(self):
        return self._size

    @property
    def content(self):
        return self._content

    @property
    def nplike(self):
        return self._content.nplike

    @property
    def nonvirtual_nplike(self):
        return self._content.nonvirtual_nplike

    Form = RegularForm

    @property
    def form(self):
        return self.Form(
            self._content.form,
            self._size,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    def __len__(self):
        return self._length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<RegularArray size="]
        out.append(repr(str(self._size)))
        out.append(" len=")
        out.append(repr(str(len(self))))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</RegularArray>")
        out.append(post)
        return "".join(out)

    def toListOffsetArray64(self, start_at_zero=False):
        offsets = self._compact_offsets64(start_at_zero)
        return self._broadcast_tooffsets64(offsets)

    def toRegularArray(self):
        return self

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)):
            raise NestedIndexError(self, where)
        start, stop = (where) * self._size, (where + 1) * self._size
        return self._content._getitem_range(slice(start, stop))

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        zeros_length = stop - start
        substart, substop = start * self._size, stop * self._size
        return RegularArray(
            self._content._getitem_range(slice(substart, substop)),
            self._size,
            zeros_length,
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        return RegularArray(
            self._content._getitem_field(where, only_fields),
            self._size,
            self._length,
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return RegularArray(
            self._content._getitem_fields(where, only_fields),
            self._size,
            self._length,
            self._fields_identifier(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        nplike, where = carry.nplike, carry.data

        copied = allow_lazy == "copied"
        if not issubclass(where.dtype.type, np.int64):
            where = where.astype(np.int64)
            copied = True

        negative = where < 0
        if nplike.any(negative):
            if not copied:
                where = where.copy()
                copied = True
            where[negative] += self._length

        if nplike.any(where >= self._length):
            raise NestedIndexError(self, where)

        nextcarry = ak._v2.index.Index64.empty(len(where) * self._size, nplike)

        self._handle_error(
            nplike[
                "awkward_RegularArray_getitem_carry",
                nextcarry.dtype.type,
                where.dtype.type,
            ](
                nextcarry.to(nplike),
                where,
                len(where),
                self._size,
            ),
            carry,
        )

        return RegularArray(
            self._content._carry(nextcarry, allow_lazy, exception),
            self._size,
            len(where),
            self._carry_identifier(carry, exception),
            self._parameters,
        )

    def _compact_offsets64(self, start_at_zero):
        nplike = self.nplike
        out = ak._v2.index.Index64.empty(self._length + 1, self.nplike)
        self._handle_error(
            nplike["awkward_RegularArray_compact_offsets", out.dtype.type](
                out.to(nplike),
                self._length,
                self._size,
            )
        )
        return out

    def _broadcast_tooffsets64(self, offsets):
        nplike = self.nplike
        if len(offsets) == 0 or offsets[0] != 0:
            raise AssertionError(
                "broadcast_tooffsets64 can only be used with offsets that start at 0, not {0}".format(
                    "(empty)" if len(offsets) == 0 else str(offsets[0])
                )
            )

        if len(offsets) - 1 != self._length:
            raise AssertionError(
                "cannot broadcast RegularArray of length {0} to length {1}".format(
                    self._length, len(offsets) - 1
                )
            )

        if self._identifier is not None:
            identifier = self._identifier[slice(0, len(offsets) - 1)]
        else:
            identifier = self._identifier

        if self._size == 1:
            carrylen = offsets[-1]
            nextcarry = ak._v2.index.Index64.empty(carrylen, nplike)
            self._handle_error(
                nplike[
                    "awkward_RegularArray_broadcast_tooffsets_size1",
                    nextcarry.dtype.type,
                    offsets.dtype.type,
                ](
                    nextcarry.to(nplike),
                    offsets.to(nplike),
                    len(offsets),
                )
            )
            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                offsets, nextcontent, identifier, self._parameters
            )

        else:
            self._handle_error(
                nplike["awkward_RegularArray_broadcast_tooffsets", offsets.dtype.type](
                    offsets.to(nplike),
                    len(offsets),
                    self._size,
                )
            )
            return ak._v2.contents.listoffsetarray.ListOffsetArray(
                offsets, self._content, self._identifier, self._parameters
            )

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        out = self.toListOffsetArray64(True)
        return out._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

    def maybe_to_nplike(self, nplike):
        out = self._content.maybe_to_nplike(nplike)
        if out is None:
            return out
        else:
            return out.reshape((len(self), -1) + out.shape[1:])

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, int):
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            nextcarry = ak._v2.index.Index64.empty(self._length, self.nplike)
            self._handle_error(
                self.nplike[
                    "awkward_RegularArray_getitem_next_at", nextcarry.dtype.type
                ](
                    nextcarry.to(self.nplike),
                    head,
                    self._length,
                    self._size,
                ),
                head,
            )
            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
            return nextcontent._getitem_next(nexthead, nexttail, advanced)

        elif isinstance(head, slice):
            nexthead, nexttail = ak._v2._slicing.headtail(tail)
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

            nextcarry = ak._v2.index.Index64.empty(self._length * nextsize, self.nplike)
            self._handle_error(
                self.nplike[
                    "awkward_RegularArray_getitem_next_range",
                    nextcarry.dtype.type,
                ](
                    nextcarry.to(self.nplike),
                    start,
                    step,
                    self._length,
                    self._size,
                    nextsize,
                ),
                head,
            )

            nextcontent = self._content._carry(nextcarry, True, NestedIndexError)

            if advanced is None or len(advanced) == 0:
                return RegularArray(
                    nextcontent._getitem_next(nexthead, nexttail, advanced),
                    nextsize,
                    self._length,
                    self._identifier,
                    self._parameters,
                )
            else:
                nplike = ak.nplike.of(advanced)
                nextadvanced = ak._v2.index.Index64.empty(
                    self._length * nextsize, nplike
                )
                self._handle_error(
                    nplike[
                        "awkward_RegularArray_getitem_next_range_spreadadvanced",
                        nextadvanced.dtype.type,
                        advanced.dtype.type,
                    ](
                        nextadvanced.to(nplike),
                        advanced.to(nplike),
                        self._length,
                        nextsize,
                    ),
                    head,
                )
                return RegularArray(
                    nextcontent._getitem_next(nexthead, nexttail, nextadvanced),
                    nextsize,
                    self._length,
                    self._identifier,
                    self._parameters,
                )

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.index.Index64):
            nplike = head.nplike

            nexthead, nexttail = ak._v2._slicing.headtail(tail)
            flathead = nplike.asarray(head.data.reshape(-1))

            regular_flathead = ak._v2.index.Index64.empty(len(flathead), nplike)
            self._handle_error(
                nplike[
                    "awkward_RegularArray_getitem_next_array_regularize",
                    regular_flathead.dtype.type,
                    flathead.dtype.type,
                ](
                    regular_flathead.to(nplike),
                    flathead,
                    len(flathead),
                    self._size,
                ),
                head,
            )

            if advanced is None or len(advanced) == 0:
                nextcarry = ak._v2.index.Index64.empty(
                    self._length * len(flathead), nplike
                )
                nextadvanced = ak._v2.index.Index64.empty(
                    self._length * len(flathead), nplike
                )
                self._handle_error(
                    nplike[
                        "awkward_RegularArray_getitem_next_array",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.to(nplike),
                        nextadvanced.to(nplike),
                        regular_flathead.to(nplike),
                        self._length,
                        len(regular_flathead),
                        self._size,
                    ),
                    head,
                )
                nextcontent = self._content._carry(nextcarry, True, NestedIndexError)

                out = nextcontent._getitem_next(nexthead, nexttail, nextadvanced)
                if advanced is None:
                    return ak._v2._slicing.getitem_next_array_wrap(
                        out, head.metadata.get("shape", (len(head),))
                    )
                else:
                    return out

            elif self._size == 0:
                nextcarry = ak._v2.index.Index64.empty(0, nplike)
                nextadvanced = ak._v2.index.Index64.empty(0, nplike)
                nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
                return nextcontent._getitem_next(nexthead, nexttail, nextadvanced)

            else:
                nextcarry = ak._v2.index.Index64.empty(self._length, nplike)
                nextadvanced = ak._v2.index.Index64.empty(self._length, nplike)
                self._handle_error(
                    nplike[
                        "awkward_RegularArray_getitem_next_array_advanced",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        advanced.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.to(nplike),
                        nextadvanced.to(nplike),
                        advanced.to(nplike),
                        regular_flathead.to(nplike),
                        self._length,
                        len(regular_flathead),
                        self._size,
                    ),
                    head,
                )
                nextcontent = self._content._carry(nextcarry, True, NestedIndexError)
                return nextcontent._getitem_next(nexthead, nexttail, nextadvanced)

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            nplike = head.nplike

            if advanced is not None:
                raise NestedIndexError(
                    self,
                    head,
                    "cannot mix jagged slice with NumPy-style advanced indexing",
                )

            if len(head) != self._size:
                raise NestedIndexError(
                    self,
                    head,
                    "cannot fit jagged slice with length {0} into {1} of size {2}".format(
                        len(head), type(self).__name__, self._size
                    ),
                )

            regularlength = self._length
            singleoffsets = head._offsets
            multistarts = ak._v2.index.Index64.empty(len(head) * regularlength, nplike)
            multistops = ak._v2.index.Index64.empty(len(head) * regularlength, nplike)

            self._handle_error(
                nplike[
                    "awkward_RegularArray_getitem_jagged_expand",
                    multistarts.dtype.type,
                    multistops.dtype.type,
                    singleoffsets.dtype.type,
                ](
                    multistarts.to(nplike),
                    multistops.to(nplike),
                    singleoffsets.to(nplike),
                    len(head),
                    regularlength,
                ),
            )
            down = self._content._getitem_next_jagged(
                multistarts, multistops, head._content, tail
            )

            return RegularArray(down, len(head), self._length, None, self._parameters)

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def mergemany(self, others):
        if len(others) == 0:
            return self
        return self.toListOffsetArray64(True).mergemany(others)

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        elif posaxis == depth + 1:
            localindex = ak._v2.index.Index64.empty(
                self._length * self._size, self.nplike
            )
            self._handle_error(
                localindex.nplike["awkward_RegularArray_localindex", np.int64](
                    localindex.to(localindex.nplike),
                    self._size,
                    self._length,
                )
            )

            return ak._v2.contents.RegularArray(
                ak._v2.contents.numpyarray.NumpyArray(localindex),
                self._size,
                self._length,
                self._identifier,
                self._parameters,
            )
        else:
            return ak._v2.contents.RegularArray(
                self._content._localindex(posaxis, depth + 1),
                self._size,
                self._length,
                self._identifier,
                self._parameters,
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
        if self._length == 0:
            return ak._v2.contents.NumpyArray(np.empty(0, np.int64))

        next = self.toListOffsetArray64(True)
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

        if isinstance(out, ak._v2.contents.RegularArray):
            if isinstance(out._content, ak._v2.contents.ListOffsetArray):
                return ak._v2.contents.RegularArray(
                    out._content.toRegularArray(),
                    out._size,
                    out._length,
                    None,
                    out._parameters,
                )

        return out

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        if self._length == 0:
            return self

        out = self.toListOffsetArray64(True)._sort_next(
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
        # if isinstance(out, ak._v2.contents.RegularArray):
        #     if isinstance(out._content, ak._v2.contents.ListOffsetArray):
        #         return ak._v2.contents.RegularArray(
        #             out._content.toRegularArray(),
        #             out._size,
        #             out._length,
        #             None,
        #             out._parameters,
        #         )

        return out

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        elif posaxis == depth + 1:
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise ValueError(
                    "ak.combinations does not compute combinations of the characters of a string; please split it into lists"
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

            totallen = combinationslen * len(self)
            tocarryraw = self.nplike.empty(n, dtype=np.intp)
            tocarry = []
            for i in range(n):
                ptr = ak._v2.index.Index64.empty(totallen, self.nplike, dtype=np.int64)
                tocarry.append(ptr)
                tocarryraw[i] = ptr.ptr

            toindex = ak._v2.index.Index64.empty(n, self.nplike, dtype=np.int64)
            fromindex = ak._v2.index.Index64.empty(n, self.nplike, dtype=np.int64)

            if self._size != 0:
                self._handle_error(
                    self.nplike[
                        "awkward_RegularArray_combinations_64",
                        np.int64,
                        toindex.to(self.nplike).dtype.type,
                        fromindex.to(self.nplike).dtype.type,
                    ](
                        tocarryraw,
                        toindex.to(self.nplike),
                        fromindex.to(self.nplike),
                        n,
                        replacement,
                        self._size,
                        len(self),
                    )
                )

            contents = []
            for ptr in tocarry:
                contents.append(self._content._carry(ptr, True, NestedIndexError))
            recordarray = ak._v2.contents.recordarray.RecordArray(
                contents, recordlookup, parameters=parameters
            )
            return ak._v2.contents.regulararray.RegularArray(
                recordarray,
                combinationslen,
                len(self),
                self._identifier,
                self._parameters,
            )
        else:
            next = self._content._getitem_range(
                slice(0, len(self) * self._size)
            )._combinations(
                n, replacement, recordlookup, parameters, posaxis, depth + 1
            )
            return ak._v2.contents.regulararray.RegularArray(
                next, self._size, len(self), self._identifier, self._parameters
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
        out = self.toListOffsetArray64(True)._reduce_next(
            reducer,
            negaxis,
            starts,
            shifts,
            parents,
            outlength,
            mask,
            keepdims,
        )

        if not self._content.dimension_optiontype:
            branch, depth = self._content.branch_depth
            convert_shallow = negaxis == depth
            convert_deep = negaxis + 2 == depth

            if keepdims:
                convert_shallow = False
                convert_deep = True

            if convert_deep:
                if isinstance(out, ak._v2.contents.ListOffsetArray):
                    if isinstance(out.content, ak._v2.contents.ListOffsetArray):
                        out = ak._v2.contents.ListOffsetArray(
                            out.offsets,
                            out.content.toRegularArray(),
                            out.identifier,
                            out.parameters,
                        )
                    elif isinstance(out.content, ak._v2.contents.ListArray):
                        out = ak._v2.contents.ListOffsetArray(
                            out.offsets,
                            out.content.toRegularArray(),
                            out.identifier,
                            out.parameters,
                        )
                elif isinstance(out, ak._v2.contents.ListArray):
                    if isinstance(out.content, ak._v2.contents.ListOffsetArray):
                        out = ak._v2.contents.ListOffsetArray(
                            out.offsets,
                            out.content.toRegularArray(),
                            out.identifier,
                            out.parameters,
                        )
                    elif isinstance(out.content, ak._v2.contents.ListArray):
                        out = ak._v2.contents.ListOffsetArray(
                            out.offsets,
                            out.content.toRegularArray(),
                            out.identifier,
                            out.parameters,
                        )

            if convert_shallow:
                if isinstance(out, ak._v2.contents.ListOffsetArray):
                    out = out.toRegularArray()
                elif isinstance(out, ak._v2.contents.ListArray):
                    out = out.toRegularArray()

        return out

    def _validityerror(self, path):
        if self.size < 0:
            return 'at {0} ("{1}"): size < 0'.format(path, type(self))
        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            return ""
        else:
            return self.content.validityerror(path + ".content")
