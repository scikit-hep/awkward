# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.numpyform import NumpyForm
from awkward._v2.forms.form import _parameters_equal
from awkward._v2.types.numpytype import primitive_to_dtype

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class NumpyArray(Content):
    is_NumpyType = True

    def __init__(self, data, identifier=None, parameters=None, nplike=None):
        self._nplike = ak.nplike.of(data) if nplike is None else nplike
        self._data = self._nplike.asarray(data)

        ak._v2.types.numpytype.dtype_to_primitive(self._data.dtype)
        if len(self._data.shape) == 0:
            raise TypeError(
                "{0} 'data' must be an array, not {1}".format(
                    type(self).__name__, repr(data)
                )
            )

        self._init(identifier, parameters)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def inner_shape(self):
        return self._data.shape[1:]

    @property
    def strides(self):
        return self._data.strides

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def nplike(self):
        return self._nplike

    @property
    def ptr(self):
        return self._data.ctypes.data

    Form = NumpyForm

    def _form_with_key(self, getkey):
        return self.Form(
            ak._v2.types.numpytype.dtype_to_primitive(self._data.dtype),
            self._data.shape[1:],
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=getkey(self),
        )

    def _to_buffers(self, form, getkey, container, nplike):
        assert isinstance(form, self.Form)
        key = getkey(self, form, "data")
        container[key] = ak._v2._util.little_endian(self.to(nplike))

    @property
    def typetracer(self):
        return NumpyArray(
            self.to(ak._v2._typetracer.TypeTracer.instance()),
            self._typetracer_identifier(),
            self._parameters,
            nplike=ak._v2._typetracer.TypeTracer.instance(),
        )

    def __len__(self):
        return len(self._data)

    def to(self, nplike):
        return nplike.asarray(self._data)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<NumpyArray dtype="]
        out.append(repr(str(self.dtype)))
        if len(self._data.shape) == 1:
            out.append(" len=" + repr(str(self._data.shape[0])))
        else:
            out.append(
                " shape='({0})'".format(", ".join(str(x) for x in self._data.shape))
            )

        extra = self._repr_extra(indent + "    ")
        arraystr_lines = self._nplike.array_str(self._data, max_line_width=30).split(
            "\n"
        )
        if len(extra) != 0 or len(arraystr_lines) > 1:
            arraystr_lines = self._nplike.array_str(
                self._data, max_line_width=max(80 - len(indent) - 4, 40)
            ).split("\n")
            if len(arraystr_lines) > 5:
                arraystr_lines = arraystr_lines[:2] + [" ..."] + arraystr_lines[-2:]
            out.append(">")
            out.extend(extra)
            out.append("\n" + indent + "    ")
            out.append(("\n" + indent + "    ").join(arraystr_lines))
            out.append("\n" + indent + "</NumpyArray>")
        else:
            out.append(">")
            out.append(arraystr_lines[0])
            out.append("</NumpyArray>")

        out.append(post)
        return "".join(out)

    def merge_parameters(self, parameters):
        return NumpyArray(
            self._data,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
            self._nplike,
        )

    def toRegularArray(self):
        if len(self._data.shape) == 1:
            return self
        else:
            return ak._v2.contents.RegularArray(
                NumpyArray(
                    self._data.reshape((-1,) + self._data.shape[2:]),
                    None,
                    None,
                    nplike=self._nplike,
                ).toRegularArray(),
                self._data.shape[1],
                self._data.shape[0],
                self._identifier,
                self._parameters,
            )

    def maybe_to_nplike(self, nplike):
        return nplike.asarray(self._data)

    def _getitem_nothing(self):
        tmp = self._data[0:0]
        return NumpyArray(
            tmp.reshape((0,) + tmp.shape[2:]),
            self._range_identifier(0, 0),
            None,
            nplike=self._nplike,
        )

    def _getitem_at(self, where):
        try:
            out = self._data[where]
        except IndexError as err:
            raise NestedIndexError(self, where, str(err))

        if hasattr(out, "shape") and len(out.shape) != 0:
            return NumpyArray(out, None, None, nplike=self._nplike)
        else:
            return out

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1

        try:
            out = self._data[where]
        except IndexError as err:
            raise NestedIndexError(self, where, str(err))

        return NumpyArray(
            out,
            self._range_identifier(start, stop),
            self._parameters,
            nplike=self._nplike,
        )

    def _getitem_field(self, where, only_fields=()):
        raise NestedIndexError(self, where, "not an array of records")

    def _getitem_fields(self, where, only_fields=()):
        if len(where) == 0:
            return self._getitem_range(slice(0, 0))
        raise NestedIndexError(self, where, "not an array of records")

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)
        try:
            nextdata = self._data[carry.data]
        except IndexError as err:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, str(err))
            else:
                raise exception(str(err))

        return NumpyArray(
            nextdata,
            self._carry_identifier(carry, exception),
            self._parameters,
            nplike=self._nplike,
        )

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        if self._data.ndim == 1:
            raise NestedIndexError(
                self,
                ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
                "too many jagged slice dimensions for array",
            )
        else:
            next = self.toRegularArray()
            return next._getitem_next_jagged(
                slicestarts, slicestops, slicecontent, tail
            )

    def _getitem_next(self, head, tail, advanced):
        nplike = self._nplike

        if head == ():
            return self

        elif isinstance(head, int):
            where = (slice(None), head) + tail

            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))

            if hasattr(out, "shape") and len(out.shape) != 0:
                return NumpyArray(out, None, None, nplike=nplike)
            else:
                return out

        elif isinstance(head, slice) or head is np.newaxis or head is Ellipsis:
            where = (slice(None), head) + tail
            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))
            out2 = NumpyArray(out, None, self._parameters, nplike=nplike)
            return out2

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif isinstance(head, ak._v2.index.Index64):
            if advanced is None:
                where = (slice(None), head.data) + tail
            else:
                where = (nplike.asarray(advanced.data), head.data) + tail

            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))

            return NumpyArray(out, None, self._parameters, nplike=nplike)

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            where = (slice(None), head) + tail
            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))
            out2 = NumpyArray(out, None, self._parameters, nplike=nplike)
            return out2

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            next = self.toRegularArray()
            return next._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            out = ak._v2.index.Index64.empty(1, self.nplike)
            out[0] = len(self)
            return ak._v2.contents.numpyarray.NumpyArray(out)[0]
        shape = []
        reps = 1
        size = len(self)
        i = 0
        while i < self._data.ndim - 1 and depth < posaxis:
            shape.append(self.shape[i])
            reps *= self.shape[i]
            size = self.shape[i + 1]
            i += 1
            depth += 1
        if posaxis > depth:
            raise np.AxisError(
                "axis={0} exceeds the depth of this array ({1})".format(axis, depth)
            )

        tonum = ak._v2.index.Index64.empty(reps, self.nplike)
        self._handle_error(
            self.nplike["awkward_RegularArray_num", tonum.dtype.type](
                tonum.to(self.nplike), size, reps
            )
        )
        return ak._v2.contents.numpyarray.NumpyArray(
            tonum.data.reshape(shape), None, self.parameters
        )

    def _offsets_and_flattened(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            raise np.AxisError(self, "axis=0 not allowed for flatten")

        elif len(self.shape) != 1:
            return self.toRegularArray()._offsets_and_flattened(posaxis, depth)

        else:
            raise np.AxisError(self, "axis out of range for flatten")

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

        elif isinstance(
            other,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):
            return self.mergeable(other.content, mergebool)

        if self._data.ndim == 0:
            return False

        if isinstance(other, ak._v2.contents.numpyarray.NumpyArray):
            if self._data.ndim != other.data.ndim:
                return False

            if self.dtype != other.dtype and (
                self.dtype == np.datetime64 or other.dtype == np.datetime64
            ):
                return False

            if self.dtype != other.dtype and (
                self.dtype == np.timedelta64 or other.dtype == np.timedelta64
            ):
                return False

            if len(self.shape) > 1 and len(self.shape) != (other.shape):
                return False

            return True
        else:
            return False

    def mergemany(self, others):
        if len(others) == 0:
            return self
        head, tail = self._merging_strategy(others)

        contiguous_arrays = []

        for array in head:
            parameters = ak._v2._util.merge_parameters(
                self._parameters, array._parameters
            )
            if isinstance(array, ak._v2.contents.emptyarray.EmptyArray):
                pass
            elif isinstance(array, ak._v2.contents.numpyarray.NumpyArray):
                contiguous_arrays.append(array.data)
            else:
                raise AssertionError(
                    "cannot merge "
                    + type(self).__name__
                    + " with "
                    + type(array).__name__
                )

        contiguous_arrays = self.nplike.concatenate(contiguous_arrays)

        next = NumpyArray(contiguous_arrays, self._identifier, parameters)

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed.mergemany(tail[1:])

    def fillna(self, value):
        return self

    def _localindex(self, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        elif len(self.shape) <= 1:
            raise np.AxisError(self, "'axis' out of range for localindex")
        else:
            return self.toRegularArray()._localindex(posaxis, depth)

    def contiguous(self):
        if self.is_contiguous:
            return self
        else:
            return ak._v2.contents.NumpyArray(
                self._nplike.ascontiguousarray(self._data),
                self._identifier,
                self._parameters,
                self._nplike,
            )

    @property
    def is_contiguous(self):
        if isinstance(self._nplike, ak._v2._typetracer.TypeTracer):
            return True

        # Alternatively, self._data.flags["C_CONTIGUOUS"], but the following assumes
        # less of the nplike.

        x = self._data.dtype.itemsize

        for i in range(len(self._data.shape), 0, -1):
            if x != self._data.strides[i - 1]:
                return False
            else:
                x = x * self._data.shape[i - 1]

        return True

    def _subranges_equal(self, starts, stops, length, sorted=True):
        nplike = self.nplike

        is_equal = ak._v2.index.Index64.zeros(1, nplike)

        tmp = nplike.empty(length, self.dtype)
        self._handle_error(
            nplike[  # noqa: E231
                "awkward_NumpyArray_fill",
                self.dtype.type,
                self._data.dtype.type,
            ](
                tmp,
                0,
                self._data,
                length,
            )
        )

        if not sorted:
            tmp_beg_ptr = ak._v2.index.Index64.empty(ak._util.kMaxLevels, nplike)
            tmp_end_ptr = ak._v2.index.Index64.empty(ak._util.kMaxLevels, nplike)

            self._handle_error(
                nplike[
                    "awkward_quick_sort",
                    self.dtype.type,
                    tmp_beg_ptr.dtype.type,
                    tmp_end_ptr.dtype.type,
                    starts.dtype.type,
                    stops.dtype.type,
                ](
                    tmp,
                    tmp_beg_ptr.to(nplike),
                    tmp_end_ptr.to(nplike),
                    starts.to(nplike),
                    stops.to(nplike),
                    True,
                    len(starts),
                    ak._util.kMaxLevels,
                )
            )
        self._handle_error(
            nplike[
                "awkward_NumpyArray_subrange_equal",
                self.dtype.type,
                starts.dtype.type,
                stops.dtype.type,
                np.bool_,
            ](
                tmp,
                starts.to(nplike),
                stops.to(nplike),
                len(starts),
                is_equal.to(nplike),
            )
        )

        return True if is_equal[0] == 1 else False

    def _as_unique_strings(self, offsets):
        nplike = self.nplike

        outoffsets = ak._v2.index.Index64.empty(len(offsets), nplike)
        out = nplike.empty(self.shape[0], self.dtype)

        self._handle_error(
            nplike[
                "awkward_NumpyArray_sort_asstrings_uint8",
                self.dtype.type,
                self._data.dtype.type,
                offsets._data.dtype.type,
                outoffsets.dtype.type,
            ](
                out,
                self._data,
                offsets.to(nplike),
                len(offsets),
                outoffsets.to(nplike),
                True,
                False,
            )
        )

        outlength = ak._v2.index.Index64.empty(1, nplike)
        nextoffsets = ak._v2.index.Index64.empty(len(offsets), nplike)
        self._handle_error(
            nplike[
                "awkward_NumpyArray_unique_strings",
                self.dtype.type,
                outoffsets.dtype.type,
                nextoffsets.dtype.type,
                outlength.dtype.type,
            ](
                out,
                outoffsets.to(nplike),
                len(offsets),
                nextoffsets.to(nplike),
                outlength.to(nplike),
            )
        )
        out2 = NumpyArray(out, None, self._parameters, nplike=nplike)

        return out2, nextoffsets[: outlength[0]]

    def numbers_to_type(self, name):
        dtype = primitive_to_dtype(name)
        return NumpyArray(
            self.nplike.asarray(self._data, dtype=dtype),
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if len(self._data) == 0:
            return True

        if len(self.shape) == 0:
            return True

        elif len(self.shape) != 1 or not self.is_contiguous:
            contiguous_self = self if self.is_contiguous else self.contiguous()
            return contiguous_self.toRegularArray()._is_unique(
                negaxis,
                starts,
                parents,
                outlength,
            )
        else:
            out = self._unique(negaxis, starts, parents, outlength)
            if isinstance(out, ak._v2.contents.ListOffsetArray):
                return len(out.content) == len(self)

            return len(out) == len(self._data)

    def _unique(self, negaxis, starts, parents, outlength):
        if self.shape[0] == 0:
            return self

        if len(self.shape) == 0:
            return self

        nplike = self.nplike

        if negaxis is None:
            contiguous_self = self if self.is_contiguous else self.contiguous()
            flattened_shape = 1
            for i in range(len(contiguous_self.shape)):
                flattened_shape = flattened_shape * self.shape[i]

            offsets = ak._v2.index.Index64.zeros(2, nplike)
            offsets[1] = flattened_shape
            dtype = (
                np.dtype(np.int64)
                if self._data.dtype.kind.upper() == "M"
                else self._data.dtype
            )
            out = nplike.empty(offsets[1], dtype)
            self._handle_error(
                nplike[  # noqa: E231
                    "awkward_sort",
                    dtype.type,
                    dtype.type,
                    offsets.dtype.type,
                ](
                    out,
                    contiguous_self._data,
                    offsets[1],
                    offsets.to(nplike),
                    2,
                    offsets[1],
                    True,
                    False,
                )
            )

            nextlength = ak._v2.index.Index64.zeros(1, nplike)
            self._handle_error(
                nplike[  # noqa: E231
                    "awkward_unique",
                    out.dtype.type,
                    nextlength.dtype.type,
                ](
                    out,
                    len(out),
                    nextlength.to(nplike),
                )
            )

            return ak._v2.contents.NumpyArray(
                nplike.asarray(out[: nextlength[0]], self.dtype)
            )

        # axis is not None
        if len(self.shape) != 1 or not self.is_contiguous:
            contiguous_self = self if self.is_contiguous else self.contiguous()
            return contiguous_self.toRegularArray()._unique(
                negaxis,
                starts,
                parents,
                outlength,
            )
        else:

            parents_length = len(parents)
            offsets_length = ak._v2.index.Index64.empty(1, nplike)
            self._handle_error(
                nplike[
                    "awkward_sorting_ranges_length",
                    offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    offsets_length.to(nplike),
                    parents.to(nplike),
                    parents_length,
                )
            )

            offsets = ak._v2.index.Index64.empty(offsets_length[0], nplike)
            self._handle_error(
                nplike[  # noqa: E231
                    "awkward_sorting_ranges",
                    offsets.dtype.type,
                    parents.dtype.type,
                ](
                    offsets.to(nplike),
                    offsets_length[0],
                    parents.to(nplike),
                    parents_length,
                )
            )

            out = nplike.empty(len(self._data), self.dtype)
            self._handle_error(
                nplike[
                    "awkward_sort",
                    out.dtype.type,
                    self._data.dtype.type,
                    offsets.dtype.type,
                ](
                    out,
                    self._data,
                    self.shape[0],
                    offsets.to(nplike),
                    offsets_length[0],
                    parents_length,
                    True,
                    False,
                )
            )

            nextoffsets = ak._v2.index.Index64.zeros(len(offsets), nplike)
            self._handle_error(
                nplike[
                    "awkward_unique_ranges",
                    out.dtype.type,
                    offsets.dtype.type,
                    nextoffsets.dtype.type,
                ](
                    out,
                    len(out),
                    offsets.to(nplike),
                    len(offsets),
                    nextoffsets.to(nplike),
                )
            )

            outoffsets = ak._v2.index.Index64.zeros(len(starts) + 1, nplike)

            self._handle_error(
                nplike[
                    "awkward_unique_offsets",
                    outoffsets.dtype.type,
                    nextoffsets.dtype.type,
                    starts.dtype.type,
                ](
                    outoffsets.to(nplike),
                    len(nextoffsets),
                    nextoffsets.to(nplike),
                    starts.to(nplike),
                    len(starts),
                )
            )

            return ak._v2.contents.ListOffsetArray(
                outoffsets,
                ak._v2.contents.NumpyArray(out),
                None,
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
        if self.shape[0] == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))

        if len(self.shape) == 0:
            raise TypeError(
                "{0} attempting to argsort a scalar ".format(type(self).__name__)
            )
        elif len(self.shape) != 1 or not self.is_contiguous:
            contiguous_self = self if self.is_contiguous else self.contiguous()
            return contiguous_self.toRegularArray()._argsort_next(
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

        else:
            nplike = self.nplike

            parents_length = len(parents)
            offsets_length = ak._v2.index.Index64.empty(1, nplike)
            self._handle_error(
                nplike[
                    "awkward_sorting_ranges_length",
                    offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    offsets_length.to(nplike),
                    parents.to(nplike),
                    parents_length,
                )
            )
            offsets_length = offsets_length[0]

            offsets = ak._v2.index.Index64.empty(offsets_length, nplike)
            self._handle_error(
                nplike[
                    "awkward_sorting_ranges",
                    offsets.dtype.type,
                    parents.dtype.type,
                ](
                    offsets.to(nplike),
                    offsets_length,
                    parents.to(nplike),
                    parents_length,
                )
            )

            dtype = (
                np.dtype(np.int64)
                if self._data.dtype.kind.upper() == "M"
                else self._data.dtype
            )
            nextcarry = ak._v2.index.Index64.empty(self.__len__(), nplike)
            self._handle_error(
                nplike[
                    "awkward_argsort",
                    nextcarry.dtype.type,
                    dtype.type,
                    offsets.dtype.type,
                ](
                    nextcarry.to(nplike),
                    self._data,
                    self.__len__(),
                    offsets.to(nplike),
                    offsets_length,
                    ascending,
                    stable,
                )
            )

            if shifts is not None:
                self._handle_error(
                    nplike[
                        "awkward_NumpyArray_rearrange_shifted",
                        nextcarry.dtype.type,
                        shifts.dtype.type,
                        offsets.dtype.type,
                        parents.dtype.type,
                        starts.dtype.type,
                    ](
                        nextcarry.to(nplike),
                        shifts.to(nplike),
                        len(shifts),
                        offsets.to(nplike),
                        offsets_length,
                        parents.to(nplike),
                        parents_length,
                        starts.to(nplike),
                        len(starts),
                    )
                )
            out = NumpyArray(nextcarry)
            return out

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        if len(self.shape) == 0:
            raise TypeError(
                "{0} attempting to sort a scalar ".format(type(self).__name__)
            )

        elif len(self.shape) != 1 or not self.is_contiguous:
            contiguous_self = self if self.is_contiguous else self.contiguous()
            return contiguous_self.toRegularArray()._sort_next(
                negaxis,
                starts,
                parents,
                outlength,
                ascending,
                stable,
                kind,
                order,
            )

        else:
            nplike = self.nplike

            parents_length = len(parents)
            offsets_length = ak._v2.index.Index64.empty(1, nplike)
            self._handle_error(
                nplike[
                    "awkward_sorting_ranges_length",
                    offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    offsets_length.to(nplike),
                    parents.to(nplike),
                    parents_length,
                )
            )

            offsets = ak._v2.index.Index64.zeros(offsets_length[0], nplike)

            self._handle_error(
                nplike[
                    "awkward_sorting_ranges",
                    offsets.dtype.type,
                    parents.dtype.type,
                ](
                    offsets.to(nplike),
                    offsets_length[0],
                    parents.to(nplike),
                    parents_length,
                )
            )

            dtype = (
                np.dtype(np.int64)
                if self._data.dtype.kind.upper() == "M"
                else self._data.dtype
            )
            out = nplike.empty(len(self._data), dtype)
            self._handle_error(
                nplike[  # noqa: E231
                    "awkward_sort",
                    dtype.type,
                    dtype.type,
                    offsets.dtype.type,
                ](
                    out,
                    self._data,
                    self.shape[0],
                    offsets.to(nplike),
                    offsets_length[0],
                    parents_length,
                    ascending,
                    stable,
                )
            )
            return ak._v2.contents.NumpyArray(nplike.asarray(out, self.dtype))

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        elif len(self.shape) <= 1:
            raise np.AxisError("'axis' out of range for combinations")
        else:
            return self.toRegularArray()._combinations(
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

        out = reducer.apply(self, parents, outlength)

        if reducer.needs_position:
            if shifts is None:
                self._handle_error(
                    nplike[
                        "awkward_NumpyArray_reduce_adjust_starts_64",
                        out.data.dtype.type,
                        parents.dtype.type,
                        starts.dtype.type,
                    ](
                        out.data,
                        outlength,
                        parents.to(nplike),
                        starts.to(nplike),
                    )
                )
            else:
                self._handle_error(
                    nplike[
                        "awkward_NumpyArray_reduce_adjust_starts_shifts_64",
                        out.data.dtype.type,
                        parents.dtype.type,
                        starts.dtype.type,
                        shifts.dtype.type,
                    ](
                        out.data,
                        outlength,
                        parents.to(nplike),
                        starts.to(nplike),
                        shifts.to(nplike),
                    )
                )

        if mask:
            outmask = ak._v2.index.Index8.zeros(outlength, nplike)
            self._handle_error(
                nplike[
                    "awkward_NumpyArray_reduce_mask_ByteMaskedArray_64",
                    outmask.dtype.type,
                    parents.dtype.type,
                ](
                    outmask.to(nplike),
                    parents.to(nplike),
                    len(parents),
                    outlength,
                )
            )

            out = ak._v2.contents.ByteMaskedArray(
                outmask,
                out,
                False,
                None,
                None,
            )

        if keepdims:
            out = ak._v2.contents.RegularArray(
                out,
                1,
                len(self),
                None,
                None,
            )

        return out

    def _validityerror(self, path):
        if len(self.shape) == 0:
            return 'at {0} ("{1}"): shape is zero-dimensional'.format(path, type(self))
        for i in range(len(self.shape)):
            if self.shape[i] < 0:
                return 'at {0} ("{1}"): shape[{2}] < 0'.format(path, type(self), i)
        for i in range(len(self.strides)):
            if self.strides[i] % self.dtype.itemsize != 0:
                return 'at {0} ("{1}"): shape[{2}] % itemsize != 0'.format(
                    path, type(self), i
                )
        return ""

    def _rpad(self, target, axis, depth, clip):
        if len(self.shape) == 0:
            raise ValueError("cannot rpad a scalar")
        elif len(self.shape) > 1 or not self.is_contiguous:
            return self.toRegularArray()._rpad(target, axis, depth, clip)
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis != depth:
            raise np.AxisError(
                "axis={0} exceeds the depth of this array({1})".format(axis, depth)
            )
        if not clip:
            if target < len(self):
                return self
            else:
                return self._rpad(target, posaxis, depth, clip=True)
        else:
            return self.rpad_axis0(target, clip=True)

    def _nbytes_part(self):
        result = self.data.nbytes
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        if self._data.ndim != 1:
            return self.toRegularArray()._to_arrow(
                pyarrow, mask_node, validbytes, length, options
            )

        nparray = self.to(numpy)
        storage_type = pyarrow.from_numpy_dtype(nparray.dtype)

        if issubclass(nparray.dtype.type, (bool, np.bool_)):
            nparray = ak._v2._connect.pyarrow.packbits(nparray)

        return pyarrow.Array.from_buffers(
            ak._v2._connect.pyarrow.to_awkwardarrow_type(
                storage_type, options["extensionarray"], mask_node, self
            ),
            length,
            [
                ak._v2._connect.pyarrow.to_validbits(validbytes),
                ak._v2._connect.pyarrow.to_length(nparray, length),
            ],
            null_count=ak._v2._connect.pyarrow.to_null_count(
                validbytes, options["count_nulls"]
            ),
        )

    def _to_numpy(self, allow_missing):
        out = ak.nplike.of(self).asarray(self)
        if type(out).__module__.startswith("cupy."):
            return out.get()
        else:
            return out

    def _completely_flatten(self, nplike, options):
        return [self.to(nplike).reshape(-1)]

    def _recursively_apply(
        self, action, depth, depth_context, lateral_context, options
    ):
        if self._data.ndim != 1 and options["numpy_to_regular"]:
            return self.toRegularArray()._recursively_apply(
                action, depth, depth_context, lateral_context, options
            )

        if options["return_array"]:

            def continuation():
                if options["keep_parameters"]:
                    return self
                else:
                    return NumpyArray(self._data, self._identifier, None, self._nplike)

        else:

            def continuation():
                pass

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
        return self.contiguous().toRegularArray()

    def _to_list(self, behavior):
        if self.parameter("__array__") == "byte":
            return ak._v2._util.tobytes(self._data)

        elif self.parameter("__array__") == "char":
            return ak._v2._util.tobytes(self._data).decode(errors="surrogateescape")

        else:
            out = self._to_list_custom(behavior)
            if out is not None:
                return out

            return self._data.tolist()
