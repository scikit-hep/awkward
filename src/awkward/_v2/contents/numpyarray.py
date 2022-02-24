# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

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
        if nplike is None:
            nplike = ak.nplike.of(data)
        if isinstance(data, ak._v2.index.Index):
            data = data.data
        self._data = nplike.asarray(data)

        ak._v2.types.numpytype.dtype_to_primitive(self._data.dtype)
        if len(self._data.shape) == 0:
            raise TypeError(
                "{} 'data' must be an array, not {}".format(
                    type(self).__name__, repr(data)
                )
            )

        self._init(identifier, parameters, nplike)

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
    def ptr(self):
        return self._data.ctypes.data

    def raw(self, nplike):
        return self.nplike.raw(self.data, nplike)

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
        container[key] = ak._v2._util.little_endian(self.raw(nplike))

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return NumpyArray(
            self.raw(tt),
            self._typetracer_identifier(),
            self._parameters,
            tt,
        )

    @property
    def length(self):
        return self._data.shape[0]

    def _forget_length(self):
        return NumpyArray(
            self._data.forget_length(),
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<NumpyArray dtype="]
        out.append(repr(str(self.dtype)))
        if len(self._data.shape) == 1:
            out.append(" len=" + repr(str(self._data.shape[0])))
        else:
            out.append(
                " shape='({})'".format(", ".join(str(x) for x in self._data.shape))
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
        shape = self._data.shape
        zeroslen = [1]
        for x in shape:
            zeroslen.append(zeroslen[-1] * x)

        out = NumpyArray(self._data.reshape(-1), None, None, self._nplike)
        for i in range(len(shape) - 1, 0, -1):
            out = ak._v2.contents.RegularArray(
                out, shape[i], zeroslen[i], None, None, self._nplike
            )
        out._identifier = self._identifier
        out._parameters = self._parameters
        return out

    def _nonfinite_to_union(self, nan_string, infinity_string, minus_infinity_string):
        shape = self._data.shape
        zeroslen = [1]
        for x in shape:
            zeroslen.append(zeroslen[-1] * x)

        out = NumpyArray(self._data.reshape(-1), None, None, self._nplike)

        is_nonfinite = ~self._nplike.isfinite(self._data)  # true for inf, -inf, nan
        is_posinf = is_nonfinite & (self._data > 0)  # true for inf only
        is_neginf = is_nonfinite & (self._data < 0)  # true for -inf only
        is_nan = self._nplike.isnan(self._data)  # true for nan only
        tags = self._nplike.zeros(out.length, np.int8)
        tags[is_nonfinite] = 1
        index = self._nplike.arange(out.length, dtype=np.int64)
        index[is_posinf] = 0
        index[is_neginf] = 1
        index[is_nan] = 2

        out = ak._v2.contents.unionarray.UnionArray(
            tags=ak._v2.index.Index8(tags),
            index=ak._v2.index.Index64(index),
            contents=[
                out,
                ak._v2.operations.convert.from_iter(
                    [
                        infinity_string if infinity_string is not None else "Infinity",
                        minus_infinity_string
                        if minus_infinity_string is not None
                        else "-Infinity",
                        nan_string if nan_string is not None else "NaN",
                    ],
                    highlevel=False,
                ),
            ],
        )
        for i in range(len(shape) - 1, 0, -1):
            out = ak._v2.contents.RegularArray(
                out, shape[i], zeroslen[i], None, None, self._nplike
            )
        out._identifier = self._identifier
        out._parameters = self._parameters

        return out

    def maybe_to_array(self, nplike):
        return nplike.asarray(self._data)

    def __array__(self, **kwargs):
        return numpy.asarray(self._data, **kwargs)

    def __iter__(self):
        return iter(self._data)

    def _getitem_nothing(self):
        tmp = self._data[0:0]
        return NumpyArray(
            tmp.reshape((0,) + tmp.shape[2:]),
            self._range_identifier(0, 0),
            None,
            self._nplike,
        )

    def _getitem_at(self, where):
        if not self._nplike.known_data and len(self._data.shape) == 1:
            return ak._v2._typetracer.UnknownScalar(self._data.dtype)

        try:
            out = self._data[where]
        except IndexError as err:
            raise NestedIndexError(self, where, str(err))

        if hasattr(out, "shape") and len(out.shape) != 0:
            return NumpyArray(out, None, None, self._nplike)
        else:
            return out

    def _getitem_range(self, where):
        if not self._nplike.known_shape:
            return self

        start, stop, step = where.indices(self.length)
        assert step == 1

        try:
            out = self._data[where]
        except IndexError as err:
            raise NestedIndexError(self, where, str(err))

        return NumpyArray(
            out,
            self._range_identifier(start, stop),
            self._parameters,
            self._nplike,
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
            self._nplike,
        )

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        if self._data.ndim == 1:
            raise NestedIndexError(
                self,
                ak._v2.contents.ListArray(
                    slicestarts, slicestops, slicecontent, None, None, self._nplike
                ),
                "too many jagged slice dimensions for array",
            )
        else:
            next = self.toRegularArray()
            return next._getitem_next_jagged(
                slicestarts, slicestops, slicecontent, tail
            )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, int):
            where = (slice(None), head) + tail

            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))

            if hasattr(out, "shape") and len(out.shape) != 0:
                return NumpyArray(out, None, None, self._nplike)
            else:
                return out

        elif isinstance(head, slice) or head is np.newaxis or head is Ellipsis:
            where = (slice(None), head) + tail
            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))
            out2 = NumpyArray(out, None, self._parameters, self._nplike)
            return out2

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif isinstance(head, ak._v2.index.Index64):
            if advanced is None:
                where = (slice(None), head.data) + tail
            else:
                where = (self._nplike.asarray(advanced.data), head.data) + tail

            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))

            return NumpyArray(out, None, self._parameters, self._nplike)

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            where = (slice(None), head) + tail
            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))
            out2 = NumpyArray(out, None, self._parameters, self._nplike)
            return out2

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            next = self.toRegularArray()
            return next._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def num(self, axis, depth=0):
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis == depth:
            out = ak._v2.index.Index64.empty(1, self._nplike)
            out[0] = self.length
            return ak._v2.contents.numpyarray.NumpyArray(out, None, None, self._nplike)[
                0
            ]
        shape = []
        reps = 1
        size = self.length
        i = 0
        while i < self._data.ndim - 1 and depth < posaxis:
            shape.append(self.shape[i])
            reps *= self.shape[i]
            size = self.shape[i + 1]
            i += 1
            depth += 1
        if posaxis > depth:
            raise np.AxisError(f"axis={axis} exceeds the depth of this array ({depth})")

        tonum = ak._v2.index.Index64.empty(reps, self._nplike)
        assert tonum.nplike is self._nplike
        self._handle_error(
            self._nplike["awkward_RegularArray_num", tonum.dtype.type](
                tonum.data, size, reps
            )
        )
        return ak._v2.contents.numpyarray.NumpyArray(
            tonum.data.reshape(shape), None, self.parameters, self._nplike
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
            return self.mergeable(other._content, mergebool)

        if isinstance(other, ak._v2.contents.numpyarray.NumpyArray):
            if self._data.ndim != other._data.ndim:
                return False

            matching_dtype = self._data.dtype == other._data.dtype

            if (
                not mergebool
                and not matching_dtype
                and (
                    self._data.dtype.type is np.bool_
                    or other._data.dtype.type is np.bool_
                )
            ):
                return False

            if not matching_dtype and np.datetime64 in (
                self._data.dtype,
                other._data.dtype,
            ):
                return False

            if not matching_dtype and np.timedelta64 in (
                self._data.dtype,
                other._data.dtype,
            ):
                return False

            if (
                len(self._data.shape) > 1
                and self._data.shape[1:] != other._data.shape[1:]
            ):
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
                self._parameters, array._parameters, True
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

        contiguous_arrays = self._nplike.concatenate(contiguous_arrays)

        next = NumpyArray(contiguous_arrays, self._identifier, parameters, self._nplike)

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
        is_equal = ak._v2.index.Index64.zeros(1, self._nplike)

        tmp = self._nplike.empty(length, self.dtype)
        self._handle_error(
            self._nplike[  # noqa: E231
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
            tmp_beg_ptr = ak._v2.index.Index64.empty(ak._util.kMaxLevels, self._nplike)
            tmp_end_ptr = ak._v2.index.Index64.empty(ak._util.kMaxLevels, self._nplike)

            assert (
                tmp_beg_ptr.nplike is self._nplike
                and tmp_end_ptr.nplike is self._nplike
                and starts.nplike is self._nplike
                and stops.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_quick_sort",
                    self.dtype.type,
                    tmp_beg_ptr.dtype.type,
                    tmp_end_ptr.dtype.type,
                    starts.dtype.type,
                    stops.dtype.type,
                ](
                    tmp,
                    tmp_beg_ptr.data,
                    tmp_end_ptr.data,
                    starts.data,
                    stops.data,
                    True,
                    starts.length,
                    ak._util.kMaxLevels,
                )
            )
        assert starts.nplike is self._nplike and stops.nplike is self._nplike
        self._handle_error(
            self._nplike[
                "awkward_NumpyArray_subrange_equal",
                self.dtype.type,
                starts.dtype.type,
                stops.dtype.type,
                np.bool_,
            ](
                tmp,
                starts.data,
                stops.data,
                starts.length,
                is_equal.data,
            )
        )

        return True if is_equal[0] == 1 else False

    def _as_unique_strings(self, offsets):
        outoffsets = ak._v2.index.Index64.empty(offsets.length, self._nplike)
        out = self._nplike.empty(self.shape[0], self.dtype)

        assert offsets.nplike is self._nplike and outoffsets.nplike is self._nplike
        self._handle_error(
            self._nplike[
                "awkward_NumpyArray_sort_asstrings_uint8",
                self.dtype.type,
                self._data.dtype.type,
                offsets._data.dtype.type,
                outoffsets.dtype.type,
            ](
                out,
                self._data,
                offsets.data,
                offsets.length,
                outoffsets.data,
                True,
                False,
            )
        )

        outlength = ak._v2.index.Index64.empty(1, self._nplike)
        nextoffsets = ak._v2.index.Index64.empty(offsets.length, self._nplike)
        assert (
            outoffsets.nplike is self._nplike
            and nextoffsets.nplike is self._nplike
            and outlength.nplike is self._nplike
        )
        self._handle_error(
            self._nplike[
                "awkward_NumpyArray_unique_strings",
                self.dtype.type,
                outoffsets.dtype.type,
                nextoffsets.dtype.type,
                outlength.dtype.type,
            ](
                out,
                outoffsets.data,
                offsets.length,
                nextoffsets.data,
                outlength.data,
            )
        )
        out2 = NumpyArray(out, None, self._parameters, self._nplike)

        return out2, nextoffsets[: outlength[0]]

    def numbers_to_type(self, name):
        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
            or self.parameter("__array__") == "char"
            or self.parameter("__array__") == "byte"
        ):
            return self
        else:
            dtype = primitive_to_dtype(name)
            return NumpyArray(
                self._nplike.asarray(self._data, dtype=dtype),
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self.length == 0:
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
                return out.content.length == self.length

            return out.length == self.length

    def _unique(self, negaxis, starts, parents, outlength):
        if self.shape[0] == 0:
            return self

        if len(self.shape) == 0:
            return self

        if negaxis is None:
            contiguous_self = self if self.is_contiguous else self.contiguous()
            # Python 3.8 could use math.prod
            flattened_shape = 1
            for s in contiguous_self.shape:
                flattened_shape = flattened_shape * s

            offsets = ak._v2.index.Index64.zeros(2, self._nplike)
            offsets[1] = flattened_shape
            dtype = (
                np.dtype(np.int64)
                if self._data.dtype.kind.upper() == "M"
                else self._data.dtype
            )
            out = self._nplike.empty(offsets[1], dtype)
            assert offsets.nplike is self._nplike
            self._handle_error(
                self._nplike[  # noqa: E231
                    "awkward_sort",
                    dtype.type,
                    dtype.type,
                    offsets.dtype.type,
                ](
                    out,
                    contiguous_self._data,
                    offsets[1],
                    offsets.data,
                    2,
                    offsets[1],
                    True,
                    False,
                )
            )

            nextlength = ak._v2.index.Index64.empty(1, self._nplike)
            assert nextlength.nplike is self._nplike
            self._handle_error(
                self._nplike[  # noqa: E231
                    "awkward_unique",
                    out.dtype.type,
                    nextlength.dtype.type,
                ](
                    out,
                    out.shape[0],
                    nextlength.data,
                )
            )

            return ak._v2.contents.NumpyArray(
                self._nplike.asarray(out[: nextlength[0]], self.dtype),
                None,
                None,
                self._nplike,
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

            parents_length = parents.length
            offsets_length = ak._v2.index.Index64.empty(1, self._nplike)
            assert (
                offsets_length.nplike is self._nplike and parents.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_sorting_ranges_length",
                    offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    offsets_length.data,
                    parents.data,
                    parents_length,
                )
            )

            offsets = ak._v2.index.Index64.empty(offsets_length[0], self._nplike)
            assert offsets.nplike is self._nplike and parents.nplike is self._nplike
            self._handle_error(
                self._nplike[  # noqa: E231
                    "awkward_sorting_ranges",
                    offsets.dtype.type,
                    parents.dtype.type,
                ](
                    offsets.data,
                    offsets_length[0],
                    parents.data,
                    parents_length,
                )
            )

            out = self._nplike.empty(self.length, self.dtype)
            assert offsets.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_sort",
                    out.dtype.type,
                    self._data.dtype.type,
                    offsets.dtype.type,
                ](
                    out,
                    self._data,
                    self.shape[0],
                    offsets.data,
                    offsets_length[0],
                    parents_length,
                    True,
                    False,
                )
            )

            nextoffsets = ak._v2.index.Index64.empty(offsets.length, self._nplike)
            assert offsets.nplike is self._nplike and nextoffsets.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_unique_ranges",
                    out.dtype.type,
                    offsets.dtype.type,
                    nextoffsets.dtype.type,
                ](
                    out,
                    out.shape[0],
                    offsets.data,
                    offsets.length,
                    nextoffsets.data,
                )
            )

            outoffsets = ak._v2.index.Index64.empty(starts.length + 1, self._nplike)

            assert (
                outoffsets.nplike is self._nplike
                and nextoffsets.nplike is self._nplike
                and starts.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_unique_offsets",
                    outoffsets.dtype.type,
                    nextoffsets.dtype.type,
                    starts.dtype.type,
                ](
                    outoffsets.data,
                    nextoffsets.length,
                    nextoffsets.data,
                    starts.data,
                    starts.length,
                )
            )

            return ak._v2.contents.ListOffsetArray(
                outoffsets,
                ak._v2.contents.NumpyArray(out),
                None,
                self._parameters,
                self._nplike,
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
            return ak._v2.contents.NumpyArray(
                self._nplike.empty(0, np.int64), None, None, self._nplike
            )

        if len(self.shape) == 0:
            raise TypeError(f"{type(self).__name__} attempting to argsort a scalar ")
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
            parents_length = parents.length
            offsets_length = ak._v2.index.Index64.empty(1, self._nplike)
            assert (
                offsets_length.nplike is self._nplike and parents.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_sorting_ranges_length",
                    offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    offsets_length.data,
                    parents.data,
                    parents_length,
                )
            )
            offsets_length = offsets_length[0]

            offsets = ak._v2.index.Index64.empty(offsets_length, self._nplike)
            assert offsets.nplike is self._nplike and parents.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_sorting_ranges",
                    offsets.dtype.type,
                    parents.dtype.type,
                ](
                    offsets.data,
                    offsets_length,
                    parents.data,
                    parents_length,
                )
            )

            dtype = (
                np.dtype(np.int64)
                if self._data.dtype.kind.upper() == "M"
                else self._data.dtype
            )
            nextcarry = ak._v2.index.Index64.empty(self.__len__(), self._nplike)
            assert nextcarry.nplike is self._nplike and offsets.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_argsort",
                    nextcarry.dtype.type,
                    dtype.type,
                    offsets.dtype.type,
                ](
                    nextcarry.data,
                    self._data,
                    self.__len__(),
                    offsets.data,
                    offsets_length,
                    ascending,
                    stable,
                )
            )

            if shifts is not None:
                assert (
                    nextcarry.nplike is self._nplike
                    and shifts.nplike is self._nplike
                    and offsets.nplike is self._nplike
                    and parents.nplike is self._nplike
                    and starts.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_NumpyArray_rearrange_shifted",
                        nextcarry.dtype.type,
                        shifts.dtype.type,
                        offsets.dtype.type,
                        parents.dtype.type,
                        starts.dtype.type,
                    ](
                        nextcarry.data,
                        shifts.data,
                        shifts.length,
                        offsets.data,
                        offsets_length,
                        parents.data,
                        parents_length,
                        starts.data,
                        starts.length,
                    )
                )
            out = NumpyArray(nextcarry, None, None, self._nplike)
            return out

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        if len(self.shape) == 0:
            raise TypeError(f"{type(self).__name__} attempting to sort a scalar ")

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
            parents_length = parents.length
            offsets_length = ak._v2.index.Index64.empty(1, self._nplike)
            assert (
                offsets_length.nplike is self._nplike and parents.nplike is self._nplike
            )
            self._handle_error(
                self._nplike[
                    "awkward_sorting_ranges_length",
                    offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    offsets_length.data,
                    parents.data,
                    parents_length,
                )
            )

            offsets = ak._v2.index.Index64.empty(offsets_length[0], self._nplike)

            assert offsets.nplike is self._nplike and parents.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_sorting_ranges",
                    offsets.dtype.type,
                    parents.dtype.type,
                ](
                    offsets.data,
                    offsets_length[0],
                    parents.data,
                    parents_length,
                )
            )

            dtype = (
                np.dtype(np.int64)
                if self._data.dtype.kind.upper() == "M"
                else self._data.dtype
            )
            out = self._nplike.empty(self.length, dtype)
            assert offsets.nplike is self._nplike
            self._handle_error(
                self._nplike[  # noqa: E231
                    "awkward_sort",
                    dtype.type,
                    dtype.type,
                    offsets.dtype.type,
                ](
                    out,
                    self._data,
                    self.shape[0],
                    offsets.data,
                    offsets_length[0],
                    parents_length,
                    ascending,
                    stable,
                )
            )
            return ak._v2.contents.NumpyArray(
                self._nplike.asarray(out, self.dtype), None, None, self._nplike
            )

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
        if len(self._data.shape) != 1 or not self.is_contiguous:
            return self.toRegularArray()._reduce_next(
                reducer,
                negaxis,
                starts,
                shifts,
                parents,
                outlength,
                mask,
                keepdims,
            )

        out = reducer.apply(self, parents, outlength)

        if reducer.needs_position:
            if shifts is None:
                assert (
                    out.nplike is self._nplike
                    and parents.nplike is self._nplike
                    and starts.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_NumpyArray_reduce_adjust_starts_64",
                        out.data.dtype.type,
                        parents.dtype.type,
                        starts.dtype.type,
                    ](
                        out.data,
                        outlength,
                        parents.data,
                        starts.data,
                    )
                )
            else:
                assert (
                    out.nplike is self._nplike
                    and parents.nplike is self._nplike
                    and starts.nplike is self._nplike
                    and shifts.nplike is self._nplike
                )
                self._handle_error(
                    self._nplike[
                        "awkward_NumpyArray_reduce_adjust_starts_shifts_64",
                        out.data.dtype.type,
                        parents.dtype.type,
                        starts.dtype.type,
                        shifts.dtype.type,
                    ](
                        out.data,
                        outlength,
                        parents.data,
                        starts.data,
                        shifts.data,
                    )
                )

        if mask:
            outmask = ak._v2.index.Index8.empty(outlength, self._nplike)
            assert outmask.nplike is self._nplike and parents.nplike is self._nplike
            self._handle_error(
                self._nplike[
                    "awkward_NumpyArray_reduce_mask_ByteMaskedArray_64",
                    outmask.dtype.type,
                    parents.dtype.type,
                ](
                    outmask.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )

            out = ak._v2.contents.ByteMaskedArray(
                outmask,
                out,
                False,
                None,
                None,
                self._nplike,
            )

        if keepdims:
            out = ak._v2.contents.RegularArray(
                out,
                1,
                self.length,
                None,
                None,
                self._nplike,
            )

        return out

    def _validityerror(self, path):
        if len(self.shape) == 0:
            return f'at {path} ("{type(self)}"): shape is zero-dimensional'
        for i, dim in enumerate(self.shape):
            if dim < 0:
                return f'at {path} ("{type(self)}"): shape[{i}] < 0'
        for i, stride in enumerate(self.strides):
            if stride % self.dtype.itemsize != 0:
                return f'at {path} ("{type(self)}"): shape[{i}] % itemsize != 0'
        return ""

    def _rpad(self, target, axis, depth, clip):
        if len(self.shape) == 0:
            raise ValueError("cannot rpad a scalar")
        elif len(self.shape) > 1 or not self.is_contiguous:
            return self.toRegularArray()._rpad(target, axis, depth, clip)
        posaxis = self.axis_wrap_if_negative(axis)
        if posaxis != depth:
            raise np.AxisError(f"axis={axis} exceeds the depth of this array({depth})")
        if not clip:
            if target < self.length:
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

        nparray = self.raw(numpy)
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
        out = self._nplike.asarray(self)
        if type(out).__module__.startswith("cupy."):
            return out.get()
        else:
            return out

    def _completely_flatten(self, nplike, options):
        return [self.raw(nplike).reshape(-1)]

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

    def _to_nplike(self, nplike):
        return NumpyArray(
            self.raw(nplike),
            identifier=self.identifier,
            parameters=self.parameters,
            nplike=nplike,
        )

    def _to_json_custom(
        self,
        nan_string,
        infinity_string,
        minus_infinity_string,
        complex_real_string,
        complex_imag_string,
    ):
        cls = ak._v2._util.arrayclass(self, None)
        if cls.__getitem__ is not ak._v2.highlevel.Array.__getitem__:
            array = cls(self)
            out = [None] * self.length
            for i in range(self.length):
                out[i] = array[i]
            return out

    def _to_json(
        self,
        nan_string,
        infinity_string,
        minus_infinity_string,
        complex_real_string,
        complex_imag_string,
    ):
        if (
            self.parameter("__array__") == "byte"
            or self.parameter("__array__") == "char"
        ):
            return ak._v2._util.tobytes(self._data).decode(errors="surrogateescape")

        else:
            if self.dtype == np.complex128:
                if complex_real_string is None or complex_imag_string is None:
                    raise ValueError(
                        "Complex numbers can't be converted to JSON without"
                        " setting 'complex_record_fields' "
                    )

                return ak._v2.operations.structure.zip(
                    {
                        complex_real_string: ak._v2.contents.NumpyArray(
                            self._data.real
                        ),
                        complex_imag_string: ak._v2.contents.NumpyArray(
                            self._data.imag
                        ),
                    }
                ).layout._to_json(
                    nan_string,
                    infinity_string,
                    minus_infinity_string,
                    complex_real_string,
                    complex_imag_string,
                )

            if (
                nan_string is not None
                or infinity_string is not None
                or minus_infinity_string is not None
            ):
                out = self._nonfinite_to_union(
                    nan_string, infinity_string, minus_infinity_string
                )
                return out.tolist()

            out = self._to_json_custom(
                nan_string,
                infinity_string,
                minus_infinity_string,
                complex_real_string,
                complex_imag_string,
            )
            if out is not None:
                return out

            return self._data.tolist()
