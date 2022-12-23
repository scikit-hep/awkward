# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy

import awkward as ak
from awkward._util import unset
from awkward.contents.content import Content
from awkward.forms.numpyform import NumpyForm
from awkward.types.numpytype import primitive_to_dtype
from awkward.typing import Final, Self

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


class NumpyArray(Content):
    is_numpy = True
    is_leaf = True

    def __init__(self, data, *, parameters=None, backend=None):
        if backend is None:
            backend = ak._backends.backend_of(
                data, default=ak._backends.NumpyBackend.instance()
            )
        if isinstance(data, ak.index.Index):
            data = data.data
        self._data = backend.nplike.asarray(data)

        if not isinstance(backend.nplike, ak._nplikes.Jax):
            ak.types.numpytype.dtype_to_primitive(self._data.dtype)

        if len(self._data.shape) == 0:
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'data' must be an array, not a scalar: {}".format(
                        type(self).__name__, repr(data)
                    )
                )
            )

        if parameters is not None and parameters.get("__array__") in ("char", "byte"):
            if data.dtype != np.dtype(np.uint8) or len(data.shape) != 1:
                raise ak._errors.wrap_error(
                    ValueError(
                        "{} is a {}, so its 'data' must be 1-dimensional and uint8, not {}".format(
                            type(self).__name__, parameters["__array__"], repr(data)
                        )
                    )
                )

        self._init(parameters, backend)

    @property
    def data(self):
        return self._data

    form_cls: Final = NumpyForm

    def copy(
        self,
        data=unset,
        *,
        parameters=unset,
        backend=unset,
    ):
        return NumpyArray(
            self._data if data is unset else data,
            parameters=self._parameters if parameters is unset else parameters,
            backend=self._backend if backend is unset else backend,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            data=copy.deepcopy(self._data, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(cls, data, *, parameters=None, backend=None):
        return cls(data, parameters=parameters, backend=backend)

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

    def _raw(self, nplike=None):
        return self._backend.nplike.raw(self.data, nplike)

    def _form_with_key(self, getkey):
        return self.form_cls(
            ak.types.numpytype.dtype_to_primitive(self._data.dtype),
            self._data.shape[1:],
            parameters=self._parameters,
            form_key=getkey(self),
        )

    def _to_buffers(self, form, getkey, container, backend):
        assert isinstance(form, self.form_cls)
        key = getkey(self, form, "data")
        container[key] = ak._util.little_endian(self._raw(backend.nplike))

    def _to_typetracer(self, forget_length: bool) -> Self:
        backend = ak._backends.TypeTracerBackend.instance()
        data = self._raw(backend.nplike)
        return NumpyArray(
            data.forget_length() if forget_length else data,
            parameters=self._parameters,
            backend=backend,
        )

    def _touch_data(self, recursive):
        if not self._backend.nplike.known_data:
            self._data.touch_data()

    def _touch_shape(self, recursive):
        if not self._backend.nplike.known_shape:
            self._data.touch_shape()

    @property
    def length(self):
        return self._data.shape[0]

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
        arraystr_lines = self._backend.nplike.array_str(
            self._data, max_line_width=30
        ).split("\n")

        if len(extra) != 0 or len(arraystr_lines) > 1:
            arraystr_lines = self._backend.nplike.array_str(
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

    def to_RegularArray(self):
        shape = self._data.shape
        zeroslen = [1]
        for x in shape:
            zeroslen.append(zeroslen[-1] * x)

        out = NumpyArray(self._data.reshape(-1), parameters=None, backend=self._backend)
        for i in range(len(shape) - 1, 0, -1):
            out = ak.contents.RegularArray(out, shape[i], zeroslen[i], parameters=None)
        out._parameters = self._parameters
        return out

    def maybe_to_NumpyArray(self) -> Self:
        return self

    def __array__(self, *args, **kwargs):
        return self._backend.nplike.asarray(self._data, *args, **kwargs)

    def __iter__(self):
        return iter(self._data)

    def _getitem_nothing(self):
        tmp = self._data[0:0]
        return NumpyArray(
            tmp.reshape((0,) + tmp.shape[2:]), parameters=None, backend=self._backend
        )

    def _getitem_at(self, where):
        if not self._backend.nplike.known_data and len(self._data.shape) == 1:
            self._touch_data(recursive=False)
            return ak._typetracer.UnknownScalar(self._data.dtype)

        try:
            out = self._data[where]
        except IndexError as err:
            raise ak._errors.index_error(self, where, str(err)) from err

        if hasattr(out, "shape") and len(out.shape) != 0:
            return NumpyArray(out, parameters=None, backend=self._backend)
        else:
            return out

    def _getitem_range(self, where):
        if not self._backend.nplike.known_shape:
            self._touch_shape(recursive=False)
            return self

        start, stop, step = where.indices(self.length)
        assert step == 1

        try:
            out = self._data[where]
        except IndexError as err:
            raise ak._errors.index_error(self, where, str(err)) from err

        return NumpyArray(out, parameters=self._parameters, backend=self._backend)

    def _getitem_field(self, where, only_fields=()):
        raise ak._errors.index_error(self, where, "not an array of records")

    def _getitem_fields(self, where, only_fields=()):
        if len(where) == 0:
            return self._getitem_range(slice(0, 0))
        raise ak._errors.index_error(self, where, "not an array of records")

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak.index.Index)
        try:
            nextdata = self._data[carry.data]
        except IndexError as err:
            raise ak._errors.index_error(self, carry.data, str(err)) from err
        return NumpyArray(nextdata, parameters=self._parameters, backend=self._backend)

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        if self._data.ndim == 1:
            raise ak._errors.index_error(
                self,
                ak.contents.ListArray(
                    slicestarts, slicestops, slicecontent, parameters=None
                ),
                "too many jagged slice dimensions for array",
            )
        else:
            next = self.to_RegularArray()
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
                raise ak._errors.index_error(self, (head,) + tail, str(err)) from err

            if hasattr(out, "shape") and len(out.shape) != 0:
                return NumpyArray(out, parameters=None, backend=self._backend)
            else:
                return out

        elif isinstance(head, slice) or head is np.newaxis or head is Ellipsis:
            where = (slice(None), head) + tail
            try:
                out = self._data[where]
            except IndexError as err:
                raise ak._errors.index_error(self, (head,) + tail, str(err)) from err
            out2 = NumpyArray(out, parameters=self._parameters, backend=self._backend)
            return out2

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif isinstance(head, ak.index.Index64):
            if advanced is None:
                where = (slice(None), head.data) + tail
            else:
                where = (
                    self._backend.index_nplike.asarray(advanced.data),
                    head.data,
                ) + tail

            try:
                out = self._data[where]
            except IndexError as err:
                raise ak._errors.index_error(self, (head,) + tail, str(err)) from err

            return NumpyArray(out, parameters=self._parameters, backend=self._backend)

        elif isinstance(head, ak.contents.ListOffsetArray):
            where = (slice(None), head) + tail
            try:
                out = self._data[where]
            except IndexError as err:
                raise ak._errors.index_error(self, (head,) + tail, str(err)) from err
            out2 = NumpyArray(out, parameters=self._parameters, backend=self._backend)
            return out2

        elif isinstance(head, ak.contents.IndexedOptionArray):
            next = self.to_RegularArray()
            return next._getitem_next_missing(head, tail, advanced)

        else:
            raise ak._errors.wrap_error(AssertionError(repr(head)))

    def _offsets_and_flattened(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            raise ak._errors.wrap_error(np.AxisError("axis=0 not allowed for flatten"))

        elif len(self.shape) != 1:
            return self.to_RegularArray()._offsets_and_flattened(axis, depth)

        else:
            raise ak._errors.wrap_error(
                np.AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
            )

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
            return self._mergeable(other._content, mergebool)

        elif isinstance(other, ak.contents.NumpyArray):
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
                self._data.dtype.type,
                other._data.dtype.type,
            ):
                return False

            if not matching_dtype and np.timedelta64 in (
                self._data.dtype.type,
                other._data.dtype.type,
            ):
                return False

            if (
                len(self._data.shape) > 1
                and self._data.shape[1:] != other._data.shape[1:]
            ):
                return False

            return True

        # If we have >1 dimension, promote ourselves to `RegularArray` and attempt to merge.
        elif isinstance(other, ak.contents.RegularArray) and self.purelist_depth > 1:
            as_regular_array = self.to_RegularArray()
            assert isinstance(as_regular_array, ak.contents.RegularArray)
            return as_regular_array._content._mergeable(other._content, mergebool)

        else:
            return False

    def _mergemany(self, others):
        if len(others) == 0:
            return self

        # Resolve merging against regular types by
        if any(isinstance(o, ak.contents.RegularArray) for o in others):
            return self.to_RegularArray()._mergemany(others)

        head, tail = self._merging_strategy(others)

        contiguous_arrays = []

        parameters = self._parameters
        for array in head:
            parameters = ak._util.merge_parameters(parameters, array._parameters, True)
            if isinstance(array, ak.contents.EmptyArray):
                pass
            elif isinstance(array, ak.contents.NumpyArray):
                contiguous_arrays.append(array.data)
            else:
                raise ak._errors.wrap_error(
                    AssertionError(
                        "cannot merge "
                        + type(self).__name__
                        + " with "
                        + type(array).__name__
                    )
                )

        contiguous_arrays = self._backend.nplike.concatenate(contiguous_arrays)

        next = NumpyArray(
            contiguous_arrays, parameters=parameters, backend=self._backend
        )

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed._mergemany(tail[1:])

    def _fill_none(self, value: Content) -> Content:
        return self

    def _local_index(self, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        elif len(self.shape) <= 1:
            raise ak._errors.wrap_error(
                np.AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
            )
        else:
            return self.to_RegularArray()._local_index(axis, depth)

    def to_contiguous(self) -> Self:
        if self.is_contiguous:
            return self
        else:
            return ak.contents.NumpyArray(
                self._backend.nplike.ascontiguousarray(self._data),
                parameters=self._parameters,
                backend=self._backend,
            )

    @property
    def is_contiguous(self) -> bool:
        return self._backend.nplike.is_c_contiguous(self._data)

    def _subranges_equal(self, starts, stops, length, sorted=True):
        is_equal = ak.index.Index64.zeros(1, nplike=self._backend.nplike)

        tmp = self._backend.nplike.empty(length, self.dtype)
        self._handle_error(
            self._backend[
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
            tmp_beg_ptr = ak.index.Index64.empty(
                ak._util.kMaxLevels, nplike=self._backend.index_nplike
            )
            tmp_end_ptr = ak.index.Index64.empty(
                ak._util.kMaxLevels, nplike=self._backend.index_nplike
            )

            assert (
                tmp_beg_ptr.nplike is self._backend.index_nplike
                and tmp_end_ptr.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
                and stops.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
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
        assert (
            starts.nplike is self._backend.index_nplike
            and stops.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
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
        offsets = ak.index.Index64(offsets.data, nplike=offsets.nplike)
        outoffsets = ak.index.Index64.empty(
            offsets.length, nplike=self._backend.index_nplike
        )
        out = self._backend.nplike.empty(self.shape[0], self.dtype)

        assert (
            offsets.nplike is self._backend.index_nplike
            and outoffsets.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
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

        outlength = ak.index.Index64.empty(1, self._backend.index_nplike)
        nextoffsets = ak.index.Index64.empty(offsets.length, self._backend.index_nplike)
        assert (
            outoffsets.nplike is self._backend.index_nplike
            and nextoffsets.nplike is self._backend.index_nplike
            and outlength.nplike is self._backend.index_nplike
        )
        self._handle_error(
            self._backend[
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
        out2 = NumpyArray(out, parameters=self._parameters, backend=self._backend)

        return out2, nextoffsets[: outlength[0]]

    def _numbers_to_type(self, name):
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
                self._backend.nplike.asarray(self._data, dtype=dtype),
                parameters=self._parameters,
                backend=self._backend,
            )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self.length == 0:
            return True

        elif len(self.shape) != 1 or not self.is_contiguous:
            contiguous_self = self.to_contiguous()
            return contiguous_self.to_RegularArray()._is_unique(
                negaxis,
                starts,
                parents,
                outlength,
            )
        else:
            out = self._unique(negaxis, starts, parents, outlength)
            if isinstance(out, ak.contents.ListOffsetArray):
                return out.content.length == self.length

            return out.length == self.length

    def _unique(self, negaxis, starts, parents, outlength):
        if self.shape[0] == 0:
            return self

        if len(self.shape) == 0:
            return self

        if negaxis is None:
            contiguous_self = self.to_contiguous()
            # Python 3.8 could use math.prod
            flattened_shape = 1
            for s in contiguous_self.shape:
                flattened_shape = flattened_shape * s

            offsets = ak.index.Index64.zeros(2, self._backend.index_nplike)
            offsets[1] = flattened_shape
            dtype = (
                np.dtype(np.int64)
                if self._data.dtype.kind.upper() == "M"
                else self._data.dtype
            )
            out = self._backend.nplike.empty(offsets[1], dtype)
            assert offsets.nplike is self._backend.index_nplike
            self._handle_error(
                self._backend[
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

            nextlength = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert nextlength.nplike is self._backend.index_nplike
            self._handle_error(
                self._backend[
                    "awkward_unique",
                    out.dtype.type,
                    nextlength.dtype.type,
                ](  # noqa: E231
                    out,
                    out.shape[0],
                    nextlength.data,
                )
            )

            return ak.contents.NumpyArray(
                self._backend.nplike.asarray(out[: nextlength[0]], self.dtype),
                parameters=None,
                backend=self._backend,
            )

        # axis is not None
        if len(self.shape) != 1 or not self.is_contiguous:
            contiguous_self = self.to_contiguous()
            return contiguous_self.to_RegularArray()._unique(
                negaxis,
                starts,
                parents,
                outlength,
            )
        else:

            parents_length = parents.length
            offsets_length = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert (
                offsets_length.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_sorting_ranges_length",
                    offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    offsets_length.data,
                    parents.data,
                    parents_length,
                )
            )

            offsets = ak.index.Index64.empty(
                offsets_length[0], self._backend.index_nplike
            )
            assert (
                offsets.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
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

            out = self._backend.nplike.empty(self.length, self.dtype)
            assert offsets.nplike is self._backend.index_nplike
            self._handle_error(
                self._backend[
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

            nextoffsets = ak.index.Index64.empty(
                offsets.length, self._backend.index_nplike
            )
            assert (
                offsets.nplike is self._backend.index_nplike
                and nextoffsets.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
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

            outoffsets = ak.index.Index64.empty(
                starts.length + 1, self._backend.index_nplike
            )

            assert (
                outoffsets.nplike is self._backend.index_nplike
                and nextoffsets.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
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

            return ak.contents.ListOffsetArray(
                outoffsets, ak.contents.NumpyArray(out), parameters=self._parameters
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
        if len(self.shape) == 0:
            raise ak._errors.wrap_error(
                TypeError(f"{type(self).__name__} attempting to argsort a scalar ")
            )
        elif len(self.shape) != 1 or not self.is_contiguous:
            contiguous_self = self.to_contiguous()
            return contiguous_self.to_RegularArray()._argsort_next(
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
            offsets_length = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert (
                offsets_length.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
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

            offsets = ak.index.Index64.empty(offsets_length, self._backend.index_nplike)
            assert (
                offsets.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
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
            nextcarry = ak.index.Index64.empty(
                self.__len__(), self._backend.index_nplike
            )
            assert (
                nextcarry.nplike is self._backend.index_nplike
                and offsets.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
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
                    nextcarry.nplike is self._backend.index_nplike
                    and shifts.nplike is self._backend.index_nplike
                    and offsets.nplike is self._backend.index_nplike
                    and parents.nplike is self._backend.index_nplike
                    and starts.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
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
            out = NumpyArray(nextcarry, parameters=None, backend=self._backend)
            return out

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        if len(self.shape) == 0:
            raise ak._errors.wrap_error(
                TypeError(f"{type(self).__name__} attempting to sort a scalar ")
            )

        elif len(self.shape) != 1 or not self.is_contiguous:
            contiguous_self = self.to_contiguous()
            return contiguous_self.to_RegularArray()._sort_next(
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
            offsets_length = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert (
                offsets_length.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
                    "awkward_sorting_ranges_length",
                    offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    offsets_length.data,
                    parents.data,
                    parents_length,
                )
            )

            offsets = ak.index.Index64.empty(
                offsets_length[0], self._backend.index_nplike
            )

            assert (
                offsets.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
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
            out = self._backend.nplike.empty(self.length, dtype)
            assert offsets.nplike is self._backend.index_nplike
            self._handle_error(
                self._backend[
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
            return ak.contents.NumpyArray(
                self._backend.nplike.asarray(out, self.dtype),
                parameters=None,
                backend=self._backend,
            )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        elif len(self.shape) <= 1:
            raise ak._errors.wrap_error(
                np.AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
            )
        else:
            return self.to_RegularArray()._combinations(
                n, replacement, recordlookup, parameters, axis, depth
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

        if self._data.ndim > 1:
            return self.to_RegularArray()._reduce_next(
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
        elif not self.is_contiguous:
            return self.to_contiguous()._reduce_next(
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

        # Yes, we've just tested these, but we need to be explicit that they are invariants
        assert self.is_contiguous
        assert self._data.ndim == 1

        out = self._backend.apply_reducer(reducer, self, parents, outlength)

        if reducer.needs_position:
            if shifts is None:
                assert (
                    out.backend is self._backend
                    and parents.nplike is self._backend.index_nplike
                    and starts.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
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
                    out.backend is self._backend
                    and parents.nplike is self._backend.index_nplike
                    and starts.nplike is self._backend.index_nplike
                    and shifts.nplike is self._backend.index_nplike
                )
                self._handle_error(
                    self._backend[
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
            outmask = ak.index.Index8.empty(outlength, self._backend.index_nplike)
            assert (
                outmask.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._handle_error(
                self._backend[
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

            out = ak.contents.ByteMaskedArray(outmask, out, False, parameters=None)

        if keepdims:
            out = ak.contents.RegularArray(out, 1, self.length, parameters=None)

        return out

    def _validity_error(self, path):
        if len(self.shape) == 0:
            return f'at {path} ("{type(self)}"): shape is zero-dimensional'
        for i, dim in enumerate(self.shape):
            if dim < 0:
                return f'at {path} ("{type(self)}"): shape[{i}] < 0'
        for i, stride in enumerate(self.strides):
            if stride % self.dtype.itemsize != 0:
                return f'at {path} ("{type(self)}"): shape[{i}] % itemsize != 0'
        return ""

    def _pad_none(self, target, axis, depth, clip):
        if len(self.shape) == 0:
            raise ak._errors.wrap_error(
                ValueError("cannot apply ak.pad_none to a scalar")
            )
        elif len(self.shape) > 1 or not self.is_contiguous:
            return self.to_RegularArray()._pad_none(target, axis, depth, clip)
        posaxis = ak._util.maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 != depth:
            raise ak._errors.wrap_error(
                np.AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
            )
        if not clip:
            if target < self.length:
                return self
            else:
                return self._pad_none(target, axis, depth, clip=True)
        else:
            return self._pad_none_axis0(target, clip=True)

    def _nbytes_part(self):
        return self.data.nbytes

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        if self._data.ndim != 1:
            return self.to_RegularArray()._to_arrow(
                pyarrow, mask_node, validbytes, length, options
            )

        nparray = self._raw(numpy)
        storage_type = pyarrow.from_numpy_dtype(nparray.dtype)

        if issubclass(nparray.dtype.type, (bool, np.bool_)):
            nparray = ak._connect.pyarrow.packbits(nparray)

        return pyarrow.Array.from_buffers(
            ak._connect.pyarrow.to_awkwardarrow_type(
                storage_type,
                options["extensionarray"],
                options["record_is_scalar"],
                mask_node,
                self,
            ),
            length,
            [
                ak._connect.pyarrow.to_validbits(validbytes),
                ak._connect.pyarrow.to_length(nparray, length),
            ],
            null_count=ak._connect.pyarrow.to_null_count(
                validbytes, options["count_nulls"]
            ),
        )

    def _to_numpy(self, allow_missing):
        out = numpy.asarray(self._data)
        if type(out).__module__.startswith("cupy."):
            return out.get()
        else:
            return out

    def _completely_flatten(self, backend, options):
        return [
            ak.contents.NumpyArray(
                self._raw(backend.nplike).reshape(-1), backend=backend
            )
        ]

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if self._data.ndim != 1 and options["numpy_to_regular"]:
            return self.to_RegularArray()._recursively_apply(
                action, behavior, depth, depth_context, lateral_context, options
            )

        if options["return_array"]:

            def continuation():
                if options["keep_parameters"]:
                    return self
                else:
                    return NumpyArray(
                        self._data, parameters=None, backend=self._backend
                    )

        else:

            def continuation():
                pass

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
        return self.to_contiguous().to_RegularArray()

    def _to_list(self, behavior, json_conversions):
        if self.parameter("__array__") == "byte":
            convert_bytes = (
                None if json_conversions is None else json_conversions["convert_bytes"]
            )
            if convert_bytes is None:
                return ak._util.tobytes(self._data)
            else:
                return convert_bytes(ak._util.tobytes(self._data))

        elif self.parameter("__array__") == "char":
            return ak._util.tobytes(self._data).decode(errors="surrogateescape")

        else:
            out = self._to_list_custom(behavior, json_conversions)
            if out is not None:
                return out

            if json_conversions is not None:
                complex_real_string = json_conversions["complex_real_string"]
                complex_imag_string = json_conversions["complex_imag_string"]
                if complex_real_string is not None:
                    if issubclass(self.dtype.type, np.complexfloating):
                        return ak.contents.RecordArray(
                            [
                                ak.contents.NumpyArray(
                                    self._data.real, backend=self._backend
                                ),
                                ak.contents.NumpyArray(
                                    self._data.imag, backend=self._backend
                                ),
                            ],
                            [complex_real_string, complex_imag_string],
                            self.length,
                            parameters=self._parameters,
                            backend=self._backend,
                        )._to_list(behavior, json_conversions)

            out = self._data.tolist()

            if json_conversions is not None:
                nan_string = json_conversions["nan_string"]
                if nan_string is not None:
                    for i in self._backend.nplike.nonzero(
                        self._backend.nplike.isnan(self._data)
                    )[0]:
                        out[i] = nan_string

                posinf_string = json_conversions["posinf_string"]
                if posinf_string is not None:
                    for i in self._backend.nplike.nonzero(self._data == np.inf)[0]:
                        out[i] = posinf_string

                neginf_string = json_conversions["neginf_string"]
                if neginf_string is not None:
                    for i in self._backend.nplike.nonzero(self._data == -np.inf)[0]:
                        out[i] = neginf_string

            return out

    def to_backend(self, backend: ak._backends.Backend) -> Self:
        return NumpyArray(
            self._raw(backend.nplike),
            parameters=self._parameters,
            backend=backend,
        )

    def _is_equal_to(self, other, index_dtype, numpyarray):
        if numpyarray:
            return (
                self._backend.nplike.array_equal(self.data, other.data)
                and self.dtype == other.dtype
                and self.is_contiguous == other.is_contiguous
                and self.shape == other.shape
            )
        else:
            return True
