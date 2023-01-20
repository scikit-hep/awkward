# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import numbers

import numpy

import awkward as ak
from awkward import _nplikes, index
from awkward._nplikes import ArrayLike, NumpyLike
from awkward._util import NDArrayOperatorsMixin
from awkward.typing import Final, Literal, Self

np = _nplikes.NumpyMetadata.instance()


class UnknownLengthType:
    def __repr__(self):
        return "UnknownLength"

    def __str__(self):
        return "??"

    def __eq__(self, other):
        return isinstance(other, UnknownLengthType)

    def __add__(self, other):
        return UnknownLength

    def __radd__(self, other):
        return UnknownLength

    def __sub__(self, other):
        return UnknownLength

    def __rsub__(self, other):
        return UnknownLength

    def __mul__(self, other):
        return UnknownLength

    def __rmul__(self, other):
        return UnknownLength

    def __truediv__(self, other):
        return UnknownLength

    def __floordiv__(self, other):
        return UnknownLength

    def __rdiv__(self, other):
        return UnknownLength

    def __rfloordiv__(self, other):
        return UnknownLength

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False


UnknownLength = UnknownLengthType()


def _emptyarray(x):
    if isinstance(x, UnknownScalar):
        return numpy.empty(0, x._dtype)
    elif hasattr(x, "dtype"):
        return numpy.empty(0, x.dtype)
    else:
        return numpy.empty(0, numpy.array(x).dtype)


class UnknownScalar:
    def __init__(self, dtype: np.dtype):
        self._dtype = np.dtype(dtype)

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return f"UnknownScalar({self._dtype!r})"

    def __str__(self):
        return f"unknown-{str(self._dtype)}"

    def __eq__(self, other):
        return isinstance(other, UnknownScalar) and self._dtype == other._dtype

    def __add__(self, other):
        return UnknownScalar((_emptyarray(self) + _emptyarray(other)).dtype)

    def __radd__(self, other):
        return UnknownScalar((_emptyarray(self) + _emptyarray(other)).dtype)

    def __sub__(self, other):
        return UnknownScalar((_emptyarray(self) - _emptyarray(other)).dtype)

    def __rsub__(self, other):
        return UnknownScalar((_emptyarray(self) - _emptyarray(other)).dtype)

    def __mul__(self, other):
        return UnknownScalar((_emptyarray(self) * _emptyarray(other)).dtype)

    def __rmul__(self, other):
        return UnknownScalar((_emptyarray(self) * _emptyarray(other)).dtype)

    def __truediv__(self, other):
        return UnknownScalar((_emptyarray(self) / _emptyarray(other)).dtype)

    def __floordiv__(self, other):
        return UnknownScalar((_emptyarray(self) // _emptyarray(other)).dtype)

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False


class MaybeNone:
    def __init__(self, content):
        self._content = content

    @property
    def content(self):
        return self._content

    def __eq__(self, other):
        if isinstance(other, MaybeNone):
            return self._content == other._content
        else:
            return False

    def __repr__(self):
        return f"MaybeNone({self._content!r})"

    def __str__(self):
        return f"maybe-{self._content}"


class OneOf:
    def __init__(self, contents):
        self._contents = contents

    @property
    def contents(self):
        return self._contents

    def __eq__(self, other):
        if isinstance(other, OneOf):
            return set(self._contents) == set(other._contents)
        else:
            return False

    def __repr__(self):
        return f"OneOf({self._contents!r})"

    def __str__(self):
        return (
            f"oneof-{'-'.join(str(x).replace('unknown-', '') for x in self._contents)}"
        )


class TypeTracerReport:
    def __init__(self):
        # maybe the order will be useful information
        self._shape_touched_set = set()
        self._shape_touched = []
        self._data_touched_set = set()
        self._data_touched = []

    def __repr__(self):
        return f"<TypeTracerReport with {len(self._shape_touched)} shape_touched, {len(self._data_touched)} data_touched>"

    @property
    def shape_touched(self):
        return self._shape_touched

    @property
    def data_touched(self):
        return self._data_touched

    def touch_shape(self, label):
        if label not in self._shape_touched_set:
            self._shape_touched_set.add(label)
            self._shape_touched.append(label)

    def touch_data(self, label):
        if label not in self._data_touched_set:
            # touching data implies that the shape will be touched as well
            # implemented here so that the codebase doesn't need to be filled
            # with calls to both methods everywhere
            self._shape_touched_set.add(label)
            self._shape_touched.append(label)
            self._data_touched_set.add(label)
            self._data_touched.append(label)


def _attach_report(layout, form, report):
    if isinstance(layout, (ak.contents.BitMaskedArray, ak.contents.ByteMaskedArray)):
        assert isinstance(form, (ak.forms.BitMaskedForm, ak.forms.ByteMaskedForm))
        layout.mask.data.form_key = form.form_key
        layout.mask.data.report = report
        _attach_report(layout.content, form.content, report)

    elif isinstance(layout, ak.contents.EmptyArray):
        assert isinstance(form, ak.forms.EmptyForm)
        layout.mask.data.form_key = form.form_key
        layout.mask.data.report = report

    elif isinstance(layout, (ak.contents.IndexedArray, ak.contents.IndexedOptionArray)):
        assert isinstance(form, (ak.forms.IndexedForm, ak.forms.IndexedOptionForm))
        layout.index.data.form_key = form.form_key
        layout.index.data.report = report
        _attach_report(layout.content, form.content, report)

    elif isinstance(layout, ak.contents.ListArray):
        assert isinstance(form, ak.forms.ListForm)
        layout.starts.data.form_key = form.form_key
        layout.starts.data.report = report
        layout.stops.data.form_key = form.form_key
        layout.stops.data.report = report
        _attach_report(layout.content, form.content, report)

    elif isinstance(layout, ak.contents.ListOffsetArray):
        assert isinstance(form, ak.forms.ListOffsetForm)
        layout.offsets.data.form_key = form.form_key
        layout.offsets.data.report = report
        _attach_report(layout.content, form.content, report)

    elif isinstance(layout, ak.contents.NumpyArray):
        assert isinstance(form, ak.forms.NumpyForm)
        layout.data.form_key = form.form_key
        layout.data.report = report

    elif isinstance(layout, ak.contents.RecordArray):
        assert isinstance(form, ak.forms.RecordForm)
        for x, y in zip(layout.contents, form.contents):
            _attach_report(x, y, report)

    elif isinstance(layout, (ak.contents.RegularArray, ak.contents.UnmaskedForm)):
        assert isinstance(form, (ak.forms.RegularForm, ak.forms.UnmaskedForm))
        _attach_report(layout.content, form.content, report)

    elif isinstance(layout, ak.contents.UnionArray):
        assert isinstance(form, ak.forms.UnionForm)
        layout.tags.data.form_key = form.form_key
        layout.tags.data.report = report
        layout.index.data.form_key = form.form_key
        layout.index.data.report = report
        for x, y in zip(layout.contents, form.contents):
            _attach_report(x, y, report)

    else:
        raise ak._errors.wrap_error(
            AssertionError(f"unrecognized layout type {type(layout)}")
        )


def typetracer_with_report(form, forget_length=True):
    layout = form.length_zero_array(highlevel=False).to_typetracer(
        forget_length=forget_length
    )
    report = TypeTracerReport()
    _attach_report(layout, form, report)
    return layout, report


def _length_after_slice(slice, original_length):
    start, stop, step = slice.indices(original_length)
    assert step != 0

    if (step > 0 and stop - start > 0) or (step < 0 and stop - start < 0):
        d, m = divmod(abs(start - stop), abs(step))
        return d + (1 if m != 0 else 0)
    else:
        return 0


class TypeTracerArray(NDArrayOperatorsMixin, ArrayLike):
    @classmethod
    def from_array(cls, array, dtype: np.dtype | None = None) -> Self:
        """
        Args:
            array: array-like object, e.g. np.ndarray, #ak.index.Index
            dtype: dtype of returned #ak._typetracer.TypeTracerArray

        Returns an #ak._typetracer.TypeTracerArray that describes the type information
        of the given array.
        """
        if isinstance(array, index.Index):
            array = array.data

        # not array-like, try and cast to a NumPy array
        elif not hasattr(array, "shape"):
            array = numpy.array(array)

        if array.dtype == np.dtype("O"):
            raise ak._errors.wrap_error(
                ValueError(
                    f"bug in Awkward Array: attempt to construct `TypeTracerArray` "
                    f"from a sequence of non-primitive types: {array}"
                )
            )

        if dtype is None:
            dtype = array.dtype

        if isinstance(array, TypeTracerArray):
            form_key = array._form_key
            report = array._report
        else:
            form_key = None
            report = None

        return cls(dtype, array.shape, form_key, report)

    def __init__(self, dtype: np.dtype, shape=None, form_key=None, report=None):
        self.form_key = form_key
        self.report = report

        self._dtype = np.dtype(dtype)
        self.shape = shape

    def __repr__(self):
        dtype = repr(self._dtype)
        shape = ""
        if self._shape != (UnknownLength,):
            shape = ", " + repr(self._shape)
        return f"TypeTracerArray({dtype}{shape})"

    @property
    def T(self) -> Self:
        return TypeTracerArray(
            self.dtype, self._shape[::-1], self.form_key, self.report
        )

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self) -> int | UnknownLength:
        size = 1
        for item in self._shape:
            if ak._util.is_integer(item):
                size += item
            else:
                return UnknownLength
        return size

    @property
    def shape(self):
        self.touch_shape()
        return self._shape

    @shape.setter
    def shape(self, value):
        if ak._util.is_integer(value):
            value = (value,)
        elif value is None or isinstance(value, (UnknownLengthType, UnknownScalar)):
            value = (UnknownLength,)
        elif not isinstance(value, tuple):
            value = tuple(value)
        self._shape = value

    @property
    def form_key(self):
        return self._form_key

    @form_key.setter
    def form_key(self, value):
        if value is not None and not isinstance(value, str):
            raise ak._errors.wrap_error(TypeError("form_key must be None or a string"))
        self._form_key = value

    @property
    def report(self):
        return self._report

    @report.setter
    def report(self, value):
        if value is not None and not isinstance(value, TypeTracerReport):
            raise ak._errors.wrap_error(
                TypeError("report must be None or a TypeTracerReport")
            )
        self._report = value

    def touch_shape(self):
        if self._report is not None:
            self._report.touch_shape(self._form_key)

    def touch_data(self):
        if self._report is not None:
            self._report.touch_data(self._form_key)

    @property
    def strides(self):
        self.touch_shape()
        out = (self._dtype.itemsize,)
        for x in self._shape[:0:-1]:
            out = (x * out[0],) + out
        return out

    @property
    def nplike(self):
        return TypeTracer.instance()

    @property
    def ndim(self):
        self.touch_shape()
        return len(self._shape)

    def astype(self, dtype: np.dtype):
        self.touch_data()
        return self.__class__(np.dtype(dtype), self._shape)

    def view(self, dtype: np.dtype):
        if (
            self.itemsize != np.dtype(dtype).itemsize
            and self._shape[-1] != UnknownLength
        ):
            last = int(
                round(self._shape[-1] * self.itemsize / np.dtype(dtype).itemsize)
            )
            shape = self._shape[:-1] + (last,)
        else:
            shape = self._shape
        dtype = np.dtype(dtype)
        return self.__class__(dtype, shape, self._form_key, self._report)

    def forget_length(self):
        return type(self)(
            self._dtype,
            (UnknownLength,) + self._shape[1:],
            self._form_key,
            self._report,
        )

    def __iter__(self):
        raise ak._errors.wrap_error(
            AssertionError(
                "bug in Awkward Array: attempt to convert TypeTracerArray into a concrete array"
            )
        )

    def __array__(self, *args, **kwargs):
        raise ak._errors.wrap_error(
            AssertionError(
                "bug in Awkward Array: attempt to convert TypeTracerArray into a concrete array"
            )
        )

    @property
    def itemsize(self):
        return self._dtype.itemsize

    class _CTypes:
        data = 0

    @property
    def ctypes(self):
        return self._CTypes

    def __len__(self):
        raise ak._errors.wrap_error(
            AssertionError(
                "bug in Awkward Array: attempt to get length of a TypeTracerArray"
            )
        )

    def __setitem__(self, where, what):
        raise ak._errors.wrap_error(
            AssertionError(
                "bug in Awkward Array: attempt to set values of a TypeTracerArray"
            )
        )

    def __getitem__(self, where):
        if isinstance(where, tuple):
            try:
                i = where.index(Ellipsis)
            except ValueError:
                pass
            else:
                before, after = where[:i], where[i + 1 :]
                missing = max(0, len(self._shape) - (len(before) + len(after)))
                where = before + (slice(None, None, None),) * missing + after

        if ak._util.is_integer(where):
            if len(self._shape) == 1:
                self.touch_data()
                if where == 0:
                    return UnknownScalar(self._dtype)
                else:
                    return UnknownScalar(self._dtype)
            else:
                self.touch_shape()
                return TypeTracerArray(
                    self._dtype, self._shape[1:], self._form_key, self._report
                )

        elif isinstance(where, slice):
            self.touch_shape()
            return TypeTracerArray(
                self._dtype,
                (UnknownLength,) + self._shape[1:],
                self._form_key,
                self._report,
            )

        elif (
            hasattr(where, "dtype")
            and hasattr(where, "shape")
            and issubclass(where.dtype.type, np.integer)
        ):
            assert len(self._shape) != 0
            self.touch_data()
            return TypeTracerArray(self._dtype, where.shape + self._shape[1:])

        elif (
            hasattr(where, "dtype")
            and hasattr(where, "shape")
            and issubclass(where.dtype.type, (np.bool_, bool))
        ):
            assert len(self._shape) != 0
            self.touch_data()
            return TypeTracerArray(self._dtype, (UnknownLength,) + self._shape[1:])

        elif isinstance(where, tuple) and any(
            hasattr(x, "dtype") and hasattr(x, "shape") for x in where
        ):
            self.touch_data()

            for num_basic, wh in enumerate(where):  # noqa: B007
                if not isinstance(wh, slice):
                    break

            if num_basic != 0:
                tmp = self.__getitem__(where[:num_basic])
                basic_shape = tmp._shape[:num_basic]
            else:
                basic_shape = ()

            shapes = []
            for j in range(num_basic, len(where)):
                wh = where[j]
                if ak._util.is_integer(wh):
                    shapes.append(numpy.array(0))
                elif hasattr(wh, "dtype") and hasattr(wh, "shape"):
                    sh = [
                        1 if isinstance(x, UnknownLengthType) else int(x)
                        for x in wh.shape
                    ]
                    shapes.append(
                        numpy.lib.stride_tricks.as_strided(
                            numpy.array(0), shape=sh, strides=[0] * len(sh)
                        )
                    )
                else:
                    raise ak._errors.wrap_error(NotImplementedError(repr(wh)))

            slicer_shape = numpy.broadcast_arrays(*shapes)[0].shape

            shape = basic_shape + slicer_shape + self._shape[num_basic + len(shapes) :]
            assert len(shape) != 0

            return TypeTracerArray(self._dtype, (UnknownLength,) + shape[1:])

        elif (
            isinstance(where, tuple)
            and len(where) > 0
            and (ak._util.is_integer(where[0]) or isinstance(where[0], slice))
        ):
            # If there are enough integer slices, this will terminate on the
            # ak._util.is_integer(where) case and end up touching data.
            self.touch_shape()

            head, tail = where[0], where[1:]
            next = self.__getitem__(head)

            inner_shape = next.shape[1:]
            after_shape = []
            for i, wh in enumerate(tail):
                if isinstance(wh, int):
                    pass
                elif isinstance(wh, slice):
                    after_shape.append(_length_after_slice(wh, inner_shape[i]))
                else:
                    raise ak._errors.wrap_error(NotImplementedError(repr(wh)))

            shape = (next._shape[0],) + tuple(after_shape)
            return TypeTracerArray(self._dtype, shape, self._form_key, self._report)

        else:
            raise ak._errors.wrap_error(NotImplementedError(repr(where)))

    def __eq__(self, other):
        self.touch_data()
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape)
        else:
            return NotImplemented

    def __ne__(self, other):
        self.touch_data()
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape)
        else:
            return NotImplemented

    def __lt__(self, other):
        self.touch_data()
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape)
        else:
            return NotImplemented

    def __le__(self, other):
        self.touch_data()
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape)
        else:
            return NotImplemented

    def __gt__(self, other):
        self.touch_data()
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape)
        else:
            return NotImplemented

    def __ge__(self, other):
        self.touch_data()
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape)
        else:
            return NotImplemented

    def reshape(self, *args):
        self.touch_shape()

        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]

        assert len(args) != 0
        assert ak._util.is_integer(args[0]) or isinstance(args[0], UnknownLengthType)
        assert all(ak._util.is_integer(x) for x in args[1:])
        assert all(x >= 0 for x in args[1:])

        return TypeTracerArray(
            self._dtype, (UnknownLength,) + args[1:], self._form_key, self._report
        )

    def copy(self):
        self.touch_data()
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.touch_data()
        replacements = [
            numpy.empty(0, x.dtype) if hasattr(x, "dtype") else x for x in inputs
        ]
        result = getattr(ufunc, method)(*replacements, **kwargs)
        return TypeTracerArray(result.dtype, shape=self._shape)

    def __bool__(self) -> bool:
        raise ak._errors.wrap_error(RuntimeError("cannot realise an unknown value"))

    def __int__(self) -> int:
        raise ak._errors.wrap_error(RuntimeError("cannot realise an unknown value"))

    def __index__(self) -> int:
        raise ak._errors.wrap_error(RuntimeError("cannot realise an unknown value"))


def try_touch_data(array):
    if isinstance(array, TypeTracerArray):
        array.touch_data()


def try_touch_shape(array):
    if isinstance(array, TypeTracerArray):
        array.touch_shape()


class TypeTracer(NumpyLike):
    known_data: Final = False
    known_shape: Final = False

    def to_rectilinear(self, array, *args, **kwargs):
        try_touch_shape(array)
        raise ak._errors.wrap_error(NotImplementedError)

    @property
    def ma(self):
        raise ak._errors.wrap_error(NotImplementedError)

    @property
    def char(self):
        raise ak._errors.wrap_error(NotImplementedError)

    @property
    def ndarray(self):
        return TypeTracerArray

    def raw(self, array: TypeTracerArray, nplike: NumpyLike) -> ArrayLike:
        if nplike is self:
            return TypeTracerArray.from_array(array)
        elif hasattr(nplike, "known_data") and nplike.known_data:
            raise ak._errors.wrap_error(
                TypeError(
                    "Converting a TypeTracer nplike to a nplike with `known_data=True` is not possible"
                )
            )
        else:
            raise ak._errors.wrap_error(
                TypeError(
                    "Invalid nplike, choose between nplike.Numpy, nplike.Cupy, nplike.Jax, or Typetracer"
                )
            )

    ############################ array creation

    def asarray(
        self,
        obj,
        *,
        dtype: numpy.dtype | None = None,
        copy: bool | None = None,
    ) -> TypeTracerArray:
        try_touch_data(obj)
        result = TypeTracerArray.from_array(obj, dtype=dtype)
        # If we want a copy, by the dtypes don't match
        if (
            not (copy is None or copy)
            and dtype is not None
            and getattr(obj, "dtype", dtype) != dtype
        ):
            raise ak._errors.wrap_error(
                ValueError(
                    "asarray was called with copy=False for an array of a different dtype"
                )
            )
        else:
            return result

    def ascontiguousarray(
        self, array: TypeTracerArray, *, dtype=None
    ) -> TypeTracerArray:
        try_touch_data(array)
        return TypeTracerArray.from_array(array, dtype=dtype)

    def frombuffer(
        self, buffer, *, dtype: np.dtype | None = None, count: int = -1
    ) -> TypeTracerArray:
        for x in (buffer, count):
            try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def zeros(
        self, shape: int | tuple[int, ...], *, dtype: np.dtype
    ) -> TypeTracerArray:
        return TypeTracerArray(dtype, shape)

    def ones(self, shape: int | tuple[int, ...], *, dtype: np.dtype) -> TypeTracerArray:
        return TypeTracerArray(dtype, shape)

    def empty(
        self, shape: int | tuple[int, ...], *, dtype: np.dtype
    ) -> TypeTracerArray:
        return TypeTracerArray(dtype, shape)

    def full(
        self, shape: int | tuple[int, ...], fill_value, *, dtype: np.dtype
    ) -> TypeTracerArray:
        array = TypeTracerArray.from_array(fill_value, dtype=dtype)
        return array.reshape(shape)

    def zeros_like(
        self, x: TypeTracerArray, *, dtype: np.dtype | None = None
    ) -> TypeTracerArray:
        try_touch_shape(x)
        if isinstance(x, UnknownScalar):
            return UnknownScalar(dtype)
        return TypeTracerArray.from_array(x, dtype=dtype)

    def ones_like(
        self, x: TypeTracerArray, *, dtype: np.dtype | None = None
    ) -> TypeTracerArray:
        try_touch_shape(x)
        return self.zeros_like(x, dtype=dtype)

    def full_like(
        self, x: TypeTracerArray, fill_value, *, dtype: np.dtype | None = None
    ) -> TypeTracerArray:
        try_touch_shape(x)
        return self.zeros_like(x, dtype=dtype)

    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: np.dtype | None = None,
    ) -> TypeTracerArray:
        try_touch_data(start)
        try_touch_data(stop)
        try_touch_data(step)
        # TODO default type computation
        assert (
            dtype is not None
        ), "internal error: calling arange without dtype (platform dependence)"
        if stop is None:
            start, stop = 0, start

        if (
            ak._util.is_integer(start)
            and ak._util.is_integer(stop)
            and ak._util.is_integer(step)
        ):
            length = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
        else:
            length = UnknownLength

        return TypeTracerArray(dtype, (length,))

    def meshgrid(
        self, *arrays: TypeTracerArray, indexing: Literal["xy", "ij"] = "xy"
    ) -> list[TypeTracerArray]:
        for x in arrays:
            try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    ############################ testing

    def array_equal(
        self, x1: TypeTracerArray, x2: TypeTracerArray, *, equal_nan: bool = False
    ) -> bool:
        try_touch_data(x1)
        try_touch_data(x2)
        return False

    def searchsorted(
        self,
        x: TypeTracerArray,
        values: TypeTracerArray,
        *,
        side: Literal["left", "right"] = "left",
        sorter: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        try_touch_data(values)
        try_touch_data(sorter)
        raise ak._errors.wrap_error(NotImplementedError)

    ############################ manipulation

    def broadcast_arrays(self, *arrays: TypeTracerArray) -> list[TypeTracerArray]:
        for x in arrays:
            try_touch_data(x)

        if len(arrays) == 0:
            return []

        next = []
        maxdim = 0
        for x in arrays:
            if not hasattr(x, "shape"):
                next.append(numpy.array(x))
            else:
                next.append(x)
                maxdim = max(maxdim, len(x.shape))

        if maxdim == 0:
            return next

        first, *rest = next
        shape = list(first.shape[1:])
        for x in rest:
            thisshape = x.shape[1:]
            if len(shape) < len(thisshape):
                shape = [1] * (len(thisshape) - len(shape)) + shape
            elif len(shape) > len(thisshape):
                thisshape = (1,) * (len(shape) - len(thisshape)) + thisshape
            for i in range(len(shape)):  # pylint: disable=consider-using-enumerate
                if shape[i] == 1 and thisshape[i] != 1:
                    shape[i] = thisshape[i]
                elif shape[i] != 1 and thisshape[i] != 1 and shape[i] != thisshape[i]:
                    raise ak._errors.wrap_error(
                        ValueError(
                            "shape mismatch: objects cannot be broadcast to a single shape"
                        )
                    )

        return [
            TypeTracerArray(x.dtype, [UnknownLength] + shape) for x in [first] + rest
        ]

    def add(
        self,
        x1: TypeTracerArray,
        x2: TypeTracerArray,
        *,
        maybe_reuse: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x1)
        try_touch_data(x2)

        is_array = False
        if isinstance(x1, TypeTracerArray):
            is_array = True
            x1 = x1[0]
        if isinstance(x2, TypeTracerArray):
            is_array = True
            x2 = x2[0]
        out = x1 + x2
        if is_array:
            return TypeTracerArray(out.dtype)
        else:
            return out

    def cumsum(
        self,
        x: TypeTracerArray,
        *,
        axis: int | None = None,
        maybe_reuse: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def nonzero(self, x: TypeTracerArray) -> tuple[TypeTracerArray, ...]:
        # array
        try_touch_data(x)
        return (TypeTracerArray(np.int64, (UnknownLength,)),) * len(x.shape)

    def unique_values(self, x: TypeTracerArray) -> TypeTracerArray:
        try_touch_data(x)
        return TypeTracerArray(x.dtype)

    def concat(self, arrays, *, axis: int | None = 0) -> TypeTracerArray:
        if axis is None:
            assert all(x.ndim == 1 for x in arrays)
        elif axis != 0:
            raise ak._errors.wrap_error(NotImplementedError("concat with axis != 0"))
        for x in arrays:
            try_touch_data(x)

        inner_shape = None
        emptyarrays = []
        for x in arrays:
            if inner_shape is None:
                inner_shape = x.shape[1:]
            elif inner_shape != x.shape[1:]:
                raise ak._errors.wrap_error(
                    ValueError(
                        "inner dimensions don't match in concatenate: {} vs {}".format(
                            inner_shape, x.shape[1:]
                        )
                    )
                )
            emptyarrays.append(_emptyarray(x))

        if inner_shape is None:
            raise ak._errors.wrap_error(
                ValueError("need at least one array to concatenate")
            )

        return TypeTracerArray(
            numpy.concatenate(emptyarrays).dtype, (UnknownLength,) + inner_shape
        )

    def repeat(
        self,
        x: TypeTracerArray,
        repeats: TypeTracerArray | int,
        *,
        axis: int | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        try_touch_data(repeats)
        raise ak._errors.wrap_error(NotImplementedError)

    def tile(self, x: TypeTracerArray, reps: int) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def stack(
        self,
        arrays: list[TypeTracerArray] | tuple[TypeTracerArray, ...],
        *,
        axis: int = 0,
    ) -> TypeTracerArray:
        for x in arrays:
            try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def packbits(
        self, array: TypeTracerArray, *, axis=None, bitorder="big"
    ) -> TypeTracerArray:
        try_touch_data(array)
        raise ak._errors.wrap_error(NotImplementedError)

    def unpackbits(
        self, array: TypeTracerArray, *, axis=None, count=None, bitorder="big"
    ) -> TypeTracerArray:
        try_touch_data(array)
        raise ak._errors.wrap_error(NotImplementedError)

    ############################ ufuncs

    def sqrt(
        self, x: TypeTracerArray, maybe_reuse: TypeTracerArray | None = None
    ) -> TypeTracerArray:
        try_touch_data(x)
        # array
        raise ak._errors.wrap_error(NotImplementedError)

    def exp(
        self, x: TypeTracerArray, maybe_reuse: TypeTracerArray | None = None
    ) -> TypeTracerArray:
        try_touch_data(x)
        # array
        raise ak._errors.wrap_error(NotImplementedError)

    def true_divide(
        self,
        x1: TypeTracerArray,
        x2: TypeTracerArray,
        maybe_reuse: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x1)
        try_touch_data(x2)
        # array1, array2
        raise ak._errors.wrap_error(NotImplementedError)

    def logical_and(
        self,
        x1: TypeTracerArray,
        x2: TypeTracerArray,
        maybe_reuse: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x1)
        try_touch_data(x2)

        dtype = np.bool_

        is_array = False
        if isinstance(x1, TypeTracerArray):
            is_array = True
        if isinstance(x2, TypeTracerArray):
            is_array = True
        if is_array:
            return TypeTracerArray(dtype)
        else:
            return UnknownScalar(dtype)

    def logical_or(
        self,
        x1: TypeTracerArray,
        x2: TypeTracerArray,
        maybe_reuse: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x1)
        try_touch_data(x2)
        dtype = np.bool_

        is_array = False
        if isinstance(x1, TypeTracerArray):
            is_array = True
        if isinstance(x2, TypeTracerArray):
            is_array = True
        if is_array:
            return TypeTracerArray(dtype)
        else:
            return UnknownScalar(dtype)

    def logical_not(
        self, x: TypeTracerArray, maybe_reuse: TypeTracerArray | None = None
    ) -> TypeTracerArray:
        try_touch_data(x)
        dtype = np.bool_

        is_array = False
        if isinstance(x, TypeTracerArray):
            is_array = True
        if is_array:
            return TypeTracerArray(dtype)
        else:
            return UnknownScalar(dtype)

    ############################ almost-ufuncs

    def nan_to_num(
        self,
        x: TypeTracerArray,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        # array, copy=True, nan=0.0, posinf=None, neginf=None
        raise ak._errors.wrap_error(NotImplementedError)

    def isclose(
        self,
        x1: TypeTracerArray,
        x2: TypeTracerArray,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> TypeTracerArray:
        try_touch_data(x1)
        try_touch_data(x2)
        # a, b, rtol=1e-05, atol=1e-08, equal_nan=False
        raise ak._errors.wrap_error(NotImplementedError)

    ############################ reducers

    def all(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_reuse: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        if axis is None:
            return UnknownScalar(np.bool_)
        else:
            raise ak._errors.wrap_error(NotImplementedError)

    def any(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_reuse: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        if axis is None:
            return UnknownScalar(np.bool_)
        else:
            raise ak._errors.wrap_error(NotImplementedError)

    def count_nonzero(
        self, x: TypeTracerArray, *, axis: int | None = None, keepdims: bool = False
    ) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def min(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_reuse: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def max(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_reuse: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def array_str(
        self,
        x: TypeTracerArray,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        try_touch_data(x)
        return "[?? ... ??]"

    def can_cast(
        self, from_: np.dtype | TypeTracerArray, to: np.dtype | TypeTracerArray
    ) -> bool:
        return numpy.can_cast(from_, to, casting="same_kind")

    @classmethod
    def is_own_array(cls, obj) -> bool:
        return isinstance(obj, TypeTracerArray)

    def is_c_contiguous(self, array: TypeTracerArray) -> bool:
        return True
