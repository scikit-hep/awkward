# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from numbers import Number

import numpy

import awkward as ak
from awkward._errors import wrap_error
from awkward._nplikes.numpylike import ArrayLike, NumpyLike, NumpyMetadata, ShapeItem
from awkward._util import NDArrayOperatorsMixin, is_non_string_like_sequence
from awkward.typing import (
    Any,
    Final,
    Literal,
    Self,
    SupportsIndex,
    SupportsInt,
    TypeVar,
)

np = NumpyMetadata.instance()


def is_unknown_length(array: Any) -> bool:
    return array is None


def is_unknown_scalar(array: Any) -> bool:
    return isinstance(array, TypeTracerArray) and array.ndim == 0


def is_unknown_integer(array: Any) -> bool:
    return is_unknown_scalar(array) and np.issubdtype(array.dtype, np.integer)


def is_unknown_array(array: Any) -> bool:
    return isinstance(array, TypeTracerArray) and array.ndim > 0


T = TypeVar("T")
S = TypeVar("S")


def ensure_known_scalar(value: T, default: S) -> T | S:
    assert not is_unknown_scalar(default)
    return default if is_unknown_scalar(value) else value


def _emptyarray(x):
    if is_unknown_scalar(x):
        return numpy.empty(0, x._dtype)
    elif hasattr(x, "dtype"):
        return numpy.empty(0, x.dtype)
    else:
        return numpy.empty(0, numpy.array(x).dtype)


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
        return f"?{self._content}"


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


def _attach_report(layout, form, report: TypeTracerReport):
    if isinstance(layout, (ak.contents.BitMaskedArray, ak.contents.ByteMaskedArray)):
        assert isinstance(form, (ak.forms.BitMaskedForm, ak.forms.ByteMaskedForm))
        layout.mask.data.form_key = form.form_key
        layout.mask.data.report = report
        _attach_report(layout.content, form.content, report)

    elif isinstance(layout, ak.contents.EmptyArray):
        assert isinstance(form, ak.forms.EmptyForm)

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

    elif isinstance(layout, (ak.contents.RegularArray, ak.contents.UnmaskedArray)):
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
    _dtype: numpy.dtype
    _shape: tuple[ShapeItem, ...]

    def __new__(cls, *args, **kwargs):
        raise wrap_error(
            TypeError(
                "internal_error: the `TypeTracer` nplike's `TypeTracerArray` object should never be directly instantiated"
            )
        )

    def __reduce__(self):
        # Fix pickling, as we ban `__new__`
        return object.__new__, (type(self),), vars(self)

    @classmethod
    def _new(
        cls,
        dtype: np.dtype,
        shape: tuple[ShapeItem, ...],
        form_key: str | None = None,
        report: TypeTracerReport | None = None,
    ):
        self = super().__new__(cls)
        self.form_key = form_key
        self.report = report

        if not isinstance(shape, tuple):
            raise wrap_error(TypeError("typetracer shape must be a tuple"))
        self._shape = shape
        self._dtype = np.dtype(dtype)

        return self

    def __repr__(self):
        dtype = repr(self._dtype)
        if self.shape is None:
            shape = ""
        else:
            shape = ", shape=" + repr(self._shape)
        return f"TypeTracerArray({dtype}{shape})"

    def __str__(self):
        if self.ndim == 0:
            return "##"

        else:
            return repr(self)

    @property
    def T(self) -> Self:
        return TypeTracerArray._new(
            self.dtype, self._shape[::-1], self.form_key, self.report
        )

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self) -> ShapeItem:
        size = 1
        for item in self._shape:
            if ak._util.is_integer(item):
                size *= item
            else:
                return None
        return size

    @property
    def shape(self) -> tuple[ShapeItem, ...]:
        self.touch_shape()
        return self._shape

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
    def nplike(self) -> TypeTracer:
        return TypeTracer.instance()

    @property
    def ndim(self) -> int:
        self.touch_shape()
        return len(self._shape)

    def view(self, dtype: np.dtype) -> Self:
        if self.itemsize != np.dtype(dtype).itemsize and self._shape[-1] is not None:
            last = int(
                round(self._shape[-1] * self.itemsize / np.dtype(dtype).itemsize)
            )
            shape = self._shape[:-1] + (last,)
        else:
            shape = self._shape
        dtype = np.dtype(dtype)
        return self._new(
            dtype, shape=shape, form_key=self._form_key, report=self._report
        )

    def forget_length(self) -> Self:
        return self._new(
            self._dtype,
            (None,) + self._shape[1:],
            self._form_key,
            self._report,
        )

    def __iter__(self):
        raise ak._errors.wrap_error(
            AssertionError(
                "bug in Awkward Array: attempt to convert TypeTracerArray into a concrete array"
            )
        )

    def __array__(self, dtype=None):
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

    def _resolve_slice_length(self, length, slice_):
        if length is None:
            return None
        elif any(
            is_unknown_scalar(x) for x in (slice_.start, slice_.stop, slice_.step)
        ):
            return None
        else:
            start, stop, step = slice_.indices(length)
            return min((stop - start) // step, length)

    def __getitem__(
        self,
        key: SupportsIndex
        | slice
        | Ellipsis
        | tuple[SupportsIndex | slice | Ellipsis | ArrayLike, ...]
        | ArrayLike,
    ) -> Self:  # noqa: F811
        if not isinstance(key, tuple):
            key = (key,)

        # 1. Validate slice items
        has_seen_ellipsis = 0
        n_basic_non_ellipsis = 0
        n_advanced = 0
        for item in key:
            # Basic indexing
            if isinstance(item, (slice, int)) or is_unknown_integer(item):
                n_basic_non_ellipsis += 1
            # Advanced indexing
            elif isinstance(item, TypeTracerArray) and (
                np.issubdtype(item.dtype, np.integer)
                or np.issubdtype(item.dtype, np.bool_)
            ):
                n_advanced += 1
            # Basic ellipsis
            elif item is Ellipsis:
                if not has_seen_ellipsis:
                    has_seen_ellipsis = True
                else:
                    raise wrap_error(
                        NotImplementedError(
                            "only one ellipsis value permitted for advanced index"
                        )
                    )
            # Basic newaxis
            elif item is np.newaxis:
                pass
            else:
                raise wrap_error(
                    NotImplementedError(
                        "only integer, unknown scalar, slice, ellipsis, or array indices are permitted"
                    )
                )

        # 2. Normalise Ellipsis and boolean arrays
        key_parts = []
        for item in key:
            if item is Ellipsis:
                n_missing_dims = self.ndim - n_advanced - n_basic_non_ellipsis
                key_parts.extend((slice(None),) * n_missing_dims)
            elif is_unknown_array(item) and np.issubdtype(item, np.bool_):
                key_parts.append(self.nplike.nonzero(item)[0])
            else:
                key_parts.append(item)
        key = tuple(key_parts)

        # 3. Apply Indexing
        advanced_is_at_front = False
        previous_item_is_basic = True
        advanced_shapes = []
        adjacent_advanced_shape = []
        result_shape_parts = []
        iter_shape = iter(self.shape)
        for item in key:
            # New axes don't reference existing dimensions
            if item is np.newaxis:
                result_shape_parts.append((1,))
                previous_item_is_basic = True
            # Otherwise, consume the dimension
            else:
                dimension_length = next(iter_shape)
                # Advanced index
                if n_advanced and (
                    isinstance(item, int)
                    or is_unknown_integer(item)
                    or is_unknown_array(item)
                ):
                    if is_unknown_scalar(item):
                        item = self.nplike.promote_scalar(item)

                    # If this is the first advanced index, insert the location
                    if not advanced_shapes:
                        result_shape_parts.append(adjacent_advanced_shape)
                    # If a previous item was basic and we have an advanced shape
                    # we have a split index
                    elif previous_item_is_basic:
                        advanced_is_at_front = True

                    advanced_shapes.append(item.shape)
                    previous_item_is_basic = False
                # Slice
                elif isinstance(item, slice):
                    slice_length = self._resolve_slice_length(dimension_length, item)
                    result_shape_parts.append((slice_length,))
                    previous_item_is_basic = True
                # Integer
                elif isinstance(item, int) or is_unknown_integer(item):
                    item = self.nplike.promote_scalar(item)

                    if is_unknown_length(dimension_length) or is_unknown_integer(item):
                        continue

                    if not 0 <= item < dimension_length:
                        raise wrap_error(
                            NotImplementedError("integer index out of bounds")
                        )

        advanced_shape = self.nplike.broadcast_shapes(*advanced_shapes)
        if advanced_is_at_front:
            result_shape_parts.insert(0, advanced_shape)
        else:
            adjacent_advanced_shape[:] = advanced_shape

        broadcast_shape = tuple(i for p in result_shape_parts for i in p)
        result_shape = broadcast_shape + tuple(iter_shape)

        return self._new(
            self._dtype,
            result_shape,
            self._form_key,
            self._report,
        )

    def __setitem__(
        self,
        key: SupportsIndex
        | slice
        | Ellipsis
        | tuple[SupportsIndex | slice | Ellipsis | ArrayLike, ...]
        | ArrayLike,
        value: int | float | bool | complex | ArrayLike,
    ):  # noqa: F811        existing_value = self.__getitem__(key)
        existing_value = self.__getitem__(key)
        if isinstance(value, TypeTracerArray) and value.ndim > existing_value.ndim:
            raise wrap_error(ValueError("cannot assign shape larger than destination"))

    def copy(self):
        self.touch_data()
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # raise ak._errors.wrap_error(
        #     RuntimeError(
        #         "TypeTracerArray objects should not be used directly with ufuncs"
        #     )
        # )
        kwargs.pop("out", None)

        if method != "__call__" or len(inputs) == 0:
            raise ak._errors.wrap_error(NotImplementedError)

        if len(kwargs) > 0:
            raise ak._errors.wrap_error(
                ValueError("TypeTracerArray does not support kwargs for ufuncs")
            )
        return self.nplike._apply_ufunc(ufunc, *inputs)

    def __bool__(self) -> bool:
        raise ak._errors.wrap_error(RuntimeError("cannot realise an unknown value"))

    def __int__(self) -> int:
        raise ak._errors.wrap_error(RuntimeError("cannot realise an unknown value"))

    def __index__(self) -> int:
        raise ak._errors.wrap_error(RuntimeError("cannot realise an unknown value"))


def _scalar_type_of(obj) -> numpy.dtype:
    if is_unknown_scalar(obj):
        return obj.dtype
    else:
        return numpy.obj2sctype(obj)


def try_touch_data(array):
    if isinstance(array, TypeTracerArray):
        array.touch_data()


def try_touch_shape(array):
    if isinstance(array, TypeTracerArray):
        array.touch_shape()


class TypeTracer(NumpyLike):
    known_data: Final = False
    known_shape: Final = False
    is_eager: Final = True

    def _apply_ufunc(self, ufunc, *inputs):
        for x in inputs:
            try_touch_data(x)

        broadcasted = self.broadcast_arrays(*inputs)
        placeholders = [numpy.empty(0, x.dtype) for x in broadcasted]

        result = ufunc(*placeholders)
        return TypeTracerArray._new(result.dtype, shape=broadcasted[0].shape)

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

    ############################ array creation

    def asarray(
        self,
        obj,
        *,
        dtype: numpy.dtype | None = None,
        copy: bool | None = None,
    ) -> TypeTracerArray:
        try_touch_data(obj)

        if isinstance(obj, ak.index.Index):
            obj = obj.data

        if isinstance(obj, TypeTracerArray):
            form_key = obj._form_key
            report = obj._report

            if dtype is None:
                return obj
            elif copy is False and dtype != obj.dtype:
                raise ak._errors.wrap_error(
                    ValueError(
                        "asarray was called with copy=False for an array of a different dtype"
                    )
                )
            else:
                return TypeTracerArray._new(
                    dtype, obj.shape, form_key=form_key, report=report
                )
        else:
            # Convert NumPy generics to scalars
            if isinstance(obj, np.generic):
                obj = numpy.asarray(obj)

            # Support array-like objects
            if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                if obj.dtype.kind == "S":
                    raise ak._errors.wrap_error(
                        TypeError("TypeTracerArray cannot be created from strings")
                    )
                elif copy is False and dtype != obj.dtype:
                    raise ak._errors.wrap_error(
                        ValueError(
                            "asarray was called with copy=False for an array of a different dtype"
                        )
                    )
                else:
                    return TypeTracerArray._new(obj.dtype, obj.shape)
            # Python objects
            elif isinstance(obj, (Number, bool)):
                as_array = numpy.asarray(obj)
                return TypeTracerArray._new(as_array.dtype, ())

            elif is_non_string_like_sequence(obj):
                assert not any(is_non_string_like_sequence(x) for x in obj)
                shape = (len(obj),)
                result_type = numpy.result_type(*obj)  # TODO: result_type
                return TypeTracerArray._new(result_type, shape)
            else:
                raise wrap_error(TypeError)

    def ascontiguousarray(
        self, x: ArrayLike, *, dtype: numpy.dtype | None = None
    ) -> TypeTracerArray:
        try_touch_data(x)
        return TypeTracerArray._new(dtype or x.dtype, shape=x.shape)

    def frombuffer(
        self, buffer, *, dtype: np.dtype | None = None, count: int = -1
    ) -> TypeTracerArray:
        for x in (buffer, count):
            try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def zeros(
        self, shape: ShapeItem | tuple[ShapeItem, ...], *, dtype: np.dtype | None = None
    ) -> TypeTracerArray:
        if not isinstance(shape, tuple):
            shape = (shape,)
        return TypeTracerArray._new(dtype, shape)

    def ones(
        self, shape: ShapeItem | tuple[ShapeItem, ...], *, dtype: np.dtype | None = None
    ) -> TypeTracerArray:
        if not isinstance(shape, tuple):
            shape = (shape,)
        return TypeTracerArray._new(dtype, shape)

    def empty(
        self, shape: ShapeItem | tuple[ShapeItem, ...], *, dtype: np.dtype | None = None
    ) -> TypeTracerArray:
        if not isinstance(shape, tuple):
            shape = (shape,)
        return TypeTracerArray._new(dtype, shape)

    def full(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        fill_value,
        *,
        dtype: np.dtype | None = None,
    ) -> TypeTracerArray:
        if not isinstance(shape, tuple):
            shape = (shape,)
        dtype = _scalar_type_of(fill_value) if dtype is None else dtype
        return TypeTracerArray._new(dtype, shape)

    def zeros_like(
        self, x: ArrayLike, *, dtype: np.dtype | None = None
    ) -> TypeTracerArray:
        try_touch_shape(x)
        if is_unknown_scalar(x):
            return TypeTracerArray._new(dtype or x.dtype, shape=())
        else:
            return TypeTracerArray._new(dtype or x.dtype, shape=x.shape)

    def ones_like(
        self, x: ArrayLike, *, dtype: np.dtype | None = None
    ) -> TypeTracerArray:
        try_touch_shape(x)
        return self.zeros_like(x, dtype=dtype)

    def full_like(
        self, x: ArrayLike, fill_value, *, dtype: np.dtype | None = None
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
        if stop is None:
            start, stop = 0, start

        if (
            ak._util.is_integer(start)
            and ak._util.is_integer(stop)
            and ak._util.is_integer(step)
        ):
            length = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
        else:
            length = None

        default_int_type = np.int64 if (ak._util.win or ak._util.bits32) else np.int32
        return TypeTracerArray._new(dtype or default_int_type, (length,))

    def meshgrid(
        self, *arrays: ArrayLike, indexing: Literal["xy", "ij"] = "xy"
    ) -> list[TypeTracerArray]:
        for x in arrays:
            try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    ############################ testing

    def array_equal(
        self, x1: ArrayLike, x2: ArrayLike, *, equal_nan: bool = False
    ) -> bool:
        try_touch_data(x1)
        try_touch_data(x2)
        return False

    def searchsorted(
        self,
        x: ArrayLike,
        values: ArrayLike,
        *,
        side: Literal["left", "right"] = "left",
        sorter: ArrayLike | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        try_touch_data(values)
        try_touch_data(sorter)
        raise ak._errors.wrap_error(NotImplementedError)

    ############################ manipulation

    def promote_scalar(self, obj) -> TypeTracerArray:
        if is_unknown_scalar(obj):
            return obj
        elif isinstance(obj, (Number, bool)):
            # TODO: statically define these types for all nplikes
            as_array = numpy.asarray(obj)
            return TypeTracerArray._new(as_array.dtype, ())
        else:
            raise wrap_error(TypeError(f"expected scalar type, received {obj}"))

    def shape_item_as_scalar(self, x1: ShapeItem) -> TypeTracerArray:
        if x1 is None:
            return TypeTracerArray._new(np.int64, shape=())
        elif isinstance(x1, int):
            return TypeTracerArray._new(np.int64, shape=())
        else:
            raise wrap_error(TypeError(f"expected None or int type, received {x1}"))

    def scalar_as_shape_item(self, x1) -> ShapeItem:
        if x1 is None:
            return None
        elif is_unknown_scalar(x1) and np.issubdtype(x1.dtype, np.integer):
            return None
        else:
            return int(x1)

    def sub_shape_item(self, x1: ShapeItem, x2: ShapeItem) -> ShapeItem:
        if x1 is None:
            return None
        if x2 is None:
            return None
        assert x1 >= 0
        assert x2 >= 0
        result = x1 - x2
        assert result >= 0
        return result

    def add_shape_item(self, x1: ShapeItem, x2: ShapeItem) -> ShapeItem:
        if x1 is None:
            return None
        if x2 is None:
            return None
        assert x1 >= 0
        assert x2 >= 0
        result = x1 + x2
        return result

    def mul_shape_item(self, x1: ShapeItem, x2: ShapeItem) -> ShapeItem:
        if x1 is None:
            return None
        if x2 is None:
            return None
        assert x1 >= 0
        assert x2 >= 0
        result = x1 * x2
        return result

    def div_shape_item(self, x1: ShapeItem, x2: ShapeItem) -> ShapeItem:
        if x1 is None:
            return None
        if x2 is None:
            return None
        assert x1 >= 0
        assert x2 >= 0
        result = x1 // x2
        assert result * x2 == x1
        return result

    def broadcast_shapes(
        self, *shapes: tuple[SupportsInt, ...]
    ) -> tuple[SupportsInt, ...]:
        ndim = max([len(s) for s in shapes], default=0)
        result: list[SupportsInt] = [1] * ndim

        for shape in shapes:
            # Right broadcasting
            missing_dim = ndim - len(shape)
            if missing_dim > 0:
                head: tuple[int, ...] = (1,) * missing_dim
                shape = head + shape

            # Fail if we absolutely know the shapes aren't compatible
            for i, item in enumerate(shape):
                # Item is unknown, take it
                if is_unknown_scalar(item):
                    result[i] = item
                # Existing item is unknown, keep it
                elif is_unknown_scalar(result[i]):
                    continue
                # Items match, continue
                elif result[i] == item:
                    continue
                # Item is broadcastable, take existing
                elif item == 1:
                    continue
                # Existing is broadcastable, take it
                elif result[i] == 1:
                    result[i] = item
                else:
                    raise wrap_error(
                        ValueError(
                            "known component of shape does not match broadcast result"
                        )
                    )
        return tuple(result)

    def broadcast_arrays(self, *arrays: ArrayLike) -> list[TypeTracerArray]:
        for x in arrays:
            try_touch_data(x)

        if len(arrays) == 0:
            return []

        all_arrays = []
        for x in arrays:
            if not hasattr(x, "shape"):
                x = self.promote_scalar(x)
            all_arrays.append(x)

        shapes = [x.shape for x in all_arrays]
        shape = self.broadcast_shapes(*shapes)

        return [TypeTracerArray._new(x.dtype, shape=shape) for x in all_arrays]

    def broadcast_to(
        self, x: ArrayLike, shape: tuple[SupportsInt, ...]
    ) -> TypeTracerArray:
        raise ak._errors.wrap_error(NotImplementedError)

    def reshape(
        self, x: ArrayLike, shape: tuple[int, ...], *, copy: bool | None = None
    ) -> TypeTracerArray:
        x.touch_shape()

        size = x.size

        # Validate new shape to ensure that it only contains at-most one placeholder
        n_placeholders = 0
        new_size = 1
        for item in shape:
            if item is None:
                # Size is no longer defined
                new_size = None
            elif not ak._util.is_integer(item):
                raise wrap_error(
                    ValueError(
                        "shape must be comprised of positive integers, -1 (for placeholders), or unknown lengths"
                    )
                )
            elif item == -1:
                if n_placeholders == 1:
                    raise wrap_error(
                        ValueError("only one placeholder dimension permitted per shape")
                    )
                n_placeholders += 1
            elif item == 0:
                raise wrap_error(ValueError("shape items cannot be zero"))
            else:
                new_size = self.mul_shape_item(new_size, item)

        # Populate placeholders
        new_shape = [*shape]
        for i, item in enumerate(shape):
            if item == -1:
                new_shape[i] = self.div_shape_item(size, new_size)
                break

        return TypeTracerArray._new(x.dtype, tuple(new_shape), x.form_key, x.report)

    def cumsum(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def nonzero(self, x: ArrayLike) -> tuple[TypeTracerArray, ...]:
        # array
        try_touch_data(x)
        return (TypeTracerArray._new(np.int64, (None,)),) * len(x.shape)

    def unique_values(self, x: ArrayLike) -> TypeTracerArray:
        try_touch_data(x)
        return TypeTracerArray._new(x.dtype)

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

        return TypeTracerArray._new(
            numpy.concatenate(emptyarrays).dtype, (None,) + inner_shape
        )

    def repeat(
        self,
        x: ArrayLike,
        repeats: ArrayLike | int,
        *,
        axis: int | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        try_touch_data(repeats)
        raise ak._errors.wrap_error(NotImplementedError)

    def tile(self, x: ArrayLike, reps: int) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def stack(
        self,
        arrays: list[ArrayLike] | tuple[ArrayLike, ...],
        *,
        axis: int = 0,
    ) -> TypeTracerArray:
        for x in arrays:
            try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def packbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def unpackbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    ############################ ufuncs

    def add(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        return self._apply_ufunc(numpy.add, x1, x2)

    def logical_and(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        return self._apply_ufunc(numpy.sqrt, x1, x2)

    def logical_or(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        return self._apply_ufunc(numpy.sqrt, x1, x2)

    def logical_not(
        self, x: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> TypeTracerArray:
        return self._apply_ufunc(numpy.sqrt, x)

    def sqrt(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> TypeTracerArray:
        return self._apply_ufunc(numpy.sqrt, x)

    def exp(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> TypeTracerArray:
        return self._apply_ufunc(numpy.exp, x)

    def divide(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        return self._apply_ufunc(numpy.sqrt, x1, x2)

    ############################ almost-ufuncs

    def nan_to_num(
        self,
        x: ArrayLike,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def isclose(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> TypeTracerArray:
        try_touch_data(x1)
        try_touch_data(x2)
        out, _ = self.broadcast_arrays(x1, x2)
        return TypeTracerArray._new(np.bool_, shape=out.shape)

    def isnan(self, x: ArrayLike) -> TypeTracerArray:
        try_touch_data(x)
        return TypeTracerArray._new(np.bool_, shape=x.shape)

    ############################ reducers

    def all(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        if axis is None:
            return TypeTracerArray._new(np.bool_, shape=())
        else:
            raise ak._errors.wrap_error(NotImplementedError)

    def any(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        if axis is None:
            return TypeTracerArray._new(np.bool_, shape=())
        else:
            raise ak._errors.wrap_error(NotImplementedError)

    def count_nonzero(
        self, x: ArrayLike, *, axis: int | None = None, keepdims: bool = False
    ) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def min(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def max(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        try_touch_data(x)
        raise ak._errors.wrap_error(NotImplementedError)

    def array_str(
        self,
        x: ArrayLike,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        try_touch_data(x)
        return "[## ... ##]"

    def astype(
        self, x: ArrayLike, dtype: numpy.dtype, *, copy: bool | None = True
    ) -> TypeTracerArray:
        x.touch_data()
        return TypeTracerArray._new(np.dtype(dtype), x.shape)

    def can_cast(self, from_: np.dtype | ArrayLike, to: np.dtype | ArrayLike) -> bool:
        return numpy.can_cast(from_, to, casting="same_kind")

    @classmethod
    def is_own_array(cls, obj) -> bool:
        return isinstance(obj, TypeTracerArray)

    def is_c_contiguous(self, x: ArrayLike) -> bool:
        return True
