# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from numbers import Number

import numpy

import awkward as ak
from awkward._nplikes.dispatch import register_nplike
from awkward._nplikes.numpylike import (
    ArrayLike,
    IndexType,
    NumpyLike,
    NumpyMetadata,
    UniqueAllResult,
)
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._operators import NDArrayOperatorsMixin
from awkward._regularize import is_integer, is_non_string_like_sequence
from awkward._typing import (
    Any,
    Final,
    Literal,
    Self,
    SupportsIndex,
    TypeVar,
)

np = NumpyMetadata.instance()


def is_unknown_length(array: Any) -> bool:
    return array is unknown_length


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


class TypeTracerArray(NDArrayOperatorsMixin, ArrayLike):
    _dtype: numpy.dtype
    _shape: tuple[ShapeItem, ...]

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "internal_error: the `TypeTracer` nplike's `TypeTracerArray` object should never be directly instantiated"
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
            raise TypeError("typetracer shape must be a tuple")
        if any(is_unknown_scalar(x) for x in shape):
            raise TypeError("typetracer shape must be integers or unknown-length")
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
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def size(self) -> ShapeItem:
        size = 1
        for item in self._shape:
            size *= item
        return size

    @property
    def shape(self) -> tuple[ShapeItem, ...]:
        self.touch_shape()
        return self._shape

    @property
    def form_key(self) -> str | None:
        return self._form_key

    @form_key.setter
    def form_key(self, value: str | None):
        if value is not None and not isinstance(value, str):
            raise TypeError("form_key must be None or a string")
        self._form_key = value

    @property
    def report(self) -> TypeTracerReport | None:
        return self._report

    @report.setter
    def report(self, value: TypeTracerReport | None):
        if value is not None and not isinstance(value, TypeTracerReport):
            raise TypeError("report must be None or a TypeTracerReport")
        self._report = value

    def touch_shape(self):
        if self._report is not None:
            self._report.touch_shape(self._form_key)

    def touch_data(self):
        if self._report is not None:
            self._report.touch_data(self._form_key)

    @property
    def nplike(self) -> TypeTracer:
        return TypeTracer.instance()

    @property
    def ndim(self) -> int:
        self.touch_shape()
        return len(self._shape)

    @property
    def nbytes(self) -> ShapeItem:
        return self.size * self._dtype.itemsize

    def view(self, dtype: np.dtype) -> Self:
        dtype = np.dtype(dtype)
        if len(self._shape) >= 1:
            last, remainder = divmod(
                self._shape[-1] * self._dtype.itemsize, dtype.itemsize
            )
            if remainder is not unknown_length and remainder != 0:
                raise ValueError(
                    "new size of array with larger dtype must be a "
                    "divisor of the total size in bytes (of the last axis of the array)"
                )
            shape = self._shape[:-1] + (last,)
        else:
            shape = self._shape
        return self._new(
            dtype, shape=shape, form_key=self._form_key, report=self._report
        )

    def forget_length(self) -> Self:
        return self._new(
            self._dtype,
            (unknown_length,) + self._shape[1:],
            self._form_key,
            self._report,
        )

    def __iter__(self):
        raise AssertionError(
            "bug in Awkward Array: attempt to convert TypeTracerArray into a concrete array"
        )

    def __array__(self, dtype=None):
        raise AssertionError(
            "bug in Awkward Array: attempt to convert TypeTracerArray into a concrete array"
        )

    class _CTypes:
        data = 0

    @property
    def ctypes(self):
        return self._CTypes

    def __len__(self):
        raise AssertionError(
            "bug in Awkward Array: attempt to get length of a TypeTracerArray"
        )

    def __getitem__(
        self,
        key: SupportsIndex
        | slice
        | Ellipsis
        | tuple[SupportsIndex | slice | Ellipsis | ArrayLike, ...]
        | ArrayLike,
    ) -> Self | int | float | bool | complex:
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
                    raise NotImplementedError(
                        "only one ellipsis value permitted for advanced index"
                    )
            # Basic newaxis
            elif item is np.newaxis:
                pass
            else:
                raise NotImplementedError(
                    "only integer, unknown scalar, slice, ellipsis, or array indices are permitted"
                )

        n_dim_index = n_basic_non_ellipsis + n_advanced
        if n_dim_index > self.ndim:
            raise IndexError(
                f"too many indices for array: array is {self.ndim}-dimensional, but {n_dim_index} were indexed"
            )

        # 2. Normalise Ellipsis and boolean arrays
        key_parts = []
        for item in key:
            if item is Ellipsis:
                # How many more dimensions do we have than the index provides
                n_missing_dims = self.ndim - n_dim_index
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
                    try_touch_data(item)
                    try_touch_data(self)

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
                    (
                        start,
                        stop,
                        step,
                        slice_length,
                    ) = self.nplike.derive_slice_for_length(item, dimension_length)
                    result_shape_parts.append((slice_length,))
                    previous_item_is_basic = True
                # Integer
                elif isinstance(item, int) or is_unknown_integer(item):
                    try_touch_data(item)
                    try_touch_data(self)

                    item = self.nplike.promote_scalar(item)

                    if is_unknown_length(dimension_length) or is_unknown_integer(item):
                        continue

                    if not 0 <= item < dimension_length:
                        raise NotImplementedError("integer index out of bounds")

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
    ):
        existing_value = self.__getitem__(key)
        if isinstance(value, TypeTracerArray) and value.ndim > existing_value.ndim:
            raise ValueError("cannot assign shape larger than destination")

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
            raise NotImplementedError

        if len(kwargs) > 0:
            raise ValueError("TypeTracerArray does not support kwargs for ufuncs")
        return self.nplike._apply_ufunc(ufunc, *inputs)

    def __bool__(self) -> bool:
        raise RuntimeError("cannot realise an unknown value")

    def __int__(self) -> int:
        raise RuntimeError("cannot realise an unknown value")

    def __index__(self) -> int:
        raise RuntimeError("cannot realise an unknown value")


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


@register_nplike
class TypeTracer(NumpyLike):
    known_data: Final = False
    is_eager: Final = True
    supports_structured_dtypes: Final = True

    def _apply_ufunc(self, ufunc, *inputs):
        for x in inputs:
            assert not isinstance(x, PlaceholderArray)
            try_touch_data(x)

        inputs = [x.content if isinstance(x, MaybeNone) else x for x in inputs]

        broadcasted = self.broadcast_arrays(*inputs)
        placeholders = [numpy.empty(0, x.dtype) for x in broadcasted]

        result = ufunc(*placeholders)
        if isinstance(result, numpy.ndarray):
            return TypeTracerArray._new(result.dtype, shape=broadcasted[0].shape)
        elif isinstance(result, tuple):
            return (
                TypeTracerArray._new(x.dtype, shape=b.shape)
                for x, b in zip(result, broadcasted)
            )
        else:
            raise TypeError

    def _axis_is_valid(self, axis: int, ndim: int) -> bool:
        if axis < 0:
            axis = axis + ndim
        return 0 <= axis < ndim

    @property
    def ma(self):
        raise NotImplementedError

    @property
    def char(self):
        raise NotImplementedError

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
        assert not isinstance(obj, PlaceholderArray)

        if isinstance(obj, ak.index.Index):
            obj = obj.data

        if isinstance(obj, TypeTracerArray):
            form_key = obj._form_key
            report = obj._report

            if dtype is None:
                return obj
            elif dtype == obj.dtype:
                return TypeTracerArray._new(
                    dtype, obj.shape, form_key=form_key, report=report
                )
            elif copy is False:
                raise ValueError(
                    "asarray was called with copy=False for an array of a different dtype"
                )
            else:
                try_touch_data(obj)
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
                    raise TypeError("TypeTracerArray cannot be created from strings")
                elif copy is False and dtype != obj.dtype:
                    raise ValueError(
                        "asarray was called with copy=False for an array of a different dtype"
                    )
                else:
                    return TypeTracerArray._new(obj.dtype, obj.shape)
            # Python objects
            elif isinstance(obj, (Number, bool)):
                as_array = numpy.asarray(obj)
                return TypeTracerArray._new(as_array.dtype, ())

            elif is_non_string_like_sequence(obj):
                shape = []
                flat_items = []
                has_seen_leaf = False

                # DFS walk into sequence, construct shape, then validate
                # remainder of the sequence against this shape.
                def populate_shape_and_items(node, dim):
                    nonlocal has_seen_leaf

                    # If we've already computed the shape,
                    # ensure this item matches!
                    if has_seen_leaf:
                        if len(node) != shape[dim - 1]:
                            raise ValueError(
                                f"sequence at dimension {dim} does not match shape {shape[dim-1]}"
                            )
                    else:
                        shape.append(len(node))

                    if isinstance(node, TypeTracerArray):
                        raise AssertionError(
                            "typetracer arrays inside sequences not currently supported"
                        )
                    # Found leaf!
                    elif len(node) == 0 or not is_non_string_like_sequence(node[0]):
                        has_seen_leaf = True
                        flat_items.extend(
                            [
                                item.dtype if is_unknown_scalar(item) else item
                                for item in node
                            ]
                        )

                    # Keep recursing!
                    else:
                        for child in node:
                            populate_shape_and_items(child, dim + 1)

                populate_shape_and_items(obj, 1)
                if dtype is None:
                    dtype = numpy.result_type(*flat_items)
                return TypeTracerArray._new(dtype, shape=tuple(shape))
            else:
                raise TypeError

    def ascontiguousarray(self, x: ArrayLike) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        return TypeTracerArray._new(
            x.dtype, shape=x.shape, form_key=x.form_key, report=x.report
        )

    def frombuffer(
        self, buffer, *, dtype: np.dtype | None = None, count: int = -1
    ) -> TypeTracerArray:
        for x in (buffer, count):
            assert not isinstance(x, PlaceholderArray)
            try_touch_data(x)
        raise NotImplementedError

    def from_dlpack(self, x: Any) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        raise NotImplementedError

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
        assert not isinstance(fill_value, PlaceholderArray)
        if not isinstance(shape, tuple):
            shape = (shape,)
        dtype = _scalar_type_of(fill_value) if dtype is None else dtype
        return TypeTracerArray._new(dtype, shape)

    def zeros_like(
        self, x: ArrayLike, *, dtype: np.dtype | None = None
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        try_touch_shape(x)
        if is_unknown_scalar(x):
            return TypeTracerArray._new(dtype or x.dtype, shape=())
        else:
            return TypeTracerArray._new(dtype or x.dtype, shape=x.shape)

    def ones_like(
        self, x: ArrayLike, *, dtype: np.dtype | None = None
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        try_touch_shape(x)
        return self.zeros_like(x, dtype=dtype)

    def full_like(
        self, x: ArrayLike, fill_value, *, dtype: np.dtype | None = None
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
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
        assert not isinstance(start, PlaceholderArray)
        assert not isinstance(stop, PlaceholderArray)
        assert not isinstance(step, PlaceholderArray)
        try_touch_data(start)
        try_touch_data(stop)
        try_touch_data(step)
        if stop is None:
            start, stop = 0, start

        if is_integer(start) and is_integer(stop) and is_integer(step):
            length = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
        else:
            length = unknown_length

        default_int_type = np.int64 if (ak._util.win or ak._util.bits32) else np.int32
        return TypeTracerArray._new(dtype or default_int_type, (length,))

    def meshgrid(
        self, *arrays: ArrayLike, indexing: Literal["xy", "ij"] = "xy"
    ) -> list[TypeTracerArray]:
        for x in arrays:
            assert not isinstance(x, PlaceholderArray)
            try_touch_data(x)

            assert x.ndim == 1

        shape = tuple(x.size for x in arrays)
        if indexing == "xy":
            shape[:2] = shape[1], shape[0]

        dtype = numpy.result_type(*arrays)
        return [TypeTracerArray._new(dtype, shape=shape) for _ in arrays]

    ############################ testing

    def array_equal(
        self, x1: ArrayLike, x2: ArrayLike, *, equal_nan: bool = False
    ) -> TypeTracerArray:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        try_touch_data(x1)
        try_touch_data(x2)
        return TypeTracerArray._new(np.bool_, shape=())

    def searchsorted(
        self,
        x: ArrayLike,
        values: ArrayLike,
        *,
        side: Literal["left", "right"] = "left",
        sorter: ArrayLike | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        assert not isinstance(values, PlaceholderArray)
        assert not isinstance(sorter, PlaceholderArray)
        try_touch_data(x)
        try_touch_data(values)
        try_touch_data(sorter)
        if (
            not (
                is_unknown_length(x.size)
                or sorter is None
                or is_unknown_length(sorter.size)
            )
            and x.size != sorter.size
        ):
            raise ValueError("x.size should equal sorter.size")

        return TypeTracerArray._new(x.dtype, (values.size,))

    ############################ manipulation

    def promote_scalar(self, obj) -> TypeTracerArray:
        assert not isinstance(obj, PlaceholderArray)
        if is_unknown_scalar(obj):
            return obj
        elif isinstance(obj, (Number, bool)):
            # TODO: statically define these types for all nplikes
            as_array = numpy.asarray(obj)
            return TypeTracerArray._new(as_array.dtype, ())
        else:
            raise TypeError(f"expected scalar type, received {obj}")

    def shape_item_as_index(self, x1: ShapeItem) -> IndexType:
        if x1 is unknown_length:
            return TypeTracerArray._new(np.int64, shape=())
        elif isinstance(x1, int):
            return x1
        else:
            raise TypeError(f"expected None or int type, received {x1}")

    def index_as_shape_item(self, x1: IndexType) -> ShapeItem:
        if is_unknown_scalar(x1) and np.issubdtype(x1.dtype, np.integer):
            return unknown_length
        else:
            return int(x1)

    def regularize_index_for_length(
        self, index: IndexType, length: ShapeItem
    ) -> IndexType:
        """
        Args:
            index: index value
            length: length of array

        Returns regularized index that is guaranteed to be in-bounds.
        """
        # Unknown indices are already regularized
        if is_unknown_scalar(index):
            return index

        # Without a known length the result must be unknown, as we cannot regularize the index
        length_scalar = self.shape_item_as_index(length)
        if length is unknown_length:
            return length_scalar

        # We have known length and index
        if index < 0:
            index = index + length

        if 0 <= index < length:
            return index
        else:
            raise IndexError(f"index value out of bounds (0, {length}): {index}")

    def derive_slice_for_length(
        self, slice_: slice, length: ShapeItem
    ) -> tuple[IndexType, IndexType, IndexType, ShapeItem]:
        """
        Args:
            slice_: normalized slice object
            length: length of layout

        Return a tuple of (start, stop, step, length) indices into a layout, suitable for
        `_getitem_range` (if step == 1). Normalize lengths to fit length of array,
        and for arrays with unknown lengths, these offsets become none.
        """
        start = slice_.start
        stop = slice_.stop
        step = slice_.step

        # Unknown lengths mean that the slice index is unknown
        length_scalar = self.shape_item_as_index(length)
        if length is unknown_length:
            return length_scalar, length_scalar, step, length
        else:
            # Normalise `None` values
            if step is None:
                step = 1

            if start is None:
                # `step` is unknown → `start` is unknown
                if is_unknown_scalar(step):
                    start = step
                elif step < 0:
                    start = length_scalar - 1
                else:
                    start = 0
            # Normalise negative integers
            elif not is_unknown_scalar(start):
                if start < 0:
                    start = start + length_scalar
                # Clamp values into length bounds
                if is_unknown_scalar(length_scalar):
                    start = length_scalar
                else:
                    start = min(max(start, 0), length_scalar)

            if stop is None:
                # `step` is unknown → `stop` is unknown
                if is_unknown_scalar(step):
                    stop = step
                elif step < 0:
                    stop = -1
                else:
                    stop = length_scalar
            # Normalise negative integers
            elif not is_unknown_scalar(stop):
                if stop < 0:
                    stop = stop + length_scalar
                # Clamp values into length bounds
                if is_unknown_scalar(length_scalar):
                    stop = length_scalar
                else:
                    stop = min(max(stop, 0), length_scalar)

            # Compute the length of the slice for downstream use
            slice_length, remainder = divmod((stop - start), step)
            if not is_unknown_scalar(slice_length):
                # Take ceiling of division
                if remainder != 0:
                    slice_length += 1

                slice_length = max(0, slice_length)

            return start, stop, step, self.index_as_shape_item(slice_length)

    def broadcast_shapes(self, *shapes: tuple[ShapeItem, ...]) -> tuple[ShapeItem, ...]:
        ndim = max([len(s) for s in shapes], default=0)
        result: list[ShapeItem] = [1] * ndim

        for shape in shapes:
            # Right broadcasting
            missing_dim = ndim - len(shape)
            if missing_dim > 0:
                head: tuple[int, ...] = (1,) * missing_dim
                shape = head + shape

            # Fail if we absolutely know the shapes aren't compatible
            for i, item in enumerate(shape):
                # Item is unknown, take it
                if is_unknown_length(item):
                    result[i] = item
                # Existing item is unknown, keep it
                elif is_unknown_length(result[i]):
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
                    raise ValueError(
                        "known component of shape does not match broadcast result"
                    )
        return tuple(result)

    def broadcast_arrays(self, *arrays: ArrayLike) -> list[TypeTracerArray]:
        for x in arrays:
            assert not isinstance(x, PlaceholderArray)
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
        self, x: ArrayLike, shape: tuple[ShapeItem, ...]
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        new_shape = self.broadcast_shapes(x.shape, shape)
        # broadcast_to is asymmetric, whilst broadcast_shapes is not
        # rather than implement broadcasting logic here, let's just santitise the result
        # the above broadcasting result can either be equal to `shape`, have greater number dimensions,
        # and/or have differing dimensions we only want the case where the shape is equal
        if len(new_shape) != len(shape):
            raise ValueError

        for result, intended in zip(new_shape, shape):
            if intended is unknown_length:
                continue
            if result is unknown_length:
                continue
            if intended != result:
                raise ValueError
        return TypeTracerArray._new(x.dtype, shape=new_shape)

    def reshape(
        self, x: ArrayLike, shape: tuple[ShapeItem, ...], *, copy: bool | None = None
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        x.touch_shape()

        size = x.size

        # Validate new shape to ensure that it only contains at-most one placeholder
        n_placeholders = 0
        new_size = 1
        for item in shape:
            if item is unknown_length:
                # Size is no longer defined
                new_size = unknown_length
            elif not is_integer(item):
                raise ValueError(
                    "shape must be comprised of positive integers, -1 (for placeholders), or unknown lengths"
                )
            elif item == -1:
                if n_placeholders == 1:
                    raise ValueError(
                        "only one placeholder dimension permitted per shape"
                    )
                n_placeholders += 1
            elif item == 0:
                raise ValueError("shape items cannot be zero")
            else:
                new_size *= item

        # Populate placeholders
        new_shape = [*shape]
        for i, item in enumerate(shape):
            if item == -1:
                new_shape[i] = size // new_size
                break

        return TypeTracerArray._new(x.dtype, tuple(new_shape), x.form_key, x.report)

    def cumsum(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        if axis is None:
            return TypeTracerArray._new(x.dtype, (x.size,))
        else:
            assert self._axis_is_valid(axis, x.ndim)
            return TypeTracerArray._new(x.dtype, x.shape)

    def nonzero(self, x: ArrayLike) -> tuple[TypeTracerArray, ...]:
        assert not isinstance(x, PlaceholderArray)
        # array
        try_touch_data(x)
        return (TypeTracerArray._new(np.int64, (unknown_length,)),) * len(x.shape)

    def where(
        self, condition: ArrayLike, x1: ArrayLike, x2: ArrayLike
    ) -> TypeTracerArray:
        assert not isinstance(condition, PlaceholderArray)
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        condition, x1, x2 = self.broadcast_arrays(condition, x1, x2)
        result_dtype = numpy.result_type(x1, x2)
        return TypeTracerArray._new(result_dtype, shape=condition.shape)

    def unique_values(self, x: ArrayLike) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        return TypeTracerArray._new(x.dtype, shape=(unknown_length,))

    def unique_all(self, x: ArrayLike) -> UniqueAllResult:
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        return UniqueAllResult(
            TypeTracerArray._new(x.dtype, shape=(unknown_length,)),
            TypeTracerArray._new(np.int64, shape=(unknown_length,)),
            TypeTracerArray._new(np.int64, shape=x.shape),
            TypeTracerArray._new(np.int64, shape=(unknown_length,)),
        )

    def sort(
        self,
        x: ArrayLike,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        return TypeTracerArray._new(x.dtype, shape=x.shape)

    def concat(self, arrays, *, axis: int | None = 0) -> TypeTracerArray:
        if axis is None:
            assert all(x.ndim == 1 for x in arrays)
        elif axis != 0:
            raise NotImplementedError("concat with axis != 0")
        for x in arrays:
            try_touch_data(x)

        inner_shape = None
        emptyarrays = []
        for x in arrays:
            assert not isinstance(x, PlaceholderArray)
            if inner_shape is None:
                inner_shape = x.shape[1:]
            elif inner_shape != x.shape[1:]:
                raise ValueError(
                    "inner dimensions don't match in concatenate: {} vs {}".format(
                        inner_shape, x.shape[1:]
                    )
                )
            emptyarrays.append(_emptyarray(x))

        if inner_shape is None:
            raise ValueError("need at least one array to concatenate")

        return TypeTracerArray._new(
            numpy.concatenate(emptyarrays).dtype, (unknown_length, *inner_shape)
        )

    def repeat(
        self,
        x: ArrayLike,
        repeats: ArrayLike | int,
        *,
        axis: int | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        assert not isinstance(repeats, PlaceholderArray)
        try_touch_data(x)
        try_touch_data(repeats)

        if axis is None:
            size = x.size
            if is_unknown_array(repeats):
                size = unknown_length
            else:
                size = size * self.index_as_shape_item(repeats)
            return TypeTracerArray._new(x.dtype, (size,))
        else:
            shape = list(x.shape)
            if isinstance(repeats, TypeTracerArray) and repeats.ndim > 0:
                raise NotImplementedError
            else:
                shape[axis] = shape[axis] * self.index_as_shape_item(repeats)
            return TypeTracerArray._new(x.dtype, shape=tuple(shape))

    def stack(
        self,
        arrays: list[ArrayLike] | tuple[ArrayLike, ...],
        *,
        axis: int = 0,
    ) -> TypeTracerArray:
        for x in arrays:
            assert not isinstance(x, PlaceholderArray)
            try_touch_data(x)
        raise NotImplementedError

    def packbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        raise NotImplementedError

    def unpackbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        raise NotImplementedError

    def strides(self, x: ArrayLike) -> tuple[ShapeItem, ...]:
        assert not isinstance(x, PlaceholderArray)
        x.touch_shape()
        out = (x._dtype.itemsize,)
        for item in reversed(x._shape):
            out = (item * out[0], *out)
        return out

    ############################ ufuncs

    def add(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x1, PlaceholderArray)
        return self._apply_ufunc(numpy.add, x1, x2)

    def logical_and(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x1, PlaceholderArray)
        return self._apply_ufunc(numpy.logical_and, x1, x2)

    def logical_or(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        return self._apply_ufunc(numpy.logical_or, x1, x2)

    def logical_not(
        self, x: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        return self._apply_ufunc(numpy.logical_not, x)

    def sqrt(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        return self._apply_ufunc(numpy.sqrt, x)

    def exp(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        return self._apply_ufunc(numpy.exp, x)

    def divide(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        return self._apply_ufunc(numpy.divide, x1, x2)

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
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        return TypeTracerArray._new(x.dtype, shape=x.shape)

    def isclose(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> TypeTracerArray:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        try_touch_data(x1)
        try_touch_data(x2)
        out, _ = self.broadcast_arrays(x1, x2)
        return TypeTracerArray._new(np.bool_, shape=out.shape)

    def isnan(self, x: ArrayLike) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
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
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        if axis is None:
            return TypeTracerArray._new(np.bool_, shape=())
        else:
            raise NotImplementedError

    def any(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        if axis is None:
            return TypeTracerArray._new(np.bool_, shape=())
        else:
            raise NotImplementedError

    def count_nonzero(
        self, x: ArrayLike, *, axis: int | None = None, keepdims: bool = False
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        if axis is None:
            return TypeTracerArray._new(np.intp, shape=())
        else:
            raise NotImplementedError

    def min(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        raise NotImplementedError

    def max(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        if axis is None:
            return TypeTracerArray._new(x.dtype, shape=())
        else:
            raise NotImplementedError

    def array_str(
        self,
        x: ArrayLike,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        assert not isinstance(x, PlaceholderArray)
        try_touch_data(x)
        return "[## ... ##]"

    def astype(
        self, x: ArrayLike, dtype: numpy.dtype, *, copy: bool | None = True
    ) -> TypeTracerArray:
        assert not isinstance(x, PlaceholderArray)
        x.touch_data()
        return TypeTracerArray._new(np.dtype(dtype), x.shape)

    def can_cast(self, from_: np.dtype | ArrayLike, to: np.dtype | ArrayLike) -> bool:
        return numpy.can_cast(from_, to, casting="same_kind")

    @classmethod
    def is_own_array_type(cls, type_: type) -> bool:
        return issubclass(type_, TypeTracerArray)

    @classmethod
    def is_own_array(cls, obj) -> bool:
        return cls.is_own_array_type(type(obj))

    def is_c_contiguous(self, x: ArrayLike) -> bool:
        assert not isinstance(x, PlaceholderArray)
        return True

    def __dlpack_device__(self) -> tuple[int, int]:
        raise NotImplementedError

    def __dlpack__(self, stream=None):
        raise NotImplementedError


def _attach_report(
    layout: ak.contents.Content, form: ak.forms.Form, report: TypeTracerReport
):
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
        raise AssertionError(f"unrecognized layout type {type(layout)}")


def typetracer_with_report(
    form: ak.forms.Form, forget_length: bool = True
) -> tuple[ak.contents.Content, TypeTracerReport]:
    layout = form.length_zero_array(highlevel=False).to_typetracer(
        forget_length=forget_length
    )
    report = TypeTracerReport()
    _attach_report(layout, form, report)
    return layout, report
