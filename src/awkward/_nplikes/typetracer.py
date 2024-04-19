# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Collection, Sequence, Set
from numbers import Number
from typing import Callable, Iterator

import numpy

import awkward as ak
from awkward._nplikes.dispatch import register_nplike
from awkward._nplikes.numpy_like import (
    ArrayLike,
    IndexType,
    NumpyLike,
    NumpyMetadata,
    UfuncLike,
    UniqueAllResult,
)
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._operators import NDArrayOperatorsMixin
from awkward._regularize import is_integer, is_non_string_like_sequence
from awkward._typing import (
    TYPE_CHECKING,
    Any,
    DType,
    Final,
    Literal,
    Self,
    SupportsIndex,
    TypeGuard,
    TypeVar,
    cast,
)

if TYPE_CHECKING:
    from types import EllipsisType

    from numpy.typing import DTypeLike

    from awkward.contents.content import Content
    from awkward.forms.form import Form


np = NumpyMetadata.instance()


def is_unknown_length(array: Any) -> bool:
    return array is unknown_length


def is_unknown_scalar(array: Any) -> TypeGuard[TypeTracerArray]:
    return isinstance(array, TypeTracerArray) and array.ndim == 0


def is_unknown_integer(array: Any) -> TypeGuard[TypeTracerArray]:
    return is_unknown_scalar(array) and np.issubdtype(array.dtype, np.integer)


def is_unknown_array(array: Any) -> TypeGuard[TypeTracerArray]:
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


class ImmutableBitSet(Set):
    def __init__(self, byteset: FillableByteSet):
        self._labels: dict[str, int] = byteset._labels
        if not byteset._is_filled.any():
            self._is_filled = None
        else:
            self._is_filled = numpy.packbits(byteset._is_filled)

    def __contains__(self, label: object) -> bool:
        if self._is_filled is None or label not in self._labels:
            return False
        else:
            assert isinstance(label, str)
            return numpy.unpackbits(self._is_filled)[self._labels[label]]

    def __iter__(self) -> Iterator[str]:
        if self._is_filled is not None:
            is_filled = numpy.unpackbits(self._is_filled)
            for label, index in self._labels.items():
                if is_filled[index]:
                    yield label

    def __len__(self) -> int:
        if self._is_filled is None:
            return 0
        else:
            return int(numpy.unpackbits(self._is_filled).sum())


class FillableByteSet(Set):
    # friend class ImmutableBitSet

    def __init__(self, labels: Collection[str]):
        self._labels = {label: i for i, label in enumerate(labels)}
        self._is_filled = numpy.zeros(len(labels), dtype=numpy.bool_)

    def add(self, label: str) -> None:
        self._is_filled[self._labels[label]] = True

    def to_bitset(self) -> ImmutableBitSet:
        return ImmutableBitSet(self)

    def clear(self) -> None:
        self._is_filled.fill(False)

    def __contains__(self, label: object) -> bool:
        if label not in self._labels:
            return False
        else:
            assert isinstance(label, str)
            return self._is_filled[self._labels[label]]

    def __iter__(self) -> Iterator[str]:
        for label, index in self._labels.items():
            if self._is_filled[index]:
                yield label

    def __len__(self) -> int:
        return int(self._is_filled.sum())


class TypeTracerReport:
    def __init__(self):
        self._shape_touched_set = set()
        self._data_touched_set = set()
        self._node_id_to_shape_touched: dict[str, ImmutableBitSet] = {}
        self._node_id_to_data_touched: dict[str, ImmutableBitSet] = {}

    def __repr__(self):
        return (
            f"<TypeTracerReport with {len(self._shape_touched_set)} shape_touched, "
            f"{len(self._data_touched_set)} data_touched>"
        )

    def set_labels(self, labels: Collection[str]):
        self._shape_touched_set = FillableByteSet(labels)
        self._data_touched_set = FillableByteSet(labels)

    @property
    def shape_touched(self) -> list[str]:
        return list(self._shape_touched_set)

    @property
    def data_touched(self) -> list[str]:
        return list(self._data_touched_set)

    def touch_shape(self, label: str) -> None:
        self._shape_touched_set.add(label)

    def touch_data(self, label: str) -> None:
        # Touching data implies that the shape will be touched as well
        # implemented here so that the codebase doesn't need to be filled
        # with calls to both methods everywhere
        self._shape_touched_set.add(label)
        self._data_touched_set.add(label)

    def commit(self, node_id: str) -> None:
        assert isinstance(self._shape_touched_set, FillableByteSet)
        assert isinstance(self._data_touched_set, FillableByteSet)
        self._node_id_to_shape_touched[node_id] = self._shape_touched_set.to_bitset()
        self._node_id_to_data_touched[node_id] = self._data_touched_set.to_bitset()
        self._shape_touched_set.clear()
        self._data_touched_set.clear()

    def shape_touched_in(self, node_ids: Collection[str]) -> list[str]:
        out: set[str] = set()
        for node_id in node_ids:
            out.update(self._node_id_to_shape_touched[node_id])
        return list(out)

    def data_touched_in(self, node_ids: Collection[str]) -> list[str]:
        out: set[str] = set()
        for node_id in node_ids:
            tmp = self._node_id_to_data_touched.get(node_id)
            if tmp is not None:
                out.update(tmp)
        return list(out)


class TypeTracerArray(NDArrayOperatorsMixin, ArrayLike):
    _dtype: numpy.dtype
    _shape: tuple[ShapeItem, ...]

    runtime_typechecks = True

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
        dtype: DType,
        shape: tuple[ShapeItem, ...],
        form_key: str | None = None,
        report: TypeTracerReport | None = None,
    ):
        self = super().__new__(cls)
        self.form_key = form_key
        self.report = report

        if cls.runtime_typechecks:
            if not isinstance(shape, tuple):
                raise TypeError("typetracer shape must be a tuple")
            if not all(isinstance(x, int) or x is unknown_length for x in shape):
                raise TypeError("typetracer shape must be integers or unknown-length")
            if not isinstance(dtype, np.dtype):
                raise TypeError("typetracer dtype must be an instance of np.dtype")
        self._shape = shape
        self._dtype = dtype

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
    def dtype(self) -> DType:
        return self._dtype

    @property
    def size(self) -> ShapeItem:
        size: ShapeItem = 1
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
        return len(self._shape)

    @property
    def nbytes(self) -> ShapeItem:
        return self.size * self._dtype.itemsize

    def view(self, dtype: DTypeLike) -> Self:
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
        | EllipsisType
        | tuple[SupportsIndex | slice | EllipsisType | ArrayLike, ...]
        | ArrayLike,
    ) -> Self:
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
        key_parts: list[SupportsIndex | slice | ArrayLike] = []
        for item in key:
            if item is Ellipsis:
                # How many more dimensions do we have than the index provides
                n_missing_dims = self.ndim - n_dim_index
                key_parts.extend((slice(None),) * n_missing_dims)
            elif is_unknown_array(item) and np.issubdtype(item, np.bool_):
                key_parts.append(self.nplike.nonzero(item)[0])
            else:
                key_parts.append(item)  # type: ignore[arg-type]
        key = tuple(key_parts)

        # 3. Apply Indexing
        advanced_is_at_front = False
        previous_item_is_basic = True
        advanced_shapes: list[tuple[ShapeItem, ...]] = []
        adjacent_advanced_shape: list[ShapeItem] = []
        result_shape_parts: list[Sequence[ShapeItem]] = []
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

                    item = self.nplike.asarray(item)

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

                    item = self.nplike.asarray(item)

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
        | EllipsisType
        | tuple[SupportsIndex | slice | EllipsisType | ArrayLike, ...]
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

        if len(kwargs) > 0:
            raise ValueError("TypeTracerArray does not support kwargs for ufuncs")
        return self.nplike.apply_ufunc(ufunc, method, inputs, kwargs)

    def __bool__(self) -> bool:
        raise RuntimeError("cannot realise an unknown value")

    def __int__(self) -> int:
        raise RuntimeError("cannot realise an unknown value")

    def __index__(self) -> int:
        raise RuntimeError("cannot realise an unknown value")

    def __dlpack_device__(self) -> tuple[int, int]:
        raise RuntimeError("cannot realise an unknown value")

    def __dlpack__(self, stream: Any = None) -> Any:
        raise RuntimeError("cannot realise an unknown value")


def _scalar_type_of(obj) -> DType:
    if is_unknown_scalar(obj):
        return obj.dtype
    else:
        return numpy.array(obj).dtype


def try_touch_data(array: Any):
    if isinstance(array, TypeTracerArray):
        array.touch_data()


def try_touch_shape(array: Any):
    if isinstance(array, TypeTracerArray):
        array.touch_shape()


@register_nplike
class TypeTracer(NumpyLike[TypeTracerArray]):
    known_data: Final = False
    is_eager: Final = True
    supports_structured_dtypes: Final = True

    def apply_ufunc(
        self,
        ufunc: UfuncLike,
        method: str,
        args: list[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> TypeTracerArray | tuple[TypeTracerArray, ...]:
        if method != "__call__" or len(args) == 0:
            raise NotImplementedError

        if hasattr(ufunc, "resolve_dtypes"):
            return self._apply_ufunc_nep_50(ufunc, method, args, kwargs)
        else:
            return self._apply_ufunc_legacy(ufunc, method, args, kwargs)

    def _get_nep_50_dtype(
        self, obj: Any
    ) -> DType | type[int] | type[complex] | type[float]:
        if hasattr(obj, "dtype"):
            return obj.dtype
        elif isinstance(obj, bool):
            return np.dtype(np.bool_)
        else:
            assert isinstance(obj, (int, complex, float))
            return type(obj)

    def _apply_ufunc_nep_50(
        self,
        ufunc: UfuncLike,
        method: str,
        args: Sequence[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> TypeTracerArray | tuple[TypeTracerArray, ...]:
        for x in args:
            try_touch_data(x)

        # Unwrap options, assume they don't occur
        args = [x.content if isinstance(x, MaybeNone) else x for x in args]
        # Determine input argument dtypes
        input_arg_dtypes = [self._get_nep_50_dtype(obj) for obj in args]
        # Resolve these for the given ufunc
        arg_dtypes = tuple(input_arg_dtypes + [None] * ufunc.nout)
        resolved_dtypes = ufunc.resolve_dtypes(arg_dtypes)
        # Interpret the arguments under these dtypes
        resolved_args = [
            self.asarray(arg, dtype=dtype) for arg, dtype in zip(args, resolved_dtypes)
        ]
        # Broadcast to ensure all-scalar or all-nd-array
        broadcasted_args = self.broadcast_arrays(*resolved_args)
        broadcasted_shape = broadcasted_args[0].shape
        result_dtypes = resolved_dtypes[ufunc.nin :]

        if len(result_dtypes) == 1:
            return TypeTracerArray._new(result_dtypes[0], shape=broadcasted_shape)
        else:
            return tuple(
                TypeTracerArray._new(dtype, shape=broadcasted_shape)
                for dtype in result_dtypes
            )

    def _apply_ufunc_legacy(
        self,
        ufunc: UfuncLike,
        method: str,
        args: Sequence[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> TypeTracerArray | tuple[TypeTracerArray, ...]:
        for x in args:
            try_touch_data(x)

        # Unwrap options, assume they don't occur
        args = [x.content if isinstance(x, MaybeNone) else x for x in args]
        # Convert np.generic to scalar arrays
        resolved_args = [
            self.asarray(arg, dtype=arg.dtype if hasattr(arg, "dtype") else None)
            for arg in args
        ]
        # Broadcast all inputs together
        broadcasted_args = self.broadcast_arrays(*resolved_args)
        broadcasted_shape = broadcasted_args[0].shape
        # Choose the broadcasted argument if it wasn't a Python scalar
        non_generic_value_promoted_args = [
            y if hasattr(x, "ndim") else x for x, y in zip(args, broadcasted_args)
        ]
        # Build proxy (empty) arrays
        proxy_args = [
            (numpy.empty(0, dtype=x.dtype) if hasattr(x, "dtype") else x)
            for x in non_generic_value_promoted_args
        ]
        # Determine result dtype from proxy call
        proxy_result = ufunc(*proxy_args, **(kwargs or {}))
        if ufunc.nout == 1:
            result_dtypes = [proxy_result.dtype]
        else:
            assert isinstance(proxy_result, tuple)
            result_dtypes = [x.dtype for x in proxy_result]

        if len(result_dtypes) == 1:
            return TypeTracerArray._new(result_dtypes[0], shape=broadcasted_shape)
        else:
            return tuple(
                TypeTracerArray._new(dtype, shape=broadcasted_shape)
                for dtype in result_dtypes
            )

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
        dtype: DTypeLike | None = None,
        copy: bool | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(obj, PlaceholderArray)

        if dtype is not None:
            dtype = np.dtype(dtype)

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
                shape: list[ShapeItem] = []
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
                                f"sequence at dimension {dim} does not match shape {shape[dim - 1]}"
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

    def ascontiguousarray(
        self, x: TypeTracerArray | PlaceholderArray
    ) -> TypeTracerArray | PlaceholderArray:
        assert isinstance(x, TypeTracerArray)
        return TypeTracerArray._new(
            x.dtype, shape=x.shape, form_key=x.form_key, report=x.report
        )

    def frombuffer(
        self, buffer, *, dtype: DTypeLike | None = None, count: ShapeItem = -1
    ) -> TypeTracerArray:
        assert not isinstance(buffer, PlaceholderArray)
        try_touch_data(buffer)
        try_touch_data(count)

        if isinstance(buffer, TypeTracerArray) or is_unknown_scalar(count):
            raise NotImplementedError
        else:
            return self.asarray(numpy.frombuffer(buffer, dtype=dtype, count=count))

    def from_dlpack(self, x: Any) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        raise NotImplementedError

    def zeros(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        *,
        dtype: DTypeLike | None = None,
    ) -> TypeTracerArray:
        if not isinstance(shape, tuple):
            shape = (shape,)
        if dtype is None:
            dtype = np.dtype(np.finfo(float).dtype)
        else:
            dtype = np.dtype(dtype)
        return TypeTracerArray._new(dtype, shape)

    def ones(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        *,
        dtype: DTypeLike | None = None,
    ) -> TypeTracerArray:
        if not isinstance(shape, tuple):
            shape = (shape,)
        if dtype is None:
            dtype = np.dtype(np.finfo(float).dtype)
        else:
            dtype = np.dtype(dtype)
        return TypeTracerArray._new(dtype, shape)

    def empty(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        *,
        dtype: DTypeLike | None = None,
    ) -> TypeTracerArray:
        if not isinstance(shape, tuple):
            shape = (shape,)
        if dtype is None:
            dtype = np.dtype(np.finfo(float).dtype)
        else:
            dtype = np.dtype(dtype)
        return TypeTracerArray._new(dtype, shape)

    def full(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        fill_value,
        *,
        dtype: DTypeLike | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(fill_value, PlaceholderArray)
        if not isinstance(shape, tuple):
            shape = (shape,)
        dtype = _scalar_type_of(fill_value) if dtype is None else np.dtype(dtype)
        return TypeTracerArray._new(dtype, shape)

    def zeros_like(
        self, x: TypeTracerArray | PlaceholderArray, *, dtype: DTypeLike | None = None
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_shape(x)
        if dtype is None:
            dtype = x.dtype
        else:
            dtype = np.dtype(dtype)
        if is_unknown_scalar(x):
            return TypeTracerArray._new(dtype, shape=())
        else:
            return TypeTracerArray._new(dtype, shape=x.shape)

    def ones_like(
        self, x: TypeTracerArray | PlaceholderArray, *, dtype: DTypeLike | None = None
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_shape(x)
        return self.zeros_like(x, dtype=dtype)

    def full_like(
        self,
        x: TypeTracerArray | PlaceholderArray,
        fill_value,
        *,
        dtype: DTypeLike | None = None,
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_shape(x)
        return self.zeros_like(x, dtype=dtype)

    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: DTypeLike | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(start, PlaceholderArray)
        assert not isinstance(stop, PlaceholderArray)
        assert not isinstance(step, PlaceholderArray)
        try_touch_data(start)
        try_touch_data(stop)
        try_touch_data(step)
        if stop is None:
            start, stop = 0, start

        length: ShapeItem
        if is_integer(start) and is_integer(stop) and is_integer(step):
            length = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)  # type: ignore[assignment]
        else:
            length = unknown_length

        if dtype is None:
            dtype = np.dtype(np.iinfo(int).dtype)
        else:
            dtype = np.dtype(dtype)
        return TypeTracerArray._new(dtype, (length,))

    def meshgrid(
        self, *arrays: TypeTracerArray, indexing: Literal["xy", "ij"] = "xy"
    ) -> list[TypeTracerArray]:
        for x in arrays:
            assert isinstance(x, TypeTracerArray)
            try_touch_data(x)

            assert x.ndim == 1

        shape: list[ShapeItem] = [x.size for x in arrays]
        if indexing == "xy":
            shape[:2] = shape[1], shape[0]

        dtype = numpy.result_type(*arrays)
        return [TypeTracerArray._new(dtype, shape=tuple(shape)) for _ in arrays]

    ############################ testing

    def array_equal(
        self, x1: TypeTracerArray, x2: TypeTracerArray, *, equal_nan: bool = False
    ) -> bool:
        raise RuntimeError

    def searchsorted(
        self,
        x: TypeTracerArray,
        values: TypeTracerArray,
        *,
        side: Literal["left", "right"] = "left",
        sorter: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
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
    def shape_item_as_index(self, x1: ShapeItem) -> IndexType:
        if x1 is unknown_length:
            return TypeTracerArray._new(np.dtype(np.int64), shape=())
        elif isinstance(x1, int):
            return x1
        else:
            raise TypeError(f"expected unknown_length or int type, received {x1}")

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

        length = cast(int, length)
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
        ndim = max((len(s) for s in shapes), default=0)
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

    def broadcast_arrays(self, *arrays: TypeTracerArray) -> list[TypeTracerArray]:
        for x in arrays:
            assert isinstance(x, TypeTracerArray)
            try_touch_data(x)

        if len(arrays) == 0:
            return []

        all_arrays = []
        for x in arrays:
            if not hasattr(x, "shape"):
                x = self.asarray(x)
            all_arrays.append(x)

        shapes = [x.shape for x in all_arrays]
        shape = self.broadcast_shapes(*shapes)

        return [TypeTracerArray._new(x.dtype, shape=shape) for x in all_arrays]

    def broadcast_to(
        self, x: TypeTracerArray, shape: tuple[ShapeItem, ...]
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
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
        self,
        x: TypeTracerArray | PlaceholderArray,
        shape: tuple[ShapeItem, ...],
        *,
        copy: bool | None = None,
    ) -> TypeTracerArray | PlaceholderArray:
        assert isinstance(x, TypeTracerArray)
        x.touch_shape()

        size = x.size

        # Validate new shape to ensure that it only contains at-most one placeholder
        n_placeholders = 0
        new_size: ShapeItem = 1
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
        x: TypeTracerArray,
        *,
        axis: int | None = None,
        maybe_out: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        if axis is None:
            return TypeTracerArray._new(x.dtype, (x.size,))
        else:
            assert self._axis_is_valid(axis, x.ndim)
            return TypeTracerArray._new(x.dtype, x.shape)

    def nonzero(self, x: TypeTracerArray) -> tuple[TypeTracerArray, ...]:
        assert isinstance(x, TypeTracerArray)
        # array
        try_touch_data(x)
        return (TypeTracerArray._new(np.dtype(np.int64), (unknown_length,)),) * len(
            x.shape
        )

    def where(
        self, condition: TypeTracerArray, x1: TypeTracerArray, x2: TypeTracerArray
    ) -> TypeTracerArray:
        assert not isinstance(condition, PlaceholderArray)
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        condition, x1, x2 = self.broadcast_arrays(condition, x1, x2)
        result_dtype = numpy.result_type(x1, x2)
        return TypeTracerArray._new(result_dtype, shape=condition.shape)

    def unique_values(self, x: TypeTracerArray) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        return TypeTracerArray._new(x.dtype, shape=(unknown_length,))

    def unique_all(self, x: TypeTracerArray) -> UniqueAllResult:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        return UniqueAllResult(
            TypeTracerArray._new(x.dtype, shape=(unknown_length,)),
            TypeTracerArray._new(np.dtype(np.int64), shape=(unknown_length,)),
            TypeTracerArray._new(np.dtype(np.int64), shape=x.shape),
            TypeTracerArray._new(np.dtype(np.int64), shape=(unknown_length,)),
        )

    def sort(
        self,
        x: TypeTracerArray,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
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
            assert isinstance(x, TypeTracerArray)
            if inner_shape is None:
                inner_shape = x.shape[1:]
            elif inner_shape != x.shape[1:]:
                raise ValueError(
                    f"inner dimensions don't match in concatenate: {inner_shape} vs {x.shape[1:]}"
                )
            emptyarrays.append(_emptyarray(x))

        if inner_shape is None:
            raise ValueError("need at least one array to concatenate")

        return TypeTracerArray._new(
            numpy.concatenate(emptyarrays).dtype, (unknown_length, *inner_shape)
        )

    def repeat(
        self,
        x: TypeTracerArray,
        repeats: TypeTracerArray | int,
        *,
        axis: int | None = None,
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
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
        arrays: list[TypeTracerArray] | tuple[TypeTracerArray, ...],
        *,
        axis: int = 0,
    ) -> TypeTracerArray:
        # Ensure all arrays have same ndim
        ndim = arrays[0].ndim
        assert all(x.ndim == ndim for x in arrays[1:])

        if axis is None:
            assert all(x.ndim == 1 for x in arrays)
        elif axis < 0:
            axis = ndim + axis
        if not 0 <= axis < ndim:
            raise ValueError(axis)

        for x in arrays:
            assert isinstance(x, TypeTracerArray)
            try_touch_data(x)

        emptyarrays = [numpy.empty_like((0,) * ndim, dtype=a.dtype) for a in arrays]
        result = numpy.stack(emptyarrays, axis=axis)
        return TypeTracerArray._new(result.dtype, result.shape)

    def packbits(
        self,
        x: TypeTracerArray,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        raise NotImplementedError

    def unpackbits(
        self,
        x: TypeTracerArray,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        raise NotImplementedError

    def strides(self, x: TypeTracerArray | PlaceholderArray) -> tuple[ShapeItem, ...]:
        assert isinstance(x, TypeTracerArray)
        x.touch_shape()
        out: tuple[ShapeItem, ...] = (x._dtype.itemsize,)
        for item in reversed(x._shape):
            out = (item * out[0], *out)
        return out

    ############################ ufuncs

    def add(
        self,
        x1: TypeTracerArray,
        x2: TypeTracerArray,
        maybe_out: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x1, PlaceholderArray)
        return self.apply_ufunc(numpy.add, "__call__", (x1, x2))  # type: ignore[arg-type,return-value]

    def logical_and(
        self,
        x1: TypeTracerArray,
        x2: TypeTracerArray,
        maybe_out: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x1, PlaceholderArray)
        return self.apply_ufunc(numpy.logical_and, "__call__", (x1, x2))  # type: ignore[arg-type,return-value]

    def logical_or(
        self,
        x1: TypeTracerArray,
        x2: TypeTracerArray,
        maybe_out: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        return self.apply_ufunc(numpy.logical_or, "__call__", (x1, x2))  # type: ignore[arg-type,return-value]

    def logical_not(
        self, x: TypeTracerArray, maybe_out: TypeTracerArray | None = None
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        return self.apply_ufunc(numpy.logical_not, "__call__", (x,))  # type: ignore[arg-type,return-value]

    def sqrt(
        self, x: TypeTracerArray, maybe_out: TypeTracerArray | None = None
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        return self.apply_ufunc(numpy.sqrt, "__call__", (x,))  # type: ignore[arg-type,return-value]

    def exp(
        self, x: TypeTracerArray, maybe_out: TypeTracerArray | None = None
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        return self.apply_ufunc(numpy.exp, "__call__", (x,))  # type: ignore[arg-type,return-value]

    def divide(
        self,
        x1: TypeTracerArray,
        x2: TypeTracerArray,
        maybe_out: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        return self.apply_ufunc(numpy.divide, "__call__", (x1, x2))  # type: ignore[arg-type,return-value]

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
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        return TypeTracerArray._new(x.dtype, shape=x.shape)

    def isclose(
        self,
        x1: TypeTracerArray,
        x2: TypeTracerArray,
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
        return TypeTracerArray._new(np.dtype(np.bool_), shape=out.shape)

    def isnan(self, x: TypeTracerArray) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        return TypeTracerArray._new(np.dtype(np.bool_), shape=x.shape)

    def real(self, x: TypeTracerArray) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        real_type = numpy.real(numpy.zeros(0, dtype=x.dtype)).dtype
        return TypeTracerArray._new(real_type, shape=x.shape)

    def imag(self, x: TypeTracerArray) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        real_type = numpy.imag(numpy.zeros(0, dtype=x.dtype)).dtype
        return TypeTracerArray._new(real_type, shape=x.shape)

    def angle(self, x: TypeTracerArray, deg: bool = False) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        float_type = numpy.angle(numpy.zeros(0, dtype=x.dtype)).dtype
        return TypeTracerArray._new(float_type, shape=x.shape)

    def round(
        self,
        x: TypeTracerArray,
        decimals: int = 0,
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        return TypeTracerArray._new(x.dtype, shape=x.shape)

    ############################ reducers

    def all(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)

        if isinstance(axis, tuple):
            raise NotImplementedError
        if maybe_out is not None:
            raise NotImplementedError

        if axis is None:
            return self.all(
                cast(TypeTracerArray, self.reshape(x, (-1,))),
                axis=axis,
                keepdims=keepdims,
                maybe_out=maybe_out,
            )

        if axis < 0:
            axis = axis + x.ndim

        assert 0 <= axis < x.ndim

        if keepdims:
            next_shape = list(x.shape)
            next_shape[axis] = 1
            return TypeTracerArray._new(np.dtype(np.bool_), shape=tuple(next_shape))
        else:
            next_shape = list(x.shape)
            del next_shape[axis]
            return TypeTracerArray._new(np.dtype(np.bool_), shape=tuple(next_shape))

    def any(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        return self.all(x, axis=axis, keepdims=keepdims, maybe_out=maybe_out)

    def count_nonzero(
        self, x: TypeTracerArray, *, axis: int | tuple[int, ...] | None = None
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        if axis is None:
            return TypeTracerArray._new(np.dtype(np.intp), shape=())
        else:
            raise NotImplementedError

    def min(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)

        if isinstance(axis, tuple):
            raise NotImplementedError
        if maybe_out is not None:
            raise NotImplementedError

        if axis is None:
            return self.min(
                cast(TypeTracerArray, self.reshape(x, (-1,))),
                axis=axis,
                keepdims=keepdims,
                maybe_out=maybe_out,
            )

        if axis < 0:
            axis = axis + x.ndim

        assert 0 <= axis < x.ndim

        if keepdims:
            next_shape = list(x.shape)
            next_shape[axis] = 1
            return TypeTracerArray._new(x.dtype, shape=tuple(next_shape))
        else:
            next_shape = list(x.shape)
            del next_shape[axis]
            return TypeTracerArray._new(x.dtype, shape=tuple(next_shape))

    def max(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: TypeTracerArray | None = None,
    ) -> TypeTracerArray:
        return self.min(x, axis=axis, keepdims=keepdims, maybe_out=maybe_out)

    def array_str(
        self,
        x: TypeTracerArray,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        assert isinstance(x, TypeTracerArray)
        try_touch_data(x)
        return "[## ... ##]"

    def astype(
        self, x: TypeTracerArray, dtype: DTypeLike, *, copy: bool | None = True
    ) -> TypeTracerArray:
        assert isinstance(x, TypeTracerArray)
        x.touch_data()
        return TypeTracerArray._new(np.dtype(dtype), x.shape)

    def can_cast(
        self, from_: DTypeLike | TypeTracerArray, to: DTypeLike | TypeTracerArray
    ) -> bool:
        return numpy.can_cast(from_, to, casting="same_kind")

    @classmethod
    def is_own_array_type(cls, type_: type) -> bool:
        return issubclass(type_, TypeTracerArray)

    @classmethod
    def is_own_array(cls, obj) -> bool:
        return cls.is_own_array_type(type(obj))

    def is_c_contiguous(self, x: TypeTracerArray | PlaceholderArray) -> bool:
        assert isinstance(x, TypeTracerArray)
        return True

    def __dlpack_device__(self) -> tuple[int, int]:
        raise NotImplementedError

    def __dlpack__(self, stream=None):
        raise NotImplementedError


def _attach_report(
    layout: Content,
    form: Form,
    report: TypeTracerReport,
    getkey: Callable[[Form, str], str],
):
    if isinstance(layout, (ak.contents.BitMaskedArray, ak.contents.ByteMaskedArray)):
        assert isinstance(form, (ak.forms.BitMaskedForm, ak.forms.ByteMaskedForm))
        layout.mask.data.form_key = getkey(form, "mask")  # type: ignore[attr-defined]
        layout.mask.data.report = report  # type: ignore[attr-defined]
        _attach_report(layout.content, form.content, report, getkey)

    elif isinstance(layout, ak.contents.EmptyArray):
        assert isinstance(form, ak.forms.EmptyForm)

    elif isinstance(layout, (ak.contents.IndexedArray, ak.contents.IndexedOptionArray)):
        assert isinstance(form, (ak.forms.IndexedForm, ak.forms.IndexedOptionForm))
        layout.index.data.form_key = getkey(form, "index")  # type: ignore[attr-defined]
        layout.index.data.report = report  # type: ignore[attr-defined]
        _attach_report(layout.content, form.content, report, getkey)

    elif isinstance(layout, ak.contents.ListArray):
        assert isinstance(form, ak.forms.ListForm)
        layout.starts.data.form_key = getkey(form, "starts")  # type: ignore[attr-defined]
        layout.starts.data.report = report  # type: ignore[attr-defined]
        layout.stops.data.form_key = getkey(form, "stops")  # type: ignore[attr-defined]
        layout.stops.data.report = report  # type: ignore[attr-defined]
        _attach_report(layout.content, form.content, report, getkey)

    elif isinstance(layout, ak.contents.ListOffsetArray):
        assert isinstance(form, ak.forms.ListOffsetForm)
        layout.offsets.data.form_key = getkey(form, "offsets")  # type: ignore[attr-defined]
        layout.offsets.data.report = report  # type: ignore[attr-defined]
        _attach_report(layout.content, form.content, report, getkey)

    elif isinstance(layout, ak.contents.NumpyArray):
        assert isinstance(form, ak.forms.NumpyForm)
        layout.data.form_key = getkey(form, "data")  # type: ignore[attr-defined]
        layout.data.report = report  # type: ignore[attr-defined]

    elif isinstance(layout, ak.contents.RecordArray):
        assert isinstance(form, ak.forms.RecordForm)
        for x, y in zip(layout.contents, form.contents):
            _attach_report(x, y, report, getkey)

    elif isinstance(layout, (ak.contents.RegularArray, ak.contents.UnmaskedArray)):
        assert isinstance(form, (ak.forms.RegularForm, ak.forms.UnmaskedForm))
        _attach_report(layout.content, form.content, report, getkey)

    elif isinstance(layout, ak.contents.UnionArray):
        assert isinstance(form, ak.forms.UnionForm)
        layout.tags.data.form_key = getkey(form, "tags")  # type: ignore[attr-defined]
        layout.tags.data.report = report  # type: ignore[attr-defined]
        layout.index.data.form_key = getkey(form, "index")  # type: ignore[attr-defined]
        layout.index.data.report = report  # type: ignore[attr-defined]
        for x, y in zip(layout.contents, form.contents):
            _attach_report(x, y, report, getkey)

    else:
        raise AssertionError(f"unrecognized layout type {type(layout)}")


def typetracer_with_report(
    form: ak.forms.Form, getkey: Callable[[Form, str], str]
) -> tuple[ak.contents.Content, TypeTracerReport]:
    layout = form.length_zero_array().to_typetracer(forget_length=True)
    report = TypeTracerReport()
    _attach_report(layout, form, report, getkey)

    # Optimisation: identify buffer keys ahead of time, and register them with report
    def buffer_key(form_key, attribute, form):
        return getkey(form, attribute)

    report.set_labels(form.expected_from_buffers(buffer_key))

    return layout, report
