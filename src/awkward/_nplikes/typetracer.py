# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

import numpy

from awkward import _errors
from awkward._nplikes import Array, NumpyLike, metadata
from awkward._nplikes.numpylike import ErrorStateLiteral
from awkward._util import is_non_string_like_sequence
from awkward.typing import (
    ContextManager,
    Final,
    Literal,
    Self,
    SupportsIndex,
    SupportsInt,
    TypeVar,
    final,
    overload,
)


def unknown_scalar(dtype: metadata.dtype) -> TypeTracerArray:
    nplike: TypeTracer = TypeTracer.instance()  # type: ignore
    return TypeTracerArray._new(dtype, (), nplike=nplike)


def is_unknown_scalar(x) -> bool:
    return isinstance(x, TypeTracerArray) and x.ndim == 0


M = TypeVar("M")
K = TypeVar("K")


class TypeTracerArray(Array):
    _dtype: dtype
    _nplike: TypeTracer
    _shape: tuple[SupportsInt, ...]

    @property
    def dtype(self) -> metadata.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> tuple[SupportsInt, ...]:
        return self._shape

    @property
    def size(self) -> SupportsInt:
        size = 1
        for component in self._shape:
            if is_unknown_scalar(component):
                return unknown_scalar(metadata.int64)
            else:
                size *= component
        return size

    @property
    def T(self) -> TypeTracerArray:
        return self._new(self._dtype, shape=self._shape[::-1], nplike=self._nplike)

    @property
    def known_shape(self) -> bool:
        return not any(is_unknown_scalar(x) for x in self._shape)

    @classmethod
    def _new(
        cls,
        dtype: dtype,
        shape: tuple[SupportsInt, ...],
        nplike: TypeTracer,
    ) -> TypeTracerArray:
        assert is_non_string_like_sequence(shape)
        self = super().__new__(cls)
        self._dtype = dtype
        self._shape = shape
        self._nplike = nplike
        return self

    def __new__(cls, *args, **kwargs):
        raise _errors.wrap_error(
            TypeError(
                "internal_error: the `TypeTracer` nplike's `TypeTracerArray` object should never be directly instantiated"
            )
        )

    def __repr__(self):
        return f"TypeTracerArray({self._dtype!r})"

    def __str__(self):
        if self.ndim == 0:
            return f"{self._dtype!r}??"
        else:
            return f"TypeTracerArray({self._dtype!r})"

    @overload
    def __getitem__(
        self, index: SupportsIndex
    ) -> int | float | complex | str | bytes | bytes:
        ...

    @overload
    def __getitem__(
        self, index: slice | Ellipsis | tuple[SupportsIndex | slice | Ellipsis, ...]
    ) -> Self:
        ...

    def __getitem__(self, index):
        if isinstance(index, SupportsIndex):
            return self._new(self._dtype, (), nplike=self._nplike)
        else:
            raise _errors.wrap_error(NotImplementedError)

    def __bool__(self) -> bool:
        raise _errors.wrap_error(RuntimeError("cannot realise an unknown value"))

    def __int__(self) -> int:
        raise _errors.wrap_error(RuntimeError("cannot realise an unknown value"))

    def __index__(self) -> int:
        raise _errors.wrap_error(RuntimeError("cannot realise an unknown value"))

    def touch_data(self):
        raise _errors.wrap_error(NotImplementedError)

    def touch_shape(self):
        raise _errors.wrap_error(NotImplementedError)

    def __add__(
        self: Array, other: int | float | complex | TypeTracerArray
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __sub__(
        self: Array, other: int | float | complex | TypeTracerArray
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __truediv__(
        self: Array, other: int | float | complex | TypeTracerArray
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __floordiv__(
        self: Array, other: int | float | complex | TypeTracerArray
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __mod__(
        self: Array, other: int | float | complex | TypeTracerArray
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __mul__(
        self: Array, other: int | float | complex | TypeTracerArray
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __pow__(
        self: Array, other: int | float | complex | TypeTracerArray
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __xor__(self: Array, other: int | bool | TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __and__(self: Array, other: int | bool | TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __or__(self: Array, other: int | bool | TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __lt__(
        self: Array,
        other: int | float | complex | str | bytes | TypeTracerArray,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __le__(
        self: Array,
        other: int | float | complex | str | bytes | TypeTracerArray,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __gt__(
        self: Array,
        other: int | float | complex | str | bytes | TypeTracerArray,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __ge__(
        self: Array,
        other: int | float | complex | str | bytes | TypeTracerArray,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __eq__(  # type: ignore[override]
        self: Array,
        other: int | float | bool | complex | str | bytes | TypeTracerArray,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __ne__(  # type: ignore[override]
        self: Array,
        other: int | float | bool | complex | str | bytes | TypeTracerArray,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __abs__(self: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __neg__(self: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __pos__(self: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def __invert__(self: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)


T = TypeVar("T")


@final
class TypeTracer(NumpyLike):
    is_eager: Final = True
    known_data: Final = False
    known_shape: Final = False

    default_float_dtype = metadata.float64
    default_int_dtype = metadata.int64

    def asarray(
        self,
        obj,
        *,
        dtype: metadata.dtype | None = None,
        copy: bool | None = None,
    ) -> TypeTracerArray:
        metadata.ensure_valid_dtype(dtype)
        if isinstance(obj, TypeTracerArray):
            if dtype is None:
                return obj
            else:
                return self.astype(obj, dtype)
        elif hasattr(obj, "shape") and hasattr(obj, "dtype"):
            return TypeTracerArray._new(obj.dtype, obj.shape, nplike=self)
        elif is_non_string_like_sequence(obj):
            assert not any(is_non_string_like_sequence(x) for x in obj)
            shape = (len(obj),)
            result_type = self.result_type(*obj)
            return TypeTracerArray._new(result_type, shape, nplike=self)
        # elif is_python_scalar(obj):
        #     ...
        else:
            raise _errors.wrap_error(TypeError)

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: metadata.dtype | None = None,
    ) -> TypeTracerArray:
        metadata.ensure_valid_dtype(dtype)
        return TypeTracerArray._new(
            dtype or self.default_float_dtype, shape, nplike=self
        )

    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: metadata.dtype | None = None,
    ) -> TypeTracerArray:
        metadata.ensure_valid_dtype(dtype)
        return TypeTracerArray._new(
            dtype or self.default_float_dtype, shape, nplike=self
        )

    def empty(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: metadata.dtype | None = None,
    ) -> TypeTracerArray:
        metadata.ensure_valid_dtype(dtype)
        return TypeTracerArray._new(
            dtype or self.default_float_dtype, shape, nplike=self
        )

    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value,
        *,
        dtype: metadata.dtype | None = None,
    ) -> TypeTracerArray:
        dtype = dtype or metadata.default_dtype(fill_value)
        metadata.ensure_valid_dtype(dtype)
        return TypeTracerArray._new(dtype, shape, nplike=self)

    def zeros_like(
        self, x: Array, *, dtype: metadata.dtype | None = None
    ) -> TypeTracerArray:
        return self.asarray(x, dtype=dtype)

    def ones_like(
        self, x: Array, *, dtype: metadata.dtype | None = None
    ) -> TypeTracerArray:
        return self.asarray(x, dtype=dtype)

    def full_like(
        self, x: Array, fill_value, *, dtype: metadata.dtype | None = None
    ) -> TypeTracerArray:
        return self.asarray(x, dtype=dtype)

    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: metadata.dtype | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def meshgrid(
        self, *arrays: Array, indexing: Literal["xy", "ij"] = "xy"
    ) -> list[TypeTracerArray]:
        if not all(x.ndim == 1 for x in arrays):
            raise _errors.wrap_error(ValueError)

        lengths = [x.shape[0] for x in arrays]
        if indexing == "xy":
            lengths[0], lengths[1] = lengths[1], lengths[0]
        shape = tuple(lengths)
        return [TypeTracerArray._new(x.dtype, shape, nplike=self) for x in arrays]

    ############################ data type functions

    def astype(
        self, x: Array, dtype: metadata.dtype, *, copy: bool = True
    ) -> TypeTracerArray:
        metadata.ensure_valid_dtype(dtype)
        return TypeTracerArray._new(dtype, shape=x.shape, nplike=self)

    def broadcast_arrays(self, *arrays: Array) -> list[TypeTracerArray]:
        shape = self.broadcast_shapes(*[a.shape for a in arrays])
        return [TypeTracerArray._new(a.dtype, shape, nplike=self) for a in arrays]

    def broadcast_to(self, x: Array, shape: tuple[SupportsInt, ...]) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ searching functions

    def nonzero(self, x: Array) -> tuple[TypeTracerArray, ...]:
        shape = (unknown_scalar(metadata.int64),)
        return (TypeTracerArray(metadata.int64, shape, nplike=self),) * x.ndim

    def where(self, condition: Array, x1: Array, x2: Array) -> TypeTracerArray:
        shape = self.broadcast_shapes(condition.shape, x1.shape, x2.shape)
        dtype = self.result_type(x1, x2)
        return TypeTracerArray._new(dtype, shape, nplike=self)

    ############################ set functions

    def unique_counts(self, x: Array) -> tuple[TypeTracerArray, TypeTracerArray]:
        shape = (unknown_scalar(metadata.int64),)
        return (
            TypeTracerArray(x.dtype, shape, nplike=self),
            TypeTracerArray(metadata.int64, shape, nplike=self),
        )

    def unique_values(self, x: Array) -> TypeTracerArray:
        shape = (unknown_scalar(metadata.int64),)
        return TypeTracerArray(x.dtype, shape, nplike=self)

    ############################ manipulation functions

    def _add_maybe_unknown(self, x1, x2) -> TypeTracerArray | T:
        if is_unknown_scalar(x1) or is_unknown_scalar(x2):
            return unknown_scalar(self.result_type(x1, x2))
        else:
            return x1 + x2

    def _compute_flattened_size(self, *arrays):
        # Compute flat size
        size = 0
        for array in arrays:
            size = self._add_maybe_unknown(size, array.size)
        return size

    @staticmethod
    def _normalise_axis(array: Array, axis: int) -> int:
        if axis >= 0:
            return axis
        else:
            return array.ndim + axis

    def concat(
        self,
        arrays: list[TypeTracerArray] | tuple[TypeTracerArray, ...],
        *,
        axis: int | None = 0,
    ) -> TypeTracerArray:
        dtype = self.result_type(*arrays)
        if axis is None:
            size = self._compute_flattened_size(*arrays)
            return TypeTracerArray._new(dtype, (size,), nplike=self)
        else:
            array, *others = arrays

            # Check dimensions
            if not all(o.ndim == array.ndim for o in others):
                raise _errors.wrap_error(
                    ValueError("arrays must have equal dimensions")
                )

            shape = array.shape
            axis = self._normalise_axis(array, axis)

            for other in others:
                for ax, dim in other.shape:
                    if ax == axis:
                        shape[ax] = self._add_maybe_unknown(shape[ax], dim)
                        continue
                    elif is_unknown_scalar(shape[ax]) or is_unknown_scalar(dim):
                        shape[ax] = unknown_scalar(metadata.int64)
                        continue
                    elif shape[ax] != dim:
                        raise _errors.wrap_error(
                            ValueError("arrays must have compatible shapes")
                        )

            return TypeTracerArray._new(dtype, tuple(shape), nplike=self)

    def stack(
        self,
        arrays: list[TypeTracerArray] | tuple[TypeTracerArray, ...],
        *,
        axis: int = 0,
    ) -> TypeTracerArray:
        dtype = self.result_type(*arrays)
        if axis is None:
            size = self._compute_flattened_size(*arrays)
            return TypeTracerArray._new(dtype, (size,), nplike=self)
        else:
            array, *others = arrays

            # Check dimensions
            if not all(o.ndim == array.ndim for o in others):
                raise _errors.wrap_error(
                    ValueError("arrays must have equal dimensions")
                )

            shape = array.shape
            axis = self._normalise_axis(array, axis)

            for other in others:
                for ax, dim in other.shape:
                    if is_unknown_scalar(shape[ax]) or is_unknown_scalar(dim):
                        shape[ax] = unknown_scalar(metadata.int64)
                        continue
                    elif shape[ax] != dim:
                        raise _errors.wrap_error(
                            ValueError("arrays must have compatible shapes")
                        )
            shape.insert(axis, len(arrays))
            return TypeTracerArray._new(dtype, tuple(shape), nplike=self)

    ############################ ufuncs

    def add(self, x1: Array, x2: Array) -> TypeTracerArray:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        dtype = self.result_type(x1, x2)
        return TypeTracerArray._new(dtype, shape=shape, nplike=self)

    def subtract(self, x1: Array, x2: Array) -> TypeTracerArray:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        dtype = self.result_type(x1, x2)
        return TypeTracerArray._new(dtype, shape=shape, nplike=self)

    def multiply(self, x1: Array, x2: Array) -> TypeTracerArray:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        dtype = self.result_type(x1, x2)
        return TypeTracerArray._new(
            numpy.empty((1,), dtype=dtype), shape=shape, nplike=self
        )

    def logical_or(self, x1: Array, x2: Array) -> TypeTracerArray:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        return TypeTracerArray._new(metadata.bool_, shape=shape, nplike=self)

    def logical_and(self, x1: Array, x2: Array) -> TypeTracerArray:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        return TypeTracerArray._new(metadata.bool_, shape=shape, nplike=self)

    def logical_not(self, x: Array) -> TypeTracerArray:
        return TypeTracerArray._new(metadata.bool_, shape=x.shape, nplike=self)

    def sqrt(self, x: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def exp(self, x: Array) -> TypeTracerArray:
        dtype = self.result_type(metadata.float64, x.dtype)
        return TypeTracerArray._new(dtype, shape=x.shape, nplike=self)

    def negative(self, x: Array) -> TypeTracerArray:
        return TypeTracerArray._new(x.dtype, shape=x.shape, nplike=self)

    def divide(self, x1: Array, x2: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def floor_divide(self, x1: Array, x2: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def power(self, x1: Array, x2: Array) -> TypeTracerArray:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        dtype = self.result_type(x1, x2)
        return TypeTracerArray._new(dtype, shape=shape, nplike=self)

    def greater(self, x1: Array, x2: Array) -> Array:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        return TypeTracerArray._new(metadata.bool_, shape=shape, nplike=self)

    def greater_equal(self, x1: Array, x2: Array) -> Array:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        return TypeTracerArray._new(metadata.bool_, shape=shape, nplike=self)

    def lesser(self, x1: Array, x2: Array) -> Array:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        return TypeTracerArray._new(metadata.bool_, shape=shape, nplike=self)

    def lesser_equal(self, x1: Array, x2: Array) -> Array:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        return TypeTracerArray._new(metadata.bool_, shape=shape, nplike=self)

    def equal(self, x1: Array, x2: Array) -> Array:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        return TypeTracerArray._new(metadata.bool_, shape=shape, nplike=self)

    def isnan(self, x: Array) -> TypeTracerArray:
        return TypeTracerArray._new(metadata.bool_, shape=x.shape, nplike=self)

    ############################ Utility functions

    def all(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def any(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ Statistical functions

    def sum(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: metadata.dtype | None = None,
        keepdims: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def prod(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: metadata.dtype | None = None,
        keepdims: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def min(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def max(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ Searching functions

    def argmin(
        self, x: Array, *, axis: int | None = None, keepdims: bool = False
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def argmax(
        self, x: Array, *, axis: int | None = None, keepdims: bool = False
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ extensions to Array API

    def as_contiguous(self, x: Array) -> TypeTracerArray:
        return x

    def cumsum(
        self, x: Array, *, axis: int | tuple[int, ...] | None = None
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def from_buffer(self, buffer, *, dtype=None, count: int = -1) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def array_equal(
        self, x1: Array, x2: Array, *, equal_nan: bool = False
    ) -> TypeTracerArray:
        return unknown_scalar(metadata.bool_)

    def search_sorted(
        self,
        x: Array,
        values: Array,
        *,
        side: Literal["left", "right"] = "left",
        sorter=None,
    ) -> TypeTracerArray:
        return TypeTracerArray._new(metadata.int64, shape=values.shape, nplike=self)

    def repeat(
        self,
        x: Array,
        repeats: Array | int,
        *,
        axis: int | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def tile(self, x: Array, reps: int) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def pack_bits(
        self,
        x: Array,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def unpack_bits(
        self,
        x: Array,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def minimum(self, x1: Array, x2: Array) -> TypeTracerArray:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        return TypeTracerArray._new(metadata.bool_, shape=shape, nplike=self)

    def maximum(self, x1: Array, x2: Array) -> TypeTracerArray:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        return TypeTracerArray._new(metadata.bool_, shape=shape, nplike=self)

    def nan_to_num(
        self,
        x: Array,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> TypeTracerArray:
        return TypeTracerArray._new(x.dtype, shape=x.shape, nplike=self)

    def is_close(
        self,
        x1: Array,
        x2: Array,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> TypeTracerArray:
        shape = self.broadcast_shapes(x1.shape, x2.shape)
        return TypeTracerArray._new(metadata.bool_, shape=shape, nplike=self)

    def count_nonzero(
        self, x: Array, *, axis: int | None = None, keepdims: bool = False
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def array_str(
        self,
        x: Array,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        raise _errors.wrap_error(NotImplementedError)

    def is_c_contiguous(self, x: Array) -> bool:
        # TODO: should this return unknown, or should we store this
        # information?
        raise _errors.wrap_error(NotImplementedError)

    def to_rectilinear(self, x: Array):
        raise _errors.wrap_error(NotImplementedError)

    def byteswap(self, x: Array, copy: bool = False):
        return x

    def error_state(self, **kwargs: ErrorStateLiteral) -> ContextManager:
        raise _errors.wrap_error(NotImplementedError)

    @classmethod
    def is_own_array(cls, x) -> bool:
        return isinstance(x, TypeTracerArray)

    @classmethod
    def shapes_are_compatible(
        cls,
        s1: tuple[SupportsInt, ...],
        s2: tuple[SupportsInt, ...],
        *,
        assume_unknown_compatible=True,
    ) -> bool:
        """
        Args:
            s1: first shape
            s2: second shape
            assume_unknown_compatible: whether to consider unknown values as compatible

        Return
        """
        # TODO: this should be named "shapes_are_equal"
        if len(s1) != len(s2):
            return False

        result_is_known = True
        for this, that in zip(s1, s2):
            if is_unknown_scalar(this) or is_unknown_scalar(that):
                result_is_known = False
            elif this != that:
                return False

        if result_is_known or assume_unknown_compatible:
            return True
        else:
            return False

    @classmethod
    def broadcast_shapes(
        cls, *shapes: tuple[SupportsInt, ...]
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
                # Existing is broadcastable, take it
                elif result[i] == 1:
                    result[i] = item
                else:
                    raise _errors.wrap_error(
                        ValueError(
                            "known component of shape does not match broadcast result"
                        )
                    )
        return tuple(result)
