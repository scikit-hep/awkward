# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

import operator
from functools import reduce
from typing import Any, Callable, Literal, SupportsIndex, SupportsInt, TypeVar, overload

import numpy

from awkward import _errors
from awkward._nplikes import Array, NumpyLike, metadata
from awkward.typing import Self


def unknown_scalar(dtype: metadata.dtype) -> TypeTracerArray:
    nplike: TypeTracer = TypeTracer.instance()  # type: ignore
    return TypeTracerArray._new_as_scalar(numpy.zeros(1, dtype=dtype), nplike=nplike)


def is_unknown_scalar(x) -> bool:
    if isinstance(x, TypeTracerArray):
        return x.ndim == 0
    else:
        return False


M = TypeVar("M")
K = TypeVar("K")


class TypeTracerArray(Array):
    _array: numpy.ndarray
    _nplike: TypeTracer
    _shape: tuple[SupportsInt, ...]

    @property
    def dtype(self) -> metadata.dtype:
        return self._array.dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> tuple[SupportsInt, ...]:
        return self._shape

    @property
    def size(self) -> SupportsInt:
        if not self._shape:
            return 0
        else:
            return reduce(operator.mul, self._shape)

    @property
    def T(self) -> TypeTracerArray:
        return self._new(self._array, shape=self._shape[::-1], nplike=self._nplike)

    @classmethod
    def _new(
        cls,
        x: numpy.generic | numpy.ndarray,
        shape: tuple[SupportsInt, ...],
        nplike: TypeTracer,
    ) -> TypeTracerArray:
        self = super().__new__(cls)
        if isinstance(x, numpy.generic):
            x = numpy.asarray(x)

        if not isinstance(x, numpy.ndarray):
            raise _errors.wrap_error(
                TypeError(
                    "internal_error: the `Numpy` nplike's `TypeTracerArray` object must be created "
                    "from scalars or arrays",
                    type(x),
                )
            )

        if not x.shape == (1,):
            raise _errors.wrap_error(
                TypeError(
                    "internal_error: the `Numpy` nplike's `TypeTracerArray` object must be created "
                    "from shape (1,) arrays",
                    type(x),
                )
            )
        self._array = x
        self._shape = shape
        self._nplike = nplike
        return self

    @classmethod
    def _promote_scalar(
        cls,
        self: Array,
        x: bool | int | float | complex | TypeTracerArray,
    ) -> TypeTracerArray:
        if isinstance(x, float):
            array = numpy.array([0], dtype=metadata.float64)
        elif isinstance(x, bool):
            array = numpy.array([0], dtype=metadata.bool_)
        elif isinstance(x, int):
            array = numpy.array([0], dtype=metadata.int64)
        elif isinstance(x, complex):
            array = numpy.array([0], dtype=metadata.complex128)
        elif not isinstance(x, cls):
            raise _errors.wrap_error(
                TypeError(f"Expected bool, int, float, or {cls.__name__}")
            )
        else:
            return x
        return cls._new_as_scalar(array, self._nplike)

    def _handles_operand(self, other) -> bool:
        return isinstance(
            other, (int, float, complex, bool, TypeTracerArray)
        )  # TODO class for scalar primitives?

    def _invoke_binary_op(
        self: Array, other, op: Callable[[Any, Any], numpy.ndarray]
    ) -> TypeTracerArray:
        other = self._promote_scalar(other, self)
        if not self._handles_operand(other):
            return NotImplemented
        else:
            return self._new_from_binary_op(self, other, op)

    @classmethod
    def _new_as_scalar(
        cls,
        array,
        nplike: TypeTracer,
    ) -> TypeTracerArray:
        return cls._new(array, (), nplike=nplike)

    @classmethod
    def _new_from_binary_op(
        cls,
        self: Array,
        other: Array,
        op: Callable[[Any, Any], numpy.ndarray],
    ) -> TypeTracerArray:
        shape = self._nplike.broadcast_shapes(self.shape, other.shape)
        return cls._new(op(self._array, other._array), shape=shape, nplike=self._nplike)

    @classmethod
    def _new_from_unary_op(
        cls, self: Array, op: Callable[[Any], numpy.ndarray]
    ) -> TypeTracerArray:
        return cls._new(op(self._array), shape=self._shape, nplike=self._nplike)

    def __new__(cls, *args, **kwargs):
        raise _errors.wrap_error(
            TypeError(
                "internal_error: the `TypeTracer` nplike's `TypeTracerArray` object should never be directly instantiated"
            )
        )

    def __repr__(self):
        return f"TypeTracerArray({self._array.dtype!r})"

    def __str__(self):
        if self.ndim == 0:
            return f"{self._array.dtype!r}??"
        else:
            return f"TypeTracerArray({self._array.dtype!r})"

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
            return self._new_as_scalar(self._array, nplike=self._nplike)
        else:
            raise _errors.wrap_error(NotImplementedError)

    #
    # def __add__(
    #     self: Array, other: int | float | complex | TypeTracerArray
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__add__))
    #
    # def __sub__(
    #     self: Array, other: int | float | complex | TypeTracerArray
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__sub__))
    #
    # def __truediv__(
    #     self: Array, other: int | float | complex | TypeTracerArray
    # ) -> TypeTracerArray:
    #     return cast(
    #         TypeTracerArray, self._invoke_binary_op(other, operator.__truediv__)
    #     )
    #
    # def __floordiv__(
    #     self: Array, other: int | float | complex | TypeTracerArray
    # ) -> TypeTracerArray:
    #     return cast(
    #         TypeTracerArray, self._invoke_binary_op(other, operator.__floordiv__)
    #     )
    #
    # def __mod__(
    #     self: Array, other: int | float | complex | TypeTracerArray
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__mod__))
    #
    # def __mul__(
    #     self: Array, other: int | float | complex | TypeTracerArray
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__mul__))
    #
    # def __pow__(
    #     self: Array, other: int | float | complex | TypeTracerArray
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__pow__))
    #
    # def __xor__(
    #     self: Array, other: int | bool | TypeTracerArray
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__xor__))
    #
    # def __and__(
    #     self: Array, other: int | bool | TypeTracerArray
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__and__))
    #
    # def __or__(
    #     self: Array, other: int | bool | TypeTracerArray
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__or__))
    #
    # def __lt__(
    #     self: Array,
    #     other: int | float | complex | str | bytes | TypeTracerArray,
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__lt__))
    #
    # def __le__(
    #     self: Array,
    #     other: int | float | complex | str | bytes | TypeTracerArray,
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__le__))
    #
    # def __gt__(
    #     self: Array,
    #     other: int | float | complex | str | bytes | TypeTracerArray,
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__gt__))
    #
    # def __ge__(
    #     self: Array,
    #     other: int | float | complex | str | bytes | TypeTracerArray,
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__ge__))
    #
    # def __eq__(  # type: ignore[override]
    #     self: Array,
    #     other: int | float | bool | complex | str | bytes | TypeTracerArray,
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__eq__))
    #
    # def __ne__(  # type: ignore[override]
    #     self: Array,
    #     other: int | float | bool | complex | str | bytes | TypeTracerArray,
    # ) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__ne__))
    #
    # def __abs__(self: Array) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._new_from_unary_op(self, operator.__abs__))
    #
    # def __neg__(self: Array) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._new_from_unary_op(self, operator.__neg__))
    #
    # def __pos__(self: Array) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._new_from_unary_op(self, operator.__pos__))
    #
    # def __invert__(self: Array) -> TypeTracerArray:
    #     return cast(TypeTracerArray, self._new_from_unary_op(self, operator.__invert__))

    def __bool__(self) -> bool:
        raise _errors.wrap_error(RuntimeError("cannot realise an unknown value"))

    def __int__(self) -> int:
        raise _errors.wrap_error(RuntimeError("cannot realise an unknown value"))

    def __index__(self) -> int:
        raise _errors.wrap_error(RuntimeError("cannot realise an unknown value"))


class TypeTracer(NumpyLike):
    is_eager = True
    known_data = False
    known_shape = False

    def asarray(
        self,
        obj,
        *,
        dtype: metadata.dtype | None = None,
        copy: bool | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: metadata.dtype | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: metadata.dtype | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def empty(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: metadata.dtype | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value,
        *,
        dtype: metadata.dtype | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def zeros_like(
        self, x: Array, *, dtype: metadata.dtype | None = None
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def ones_like(
        self, x: Array, *, dtype: metadata.dtype | None = None
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def full_like(
        self, x: Array, fill_value, *, dtype: metadata.dtype | None = None
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

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
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ data type functions

    def astype(
        self, x: Array, dtype: metadata.dtype, *, copy: bool = True
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def broadcast_arrays(self, *arrays: Array) -> list[TypeTracerArray]:
        raise _errors.wrap_error(NotImplementedError)

    def broadcast_to(self, x: Array, shape: tuple[SupportsInt, ...]) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ searching functions

    def nonzero(self, x: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def where(self, condition: Array, x1: Array, x2: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ set functions

    def unique_counts(self, x: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def unique_values(self, x: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ manipulation functions

    def concat(
        self,
        arrays: list[TypeTracerArray] | tuple[TypeTracerArray, ...],
        *,
        axis: int | None = 0,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def stack(
        self,
        arrays: list[TypeTracerArray] | tuple[TypeTracerArray, ...],
        *,
        axis: int = 0,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ ufuncs

    def add(self, x1: Array, x2: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def subtract(self, x1: Array, x2: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def multiply(self, x1: Array, x2: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def logical_or(self, x1: Array, x2: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def logical_and(self, x1: Array, x2: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def logical_not(self, x: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def sqrt(self, x: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def exp(self, x: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def divide(self, x1: Array, x2: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def greater(self, x1: Array, x2: Array) -> Array:
        raise _errors.wrap_error(NotImplementedError)

    def greater_equal(self, x1: Array, x2: Array) -> Array:
        raise _errors.wrap_error(NotImplementedError)

    def lesser(self, x1: Array, x2: Array) -> Array:
        raise _errors.wrap_error(NotImplementedError)

    def lesser_equal(self, x1: Array, x2: Array) -> Array:
        raise _errors.wrap_error(NotImplementedError)

    def equal(self, x1: Array, x2: Array) -> Array:
        raise _errors.wrap_error(NotImplementedError)

    def isnan(self, x: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

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
        raise _errors.wrap_error(NotImplementedError)

    def cumsum(
        self, x: Array, *, axis: int | tuple[int, ...] | None = None
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def from_buffer(self, buffer, *, dtype=None, count: int = -1) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def array_equal(self, x1: Array, x2: Array, *, equal_nan: bool = False) -> bool:
        raise _errors.wrap_error(NotImplementedError)

    def search_sorted(
        self,
        x: Array,
        values: Array,
        *,
        side: Literal["left", "right"] = "left",
        sorter=None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

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
        raise _errors.wrap_error(NotImplementedError)

    def maximum(self, x1: Array, x2: Array) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def nan_to_num(
        self,
        x: Array,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def is_close(
        self,
        x1: Array,
        x2: Array,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

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
        raise _errors.wrap_error(NotImplementedError)

    def to_rectilinear(self, x: Array):
        raise _errors.wrap_error(NotImplementedError)

    def byteswap(self, x: Array, copy: bool = False):
        return x

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
    ) -> bool | TypeTracerArray:
        """
        Args:
            s1: first shape
            s2: second shape
            assume_unknown_compatible: whether to consider unknown values as compatible

        Return
        """
        if len(s1) != len(s2):
            return False

        result_is_known = True
        for this, that in zip(s1, s2):
            components_are_equal = this == that
            if is_unknown_scalar(components_are_equal):
                result_is_known = False
            elif not components_are_equal:
                return False

        if result_is_known or assume_unknown_compatible:
            return True
        else:
            return unknown_scalar(metadata.bool_)

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
