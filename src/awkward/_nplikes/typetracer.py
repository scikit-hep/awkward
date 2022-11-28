# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

import operator
from functools import reduce
from typing import Callable, Iterable, Literal, Sized, TypeVar, Union, cast

import numpy

from awkward import _errors
from awkward._nplikes import dtypes, numpylike

ShapeItem = Union[int, "TypeTracerArray"]


Shape = numpylike.Shape[ShapeItem]


class TypeTracerShape(numpylike.Shape):
    """
    A `shape` interface for `TypeTracerArray`. Unlike Numpy arrays, `TypeTracerArray.shape`
    can contain unknown values. Therefore, evaluating `shape == (...,)` is not safe if relied
    upon within Awkward internals. Therefore, we want to ensure that shape comparisons fail
    if a concrete value is required.
    """

    _items: tuple[ShapeItem, ...]

    def __init__(self, items: Iterable[ShapeItem]):
        self._items = tuple(items)

    def __add__(self, other) -> TypeTracerShape:
        if isinstance(other, tuple):
            return self.__class__(self._items + other)
        elif isinstance(other, TypeTracerShape):
            return self.__class__(self._items + other._items)
        else:
            return NotImplemented

    def __radd__(self, other) -> TypeTracerShape:
        if isinstance(other, tuple):
            return self.__class__(other + self._items)
        else:
            return NotImplemented

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(self._items[index])
        else:
            return self._items[operator.index(index)]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __eq__(self, other) -> bool | TypeTracerArray:
        if isinstance(other, Iterable) and isinstance(other, Sized):
            return TypeTracer.shapes_are_compatible(self, other)
        else:
            return NotImplemented

    def __str__(self):
        return str(self._items)


def unknown_scalar(dtype: dtypes.dtype) -> TypeTracerArray:
    nplike = TypeTracer.instance()
    return TypeTracerArray._new_as_scalar(numpy.zeros(1, dtype=dtype), nplike=nplike)


def is_unknown_scalar(x) -> bool:
    if isinstance(x, TypeTracerArray):
        return x.ndim == 0
    else:
        return False


K = TypeVar("K")


class TypeTracerArray:
    _array: numpy.ndarray
    _nplike: TypeTracer
    _shape: TypeTracerShape

    @property
    def dtype(self) -> dtypes.dtype:
        return self._array.dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> TypeTracerShape:
        return self._shape

    @property
    def size(self) -> ShapeItem:
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
        shape: Shape,
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
        self._shape = TypeTracerShape(shape)
        self._nplike = nplike
        return self

    @classmethod
    def _promote_scalar(
        cls,
        self: TypeTracerArray,
        x: bool | int | float | complex | TypeTracerArray,
    ) -> TypeTracerArray:
        if isinstance(x, float):
            array = numpy.array([0], dtype=dtypes.float64)
        elif isinstance(x, bool):
            array = numpy.array([0], dtype=dtypes.bool_)
        elif isinstance(x, int):
            array = numpy.array([0], dtype=dtypes.int64)
        elif isinstance(x, complex):
            array = numpy.array([0], dtype=dtypes.complex128)
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
        self: TypeTracerArray, other, op: Callable[[K, K], K]
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
        self: TypeTracerArray,
        other: TypeTracerArray,
        op: Callable[[K, K], K],
    ) -> TypeTracerArray:
        shape = self._nplike.broadcast_shapes(self.shape, other.shape)
        return cls._new(op(self._array, other._array), shape=shape, nplike=self._nplike)

    @classmethod
    def _new_from_unary_op(
        cls, self: TypeTracerArray, op: Callable[[K], K]
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

    def __add__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__add__))

    def __sub__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__sub__))

    def __truediv__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(
            TypeTracerArray, self._invoke_binary_op(other, operator.__truediv__)
        )

    def __floordiv__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(
            TypeTracerArray, self._invoke_binary_op(other, operator.__floordiv__)
        )

    def __mod__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__mod__))

    def __mul__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__mul__))

    def __pow__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__pow__))

    def __xor__(
        self: TypeTracerArray, other: int | bool | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__xor__))

    def __and__(
        self: TypeTracerArray, other: int | bool | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__and__))

    def __or__(
        self: TypeTracerArray, other: int | bool | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__or__))

    def __lt__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__lt__))

    def __le__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__le__))

    def __gt__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__gt__))

    def __ge__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__ge__))

    def __eq__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__eq__))

    def __ne__(
        self: TypeTracerArray, other: int | float | TypeTracerArray
    ) -> TypeTracerArray:
        return cast(TypeTracerArray, self._invoke_binary_op(other, operator.__ne__))

    def __abs__(self: TypeTracerArray) -> TypeTracerArray:
        return cast(TypeTracerArray, self._new_from_unary_op(self, operator.__abs__))

    def __neg__(self: TypeTracerArray) -> TypeTracerArray:
        return cast(TypeTracerArray, self._new_from_unary_op(self, operator.__neg__))

    def __pos__(self: TypeTracerArray) -> TypeTracerArray:
        return cast(TypeTracerArray, self._new_from_unary_op(self, operator.__pos__))

    def __invert__(self: TypeTracerArray) -> TypeTracerArray:
        return cast(TypeTracerArray, self._new_from_unary_op(self, operator.__invert__))

    def __bool__(self) -> bool:
        raise _errors.wrap_error(RuntimeError("cannot realise an unknown value"))

    def __int__(self) -> int:
        raise _errors.wrap_error(RuntimeError("cannot realise an unknown value"))

    def __index__(self) -> int:
        raise _errors.wrap_error(RuntimeError("cannot realise an unknown value"))


class TypeTracer(numpylike.NumpyLike):
    is_eager = True
    known_data = False
    known_shape = False

    def asarray(
        self,
        obj,
        *,
        dtype: dtypes.dtype | None = None,
        copy: bool | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtypes.dtype | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtypes.dtype | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def empty(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtypes.dtype | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value,
        *,
        dtype: dtypes.dtype | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def zeros_like(
        self, x: TypeTracerArray, *, dtype: dtypes.dtype | None = None
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def ones_like(
        self, x: TypeTracerArray, *, dtype: dtypes.dtype | None = None
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def full_like(
        self, x: TypeTracerArray, fill_value, *, dtype: dtypes.dtype | None = None
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: dtypes.dtype | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def meshgrid(
        self, *arrays: TypeTracerArray, indexing: Literal["xy", "ij"] = "xy"
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ data type functions

    def astype(
        self, x: TypeTracerArray, dtype: dtypes.dtype, *, copy: bool = True
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def broadcast_arrays(self, *arrays: TypeTracerArray) -> list[TypeTracerArray]:
        raise _errors.wrap_error(NotImplementedError)

    def broadcast_to(self, x: TypeTracerArray, shape: Shape) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def result_type(
        self, *arrays_and_dtypes: TypeTracerArray | dtypes.dtype
    ) -> dtypes.dtype:
        all_dtypes: list[dtypes.dtype] = []
        for item in arrays_and_dtypes:
            if hasattr(item, "shape") and hasattr(item, "dtype"):
                item = item.dtype
            if isinstance(item, dtypes.dtype):
                all_dtypes.append(item)
            else:
                raise TypeError(
                    "result_type() inputs must be array_api arrays or dtypes"
                )

        return numpy.result_type(*all_dtypes)

    ############################ searching functions

    def nonzero(self, x: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def where(
        self, condition: TypeTracerArray, x1: TypeTracerArray, x2: TypeTracerArray
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ set functions

    def unique_counts(self, x: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def unique_values(self, x: TypeTracerArray) -> TypeTracerArray:
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

    def add(self, x1: TypeTracerArray, x2: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def multiply(self, x1: TypeTracerArray, x2: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def logical_or(self, x1: TypeTracerArray, x2: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def logical_and(self, x1: TypeTracerArray, x2: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def logical_not(self, x: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def sqrt(self, x: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def exp(self, x: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def divide(self, x1: TypeTracerArray, x2: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def equal(self, x1: TypeTracerArray, x2: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def isnan(self, x: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ Utility functions

    def all(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def any(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ Statistical functions

    def sum(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: dtypes.dtype | None = None,
        keepdims: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def prod(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: dtypes.dtype | None = None,
        keepdims: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def min(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def max(
        self,
        x: TypeTracerArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ Searching functions

    def argmin(
        self, x: TypeTracerArray, *, axis: int | None = None, keepdims: bool = False
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def argmax(
        self, x: TypeTracerArray, *, axis: int | None = None, keepdims: bool = False
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    ############################ extensions to Array API

    def as_contiguous(self, x: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def cumsum(self, x: TypeTracerArray, *, axis: int | None = None) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def from_buffer(self, buffer, *, dtype=None) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def array_equal(
        self, x1: TypeTracerArray, x2: TypeTracerArray, *, equal_nan: bool = False
    ) -> bool:
        raise _errors.wrap_error(NotImplementedError)

    def search_sorted(
        self,
        x: TypeTracerArray,
        values: TypeTracerArray,
        *,
        side: Literal["left", "right"] = "left",
        sorter=None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def repeat(
        self,
        x: TypeTracerArray,
        repeats: TypeTracerArray | int,
        *,
        axis: int | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def tile(self, x: TypeTracerArray, reps: int) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def pack_bits(
        self,
        x: TypeTracerArray,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big -> Array",
    ):
        raise _errors.wrap_error(NotImplementedError)

    def unpack_bits(
        self,
        x: TypeTracerArray,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def minimum(self, x1: TypeTracerArray, x2: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def maximum(self, x1: TypeTracerArray, x2: TypeTracerArray) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def nan_to_num(
        self,
        x: TypeTracerArray,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def is_close(
        self,
        x1: TypeTracerArray,
        x2: TypeTracerArray,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def count_nonzero(
        self, x: TypeTracerArray, *, axis: int | None = None, keepdims: bool = False
    ) -> TypeTracerArray:
        raise _errors.wrap_error(NotImplementedError)

    def array_str(
        self,
        x: TypeTracerArray,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        raise _errors.wrap_error(NotImplementedError)

    def is_c_contiguous(self, x: TypeTracerArray) -> bool:
        raise _errors.wrap_error(NotImplementedError)

    def to_rectilinear(self, array):
        raise _errors.wrap_error(NotImplementedError)

    @classmethod
    def is_own_array(cls, x) -> bool:
        raise _errors.wrap_error(NotImplementedError)

    @classmethod
    def shapes_are_compatible(
        cls, s1: Shape, s2: Shape, *, assume_unknown_compatible=True
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
            return unknown_scalar(dtypes.bool_)

    @classmethod
    def broadcast_shapes(cls, *shapes: Shape) -> TypeTracerShape:
        ndim = max([len(s) for s in shapes], default=0)
        result = [1] * ndim

        for shape in shapes:
            # Right broadcasting
            missing_dim = ndim - len(shape)
            if missing_dim > 0:
                shape = (1,) * missing_dim + shape

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
        return TypeTracerShape(result)
