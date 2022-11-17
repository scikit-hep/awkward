# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

import enum
import operator
from functools import reduce
from typing import Any, Callable, TypeVar, Union, cast

import numpy

from awkward import _errors
from awkward._nplikes import dtypes, numpylike

ShapeItem = Union[int, "TypeTracerArray"]
Shape = tuple[ShapeItem, ...]


def is_unknown_scalar(x) -> bool:
    return isinstance(x, TypeTracerArray) and x.shape == ()


def shapes_are_compatible(self: Shape, other: Shape) -> bool:
    if len(self) != len(other):
        return False

    for this, that in zip(self, other):
        components_are_equal = this == that
        if not (is_unknown_scalar(components_are_equal) or components_are_equal):
            return False
    return True


def broadcast_shapes(*shapes: Shape):
    ndim = max([len(s) for s in shapes], default=0)
    result = [1] * ndim

    for shape in shapes:
        # Right broadcasting
        missing_dim = ndim - len(shape)
        if missing_dim > 0:
            shape = (1,) * missing_dim + shape

        # Fail if we absolutely know the shapes aren't compatible
        for i, item in enumerate(shape):
            if is_unknown_scalar(item):
                result[i] = item
            elif result[i] != 1 != item:
                raise _errors.wrap_error(
                    ValueError(
                        "known component of shape does not match broadcast result"
                    )
                )
    return tuple(result)


K = TypeVar("K")


ScalarTraitInitialiser = Callable[[bool | int | float | complex], tuple[str]]


class TypeTracerArray:
    _array: numpy.ndarray
    _nplike: TypeTracer
    _shape: Shape
    _traits: set[str]

    _operator_drop_traits: dict[Any, set[str]] = {}
    _scalar_trait_initialisers: list[ScalarTraitInitialiser] = []

    @classmethod
    def drop_trait_for(cls, trait: str, *ops):
        for op in ops:
            op_traits = cls._operator_drop_traits.setdefault(op, set())
            op_traits.add(trait)

    @classmethod
    def initialises_scalar_trait(
        cls, func: ScalarTraitInitialiser
    ) -> ScalarTraitInitialiser:
        cls._scalar_trait_initialisers.append(func)
        return func

    @property
    def dtype(self) -> dtypes.dtype:
        return self._array.dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> Shape:
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

    def has_trait(self, trait: str) -> bool:
        return trait in self._traits

    @classmethod
    def _new(
        cls,
        x: numpy.generic | numpy.ndarray,
        shape: Shape,
        nplike: TypeTracer,
        *,
        traits: frozenset[str] = frozenset(),
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
        self._traits = traits
        return self

    @classmethod
    def _initialise_traits_from_scalar(
        cls, value: bool | int | float | complex
    ) -> frozenset[str]:
        traits = frozenset()
        for func in cls._scalar_trait_initialisers:
            traits |= func(value)
        return traits

    @classmethod
    def _promote_scalar(
        cls: type[TypeTracerArray],
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
        traits = self._initialise_traits_from_scalar(x)
        return cls._new_as_scalar(array, self._nplike, traits=traits)

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
        cls: type[TypeTracerArray],
        array,
        nplike: TypeTracer,
        *,
        traits: frozenset[str] = frozenset(),
    ) -> TypeTracerArray:
        return cls._new(array, (), nplike=nplike, traits=traits)

    @classmethod
    def _new_from_binary_op(
        cls: type[TypeTracerArray],
        self: TypeTracerArray,
        other: TypeTracerArray,
        op: Callable[[K, K], K],
    ) -> TypeTracerArray:
        new_traits = (self._traits & other._traits) - self._operator_drop_traits.get(
            op, set()
        )
        array = op(self._array, other._array)
        shape = broadcast_shapes(self.shape, other.shape)
        return cls._new(array, shape=shape, nplike=self._nplike, traits=new_traits)

    @classmethod
    def _new_from_unary_op(
        cls: type[TypeTracerArray], self: TypeTracerArray, op: Callable[[K], K]
    ) -> TypeTracerArray:
        new_traits = self._traits - self._operator_drop_traits.get(op, set())
        array = op(self._array)
        return cls._new(
            array, shape=self._shape, nplike=self._nplike, traits=new_traits
        )

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


class TypeTracerTraits(str, enum.Enum):
    POSITIVE = "POSITIVE"


# Don't preserve positivity between positive arrays under these operations
TypeTracerArray.drop_trait_for(
    TypeTracerTraits.POSITIVE,
    operator.sub,
    operator.inv,
    operator.neg,
    operator.ge,
    operator.le,
    operator.gt,
    operator.lt,
    operator.eq,
    operator.ne,
)


@TypeTracerArray.initialises_scalar_trait
def _initialise_scalar_trait(value):
    if isinstance(value, (int, float)) and value >= 0:
        return TypeTracerTraits.POSITIVE


class TypeTracer(numpylike.NumpyLike):
    def _broadcast_shapes(self):
        ...
