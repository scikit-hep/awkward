# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from abc import abstractmethod

from awkward._nplikes.shape import ShapeItem
from awkward._typing import (
    TYPE_CHECKING,
    Any,
    DType,
    EllipsisType,
    Protocol,
    Self,
    SupportsIndex,
    overload,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


class ArrayLike(Protocol):
    @property
    @abstractmethod
    def dtype(self) -> DType: ...

    @property
    @abstractmethod
    def ndim(self) -> int: ...

    @property
    @abstractmethod
    def shape(self) -> tuple[ShapeItem, ...]: ...

    @property
    @abstractmethod
    def strides(self) -> tuple[ShapeItem, ...]: ...

    @property
    @abstractmethod
    def size(self) -> ShapeItem: ...

    @property
    @abstractmethod
    def nbytes(self) -> ShapeItem: ...

    @property
    @abstractmethod
    def T(self) -> Self: ...

    @abstractmethod
    def __getitem__(
        self,
        key: SupportsIndex
        | slice
        | EllipsisType
        | tuple[SupportsIndex | slice | EllipsisType | Self, ...]
        | Self,
    ) -> Self: ...

    @overload
    def __setitem__(
        self,
        key: slice
        | EllipsisType
        | tuple[SupportsIndex | slice | EllipsisType, ...]
        | Self,
        value: int | float | bool | complex,
    ): ...

    @overload
    def __setitem__(
        self,
        key: SupportsIndex
        | slice
        | EllipsisType
        | tuple[SupportsIndex | slice | EllipsisType | Self, ...]
        | Self,
        value: int | float | bool | complex | Self,
    ): ...

    @abstractmethod
    def __setitem__(self, key, value): ...

    @abstractmethod
    def __bool__(self) -> bool: ...

    @abstractmethod
    def __int__(self) -> int: ...

    @abstractmethod
    def __index__(self) -> int: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def view(self, dtype: DTypeLike) -> Self: ...

    # Scalar UFUNCS
    @abstractmethod
    def __add__(self, other: int | complex | float | Self) -> Self: ...

    @abstractmethod
    def __sub__(self, other: int | complex | float | Self) -> Self: ...

    @abstractmethod
    def __mul__(self, other: int | complex | float | Self) -> Self: ...

    @abstractmethod
    def __truediv__(self, other: int | complex | float | Self) -> Self: ...

    @abstractmethod
    def __floordiv__(self, other: int | complex | float | Self) -> Self: ...

    @abstractmethod
    def __gt__(self, other: int | complex | float | Self) -> Self: ...

    @abstractmethod
    def __lt__(self, other: int | complex | float | Self) -> Self: ...

    @abstractmethod
    def __ge__(self, other: int | complex | float | Self) -> Self: ...

    @abstractmethod
    def __le__(self, other: int | complex | float | Self) -> Self: ...

    @abstractmethod
    def __eq__(self, other: int | complex | float | bool | Self) -> Self:  # type: ignore[override]
        ...

    @abstractmethod
    def __and__(self, other: int | bool | Self) -> Self: ...

    @abstractmethod
    def __or__(self, other: int | bool | Self) -> Self: ...

    @abstractmethod
    def __invert__(self) -> Self: ...


class MaterializableArray(ArrayLike):
    @abstractmethod
    def materialize(self) -> ArrayLike: ...


def maybe_materialize(
    *args: Any,
    type_: type[MaterializableArray]
    | tuple[type[MaterializableArray], ...] = MaterializableArray,
) -> tuple[Any, ...]:
    """
    Returns a tuple where all arguments that are instances of `type_` have been replaced
    by the result of calling their `.materialize()` method.

    Other `ArrayLike` or `Any` arguments are returned unchanged.

    Args:
        *args: Variable length argument list of MaterializableArray or ArrayLike or Any objects.
        type_: The class or tuple of classes to check for materialization. The default is `MaterializableArray`.

    Returns:
        tuple: A tuple where each instance of `type_` is replaced by its materialized form,
        and other ArrayLike or Any objects are returned unchanged.
    """
    return tuple(arg.materialize() if isinstance(arg, type_) else arg for arg in args)
