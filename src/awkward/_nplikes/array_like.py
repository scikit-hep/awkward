# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from abc import abstractmethod

from awkward._nplikes.shape import ShapeItem
from awkward._typing import (
    TYPE_CHECKING,
    DType,
    Protocol,
    Self,
    SupportsIndex,
    overload,
)

if TYPE_CHECKING:
    from types import EllipsisType

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
