# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

"""
An Array-API inspired array compatibility layer. Where possible, this should follow the Array API standard, making
sensible decisions where the standard is unspecified. A notable extension is the `known_shape` and `known_data`
interface; some NumpyLike's may choose not to implement operations that act on concrete shapes or data. In this case
we differ from the Array API, and allow this.

Unlike the Array API, we do not assume that different NumpyLike instances define different metadata (dtypes).
We assume that these are common across NumpyLikes.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal, Protocol, TypeVar, runtime_checkable

DTypeT = TypeVar("DTypeT")
DeviceT = TypeVar("DeviceT")


NumpyLikeSelf = TypeVar("NumpyLikeSelf", bound="NumpyLike")


@runtime_checkable
class NumpyLike(Protocol):
    known_data: bool
    known_shape: bool
    is_eager: bool

    _instance: NumpyLikeSelf

    @classmethod
    def instance(cls: type[NumpyLikeSelf]) -> NumpyLikeSelf:
        try:
            return cls._instance
        except AttributeError:
            cls._instance = cls()
            return cls._instance

    ############################ array creation

    @abstractmethod
    def asarray(
        self,
        obj,
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
        copy: bool | None = None,
    ):
        ...

    @abstractmethod
    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
    ):
        ...

    @abstractmethod
    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
    ):
        ...

    @abstractmethod
    def empty(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
    ):
        ...

    @abstractmethod
    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value,
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
    ):
        ...

    @abstractmethod
    def zeros_like(
        self, x, *, dtype: DTypeT | None = None, device: DeviceT | None = None
    ):
        ...

    @abstractmethod
    def ones_like(
        self, x, *, dtype: DTypeT | None = None, device: DeviceT | None = None
    ):
        ...

    @abstractmethod
    def full_like(
        self,
        x,
        fill_value,
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
    ):
        ...

    @abstractmethod
    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
    ):
        ...

    @abstractmethod
    def meshgrid(self, *arrays, indexing: Literal["xy", "ij"] = "xy"):
        ...

    ############################ data type functions

    @abstractmethod
    def astype(self, x, dtype, *, copy: bool = True):
        ...

    @abstractmethod
    def broadcast_arrays(self, *arrays):
        ...

    # TODO better type signature for unknown shapes
    @abstractmethod
    def broadcast_to(self, x, shape: int | tuple[int, ...]):
        ...

    @abstractmethod
    def result_type(self, *arrays_and_dtypes):
        ...

    ############################ searching functions

    @abstractmethod
    def nonzero(self, x):
        ...

    @abstractmethod
    def where(self, condition, x1, x2):
        ...

    ############################ set functions

    @abstractmethod
    def unique_counts(self, x):
        ...

    @abstractmethod
    def unique_values(self, x):
        ...

    ############################ manipulation functions

    @abstractmethod
    def concat(self, arrays, *, axis: int | None = 0):
        ...

    @abstractmethod
    def stack(self, arrays, *, axis: int = 0):
        ...

    ############################ ufuncs

    @abstractmethod
    def add(self, x1, x2):
        ...

    @abstractmethod
    def multiply(self, x1, x2):
        ...

    @abstractmethod
    def logical_or(self, x1, x2):
        ...

    @abstractmethod
    def logical_and(self, x1, x2):
        ...

    @abstractmethod
    def logical_not(self, x):
        ...

    @abstractmethod
    def sqrt(self, x):
        ...

    @abstractmethod
    def exp(self, x):
        ...

    @abstractmethod
    def divide(self, x1, x2):
        ...

    @abstractmethod
    def equal(self, x1, x2):
        ...

    @abstractmethod
    def isnan(self, x):
        ...

    ############################ Utility functions

    @abstractmethod
    def all(
        self, x, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ):
        ...

    @abstractmethod
    def any(
        self, x, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ):
        ...

    ############################ Statistical functions

    @abstractmethod
    def sum(
        self,
        x,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeT | None = None,
        keepdims: bool = False,
    ):
        ...

    @abstractmethod
    def prod(
        self,
        x,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeT | None = None,
        keepdims: bool = False,
    ):
        ...

    @abstractmethod
    def min(
        self, x, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ):
        ...

    @abstractmethod
    def max(
        self, x, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ):
        ...

    ############################ Searching functions

    @abstractmethod
    def argmin(self, x, *, axis: int | None = None, keepdims: bool = False):
        ...

    @abstractmethod
    def argmax(self, x, *, axis: int | None = None, keepdims: bool = False):
        ...

    ############################ extensions to Array API

    @abstractmethod
    def ascontiguousarray(self, x):
        ...

    @abstractmethod
    def cumsum(self, x, *, axis: int | tuple[int, ...] | None = None):
        ...

    @abstractmethod
    def frombuffer(self, buffer, *, dtype=None):  # TODO dtype: float?`
        ...

    @abstractmethod
    def array_equal(self, x1, x2, *, equal_nan: bool = False) -> bool:
        ...

    @abstractmethod
    def searchsorted(
        self, x, values, *, side: Literal["left", "right"] = "left", sorter=None
    ):
        ...

    # TODO
    @abstractmethod
    def repeat(self, x, repeats, *, axis=None):
        ...

    @abstractmethod
    def tile(self, x, reps: int):
        ...

    @abstractmethod
    def packbits(
        self, x, *, axis: int | None = None, bitorder: Literal["big", "little"] = "big"
    ):
        ...

    @abstractmethod
    def unpackbits(
        self,
        a,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ):
        ...

    @abstractmethod
    def minimum(self, x1, x2):
        ...

    @abstractmethod
    def maximum(self, x1, x2):
        ...

    @abstractmethod
    def nan_to_num(
        self,
        x,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ):
        ...

    @abstractmethod
    def isclose(
        self, x1, x2, *, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ):
        ...

    @abstractmethod
    def count_nonzero(self, x, *, axis: int | None = None, keepdims: bool = False):
        ...

    @abstractmethod
    def array_str(
        self,
        x,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        ...

    @abstractmethod
    def is_c_contiguous(self, x) -> bool:
        ...

    @abstractmethod
    def to_rectilinear(self, array):
        ...

    @classmethod
    @abstractmethod
    def is_own_array(cls, obj) -> bool:
        """
        Args:
            obj: object to test

        Return `True` if the given object is a numpy buffer, otherwise `False`.

        """
        ...
