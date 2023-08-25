# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import math

import numpy

from awkward._nplikes.numpylike import (
    ArrayLike,
    IndexType,
    NumpyLike,
    NumpyMetadata,
    UniqueAllResult,
)
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._typing import Any, Final, Literal

np = NumpyMetadata.instance()


class ArrayModuleNumpyLike(NumpyLike):
    known_data: Final = True

    ############################ array creation

    def asarray(
        self,
        obj,
        *,
        dtype: numpy.dtype | None = None,
        copy: bool | None = None,
    ) -> ArrayLike:
        if isinstance(obj, PlaceholderArray):
            assert obj.dtype == dtype or dtype is None
            return obj
        elif copy:
            return self._module.array(obj, dtype=dtype, copy=True)
        elif copy is None:
            return self._module.asarray(obj, dtype=dtype)
        else:
            if getattr(obj, "dtype", dtype) != dtype:
                raise ValueError(
                    "asarray was called with copy=False for an array of a different dtype"
                )
            else:
                return self._module.asarray(obj, dtype=dtype)

    def ascontiguousarray(self, x: ArrayLike) -> ArrayLike:
        if isinstance(x, PlaceholderArray):
            return x
        else:
            return self._module.ascontiguousarray(x)

    def frombuffer(
        self, buffer, *, dtype: np.dtype | None = None, count: int = -1
    ) -> ArrayLike:
        assert not isinstance(buffer, PlaceholderArray)
        assert not isinstance(count, PlaceholderArray)
        return self._module.frombuffer(buffer, dtype=dtype, count=count)

    def from_dlpack(self, x: Any) -> ArrayLike:
        return self._module.from_dlpack(x)

    def zeros(
        self, shape: int | tuple[int, ...], *, dtype: np.dtype | None = None
    ) -> ArrayLike:
        return self._module.zeros(shape, dtype=dtype)

    def ones(
        self, shape: int | tuple[int, ...], *, dtype: np.dtype | None = None
    ) -> ArrayLike:
        return self._module.ones(shape, dtype=dtype)

    def empty(
        self, shape: int | tuple[int, ...], *, dtype: np.dtype | None = None
    ) -> ArrayLike:
        return self._module.empty(shape, dtype=dtype)

    def full(
        self, shape: int | tuple[int, ...], fill_value, *, dtype: np.dtype | None = None
    ) -> ArrayLike:
        return self._module.full(shape, fill_value, dtype=dtype)

    def zeros_like(self, x: ArrayLike, *, dtype: np.dtype | None = None) -> ArrayLike:
        if isinstance(x, PlaceholderArray):
            return self.zeros(x.shape, dtype=dtype or x.dtype)
        else:
            return self._module.zeros_like(x, dtype=dtype)

    def ones_like(self, x: ArrayLike, *, dtype: np.dtype | None = None) -> ArrayLike:
        if isinstance(x, PlaceholderArray):
            return self.ones(x.shape, dtype=dtype or x.dtype)
        else:
            return self._module.ones_like(x, dtype=dtype)

    def full_like(
        self, x: ArrayLike, fill_value, *, dtype: np.dtype | None = None
    ) -> ArrayLike:
        if isinstance(x, PlaceholderArray):
            return self.full(x.shape, fill_value, dtype=dtype or x.dtype)
        else:
            return self._module.full_like(x, fill_value, dtype=dtype)

    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: np.dtype | None = None,
    ) -> ArrayLike:
        assert not isinstance(start, PlaceholderArray)
        assert not isinstance(stop, PlaceholderArray)
        assert not isinstance(step, PlaceholderArray)
        return self._module.arange(start, stop, step, dtype=dtype)

    def meshgrid(
        self, *arrays: ArrayLike, indexing: Literal["xy", "ij"] = "xy"
    ) -> list[ArrayLike]:
        return self._module.meshgrid(*arrays, indexing=indexing)

    ############################ testing

    def array_equal(
        self, x1: ArrayLike, x2: ArrayLike, *, equal_nan: bool = False
    ) -> ArrayLike:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        return self._module.asarray(
            self._module.array_equal(x1, x2, equal_nan=equal_nan)
        )

    def searchsorted(
        self,
        x: ArrayLike,
        values: ArrayLike,
        *,
        side: Literal["left", "right"] = "left",
        sorter: ArrayLike | None = None,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        assert not isinstance(values, PlaceholderArray)
        assert not isinstance(sorter, PlaceholderArray)
        return self._module.searchsorted(x, values, side=side, sorter=sorter)

    ############################ manipulation

    def broadcast_arrays(self, *arrays: ArrayLike) -> list[ArrayLike]:
        assert not any(isinstance(x, PlaceholderArray) for x in arrays)
        return self._module.broadcast_arrays(*arrays)

    def _compute_compatible_shape(
        self, shape: tuple[ShapeItem, ...], existing_shape: tuple[ShapeItem, ...]
    ) -> tuple[ShapeItem, ...]:
        next_shape = list(shape)
        j = None
        length_factor = 1
        for i, item in enumerate(shape):
            if item != -1:
                length_factor *= item
            elif j is not None:
                raise ValueError("can only have one unknown dimension")
            else:
                j = i
        if j is not None:
            next_shape[j] = math.prod(existing_shape) // length_factor
        return tuple(next_shape)

    def reshape(
        self, x: ArrayLike, shape: tuple[ShapeItem, ...], *, copy: bool | None = None
    ) -> ArrayLike:
        if isinstance(x, PlaceholderArray):
            next_shape = self._compute_compatible_shape(shape, x.shape)
            return PlaceholderArray(self, next_shape, x.dtype)

        if copy is None:
            return self._module.reshape(x, shape)
        elif copy:
            return self._module.reshape(self._module.copy(x, order="C"), shape)
        else:
            result = self._module.asarray(x)
            try:
                result.shape = shape
            except AttributeError:
                raise ValueError("cannot reshape array without copying") from None
            return result

    def shape_item_as_index(self, x1: ShapeItem) -> int:
        if x1 is unknown_length:
            raise TypeError("array module nplikes do not support unknown lengths")
        elif isinstance(x1, int):
            return x1
        else:
            raise TypeError(f"expected None or int type, received {x1}")

    def index_as_shape_item(self, x1: IndexType) -> int:
        return int(x1)

    def derive_slice_for_length(
        self, slice_: slice, length: ShapeItem
    ) -> tuple[IndexType, IndexType, IndexType, ShapeItem]:
        """
        Args:
            slice_: normalized slice object
            length: length of layout

        Return a tuple of (start, stop, step) indices into a layout, suitable for
        `_getitem_range` (if step == 1). Normalize lengths to fit length of array,
        and for arrays with unknown lengths, these offsets become none.
        """
        # We have known_data (therefore known shape), so we can safely convert to int
        slice_as_shape = slice(
            slice_.start
            if slice_.start is None
            else self.index_as_shape_item(slice_.start),
            slice_.stop
            if slice_.stop is None
            else self.index_as_shape_item(slice_.stop),
            slice_.step
            if slice_.step is None
            else self.index_as_shape_item(slice_.step),
        )
        start, stop, step = slice_as_shape.indices(length)
        slice_length = math.ceil((stop - start) / step)

        # Shape items are already valid indices, so we don't need to invoke
        # `shape_item_as_index`
        return start, stop, step, slice_length

    def regularize_index_for_length(
        self, index: IndexType, length: ShapeItem
    ) -> IndexType:
        """
        Args:
            index: index value
            length: length of array

        Returns regularized index that is guaranteed to be in-bounds.
        """  # We have known length and index
        if index < 0:
            index = index + length

        if 0 <= index < length:
            return index
        else:
            raise IndexError(f"index value out of bounds (0, {length}): {index}")

    def nonzero(self, x: ArrayLike) -> tuple[ArrayLike, ...]:
        assert not isinstance(x, PlaceholderArray)
        return self._module.nonzero(x)

    def where(self, condition: ArrayLike, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        assert not isinstance(condition, PlaceholderArray)
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        return self._module.where(condition, x1, x2)

    def unique_values(self, x: ArrayLike) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.unique(
            x,
            return_counts=False,
            return_index=False,
            return_inverse=False,
            equal_nan=False,
        )

    def unique_all(self, x: ArrayLike) -> UniqueAllResult:
        assert not isinstance(x, PlaceholderArray)
        values, indices, inverse_indices, counts = self._module.unique(
            x, return_counts=True, return_index=True, return_inverse=True
        )
        # np.unique() flattens inverse indices, but they need to share x's shape
        # See https://github.com/numpy/numpy/issues/20638
        inverse_indices = inverse_indices.reshape(x.shape)
        return UniqueAllResult(values, indices, inverse_indices, counts)

    def sort(
        self,
        x: ArrayLike,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        # Note: this keyword argument is different, and the default is different.
        kind = "stable" if stable else "quicksort"
        res = self._module.sort(x, axis=axis, kind=kind)
        if descending:
            return self._module.flip(res, axis=axis)
        else:
            return res

    def concat(
        self,
        arrays: list[ArrayLike] | tuple[ArrayLike, ...],
        *,
        axis: int | None = 0,
    ) -> ArrayLike:
        assert not any(isinstance(x, PlaceholderArray) for x in arrays)
        return self._module.concatenate(arrays, axis=axis, casting="same_kind")

    def repeat(
        self,
        x: ArrayLike,
        repeats: ArrayLike | int,
        *,
        axis: int | None = None,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        assert not isinstance(repeats, PlaceholderArray)
        return self._module.repeat(x, repeats=repeats, axis=axis)

    def stack(
        self,
        arrays: list[ArrayLike] | tuple[ArrayLike, ...],
        *,
        axis: int = 0,
    ) -> ArrayLike:
        assert not any(isinstance(x, PlaceholderArray) for x in arrays)
        arrays = list(arrays)
        return self._module.stack(arrays, axis=axis)

    def packbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.packbits(x, axis=axis, bitorder=bitorder)

    def unpackbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.unpackbits(x, axis=axis, count=count, bitorder=bitorder)

    def broadcast_to(self, x: ArrayLike, shape: tuple[ShapeItem, ...]) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.broadcast_to(x, shape)

    def strides(self, x: ArrayLike) -> tuple[ShapeItem, ...]:
        if isinstance(x, PlaceholderArray):
            # Assume contiguous
            strides = (x.itemsize,)
            for item in x.shape[-1:0:-1]:
                strides = (item * strides[0], *strides)
            return strides

        return x.strides

    ############################ ufuncs

    def add(
        self, x1: ArrayLike, x2: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        return self._module.add(x1, x2, out=maybe_out)

    def logical_or(
        self, x1: ArrayLike, x2: ArrayLike, *, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        return self._module.logical_or(x1, x2, out=maybe_out)

    def logical_and(
        self, x1: ArrayLike, x2: ArrayLike, *, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        return self._module.logical_and(x1, x2, out=maybe_out)

    def logical_not(
        self, x: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.logical_not(x, out=maybe_out)

    def sqrt(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.sqrt(x, out=maybe_out)

    def exp(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.exp(x, out=maybe_out)

    def divide(
        self, x1: ArrayLike, x2: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        return self._module.divide(x1, x2, out=maybe_out)

    ############################ almost-ufuncs

    def nan_to_num(
        self,
        x: ArrayLike,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.nan_to_num(
            x, copy=copy, nan=nan, posinf=posinf, neginf=neginf
        )

    def isclose(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> ArrayLike:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        return self._module.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def isnan(self, x: ArrayLike) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.isnan(x)

    def all(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.all(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def any(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.any(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def min(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.min(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def max(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.max(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def count_nonzero(
        self, x: ArrayLike, *, axis: int | None = None, keepdims: bool = False
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.count_nonzero(x, axis=axis, keepdims=keepdims)

    def cumsum(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return self._module.cumsum(x, axis=axis, out=maybe_out)

    def array_str(
        self,
        x: ArrayLike,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        assert not isinstance(x, PlaceholderArray)
        return self._module.array_str(
            x,
            max_line_width=max_line_width,
            precision=precision,
            suppress_small=suppress_small,
        )

    def astype(
        self, x: ArrayLike, dtype: numpy.dtype, *, copy: bool | None = True
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        return x.astype(dtype, copy=copy)

    def can_cast(self, from_: np.dtype | ArrayLike, to: np.dtype | ArrayLike) -> bool:
        return self._module.can_cast(from_, to, casting="same_kind")

    @classmethod
    def is_own_array(cls, obj) -> bool:
        return cls.is_own_array_type(type(obj))
