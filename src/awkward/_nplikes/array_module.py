# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import numpy

import awkward as ak
from awkward._nplikes.numpylike import ArrayLike, NumpyLike, NumpyMetadata, ShapeItem
from awkward.typing import Final, Literal, SupportsInt

np = NumpyMetadata.instance()


class ArrayModuleNumpyLike(NumpyLike):
    known_data: Final = True
    known_shape: Final = True

    ############################ array creation

    def asarray(
        self,
        obj,
        *,
        dtype: numpy.dtype | None = None,
        copy: bool | None = None,
    ) -> ArrayLike:
        if copy:
            return self._module.array(obj, dtype=dtype, copy=True)
        elif copy is None:
            return self._module.asarray(obj, dtype=dtype)
        else:
            if getattr(obj, "dtype", dtype) != dtype:
                raise ak._errors.wrap_error(
                    ValueError(
                        "asarray was called with copy=False for an array of a different dtype"
                    )
                )
            else:
                return self._module.asarray(obj, dtype=dtype)

    def ascontiguousarray(
        self, x: ArrayLike, *, dtype: np.dtype | None = None
    ) -> ArrayLike:
        return self._module.ascontiguousarray(x, dtype=dtype)

    def frombuffer(
        self, buffer, *, dtype: np.dtype | None = None, count: int = -1
    ) -> ArrayLike:
        return self._module.frombuffer(buffer, dtype=dtype, count=count)

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
        return self._module.zeros_like(x, dtype=dtype)

    def ones_like(self, x: ArrayLike, *, dtype: np.dtype | None = None) -> ArrayLike:
        return self._module.ones_like(x, dtype=dtype)

    def full_like(
        self, x: ArrayLike, fill_value, *, dtype: np.dtype | None = None
    ) -> ArrayLike:
        return self._module.full_like(x, fill_value, dtype=dtype)

    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: np.dtype | None = None,
    ) -> ArrayLike:
        return self._module.arange(start, stop, step, dtype=dtype)

    def meshgrid(
        self, *arrays: ArrayLike, indexing: Literal["xy", "ij"] = "xy"
    ) -> list[ArrayLike]:
        return self._module.meshgrid(*arrays, indexing=indexing)

    ############################ testing

    def array_equal(
        self, x1: ArrayLike, x2: ArrayLike, *, equal_nan: bool = False
    ) -> bool:
        return self._module.array_equal(x1, x2, equal_nan=equal_nan)

    def searchsorted(
        self,
        x: ArrayLike,
        values: ArrayLike,
        *,
        side: Literal["left", "right"] = "left",
        sorter: ArrayLike | None = None,
    ) -> ArrayLike:
        return self._module.searchsorted(x, values, side=side, sorter=sorter)

    ############################ manipulation

    def broadcast_arrays(self, *arrays: ArrayLike) -> list[ArrayLike]:
        return self._module.broadcast_arrays(*arrays)

    def reshape(
        self, x: ArrayLike, shape: tuple[int, ...], *, copy: bool | None = None
    ) -> ArrayLike:
        if copy is False:
            raise ak._errors.wrap_error(
                NotImplementedError(
                    "reshape was called with copy=False, which is currently not supported"
                )
            )
        result = x.reshape(shape)
        if copy and self._module.shares_memory(x, result):
            return self._module.copy(result)
        else:
            return result

    def shape_item_as_scalar(self, x1: ShapeItem):
        if x1 is None:
            raise ak._errors.wrap_error(
                TypeError("array module nplikes do not support unknown lengths")
            )
        elif isinstance(x1, int):
            return self._module.asarray(x1, dtype=np.int64)
        else:
            raise ak._errors.wrap_error(
                TypeError(f"expected None or int type, received {x1}")
            )

    def scalar_as_shape_item(self, x1) -> ShapeItem:
        if x1 is None:
            return None
        else:
            return int(x1)

    def add_shape_item(self, x1: ShapeItem, x2: ShapeItem) -> ShapeItem:
        assert x1 >= 0
        assert x2 >= 0
        return x1 + x2

    def sub_shape_item(self, x1: ShapeItem, x2: ShapeItem) -> ShapeItem:
        assert x1 >= 0
        assert x2 >= 0
        result = x1 - x2
        assert result >= 0
        return result

    def mul_shape_item(self, x1: ShapeItem, x2: ShapeItem) -> ShapeItem:
        assert x1 >= 0
        assert x2 >= 0
        return x1 * x2

    def div_shape_item(self, x1: ShapeItem, x2: ShapeItem) -> ShapeItem:
        assert x1 >= 0
        assert x2 >= 0
        result = x1 // x2
        assert result * x2 == x1
        return result

    def nonzero(self, x: ArrayLike) -> tuple[ArrayLike, ...]:
        return self._module.nonzero(x)

    def unique_values(self, x: ArrayLike) -> ArrayLike:
        return self._module.unique(
            x,
            return_counts=False,
            return_index=False,
            return_inverse=False,
            equal_nan=False,
        )

    def concat(
        self,
        arrays: list[ArrayLike] | tuple[ArrayLike, ...],
        *,
        axis: int | None = 0,
    ) -> ArrayLike:
        return self._module.concatenate(arrays, axis=axis, casting="same_kind")

    def repeat(
        self,
        x: ArrayLike,
        repeats: ArrayLike | int,
        *,
        axis: int | None = None,
    ) -> ArrayLike:
        return self._module.repeat(x, repeats=repeats, axis=axis)

    def tile(self, x: ArrayLike, reps: int) -> ArrayLike:
        return self._module.tile(x, reps)

    def stack(
        self,
        arrays: list[ArrayLike] | tuple[ArrayLike, ...],
        *,
        axis: int = 0,
    ) -> ArrayLike:
        arrays = list(arrays)
        return self._module.stack(arrays, axis=axis)

    def packbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayLike:
        return self._module.packbits(x, axis=axis, bitorder=bitorder)

    def unpackbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayLike:
        return self._module.unpackbits(x, axis=axis, count=count, bitorder=bitorder)

    def broadcast_to(self, x: ArrayLike, shape: tuple[SupportsInt, ...]) -> ArrayLike:
        return self._module.broadcast_to(x, shape)

    ############################ ufuncs

    def add(
        self, x1: ArrayLike, x2: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        return self._module.add(x1, x2, out=maybe_out)

    def logical_or(
        self, x1: ArrayLike, x2: ArrayLike, *, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        return self._module.logical_or(x1, x2, out=maybe_out)

    def logical_and(
        self, x1: ArrayLike, x2: ArrayLike, *, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        return self._module.logical_and(x1, x2, out=maybe_out)

    def logical_not(
        self, x: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        return self._module.logical_not(x, out=maybe_out)

    def sqrt(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> ArrayLike:
        return self._module.sqrt(x, out=maybe_out)

    def exp(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> ArrayLike:
        return self._module.exp(x, out=maybe_out)

    def divide(
        self, x1: ArrayLike, x2: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
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
        return self._module.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def isnan(self, x: ArrayLike) -> ArrayLike:
        return self._module.isnan(x)

    def all(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        return self._module.all(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def any(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        return self._module.any(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def min(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        return self._module.min(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def max(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        return self._module.max(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def count_nonzero(
        self, x: ArrayLike, *, axis: int | None = None, keepdims: bool = False
    ) -> ArrayLike:
        return self._module.count_nonzero(x, axis=axis, keepdims=keepdims)

    def cumsum(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        return self._module.cumsum(x, axis=axis, out=maybe_out)

    def array_str(
        self,
        x: ArrayLike,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        return self._module.array_str(
            x,
            max_line_width=max_line_width,
            precision=precision,
            suppress_small=suppress_small,
        )

    def astype(
        self, x: ArrayLike, dtype: numpy.dtype, *, copy: bool | None = True
    ) -> ArrayLike:
        return x.astype(dtype, copy=copy)

    def can_cast(self, from_: np.dtype | ArrayLike, to: np.dtype | ArrayLike) -> bool:
        return self._module.can_cast(from_, to, casting="same_kind")
