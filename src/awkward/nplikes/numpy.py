# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

import functools
from typing import Literal, TypeVar

from awkward import _errors
from awkward.nplikes import dtypes
from awkward.nplikes.numpylike import NumpyLike

DTypeT = TypeVar("DTypeT")
DeviceT = TypeVar("DeviceT")


class Numpy(NumpyLike):
    known_data: bool
    known_shape: bool
    is_eager: bool

    ############################ array creation
    @property
    def array_api(self):
        import numpy.array_api

        return numpy.array_api

    @property
    def numpy_api(self):
        import numpy

        return numpy

    def _check_dtypes(self, *array_or_dtypes, allow_none=True):
        for array_or_dtype in array_or_dtypes:
            if array_or_dtype and allow_none:
                continue

            if hasattr(array_or_dtype, "dtype"):
                array_or_dtype = array_or_dtype.dtype

            if dtypes.is_known_dtype(array_or_dtype):
                continue

            raise _errors.wrap_error(
                NotImplementedError(
                    "internal error: the given dtype {array_or_dtype!r} is not supported"
                )
            )

    def _ignore_device(self, device: DeviceT):
        if device is not None:
            raise _errors.wrap_error(
                NotImplementedError(
                    "internal error: this `NumpyLike` does not support `device`"
                )
            )

    def _expect_dtype(self, array, dtype: DTypeT):
        if array.dtype != dtype:
            raise _errors.wrap_error(
                NotImplementedError(
                    "internal error: this `NumpyLike` operation did not produce the expected dtype"
                )
            )
        return array

    def _expect_has_shape(self, array):
        if not hasattr(array, "shape"):
            raise _errors.wrap_error(
                NotImplementedError(
                    "internal error: this `NumpyLike` operation did not produce a result with a shape"
                )
            )
        return array

    _dtype_upcasts = {
        dtypes.int8: dtypes.int64,
        dtypes.int16: dtypes.int64,
        dtypes.int32: dtypes.int64,
        dtypes.int64: dtypes.int64,
        dtypes.uint8: dtypes.uint64,
        dtypes.uint16: dtypes.uint64,
        dtypes.uint32: dtypes.uint64,
        dtypes.uint64: dtypes.uint64,
        dtypes.float32: dtypes.float64,
        dtypes.complex64: dtypes.complex128,
    }

    def asarray(
        self,
        obj,
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
        copy: bool | None = None,
    ):
        self._ignore_device(device)
        self._check_dtypes(dtype)
        if copy:
            return self.numpy_api.array(obj, dtype=dtype, copy=True)
        else:
            return self.numpy_api.asarray(obj, dtype=dtype, copy=copy)

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
    ):
        self._ignore_device(device)
        self._check_dtypes(dtype)
        return self.numpy_api.zeros(shape, dtype=dtype)

    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
    ):
        self._ignore_device(device)
        self._check_dtypes(dtype)
        return self.numpy_api.ones(shape, dtype=dtype)

    def empty(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
    ):
        self._ignore_device(device)
        self._check_dtypes(dtype)
        return self.numpy_api.empty(shape, dtype=dtype)

    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value,
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
    ):
        self._ignore_device(device)
        self._check_dtypes(dtype)
        return self.numpy_api.full(shape, dtype=dtype)

    def zeros_like(
        self, x, *, dtype: DTypeT | None = None, device: DeviceT | None = None
    ):
        self._ignore_device(device)
        self._check_dtypes(x, dtype)
        return self.numpy_api.zeros_like(x, dtype=dtype, device=device)

    def ones_like(
        self, x, *, dtype: DTypeT | None = None, device: DeviceT | None = None
    ):
        self._ignore_device(device)
        self._check_dtypes(x, dtype)
        return self.numpy_api.ones_like(x, dtype=dtype, device=device)

    def full_like(
        self,
        x,
        fill_value,
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
    ):
        self._ignore_device(device)
        self._check_dtypes(x, dtype)
        return self.numpy_api.full_like(x, dtype=dtype, device=device)

    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: DTypeT | None = None,
        device: DeviceT | None = None,
    ):
        self._ignore_device(device)
        self._check_dtypes(dtype)
        return self.numpy_api.arange(start, stop, step, dtype=dtype)

    def meshgrid(self, *arrays, indexing: Literal["xy", "ij"] = "xy"):
        self._check_dtypes(*arrays)
        if len({a.dtype for a in arrays}) > 1:
            raise _errors.wrap_error(
                ValueError("all meshgrid arrays must have the same dtype")
            )
        return self.numpy_api.meshgrid(*arrays, indexing=indexing)

    ############################ data type functions

    def astype(self, x, dtype: DTypeT, *, copy: bool = True):
        self._check_dtypes(x, dtype)
        return x.astype(dtype=dtype, copy=copy)

    def broadcast_arrays(self, *arrays):
        self._check_dtypes(*arrays)
        return self.numpy_api.broadcast_arrays(*arrays)

    # TODO better type signature for unknown shapes
    def broadcast_to(self, x, shape: int | tuple[int, ...]):
        self._check_dtypes(x)
        return self.numpy_api.broadcast_to(x, shape)

    def result_type(self, *arrays_and_dtypes) -> DTypeT:
        all_dtypes = []
        for item in arrays_and_dtypes:
            if hasattr(item, "shape") and hasattr(item, "dtype"):
                item = item.dtype
            elif not isinstance(item, dtypes.dtype):
                raise TypeError(
                    "result_type() inputs must be array_api arrays or dtypes"
                )
            all_dtypes.append(item.dtype)

        if len(all_dtypes) == 0:
            raise _errors.wrap_error(ValueError("at least one array or dtype required"))

        return functools.reduce(dtypes.promote_types, all_dtypes)

    ############################ searching functions

    def nonzero(self, x):
        return self.numpy_api.nonzero(x)

    def where(self, condition, x1, x2):
        return self._expect_dtype(
            self.numpy_api.where(condition, x1, x2),
            self.result_type(x1.dtype, x2.dtype),
        )

    ############################ set functions

    def unique_counts(self, x):
        return self.numpy_api.unique(
            x,
            return_counts=True,
            return_index=False,
            return_inverse=False,
            equal_nan=False,
        )

    def unique_values(self, x):
        return self.numpy_api.unique(
            x,
            return_counts=False,
            return_index=False,
            return_inverse=False,
            equal_nan=False,
        )

    ############################ manipulation functions

    def concat(self, arrays, *, axis: int | None = 0):
        return self._expect_dtype(
            self.numpy_api.concatenate(arrays, axis=axis), self.result_type(*arrays)
        )

    def stack(self, arrays, *, axis: int = 0):
        return self._expect_dtype(
            self.numpy_api.stack(arrays, axis=axis), self.result_type(*arrays)
        )

    ############################ ufuncs

    def add(self, x1, x2):
        return self._expect_dtype(self.numpy_api.add(x1, x2), self.result_type(x1, x2))

    def multiply(self, x1, x2):
        return self._expect_dtype(
            self.numpy_api.multiply(x1, x2), self.result_type(x1, x2)
        )

    def logical_or(self, x1, x2):
        return self._expect_dtype(
            self.numpy_api.logical_or(x1, x2), self.result_type(x1, x2)
        )

    def logical_and(self, x1, x2):
        return self._expect_dtype(
            self.numpy_api.logical_and(x1, x2), self.result_type(x1, x2)
        )

    def logical_not(self, x):
        return self.numpy_api.logical_not(x)

    def sqrt(self, x):
        return self.numpy_api.sqrt(x)

    def exp(self, x):
        return self.numpy_api.exp(x)

    def divide(self, x1, x2):
        return self._expect_dtype(
            self.numpy_api.divide(x1, x2), self.result_type(x1, x2)
        )

    def equal(self, x1, x2):
        return self._expect_dtype(
            self.numpy_api.equal(x1, x2), self.result_type(x1, x2)
        )

    def isnan(self, x):
        return self.numpy_api.isnan(x)

    ############################ Utility functions

    def all(
        self, x, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ):
        return self._expect_has_shape(
            self.numpy_api.all(x, axis=axis, keepdims=keepdims),
        )

    def any(
        self, x, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ):
        return self._expect_has_shape(
            self.numpy_api.any(x, axis=axis, keepdims=keepdims),
        )

    ############################ Statistical functions

    def sum(
        self,
        x,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeT | None = None,
        keepdims: bool = False,
    ):
        if dtype not in dtypes.numeric_dtypes:
            raise _errors.wrap_error(
                TypeError(f"internal error: unsupported dtype encountered: {dtype!r}")
            )

        if dtype is None:
            dtype = self._dtype_upcasts.get(x.dtype, x.dtype)

        return self._expect_has_shape(
            self.numpy_api.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
        )

    def prod(
        self,
        x,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeT | None = None,
        keepdims: bool = False,
    ):
        if dtype not in dtypes.numeric_dtypes:
            raise _errors.wrap_error(
                TypeError(f"internal error: unsupported dtype encountered: {dtype!r}")
            )

        if dtype is None:
            dtype = self._dtype_upcasts.get(x.dtype, x.dtype)

        return self._expect_has_shape(
            self.numpy_api.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)
        )

    def min(
        self, x, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ):
        if x.dtype not in dtypes.numeric_dtypes:
            raise _errors.wrap_error(
                TypeError(f"internal error: unsupported dtype encountered: {x.dtype!r}")
            )
        return self._expect_has_shape(
            self.numpy_api.min(x, axis=axis, keepdims=keepdims)
        )

    def max(
        self, x, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ):
        if x.dtype not in dtypes.numeric_dtypes:
            raise _errors.wrap_error(
                TypeError(f"internal error: unsupported dtype encountered: {x.dtype!r}")
            )
        return self._expect_has_shape(
            self.numpy_api.max(x, axis=axis, keepdims=keepdims)
        )

    ############################ Searching functions

    def argmin(self, x, *, axis: int | None = None, keepdims: bool = False):
        return self._expect_has_shape(
            self.numpy_api.argmin(x, axis=axis, keepdims=keepdims)
        )

    def argmax(self, x, *, axis: int | None = None, keepdims: bool = False):
        return self._expect_has_shape(
            self.numpy_api.argmin(x, axis=axis, keepdims=keepdims)
        )

    ############################ extensions to Array API

    def ascontiguousarray(self, x):
        return self.numpy_api.ascontiguousarray(x)

    def cumsum(self, x, *, axis: int | None = None):
        return self._expect_has_shape(self.numpy_api.cumsum(x, axis=axis))

    def frombuffer(self, buffer, *, dtype=None):
        return self.numpy_api.frombuffer(buffer, dtype=dtype)

    def array_equal(self, x1, x2, *, equal_nan: bool = False) -> bool:
        return self.numpy_api.array_equal(x1, x2, equal_nan=equal_nan)

    def searchsorted(
        self, x, values, *, side: Literal["left", "right"] = "left", sorter=None
    ):
        return self.numpy_api.searchsorted(x, values, side=side, sorter=sorter)

    # TODO
    def repeat(self, x, repeats, *, axis: int | None = None):
        return self.numpy_api.repeat(x, repeats, axis=axis)

    def tile(self, x, reps: int):
        return self.numpy_api.tile(x, reps)

    def packbits(
        self, x, *, axis: int | None = None, bitorder: Literal["big", "little"] = "big"
    ):
        return self.numpy_api.packbits(x, axis=axis, bitorder=bitorder)

    def unpackbits(
        self,
        x,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ):
        return self.numpy_api.unpackbits(x, axis=axis, count=count, bitorder=bitorder)

    def minimum(self, x1, x2):
        return self._expect_dtype(
            self.numpy_api.minimum(x1, x2), self.result_type(x1, x2)
        )

    def maximum(self, x1, x2):
        return self._expect_dtype(
            self.numpy_api.maximum(x1, x2), self.result_type(x1, x2)
        )

    def nan_to_num(
        self,
        x,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ):
        return self.numpy_api.nan_to_num(
            x, copy=copy, nan=nan, posinf=posinf, neginf=neginf
        )

    def isclose(
        self, x1, x2, *, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ):
        return self.numpy_api.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def count_nonzero(self, x, *, axis: int | None = None, keepdims: bool = False):
        return self._expect_has_shape(
            self.numpy_api.count_nonzero(x, axis=axis, keepdims=keepdims)
        )

    def array_str(
        self,
        x,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        ...

    def is_c_contiguous(self, x) -> bool:
        return x.flags["C_CONTIGUOUS"]

    def to_rectilinear(self, array):
        raise NotImplementedError

    @classmethod
    def is_own_array(cls, x) -> bool:
        """
        Args:
            x: object to test

        Return `True` if the given object is a numpy buffer, otherwise `False`.

        """
        import numpy

        return isinstance(x, numpy.ndarray)
