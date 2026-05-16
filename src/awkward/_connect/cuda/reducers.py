# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from abc import abstractmethod

import awkward as ak  # noqa: F401  (kept for type-annotated reducer subclasses)
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._reducers import Reducer
from awkward._typing import Any as AnyType
from awkward._typing import Self, TypeVar

np = NumpyMetadata.instance()

DTypeLike = AnyType

# Registry of CUDA-specific reducer overloads. Currently empty — the CUDA
# backend dispatches reducers through `_backends/cupy.py:_get_cuda_compute_impl`
# directly into `_connect/cuda/_compute.py`'s CCCL-based helpers, so no
# subclass of `CudaComputeReducer` is registered. The scaffolding is kept
# here as the recognised home for future CUDA-specific reducer overloads
# (mirrors the populated framework in `_connect/jax/reducers.py`).
_overloads: dict[type[Reducer], type[CudaComputeReducer]] = {}


R = TypeVar("R", bound=Reducer)


def overloads(cls: type[Reducer]):
    def registrar(new_cls: type[R]) -> type[R]:
        _overloads[cls] = new_cls
        return new_cls

    return registrar


class CudaComputeReducer(Reducer):
    @classmethod
    @abstractmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        raise NotImplementedError

    @classmethod
    def _dtype_for_kernel(cls, dtype: DTypeLike) -> DTypeLike:
        dtype = np.dtype(dtype)
        if dtype.kind.upper() == "M":
            return np.dtype(np.int64)
        elif dtype == np.complex128:
            return np.dtype(np.float64)
        elif dtype == np.complex64:
            return np.dtype(np.float32)
        else:
            return dtype


def get_cuda_compute_reducer(reducer: Reducer) -> Reducer:
    """
    Returns the CUDA-specific reducer if one is registered via @overloads,
    otherwise falls back to the original reducer.
    """
    impl = _overloads.get(type(reducer))
    if impl is None:
        return reducer

    return impl.from_kernel_reducer(reducer)
