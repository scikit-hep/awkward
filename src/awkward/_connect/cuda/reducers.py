# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from abc import abstractmethod

import awkward as ak
from awkward import _reducers
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import ShapeItem
from awkward._reducers import Reducer
from awkward._typing import Any as AnyType
from awkward._typing import Final, Self, TypeVar

np = NumpyMetadata.instance()

DTypeLike = AnyType

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

    _use32 = ((ak._util.win or ak._util.bits32) and not ak._util.numpy2) or (
        ak._util.numpy2 and np.intp is np.int32
    )


# TODO: change this for a custom one?
def apply_positional_corrections(
    reduced: ak.contents.NumpyArray,
    parents: ak.index.Index | ak.index.ZeroIndex,
    offsets: ak.index.Index | ak.index.EmptyIndex,
    starts: ak.index.Index,
    shifts: ak.index.Index | None,
):
    if shifts is None:
        assert (
            parents.nplike is reduced.backend.nplike
            and starts.nplike is reduced.backend.nplike
        )
        reduced.backend.maybe_kernel_error(
            reduced.backend[
                "awkward_NumpyArray_reduce_adjust_starts_64",
                reduced.dtype.type,
                parents.dtype.type,
                starts.dtype.type,
            ](
                reduced.data,
                reduced.length,
                parents.data,
                starts.data,
            )
        )
        return reduced
    else:
        assert (
            parents.nplike is reduced.backend.nplike
            and starts.nplike is reduced.backend.nplike
            and shifts.nplike is reduced.backend.nplike
        )
        reduced.backend.maybe_kernel_error(
            reduced._backend[
                "awkward_NumpyArray_reduce_adjust_starts_shifts_64",
                reduced.dtype.type,
                parents.dtype.type,
                starts.dtype.type,
                shifts.dtype.type,
            ](
                reduced.data,
                reduced.length,
                parents.data,
                starts.data,
                shifts.data,
            )
        )
        return reduced


@overloads(_reducers.ArgMin)
class ArgMin(CudaComputeReducer):
    name: Final = "argmin"
    needs_position: Final = True
    preferred_dtype: Final = np.int64

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.ArgMin)
        return cls()

    @classmethod
    def _return_dtype(cls, given_dtype):
        return np.int64

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        offsets: ak.index.Index | ak.index.EmptyIndex,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        # View array data in kernel-supported dtype
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.int64)

        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_argmin_complex",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            from ._compute import awkward_reduce_argmin

            # should I pass kernel_array_data here too? (instead of array.data)
            result = awkward_reduce_argmin(
                result, array.data, parents.data, parents.length, starts.data, outlength
            )

        result_array = ak.contents.NumpyArray(result, backend=array.backend)
        corrected_data = apply_positional_corrections(
            result_array, parents, offsets, starts, shifts
        )
        return corrected_data


def get_cuda_compute_reducer(reducer: Reducer) -> Reducer:
    # can be deleted after all reducers are added
    if reducer.name not in ("argmin",):
        return reducer

    return _overloads[type(reducer)].from_kernel_reducer(reducer)
