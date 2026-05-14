# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._backends.backend import Backend, KernelKeyType
from awkward._backends.dispatch import register_backend
from awkward._kernels import CudaComputeKernel, CupyKernel, NumpyKernel
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Final

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@register_backend(Cupy)  # type: ignore[type-abstract]
class CupyBackend(Backend):
    name: Final[str] = "cuda"

    _cupy: Cupy

    @property
    def nplike(self) -> Cupy:
        return self._cupy

    def __init__(self):
        self._cupy = Cupy.instance()

    def __getitem__(
        self, index: KernelKeyType
    ) -> CudaComputeKernel | CupyKernel | NumpyKernel:
        from awkward._connect import cuda
        from awkward._connect.cuda import _compute as cuda_compute

        kernel_name = index[0] if index else ""

        cupy = cuda.import_cupy("Awkward Arrays with CUDA")
        _cuda_kernels = cuda.initialize_cuda_kernels(cupy)
        func = _cuda_kernels[index]

        if func is not None:
            return CupyKernel(func, index)

        if self._supports_cuda_compute(kernel_name):
            if cuda_compute.is_available():
                compute_impl = self._get_cuda_compute_impl(kernel_name)
                if compute_impl is not None:
                    return CudaComputeKernel(compute_impl, index)
            else:
                raise NotImplementedError(
                    f"Operation '{kernel_name}' requires cuda.compute but it is not available."
                )

        raise AssertionError(
            f"Operation '{kernel_name}' is not supported on CUDA backend."
        )

    # ---------------------------------------------------------
    # CUDA compute support table
    # ---------------------------------------------------------
    def _supports_cuda_compute(self, kernel_name: str) -> bool:
        return kernel_name in (
            # core reducers
            "awkward_reduce_sum",
            "awkward_reduce_sum_bool",
            "awkward_reduce_sum_bool_complex",
            "awkward_reduce_sum_bool_complex64_64",  # FIXME
            "awkward_reduce_sum_bool_complex128_64",  # FIXME
            "awkward_reduce_sum_int32_bool_64",
            "awkward_reduce_sum_int64_bool_64",
            "awkward_reduce_sum_complex",
            "awkward_reduce_max",
            "awkward_reduce_max_complex",
            "awkward_reduce_min",
            "awkward_reduce_min_complex",
            "awkward_reduce_prod",
            "awkward_reduce_prod_bool",
            "awkward_reduce_prod_complex",
            "awkward_reduce_prod_bool_complex",
            "awkward_reduce_argmax",
            "awkward_reduce_argmax_complex",
            "awkward_reduce_argmin",
            "awkward_reduce_argmin_complex",
            "awkward_reduce_count_64",
            "awkward_reduce_countnonzero",
            "awkward_reduce_countnonzero_complex",
            # indexing / structure
            "awkward_missing_repeat",
            "awkward_index_rpad_and_clip_axis0",
            "awkward_index_rpad_and_clip_axis1",
            # sort
            "awkward_sort",
        )

    # ---------------------------------------------------------
    # CUDA compute dispatch table
    # ---------------------------------------------------------
    def _get_cuda_compute_impl(self, kernel_name: str):
        from awkward._connect.cuda import _compute as cuda_compute

        return {
            "awkward_sort": cuda_compute.segmented_sort,
            "awkward_reduce_sum": cuda_compute.awkward_reduce_sum,
            "awkward_reduce_sum_bool": cuda_compute.awkward_reduce_sum_bool,
            "awkward_reduce_sum_int32_bool_64": cuda_compute.awkward_reduce_sum_int32_bool_64,
            "awkward_reduce_sum_int64_bool_64": cuda_compute.awkward_reduce_sum_int64_bool_64,
            "awkward_reduce_sum_complex": cuda_compute.awkward_reduce_sum_complex,
            "awkward_reduce_sum_bool_complex64_64": cuda_compute.awkward_reduce_sum_bool_complex64_64,
            "awkward_reduce_sum_bool_complex128_64": cuda_compute.awkward_reduce_sum_bool_complex128_64,
            "awkward_reduce_max": cuda_compute.awkward_reduce_max,
            "awkward_reduce_max_complex": cuda_compute.awkward_reduce_max_complex,
            "awkward_reduce_min": cuda_compute.awkward_reduce_min,
            "awkward_reduce_min_complex": cuda_compute.awkward_reduce_min_complex,
            "awkward_reduce_prod": cuda_compute.awkward_reduce_prod,
            "awkward_reduce_prod_bool": cuda_compute.awkward_reduce_prod_bool,
            "awkward_reduce_prod_complex": cuda_compute.awkward_reduce_prod_complex,
            "awkward_reduce_prod_bool_complex": cuda_compute.awkward_reduce_prod_bool_complex,
            "awkward_reduce_argmax": cuda_compute.awkward_reduce_argmax,
            "awkward_reduce_argmax_complex": cuda_compute.awkward_reduce_argmax_complex,
            "awkward_reduce_argmin": cuda_compute.awkward_reduce_argmin,
            "awkward_reduce_argmin_complex": cuda_compute.awkward_reduce_argmin_complex,
            "awkward_reduce_count_64": cuda_compute.awkward_reduce_count_64,
            "awkward_reduce_countnonzero": cuda_compute.awkward_reduce_countnonzero,
            "awkward_reduce_countnonzero_complex": cuda_compute.awkward_reduce_countnonzero_complex,
            "awkward_missing_repeat": cuda_compute.awkward_missing_repeat,
            "awkward_index_rpad_and_clip_axis0": cuda_compute.awkward_index_rpad_and_clip_axis0,
            "awkward_index_rpad_and_clip_axis1": cuda_compute.awkward_index_rpad_and_clip_axis1,
        }.get(kernel_name)

    def prepare_reducer(self, reducer: ak._reducers.Reducer) -> ak._reducers.Reducer:
        from awkward._connect.cuda import get_cuda_compute_reducer

        return get_cuda_compute_reducer(reducer)
