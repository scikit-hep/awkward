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
            "awkward_reduce_sum_bool_complex64_64",  # alias → _bool_complex
            "awkward_reduce_sum_bool_complex128_64",  # alias → _bool_complex
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
            "awkward_NumpyArray_reduce_adjust_starts_shifts_64",
            "awkward_NumpyArray_reduce_mask_ByteMaskedArray_64",
            "awkward_localindex",
            "awkward_IndexedArray_overlay_mask",
            "awkward_IndexedArray_reduce_next_64",
            "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64",
            "awkward_ByteMaskedArray_getitem_nextcarry",
            "awkward_ByteMaskedArray_numnull",
            "awkward_RegularArray_getitem_jagged_expand",
            "awkward_UnionArray_simplify_one",
            "awkward_ListArray_broadcast_tooffsets",
            "awkward_ListArray_localindex",
            "awkward_ListArray_compact_offsets",
            "awkward_ListArray_combinations_length",
            "awkward_ListArray_combinations",
            "awkward_reduce_countnonzero_complex",
            # indexing / structure
            "awkward_missing_repeat",
            "awkward_index_rpad_and_clip_axis0",
            "awkward_RegularArray_localindex",
            "awkward_RegularArray_getitem_next_range_spreadadvanced",
            "awkward_RegularArray_getitem_next_range",
            "awkward_RegularArray_getitem_next_array_advanced",
            "awkward_RegularArray_getitem_next_array",
            "awkward_index_rpad_and_clip_axis1",
            # sort
            "awkward_sort",
            # other kernels
            "awkward_RegularArray_getitem_carry",
            "awkward_NumpyArray_subrange_equal",
            "awkward_NumpyArray_pad_zero_to_length",
            "awkward_NumpyArray_subrange_equal_bool",
            "awkward_MaskedArray_getitem_next_jagged_project",
            "awkward_RegularArray_rpad_and_clip_axis1",
            "awkward_RegularArray_getitem_next_at",
            "awkward_ListOffsetArray_rpad_length_axis1",
            "awkward_ListOffsetArray_rpad_and_clip_axis1",
            "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64",
            "awkward_ListOffsetArray_local_preparenext_64",
            "awkward_ListArray_rpad_and_clip_length_axis1",
            "awkward_ListArray_min_range",
            "awkward_ListArray_getitem_next_range_counts",
            "awkward_ListArray_getitem_next_range_carrylength",
            "awkward_ListArray_fill",
            "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1",
            "awkward_IndexedArray_validity",
            "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
            "awkward_IndexedArray_reduce_next_fix_offsets_64",
            "awkward_IndexedArray_ranges_next_64",
            "awkward_IndexedArray_ranges_carry_next_64",
            "awkward_IndexedArray_flatten_nextcarry",
            "awkward_IndexedArray_numnull",
            "awkward_IndexedArray_numnull_unique_64",
            "awkward_IndexedArray_numnull_parents",
            "awkward_ListArray_getitem_jagged_carrylen",
            "awkward_ListArray_getitem_jagged_descend",
            "awkward_ListArray_getitem_jagged_numvalid",
            "awkward_IndexedArray_getitem_nextcarry_outindex",
            "awkward_IndexedArray_getitem_nextcarry",
            "awkward_IndexedArray_flatten_none2empty",
            "awkward_Index_nones_as_index",
            "awkward_Content_getitem_next_missing_jagged_getmaskstartstop",
            "awkward_ByteMaskedArray_toIndexedOptionArray",
            "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64",
            "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
            "awkward_ByteMaskedArray_overlay_mask",
            "awkward_ByteMaskedArray_getitem_nextcarry_outindex",
            "awkward_BitMaskedArray_to_IndexedOptionArray",
            "awkward_BitMaskedArray_to_ByteMaskedArray",
            "awkward_UnionArray_fillindex",
            "awkward_UnionArray_fillindex_count",
            "awkward_UnionArray_fillna",
            "awkward_UnionArray_filltags",
            "awkward_UnionArray_filltags_const",
            "awkward_UnionArray_project",
            "awkward_UnionArray_regular_index",
            "awkward_UnionArray_regular_index_getsize",
            "awkward_UnionArray_simplify",
            "awkward_UnionArray_validity",
            # flipped from CuPy raw kernels (cuda.compute migration, Phase 1)
            "awkward_IndexedArray_simplify",
            "awkward_ListArray_getitem_jagged_expand",
            "awkward_ListArray_getitem_next_array",
            "awkward_ListArray_getitem_next_array_advanced",
            "awkward_ListArray_getitem_next_range_spreadadvanced",
            "awkward_ListArray_validity",
            "awkward_ListOffsetArray_drop_none_indexes",
            "awkward_ListOffsetArray_flatten_offsets",
            "awkward_ListOffsetArray_toRegularArray",
            "awkward_RegularArray_getitem_next_array_regularize",
            "awkward_UnionArray_nestedfill_tags_index",
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
            "awkward_reduce_sum_bool_complex": cuda_compute.awkward_reduce_sum_bool_complex,
            "awkward_reduce_sum_bool_complex64_64": cuda_compute.awkward_reduce_sum_bool_complex,
            "awkward_reduce_sum_bool_complex128_64": cuda_compute.awkward_reduce_sum_bool_complex,
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
            "awkward_NumpyArray_reduce_adjust_starts_shifts_64": cuda_compute.awkward_NumpyArray_reduce_adjust_starts_shifts_64,
            "awkward_NumpyArray_reduce_mask_ByteMaskedArray_64": cuda_compute.awkward_NumpyArray_reduce_mask_ByteMaskedArray_64,
            "awkward_localindex": cuda_compute.awkward_localindex,
            "awkward_missing_repeat": cuda_compute.awkward_missing_repeat,
            "awkward_index_rpad_and_clip_axis0": cuda_compute.awkward_index_rpad_and_clip_axis0,
            "awkward_index_rpad_and_clip_axis1": cuda_compute.awkward_index_rpad_and_clip_axis1,
            "awkward_IndexedArray_overlay_mask": cuda_compute.awkward_IndexedArray_overlay_mask,
            "awkward_IndexedArray_reduce_next_64": cuda_compute.awkward_IndexedArray_reduce_next_64,
            "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64": cuda_compute.awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64,
            "awkward_ByteMaskedArray_getitem_nextcarry": cuda_compute.awkward_ByteMaskedArray_getitem_nextcarry,
            "awkward_ByteMaskedArray_numnull": cuda_compute.awkward_ByteMaskedArray_numnull,
            "awkward_RegularArray_getitem_jagged_expand": cuda_compute.awkward_RegularArray_getitem_jagged_expand,
            "awkward_UnionArray_simplify_one": cuda_compute.awkward_UnionArray_simplify_one,
            "awkward_ListArray_broadcast_tooffsets": cuda_compute.awkward_ListArray_broadcast_tooffsets,
            "awkward_ListArray_localindex": cuda_compute.awkward_ListArray_localindex,
            "awkward_ListArray_compact_offsets": cuda_compute.awkward_ListArray_compact_offsets,
            "awkward_ListArray_combinations_length": cuda_compute.awkward_ListArray_combinations_length,
            "awkward_ListArray_combinations": cuda_compute.awkward_ListArray_combinations,
            # indexing / structure
            "awkward_RegularArray_localindex": cuda_compute.awkward_RegularArray_localindex,
            "awkward_RegularArray_getitem_next_range_spreadadvanced": cuda_compute.awkward_RegularArray_getitem_next_range_spreadadvanced,
            "awkward_RegularArray_getitem_next_range": cuda_compute.awkward_RegularArray_getitem_next_range,
            "awkward_RegularArray_getitem_next_array_advanced": cuda_compute.awkward_RegularArray_getitem_next_array_advanced,
            "awkward_RegularArray_getitem_next_array": cuda_compute.awkward_RegularArray_getitem_next_array,
            "awkward_RegularArray_getitem_carry": cuda_compute.awkward_RegularArray_getitem_carry,
            "awkward_RegularArray_rpad_and_clip_axis1": cuda_compute.awkward_RegularArray_rpad_and_clip_axis1,
            "awkward_RegularArray_getitem_next_at": cuda_compute.awkward_RegularArray_getitem_next_at,
            "awkward_NumpyArray_subrange_equal": cuda_compute.awkward_NumpyArray_subrange_equal,
            "awkward_NumpyArray_pad_zero_to_length": cuda_compute.awkward_NumpyArray_pad_zero_to_length,
            "awkward_NumpyArray_subrange_equal_bool": cuda_compute.awkward_NumpyArray_subrange_equal_bool,
            "awkward_MaskedArray_getitem_next_jagged_project": cuda_compute.awkward_MaskedArray_getitem_next_jagged_project,
            "awkward_ListOffsetArray_rpad_length_axis1": cuda_compute.awkward_ListOffsetArray_rpad_length_axis1,
            "awkward_ListOffsetArray_rpad_and_clip_axis1": cuda_compute.awkward_ListOffsetArray_rpad_and_clip_axis1,
            "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64": cuda_compute.awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64,
            "awkward_ListOffsetArray_local_preparenext_64": cuda_compute.awkward_ListOffsetArray_local_preparenext_64,
            "awkward_ListArray_rpad_and_clip_length_axis1": cuda_compute.awkward_ListArray_rpad_and_clip_length_axis1,
            "awkward_ListArray_min_range": cuda_compute.awkward_ListArray_min_range,
            "awkward_ListArray_getitem_next_range_counts": cuda_compute.awkward_ListArray_getitem_next_range_counts,
            "awkward_ListArray_getitem_next_range_carrylength": cuda_compute.awkward_ListArray_getitem_next_range_carrylength,
            "awkward_ListArray_fill": cuda_compute.awkward_ListArray_fill,
            "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1": cuda_compute.awkward_IndexedOptionArray_rpad_and_clip_mask_axis1,
            "awkward_IndexedArray_validity": cuda_compute.awkward_IndexedArray_validity,
            "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64": cuda_compute.awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64,
            "awkward_IndexedArray_reduce_next_fix_offsets_64": cuda_compute.awkward_IndexedArray_reduce_next_fix_offsets_64,
            "awkward_IndexedArray_ranges_next_64": cuda_compute.awkward_IndexedArray_ranges_next_64,
            "awkward_IndexedArray_ranges_carry_next_64": cuda_compute.awkward_IndexedArray_ranges_carry_next_64,
            "awkward_IndexedArray_flatten_nextcarry": cuda_compute.awkward_IndexedArray_flatten_nextcarry,
            "awkward_IndexedArray_numnull": cuda_compute.awkward_IndexedArray_numnull,
            "awkward_IndexedArray_numnull_unique_64": cuda_compute.awkward_IndexedArray_numnull_unique_64,
            "awkward_IndexedArray_numnull_parents": cuda_compute.awkward_IndexedArray_numnull_parents,
            "awkward_ListArray_getitem_jagged_carrylen": cuda_compute.awkward_ListArray_getitem_jagged_carrylen,
            "awkward_ListArray_getitem_jagged_descend": cuda_compute.awkward_ListArray_getitem_jagged_descend,
            "awkward_ListArray_getitem_jagged_numvalid": cuda_compute.awkward_ListArray_getitem_jagged_numvalid,
            "awkward_IndexedArray_getitem_nextcarry_outindex": cuda_compute.awkward_IndexedArray_getitem_nextcarry_outindex,
            "awkward_IndexedArray_getitem_nextcarry": cuda_compute.awkward_IndexedArray_getitem_nextcarry,
            "awkward_IndexedArray_flatten_none2empty": cuda_compute.awkward_IndexedArray_flatten_none2empty,
            "awkward_Index_nones_as_index": cuda_compute.awkward_Index_nones_as_index,
            "awkward_Content_getitem_next_missing_jagged_getmaskstartstop": cuda_compute.awkward_Content_getitem_next_missing_jagged_getmaskstartstop,
            "awkward_ByteMaskedArray_toIndexedOptionArray": cuda_compute.awkward_ByteMaskedArray_toIndexedOptionArray,
            "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64": cuda_compute.awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64,
            "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64": cuda_compute.awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64,
            "awkward_ByteMaskedArray_overlay_mask": cuda_compute.awkward_ByteMaskedArray_overlay_mask,
            "awkward_ByteMaskedArray_getitem_nextcarry_outindex": cuda_compute.awkward_ByteMaskedArray_getitem_nextcarry_outindex,
            "awkward_BitMaskedArray_to_IndexedOptionArray": cuda_compute.awkward_BitMaskedArray_to_IndexedOptionArray,
            "awkward_BitMaskedArray_to_ByteMaskedArray": cuda_compute.awkward_BitMaskedArray_to_ByteMaskedArray,
            "awkward_UnionArray_fillindex": cuda_compute.awkward_UnionArray_fillindex,
            "awkward_UnionArray_fillindex_count": cuda_compute.awkward_UnionArray_fillindex_count,
            "awkward_UnionArray_fillna": cuda_compute.awkward_UnionArray_fillna,
            "awkward_UnionArray_filltags": cuda_compute.awkward_UnionArray_filltags,
            "awkward_UnionArray_filltags_const": cuda_compute.awkward_UnionArray_filltags_const,
            "awkward_UnionArray_project": cuda_compute.awkward_UnionArray_project,
            "awkward_UnionArray_regular_index": cuda_compute.awkward_UnionArray_regular_index,
            "awkward_UnionArray_regular_index_getsize": cuda_compute.awkward_UnionArray_regular_index_getsize,
            "awkward_UnionArray_simplify": cuda_compute.awkward_UnionArray_simplify,
            "awkward_UnionArray_validity": cuda_compute.awkward_UnionArray_validity,
            # flipped from CuPy raw kernels (cuda.compute migration, Phase 1)
            "awkward_IndexedArray_simplify": cuda_compute.awkward_IndexedArray_simplify,
            "awkward_ListArray_getitem_jagged_expand": cuda_compute.awkward_ListArray_getitem_jagged_expand,
            "awkward_ListArray_getitem_next_array": cuda_compute.awkward_ListArray_getitem_next_array,
            "awkward_ListArray_getitem_next_array_advanced": cuda_compute.awkward_ListArray_getitem_next_array_advanced,
            "awkward_ListArray_getitem_next_range_spreadadvanced": cuda_compute.awkward_ListArray_getitem_next_range_spreadadvanced,
            "awkward_ListArray_validity": cuda_compute.awkward_ListArray_validity,
            "awkward_ListOffsetArray_drop_none_indexes": cuda_compute.awkward_ListOffsetArray_drop_none_indexes,
            "awkward_ListOffsetArray_flatten_offsets": cuda_compute.awkward_ListOffsetArray_flatten_offsets,
            "awkward_ListOffsetArray_toRegularArray": cuda_compute.awkward_ListOffsetArray_toRegularArray,
            "awkward_RegularArray_getitem_next_array_regularize": cuda_compute.awkward_RegularArray_getitem_next_array_regularize,
            "awkward_UnionArray_nestedfill_tags_index": cuda_compute.awkward_UnionArray_nestedfill_tags_index,
        }.get(kernel_name)

    def prepare_reducer(self, reducer: ak._reducers.Reducer) -> ak._reducers.Reducer:
        from awkward._connect.cuda import get_cuda_compute_reducer

        return get_cuda_compute_reducer(reducer)
