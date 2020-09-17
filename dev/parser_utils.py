# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

SPEC_BLACKLIST = [
    "awkward_ListArray_combinations_step",
    "awkward_Index8_getitem_at_nowrap",
    "awkward_IndexU8_getitem_at_nowrap",
    "awkward_Index32_getitem_at_nowrap",
    "awkward_IndexU32_getitem_at_nowrap",
    "awkward_Index64_getitem_at_nowrap",
    "awkward_NumpyArraybool_getitem_at0",
    "awkward_NumpyArray8_getitem_at0",
    "awkward_NumpyArrayU8_getitem_at0",
    "awkward_NumpyArray16_getitem_at0",
    "awkward_NumpyArray32_getitem_at0",
    "awkward_NumpyArrayU32_getitem_at0",
    "awkward_NumpyArray64_getitem_at0",
    "awkward_NumpyArrayU64_getitem_at0",
    "awkward_NumpyArrayfloat32_getitem_at0",
    "awkward_NumpyArrayfloat64_getitem_at0",
    "awkward_Index8_setitem_at_nowrap",
    "awkward_IndexU8_setitem_at_nowrap",
    "awkward_Index32_setitem_at_nowrap",
    "awkward_IndexU32_setitem_at_nowrap",
    "awkward_Index64_setitem_at_nowrap",
    "awkward_regularize_rangeslice",
    "awkward_NumpyArrayU16_getitem_at0",
    "awkward_ListArray_combinations",
    "awkward_RegularArray_combinations",
    "awkward_ListOffsetArray_reduce_nonlocal_preparenext_64",
]

TEST_BLACKLIST = SPEC_BLACKLIST + [
    "awkward_ListArray_combinations",
    "awkward_RegularArray_combinations",
    "awkward_slicearray_ravel",
    "awkward_ListArray_getitem_next_range_carrylength",
    "awkward_ListArray_getitem_next_range",
    "awkward_regularize_rangeslice",
    "awkward_ListOffsetArray_rpad_and_clip_axis1",
    "awkward_ListOffsetArray_rpad_axis1",
    "awkward_UnionArray_flatten_combine",
    "awkward_UnionArray_flatten_length",
    "awkward_ListArray_getitem_jagged_numvalid",
    "awkward_NumpyArray_fill",
    "awkward_ListArray_getitem_next_range_spreadadvanced",
    "awkward_ListOffsetArray_reduce_local_nextparents_64",
    "awkward_NumpyArray_getitem_next_array_advanced",
    "awkward_ListOffsetArray_reduce_local_nextparents_64",
    "awkward_IndexedArray_overlay_mask",
    "awkward_ListOffsetArray_rpad_length_axis1",
]

SUCCESS_TEST_BLACKLIST = TEST_BLACKLIST + [
    "awkward_RegularArray_broadcast_tooffsets",
    "awkward_ListArray_validity",
    "awkward_IndexedArray_validity",
    "awkward_UnionArray_validity",
    "awkward_regularize_arrayslice",
]

PYGEN_BLACKLIST = SPEC_BLACKLIST + [
    "awkward_sorting_ranges",
    "awkward_sorting_ranges_length",
    "awkward_argsort",
    "awkward_argsort_bool",
    "awkward_argsort_int8",
    "awkward_argsort_uint8",
    "awkward_argsort_int16",
    "awkward_argsort_uint16",
    "awkward_argsort_int32",
    "awkward_argsort_uint32",
    "awkward_argsort_int64",
    "awkward_argsort_uint64",
    "awkward_argsort_float32",
    "awkward_argsort_float64",
    "awkward_sort",
    "awkward_sort_bool",
    "awkward_sort_int8",
    "awkward_sort_uint8",
    "awkward_sort_int16",
    "awkward_sort_uint16",
    "awkward_sort_int32",
    "awkward_sort_uint32",
    "awkward_sort_int64",
    "awkward_sort_uint64",
    "awkward_sort_float32",
    "awkward_sort_float64",
    "awkward_ListOffsetArray_local_preparenext_64",
    "awkward_IndexedArray_local_preparenext_64",
    "awkward_NumpyArray_sort_asstrings_uint8",
    "awkward_NumpyArray_contiguous_copy",
    "awkward_NumpyArray_contiguous_copy_64",
    "awkward_NumpyArray_getitem_next_null",
    "awkward_NumpyArray_getitem_next_null_64",
]


def indent_code(code, indent):
    finalcode = ""
    for line in code.splitlines():
        finalcode += " " * indent + line + "\n"
    return finalcode
