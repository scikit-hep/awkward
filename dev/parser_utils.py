# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

SPEC_BLACKLIST = [
    "awkward_ListArray_combinations_step",
    "awkward_regularize_rangeslice",
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
    "awkward_ListOffsetArray_reduce_nonlocal_preparenext_64",
]

TEST_BLACKLIST = PYGEN_BLACKLIST + [
    "awkward_RegularArray_combinations",
    "awkward_ListArray_combinations",
    "awkward_slicearray_ravel",
    "awkward_ListArray_getitem_next_range_carrylength",
    "awkward_ListArray_getitem_next_range",
    "awkward_ListOffsetArray_rpad_and_clip_axis1",
    "awkward_ListOffsetArray_rpad_axis1",
    "awkward_UnionArray_flatten_combine",
    "awkward_UnionArray_flatten_length",
    "awkward_ListArray_getitem_jagged_numvalid",
    "awkward_NumpyArray_fill",
    "awkward_ListArray_getitem_next_range_spreadadvanced",
    "awkward_ListOffsetArray_reduce_local_nextparents_64",
    "awkward_NumpyArray_getitem_next_array_advanced",
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


def indent_code(code, indent):
    finalcode = ""
    for line in code.splitlines():
        finalcode += " " * indent + line + "\n"
    return finalcode
