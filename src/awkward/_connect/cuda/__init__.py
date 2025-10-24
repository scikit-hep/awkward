# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import glob
import math
import os

import numpy

try:
    import cupy

    error_message = None

except ModuleNotFoundError:
    cupy = None
    error_message = """to use {0}, you must install cupy:

    pip install cupy

or

    conda install -c conda-forge cupy
"""

cuda_streamptr_to_contexts = {}
kernel_errors = {}
kernel = None

ERROR_BITS = 8
NO_ERROR = numpy.iinfo(numpy.uint64).max


dtype_to_ctype = {
    numpy.bool_: "bool",
    numpy.int8: "int8_t",
    numpy.uint8: "uint8_t",
    numpy.int16: "int16_t",
    numpy.uint16: "uint16_t",
    numpy.int32: "int32_t",
    numpy.uint32: "uint32_t",
    numpy.int64: "int64_t",
    numpy.uint64: "uint64_t",
    numpy.float32: "float",
    numpy.float64: "double",
}


def fetch_specialization(keys):
    specialized_name = keys[0].replace("'", "") + "<"

    keys = keys[1:]
    for key in keys[:-1]:
        if dtype_to_ctype.get(key) is None:
            dtype_string = repr(key)
            specialized_name = specialized_name + dtype_string.split("'")[1] + "_t, "
        else:
            specialized_name = specialized_name + dtype_to_ctype[key] + ", "

    if dtype_to_ctype.get(keys[-1]) is None:
        dtype_string = repr(keys[-1])
        specialized_name = specialized_name + dtype_string.split("'")[1] + "_t>"
    else:
        specialized_name = specialized_name + dtype_to_ctype[keys[-1]] + ">"

    return specialized_name


def fetch_template_specializations(kernel_dict):
    # These cuda kernels consist of multiple kernels that don't have templated
    # specializations of the same name (e.g. '_a', '_b').
    kernel_exclusions = [
        "awkward_Index_nones_as_index",
        "awkward_ByteMaskedArray_getitem_nextcarry",
        "awkward_ByteMaskedArray_numnull",
        "awkward_IndexedArray_numnull",
        "awkward_IndexedArray_numnull_parents",
        "awkward_ByteMaskedArray_getitem_nextcarry_outindex",
        "awkward_ByteMaskedArray_reduce_next_64",
        "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64",
        "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
        "awkward_Content_getitem_next_missing_jagged_getmaskstartstop",
        "awkward_IndexedArray_flatten_nextcarry",
        "awkward_IndexedArray_flatten_none2empty",
        "awkward_IndexedArray_getitem_nextcarry",
        "awkward_IndexedArray_getitem_nextcarry_outindex",
        "awkward_IndexedArray_index_of_nulls",
        "awkward_IndexedArray_local_preparenext_64",
        "awkward_IndexedArray_ranges_next_64",
        "awkward_IndexedArray_ranges_carry_next_64",
        "awkward_IndexedArray_reduce_next_64",
        "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64",
        "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
        "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1",
        "awkward_ListArray_broadcast_tooffsets",
        "awkward_ListArray_combinations_length",
        "awkward_ListArray_combinations",
        "awkward_RegularArray_combinations_64",
        "awkward_ListArray_compact_offsets",
        "awkward_ListArray_getitem_jagged_apply",
        "awkward_ListArray_getitem_jagged_carrylen",
        "awkward_ListArray_getitem_jagged_descend",
        "awkward_ListArray_getitem_jagged_numvalid",
        "awkward_ListArray_getitem_jagged_shrink",
        "awkward_ListArray_getitem_next_range",
        "awkward_ListArray_getitem_next_range_carrylength",
        "awkward_ListArray_getitem_next_range_counts",
        "awkward_ListArray_min_range",
        "awkward_ListArray_rpad_and_clip_length_axis1",
        "awkward_ListArray_rpad_axis1",
        "awkward_ListOffsetArray_drop_none_indexes",
        "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64",
        "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",
        "awkward_ListOffsetArray_reduce_local_outoffsets_64",
        "awkward_UnionArray_regular_index",
        "awkward_ListOffsetArray_rpad_axis1",
        "awkward_ListOffsetArray_rpad_length_axis1",
        "awkward_MaskedArray_getitem_next_jagged_project",
        "awkward_NumpyArray_rearrange_shifted",
        "awkward_RecordArray_reduce_nonlocal_outoffsets_64",
        "awkward_reduce_count_64",
        "awkward_reduce_sum",
        "awkward_reduce_sum_bool",
        "awkward_reduce_sum_bool_complex",
        "awkward_reduce_sum_complex",
        "awkward_reduce_sum_int32_bool_64",
        "awkward_reduce_sum_int64_bool_64",
        "awkward_reduce_prod",
        "awkward_reduce_prod_bool",
        "awkward_reduce_prod_bool_complex",
        "awkward_reduce_prod_complex",
        "awkward_reduce_countnonzero",
        "awkward_reduce_countnonzero_complex",
        "awkward_reduce_max",
        "awkward_reduce_max_complex",
        "awkward_reduce_min",
        "awkward_reduce_min_complex",
        "awkward_reduce_argmin",
        "awkward_reduce_argmin_complex",
        "awkward_reduce_argmax",
        "awkward_reduce_argmax_complex",
        "awkward_sorting_ranges",
        "awkward_sorting_ranges_length",
        "awkward_UnionArray_flatten_length",
        "awkward_UnionArray_flatten_combine",
        "awkward_UnionArray_nestedfill_tags_index",
        "awkward_UnionArray_project",
    ]
    template_specializations = []
    import re

    for keys, value in kernel_dict.items():
        pattern = re.compile("_[a-z]$")
        if keys[0] not in kernel_exclusions:
            if value is None:
                if pattern.search(keys[0]):
                    template_specializations.append(fetch_specialization(list(keys)))
            else:
                template_specializations.append(fetch_specialization(list(keys)))

    return template_specializations


def populate_kernel_errors(kernel_name, cu_file):
    import re

    pattern_errstring = "// message:"

    result_errstring = [_.start() for _ in re.finditer(pattern_errstring, cu_file)]

    err_strings = []
    for index in result_errstring:
        error = cu_file[index:]
        error = error[error.find('"') : error.find("\n")]
        error = error.replace('"', "")
        err_strings.append(error)

    kernel_errors[kernel_name] = err_strings


class Invocation:
    def __init__(self, name, error_context):
        self._name = name
        self._error_context = error_context

    @property
    def name(self):
        return self._name

    @property
    def error_context(self):
        return self._error_context


def import_cupy(name="Awkward Arrays with CUDA"):
    if cupy is None:
        raise ModuleNotFoundError(error_message.format(name))

    return cupy


def initialize_cuda_kernels(cupy):
    if cupy is not None:
        global kernel

        if kernel is None:
            import awkward._connect.cuda._kernel_signatures

            cuda_src = f"#define ERROR_BITS {ERROR_BITS}\n#define NO_ERROR {NO_ERROR}"

            cuda_kernels_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "cuda_kernels"
            )

            with open(
                os.path.join(cuda_kernels_path, "cuda_common.cu"),
                encoding="utf-8",
            ) as header_file:
                cuda_src = cuda_src + "\n" + header_file.read()
            for filename in glob.glob(os.path.join(cuda_kernels_path, "awkward_*.cu")):
                with open(filename, encoding="utf-8") as cu_file:
                    cu_code = cu_file.read()
                    populate_kernel_errors(
                        filename[filename.find("awkward_") : filename.find(".cu")],
                        cu_code,
                    )
                    cuda_src = cuda_src + "\n" + cu_code

            # Pass an empty Raw Module to fetch all template specializations
            template_specializations = fetch_template_specializations(
                awkward._connect.cuda._kernel_signatures.by_signature(None)
            )
            cuda_kernel_templates = cupy.RawModule(
                code=cuda_src,
                options=(
                    "--std=c++11",
                    "--diag-suppress=186",
                ),
                name_expressions=template_specializations,
            )
            kernel = awkward._connect.cuda._kernel_signatures.by_signature(
                cuda_kernel_templates
            )

        return kernel
    else:
        raise ModuleNotFoundError(error_message.format("Awkward Arrays with CUDA"))


def synchronize_cuda(stream=None):
    cupy = import_cupy("Awkward Arrays with CUDA")

    if stream is None:
        stream = cupy.cuda.get_current_stream()

    stream.synchronize()

    invocation_index = cuda_streamptr_to_contexts[stream.ptr][0].get().tolist()
    contexts = cuda_streamptr_to_contexts[stream.ptr][1]

    if invocation_index != NO_ERROR:
        invoked_kernel = contexts[int(invocation_index // math.pow(2, ERROR_BITS))]
        cuda_streamptr_to_contexts[stream.ptr] = (
            cupy.array(NO_ERROR),
            [],
        )
        if invoked_kernel.error_context is None:
            raise ValueError(
                f"{kernel_errors[invoked_kernel.name][int(invocation_index % math.pow(2, ERROR_BITS))]} in compiled CUDA code ({invoked_kernel.name})"
            )
        else:
            raise invoked_kernel.error_context.decorate_exception(
                ValueError,
                ValueError(
                    f"{kernel_errors[invoked_kernel.name][int(invocation_index % math.pow(2, ERROR_BITS))]} in compiled CUDA code ({invoked_kernel.name})"
                ),
            )
