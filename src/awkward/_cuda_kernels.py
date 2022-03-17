# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os
import glob
import math

import numpy

import awkward

kernel_specializations = {
    "awkward_ListArray32_num_64": "cuda_ListArray_num<int32_t, int64_t>",
    "awkward_ListArrayU32_num_64": "cuda_ListArray_num<uint32_t, int64_t>",
    "awkward_ListArray64_num_64": "cuda_ListArray_num<int64_t, int64_t>",
    "awkward_RegularArray_num_64": "cuda_RegularArray_num<int64_t>",
    "awkward_BitMaskedArray_to_ByteMaskedArray": "cuda_BitMaskedArray_to_ByteMaskedArray",
    "awkward_ListArray32_validity": "cuda_ListArray_validity<int32_t>",
    "awkward_ListArrayU32_validity": "cuda_ListArray_validity<uint32_t>",
    "awkward_ListArray64_validity": "cuda_ListArray_validity<int64_t>",
}

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
error_bits = 8

kernel_errors = {}

kernel = None


def populate_kernel_errors(kernel_name, cu_file):
    import re

    pattern_errstring = "// message:"

    result_errstring = [_.start() for _ in re.finditer(pattern_errstring, cu_file)]

    err_strings = list()
    for index in result_errstring:
        error = cu_file[index:]
        error = error[error.find('"') : error.find("\n")]
        error = error.replace('"', "")
        err_strings.append(error)

    kernel_errors[kernel_name] = err_strings


class Invocation:
    def __init__(self, name, error_context):
        self.name = name
        self.error_context = error_context


def import_cupy(name):
    if cupy is None:
        raise ImportError(error_message.format(name))
    return cupy


def initialize_cuda_kernels(cupy):
    if cupy is not None:
        global kernel

        if kernel is None:
            import awkward._kernel_signatures_cuda

            cuda_src = f"#define ERROR_BITS {error_bits}\n #define MAX_NUMPY_INT {numpy.iinfo(numpy.int64).max}"
            with open(
                "/home/swish/projects/awkward-1.0/src/cuda-kernels/cuda_common.cu",
                encoding="utf-8",
            ) as header_file:
                cuda_src = cuda_src + "\n" + header_file.read()
            for filename in glob.glob(
                os.path.join(
                    "/home/swish/projects/awkward-1.0/src/cuda-kernels", "awkward_*.cu"
                )
            ):
                with open(filename, encoding="utf-8") as cu_file:
                    cu_code = cu_file.read()
                    populate_kernel_errors(
                        filename[filename.find("awkward_") : filename.find(".cu")],
                        cu_code,
                    )
                    cuda_src = cuda_src + "\n" + cu_code

            cuda_kernel_templates = cupy.RawModule(
                code=cuda_src,
                options=("--std=c++11",),
                name_expressions=list(kernel_specializations.values()),
            )
            kernel = awkward._kernel_signatures_cuda.by_signature(
                cuda_kernel_templates, kernel_specializations
            )
        return kernel
    else:
        raise ImportError(error_message.format("Awkward Arrays with CUDA"))


def synchronize_cuda(stream):
    cupy = import_cupy("Awkward Arrays with CUDA")

    stream.synchronize()
    invocation_index = cuda_streamptr_to_contexts[stream.ptr][0]
    contexts = cuda_streamptr_to_contexts[stream.ptr][1]

    if invocation_index != numpy.iinfo(numpy.int64).max:
        invoked_kernel = contexts[invocation_index // math.pow(2, error_bits)]
        cuda_streamptr_to_contexts[stream.ptr] = (
            cupy.array([numpy.iinfo(numpy.int64).max], dtype=cupy.int64),
            list(),
        )
        raise awkward._v2._util.error(
            ValueError(
                f"{invoked_kernel.name} raised the following error: {kernel_errors[invoked_kernel.name][invocation_index % math.pow(2, error_bits)]}"
            ),
            invoked_kernel.error_context,
        )
