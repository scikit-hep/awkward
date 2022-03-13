# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os
import glob

kernel_specializations = {
    "awkward_ListArray32_num_64": "cuda_ListArray_num<int32_t, int64_t>",
    "awkward_ListArrayU32_num_64": "cuda_ListArray_num<uint32_t, int64_t>",
    "awkward_ListArray64_num_64": "cuda_ListArray_num<int64_t, int64_t>",
    "awkward_RegularArray_num_64": "cuda_RegularArray_num<int64_t>",
    "awkward_BitMaskedArray_to_ByteMaskedArray": "cuda_BitMaskedArray_to_ByteMaskedArray",
}

error_codes = {0: "Success", 1: "awkward_list_array_num:Success()"}

kernel = None

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


def import_cupy(name):
    if cupy is None:
        raise ImportError(error_message.format(name))
    return cupy


def initialize_cuda_kernels(cupy):
    if cupy is not None:
        global kernel

        if kernel is None:
            import awkward._kernel_signatures_cuda

            cuda_src = ""
            for filename in glob.glob(
                os.path.join(
                    "/home/swish/projects/awkward-1.0/src/cuda-kernels", "*.cu"
                )
            ):
                with open(filename, encoding="utf-8") as cu_file:
                    cuda_src = cuda_src + "\n" + cu_file.read()

            cuda_kernel_templates = cupy.RawModule(
                code=cuda_src,
                options=("--std=c++11",),
                jitify=True,
                name_expressions=list(kernel_specializations.values()),
            )
            kernel = awkward._kernel_signatures_cuda.by_signature(
                cuda_kernel_templates, kernel_specializations
            )
        return kernel
    else:
        raise ImportError(error_message.format("Awkward Arrays with CUDA"))
