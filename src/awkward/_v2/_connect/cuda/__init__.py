# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os
import glob
import math
import sys

import numpy

import awkward

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
            import awkward._v2._connect.cuda._kernel_signatures

            cuda_src = f"#define ERROR_BITS {ERROR_BITS}\n#define NO_ERROR {NO_ERROR}"

            if sys.version_info.major == 3 and sys.version_info.minor < 7:
                cuda_kernels_path = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "cuda_kernels"
                )
            else:
                import importlib.resources

                cuda_kernels_path = importlib.resources.path(
                    "awkward._v2._connect.cuda", "cuda_kernels"
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
            from awkward._v2._connect.cuda._kernel_signatures import (
                kernel_specializations,
            )

            cuda_kernel_templates = cupy.RawModule(
                code=cuda_src,
                options=("--std=c++11",),
                name_expressions=list(kernel_specializations.values()),
            )
            kernel = awkward._v2._connect.cuda._kernel_signatures.by_signature(
                cuda_kernel_templates
            )

        return kernel
    else:
        raise ImportError(error_message.format("Awkward Arrays with CUDA"))


def synchronize_cuda(stream):
    cupy = import_cupy("Awkward Arrays with CUDA")

    stream.synchronize()
    invocation_index = cuda_streamptr_to_contexts[stream.ptr][0]
    contexts = cuda_streamptr_to_contexts[stream.ptr][1]

    if invocation_index != NO_ERROR:
        invoked_kernel = contexts[invocation_index // math.pow(2, ERROR_BITS)]
        cuda_streamptr_to_contexts[stream.ptr] = (
            cupy.array(NO_ERROR),
            [],
        )
        raise awkward._v2._util.error(
            ValueError(
                f"{invoked_kernel.name} raised the following error: {kernel_errors[invoked_kernel.name][invocation_index % math.pow(2, ERROR_BITS)]}"
            ),
            invoked_kernel.error_context,
        )
