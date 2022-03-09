import cupy as cp
from pathlib import Path
import cuda_utils

cuda_source = Path(
    "/home/swish/projects/awkward-1.0/src/awkward/_v2/cuda-kernels/manual_awkward_ListArray_num.cu"
).read_text()

cuda_kernel_template = cp.RawModule(
    code=cuda_source,
    options=("--std=c++11",),
    jitify=True,
    name_expressions=[
        "cuda_ListArray_num<int32_t, int64_t>",
        "cuda_ListArray_num<uint32_t, int64_t>",
        "cuda_ListArray_num<int64_t, int64_t>",
    ],
)


def awkward_ListArray32_num_64(tonum, fromstarts, fromstops, length):
    grid = cuda_utils.calc_blocks(length)
    blocks = cuda_utils.calc_threads(length)

    cuda_kernel_template.get_function("cuda_ListArray_num<int32_t, int64_t>")(
        grid, blocks, (tonum, fromstarts, fromstops, length)
    )

    return cuda_utils.success()


def awkward_ListArrayU32_num_64(tonum, fromstarts, fromstops, length):
    grid = cuda_utils.calc_blocks(length)
    blocks = cuda_utils.calc_threads(length)

    cuda_kernel_template.get_function("cuda_ListArray_num<uint32_t, int64_t>")(
        grid, blocks, (tonum, fromstarts, fromstops, length)
    )

    return cuda_utils.success()


def awkward_ListArray64_num_64(tonum, fromstarts, fromstops, length):
    grid = cuda_utils.calc_blocks(length)
    blocks = cuda_utils.calc_threads(length)

    cuda_kernel_template.get_function("cuda_ListArray_num<int64_t, int64_t>")(
        grid, blocks, (tonum, fromstarts, fromstops, length)
    )

    return cuda_utils.success()
