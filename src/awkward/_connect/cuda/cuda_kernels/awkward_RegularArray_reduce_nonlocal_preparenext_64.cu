// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nextcarry, nextoffsets, offsets, size, length, outlength, invocation_index, err_code) = args
//     # Offsets-pipeline: derive parents from outer offsets so the kernel body
//     # (still parents-driven) runs unchanged. The outer offsets cover `length`
//     # rows total: parents = repeat(arange(outlength), per-bin counts).
//     if int(outlength) > 0 and int(length) > 0:
//         # CuPy refuses cupy.ndarray as `repeats`; use searchsorted to
//         # derive parents on-device with the same result.
//         parents = cupy.searchsorted(
//             offsets[1:int(outlength) + 1],
//             cupy.arange(int(length * size), dtype=cupy.int64),
//             side='right',
//         ).astype(cupy.int64)
//     else:
//         parents = cupy.zeros(0, dtype=cupy.int64)
//     scan_in_array = cupy.ones(length * size, dtype=cupy.int64)
//     scan_in_array = cupy.cumsum(scan_in_array)
//     # Compute nextoffsets host-side: bin-major numbering means
//     #   nextbin = bin * size + j, and each (bin, j) cell receives exactly
//     #   (offsets[bin+1] - offsets[bin]) elements (one per row in that bin).
//     # The Python reference impl matches this layout.
//     if int(outlength) > 0 and int(size) > 0:
//         per_bin_counts = (offsets[1:int(outlength) + 1] - offsets[:int(outlength)]).astype(cupy.int64)
//         per_nextbin_counts = cupy.repeat(per_bin_counts, int(size))
//         nextoffsets[0] = 0
//         nextoffsets[1:] = cupy.cumsum(per_nextbin_counts)
//     else:
//         nextoffsets[0] = 0
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_RegularArray_reduce_nonlocal_preparenext_64', nextcarry.dtype, nextoffsets.dtype, parents.dtype]))(grid, block, (nextcarry, nextoffsets, parents, size, length, scan_in_array, invocation_index, err_code))
// out["awkward_RegularArray_reduce_nonlocal_preparenext_64", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_RegularArray_reduce_nonlocal_preparenext_64(
    T* nextcarry,
    C* nextoffsets,
    const U* parents,
    int64_t size,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) / length;
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) % length;
    if (thread_id < length && thready_id < size) {
      nextcarry[scan_in_array[thready_id * length + thread_id] - 1] = thread_id * size + thready_id;
    }
  }
}
