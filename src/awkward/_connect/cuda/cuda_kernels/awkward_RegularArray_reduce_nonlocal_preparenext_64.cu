// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nextcarry, nextoffsets, offsets, size, length, outlength, invocation_index, err_code) = args
//     # Map each of the `length` rows to its outer bin. CuPy's searchsorted
//     # runs on-device; `offsets` is monotone with offsets[0] == 0 and
//     # offsets[outlength] == length, so every row lands in a valid bin.
//     if int(length) > 0 and int(outlength) > 0:
//         row_parents = cupy.searchsorted(
//             offsets[1:int(outlength) + 1],
//             cupy.arange(int(length), dtype=cupy.int64),
//             side='right',
//         ).astype(cupy.int64)
//     else:
//         row_parents = cupy.zeros(0, dtype=cupy.int64)
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
//     if int(length) > 0 and int(size) > 0:
//         cuda_kernel_templates.get_function(fetch_specialization(['awkward_RegularArray_reduce_nonlocal_preparenext_64', nextcarry.dtype, nextoffsets.dtype, offsets.dtype]))(grid, block, (nextcarry, nextoffsets, offsets, row_parents, size, length, invocation_index, err_code))
// out["awkward_RegularArray_reduce_nonlocal_preparenext_64", {dtype_specializations}] = None
// END PYTHON

// Matches the CPU kernel / Python reference: nextcarry is grouped per
// (bin, j) cell — all entries for nextbin = bin*size + j are contiguous,
// bins outermost, columns j inside each bin, rows in bin-local order.
// For row i in bin b, the destination of column j is
//   offsets[b]*size + j*(offsets[b+1] - offsets[b]) + (i - offsets[b])
// and the carried content index is i*size + j (unchanged from the
// parents-based kernel).
template <typename T, typename C, typename U>
__global__ void
awkward_RegularArray_reduce_nonlocal_preparenext_64(
    T* nextcarry,
    C* nextoffsets,
    const U* offsets,
    const int64_t* row_parents,
    int64_t size,
    int64_t length,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) / length;
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) % length;
    if (thread_id < length && thready_id < size) {
      int64_t bin = row_parents[thread_id];
      int64_t row_start = (int64_t)offsets[bin];
      int64_t count = (int64_t)offsets[bin + 1] - row_start;
      nextcarry[row_start * size + thready_id * count + (thread_id - row_start)] =
          thread_id * size + thready_id;
    }
  }
}
