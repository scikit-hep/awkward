// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nummissing, missing, nextshifts, offsets, length, starts, outer_offsets, outlength, maxcount, nextlen, nextcarry, invocation_index, err_code) = args
//     work = outlength * maxcount
//     if work > 0:
//         grid = (divmod(work + block[0] - 1, block[0])[0],)
//         cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_a", nummissing.dtype, missing.dtype, nextshifts.dtype, offsets.dtype, starts.dtype, outer_offsets.dtype, nextcarry.dtype]))(grid, block, args)
//     if nextlen > 0:
//         grid = (divmod(nextlen + block[0] - 1, block[0])[0],)
//         cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_b", nummissing.dtype, missing.dtype, nextshifts.dtype, offsets.dtype, starts.dtype, outer_offsets.dtype, nextcarry.dtype]))(grid, block, args)
// out["awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_a", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V, typename W, typename X, typename Y>
__global__ void
awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_a(
    T* nummissing,
    C* missing,
    U* nextshifts,
    const V* offsets,
    int64_t length,
    const W* starts,
    const X* outer_offsets,
    int64_t outlength,
    int64_t maxcount,
    int64_t nextlen,
    const Y* nextcarry,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength * maxcount) {
      int64_t outer_bin = thread_id / maxcount;
      int64_t column = thread_id % maxcount;
      int64_t count_missing = 0;
      // Each thread owns one column in one group. Rows remain in CPU order.
      for (int64_t row = outer_offsets[outer_bin];
           row < outer_offsets[outer_bin + 1]; row++) {
        int64_t start = offsets[row];
        int64_t count = offsets[row + 1] - start;
        if (column >= count) {
          count_missing++;
        }
        else {
          missing[start + column] = count_missing;
        }
      }
      // The canonical scratch buffer retains only the final group's totals.
      if (outer_bin == outlength - 1) {
        nummissing[column] = count_missing;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W, typename X, typename Y>
__global__ void
awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_b(
    T* nummissing,
    C* missing,
    U* nextshifts,
    const V* offsets,
    int64_t length,
    const W* starts,
    const X* outer_offsets,
    int64_t outlength,
    int64_t maxcount,
    int64_t nextlen,
    const Y* nextcarry,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < nextlen) {
      nextshifts[thread_id] = missing[nextcarry[thread_id]];
    }
  }
}
