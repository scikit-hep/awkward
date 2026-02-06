// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nummissing, missing, nextshifts, offsets, length, starts, parents, maxcount, nextlen, nextcarry, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((length + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     counts = cupy.zeros(length, dtype=cupy.int64)
//     scan_in_array = cupy.zeros(length * maxcount, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_a",
//        nummissing.dtype, missing.dtype, nextshifts.dtype, offsets.dtype, starts.dtype, parents.dtype, nextcarry.dtype]))((grid_size,), block,
//        (nummissing, missing, nextshifts, offsets, length, starts, parents, maxcount, nextlen, nextcarry, counts, scan_in_array, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_b",
//        nummissing.dtype, missing.dtype, nextshifts.dtype, offsets.dtype, starts.dtype, parents.dtype, nextcarry.dtype]))((grid_size,), block,
//        (nummissing, missing, nextshifts, offsets, length, starts, parents, maxcount, nextlen, nextcarry, counts, scan_in_array, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_c",
//        nummissing.dtype, missing.dtype, nextshifts.dtype, offsets.dtype, starts.dtype, parents.dtype, nextcarry.dtype]))((grid_size,), block,
//        (nummissing, missing, nextshifts, offsets, length, starts, parents, maxcount, nextlen, nextcarry, counts, scan_in_array, invocation_index, err_code))
//     if block[0] > 0:
//         grid_size_next = math.floor((nextlen + block[0] - 1) / block[0])
//     else:
//         grid_size_next = 1
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_d",
//         nummissing.dtype, missing.dtype, nextshifts.dtype, offsets.dtype, starts.dtype, parents.dtype, nextcarry.dtype]))((grid_size_next,), block,
//         (nextshifts, missing, nextshifts, offsets, length, starts, parents, maxcount, nextlen, nextcarry, counts, scan_in_array, invocation_index, err_code))
// out["awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_a", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_b", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_c", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_d", {dtype_specializations}] = None
// END PYTHON

template <typename T_nummissing,
	  typename T_missing,
	  typename T_nextshifts,
	  typename T_offsets,
	  typename T_starts,
	  typename T_parents,
	  typename T_nextcarry>
__global__ void awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_a(
  T_nummissing* nummissing,
  T_missing* missing,
  T_nextshifts* nextshifts,
  const T_offsets* offsets,
  int64_t length,
  const T_starts* starts,
  const T_parents* parents,
  int64_t maxcount,
  int64_t nextlen,
  const T_nextcarry* nextcarry,
  int64_t* counts,
  int64_t* scan_in_array,
  uint64_t invocation_index,
  uint64_t* err_code) {

  if (err_code[0] != NO_ERROR) {
    return;
  }

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < length) {
    int64_t start = (int64_t)offsets[thread_id];
    int64_t stop = (int64_t)offsets[thread_id + 1];
    int64_t count = stop - start;
    counts[thread_id] = count;

    // For each position j (0 to maxcount-1), mark if this item's count <= j
    // This creates a 2D logical array: scan_in_array[thread_id * maxcount + j]
    for (int64_t j = 0; j < maxcount; j++) {
      scan_in_array[thread_id * maxcount + j] = (count <= j) ? 1 : 0;
    }
  }
}

template <typename T_nummissing,
	  typename T_missing,
	  typename T_nextshifts,
	  typename T_offsets,
	  typename T_starts,
	  typename T_parents,
	  typename T_nextcarry>
__global__ void awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_b(
  T_nummissing* nummissing,
  T_missing* missing,
  T_nextshifts* nextshifts,
  const T_offsets* offsets,
  int64_t length,
  const T_starts* starts,
  const T_parents* parents,
  int64_t maxcount,
  int64_t nextlen,
  const T_nextcarry* nextcarry,
  int64_t* counts,
  int64_t* scan_in_array,
  uint64_t invocation_index,
  uint64_t* err_code) {

  if (err_code[0] != NO_ERROR) {
    return;
  }

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < maxcount) {
    int64_t j = thread_id;

    // Sequential scan for this column
    for (int64_t i = 0; i < length; i++) {
      T_parents parent_idx = parents[i];
      T_starts group_start = starts[parent_idx];

      if (i == group_start) {
        // First in group - keep value from kernel A
        // (do nothing)
      } else {
        // Accumulate from previous in same group
        int64_t prev_val = scan_in_array[(i - 1) * maxcount + j];
        int64_t curr_val = scan_in_array[i * maxcount + j];
        scan_in_array[i * maxcount + j] = prev_val + curr_val;
      }
    }
  }
}

template <typename T_nummissing,
	  typename T_missing,
	  typename T_nextshifts,
	  typename T_offsets,
	  typename T_starts,
	  typename T_parents,
	  typename T_nextcarry>
__global__ void awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_c(
  T_nummissing* nummissing,
  T_missing* missing,
  T_nextshifts* nextshifts,
  T_offsets* offsets,
  int64_t length,
  const T_starts* starts,
  const T_parents* parents,
  int64_t maxcount,
  int64_t nextlen,
  const T_nextcarry* nextcarry,
  int64_t* counts,
  int64_t* scan_in_array,
  uint64_t invocation_index,
  uint64_t* err_code) {

  if (err_code[0] != NO_ERROR) {
    return;
  }

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < length) {
    int64_t start = (int64_t)offsets[thread_id];
    int64_t count = counts[thread_id];
    int64_t parent_idx = parents[thread_id];
    int64_t group_start = starts[parent_idx];

    for (int64_t j = 0; j < count; j++) {
      int64_t num_missing;

      if (thread_id == group_start) {
        num_missing = 0;
      } else {
        int64_t inclusive_scan = scan_in_array[thread_id * maxcount + j];
        int64_t my_marker = (count <= j) ? 1 : 0;
        num_missing = inclusive_scan - my_marker;
      }

      missing[start + j] = (T_missing)num_missing;
    }
  }
}

template <typename T_nummissing,
	  typename T_missing,
	  typename T_nextshifts,
	  typename T_offsets,
	  typename T_starts,
	  typename T_parents,
	  typename T_nextcarry>
__global__ void awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64_d(
  T_nummissing* nummissing,
  T_missing* missing,
  T_nextshifts* nextshifts,
  const T_offsets* offsets,
  int64_t length,
  const T_starts* starts,
  const T_parents* parents,
  int64_t maxcount,
  int64_t nextlen,
  const T_nextcarry* nextcarry,
  int64_t* counts,
  int64_t* scan_in_array,
  uint64_t invocation_index,
  uint64_t* err_code) {

  if (err_code[0] != NO_ERROR) {
    return;
  }

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < nextlen) {
    nextshifts[thread_id] = (T_missing)missing[nextcarry[thread_id]];
  }
}
