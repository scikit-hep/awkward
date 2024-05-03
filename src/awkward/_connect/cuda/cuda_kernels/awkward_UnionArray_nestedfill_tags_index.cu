// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (totags, toindex, tmpstarts, tag, fromcounts, length, invocation_index, err_code) = args
//     if length > 0:
//         scan_in_array = cupy.zeros(int(tmpstarts[length -1] + fromcounts[length - 1]), dtype=cupy.int64)
//     else:
//         scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_UnionArray_nestedfill_tags_index_a", totags.dtype, toindex.dtype, tmpstarts.dtype, fromcounts.dtype]))(grid, block, (totags, toindex, tmpstarts, tag, fromcounts, length, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_UnionArray_nestedfill_tags_index_b", totags.dtype, toindex.dtype, tmpstarts.dtype, fromcounts.dtype]))(grid, block, (totags, toindex, tmpstarts, tag, fromcounts, length, scan_in_array, invocation_index, err_code))
// out["awkward_UnionArray_nestedfill_tags_index_a", {dtype_specializations}] = None
// out["awkward_UnionArray_nestedfill_tags_index_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_UnionArray_nestedfill_tags_index_a(
    T* totags,
    C* toindex,
    U* tmpstarts,
    T tag,
    const V* fromcounts,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      U start = tmpstarts[thread_id];
      V stop = start + fromcounts[thread_id];
      for (int64_t j = start;  j < stop;  j++) {
        scan_in_array[j] += 1;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_UnionArray_nestedfill_tags_index_b(
    T* totags,
    C* toindex,
    U* tmpstarts,
    T tag,
    const V* fromcounts,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      U start = tmpstarts[thread_id];
      V stop = start + fromcounts[thread_id];
      for (int64_t j = start;  j < stop;  j++) {
        totags[j] = tag;
        toindex[j] = (C)(scan_in_array[j] - 1);
      }
      tmpstarts[thread_id] = stop;
    }
  }
}
