// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, toindex, fromindex, n, replacement, starts, stops, length, invocation_index, err_code) = args
//     scan_in_array_offsets = cupy.zeros(length + 1, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_a", tocarry[0].dtype, toindex.dtype, fromindex.dtype, starts.dtype, stops.dtype]))(grid, block, (tocarry, toindex, fromindex, n, replacement, starts, stops, length, scan_in_array_offsets, invocation_index, err_code))
//     scan_in_array_offsets = cupy.cumsum(scan_in_array_offsets)
//     scan_in_array_parents = cupy.zeros(int(scan_in_array_offsets[length]), dtype=cupy.int64)
//     scan_in_array_local_indices = cupy.zeros(int(scan_in_array_offsets[length]), dtype=cupy.int64)
//     for i in range(1, length + 1):
//         scan_in_array_parents[scan_in_array_offsets[i - 1]:scan_in_array_offsets[i]] = i - 1
//     if int(scan_in_array_offsets[length]) < 1024:
//         block_size = int(scan_in_array_offsets[length])
//     else:
//         block_size = 1024
//     if block_size > 0:
//         grid_size = math.floor((int(scan_in_array_offsets[length]) + block_size - 1) / block_size)
//     else:
//         grid_size = 1
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_b", tocarry[0].dtype, toindex.dtype, fromindex.dtype, starts.dtype, stops.dtype]))((grid_size,), (block_size,), (tocarry, toindex, fromindex, n, replacement, starts, stops, length, scan_in_array_offsets, scan_in_array_parents, scan_in_array_local_indices, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_c", tocarry[0].dtype, toindex.dtype, fromindex.dtype, starts.dtype, stops.dtype]))((grid_size,), (block_size,), (tocarry, toindex, fromindex, n, replacement, starts, stops, length, scan_in_array_offsets, scan_in_array_parents, scan_in_array_local_indices, invocation_index, err_code))
// out["awkward_ListArray_combinations_a", {dtype_specializations}] = None
// out["awkward_ListArray_combinations_b", {dtype_specializations}] = None
// out["awkward_ListArray_combinations_c", {dtype_specializations}] = None
// END PYTHON

enum class LISTARRAY_COMBINATIONS_ERRORS {
  N_NOT_IMPLEMENTED,  // message: "not implemented for given n"
};

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ListArray_combinations_a(
    T** tocarry,
    C* toindex,
    U* fromindex,
    int64_t n,
    bool replacement,
    const V* starts,
    const W* stops,
    int64_t length,
    int64_t* scan_in_array_offsets,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if (n != 2) {
        RAISE_ERROR(LISTARRAY_COMBINATIONS_ERRORS::N_NOT_IMPLEMENTED)
      }
      int64_t counts = stops[thread_id] - starts[thread_id];
      if (replacement) {
        scan_in_array_offsets[thread_id + 1] = counts * (counts + 1) / 2;
      } else {
        scan_in_array_offsets[thread_id + 1] = counts * (counts - 1) / 2;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ListArray_combinations_b(
    T** tocarry,
    C* toindex,
    U* fromindex,
    int64_t n,
    bool replacement,
    const V* starts,
    const W* stops,
    int64_t length,
    int64_t* scan_in_array_offsets,
    int64_t* scan_in_array_parents,
    int64_t* scan_in_array_local_indices,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t offsetslength = scan_in_array_offsets[length];

    if (thread_id < offsetslength) {
      if (n != 2) {
        RAISE_ERROR(LISTARRAY_COMBINATIONS_ERRORS::N_NOT_IMPLEMENTED)
      }
      scan_in_array_local_indices[thread_id] = thread_id - scan_in_array_offsets[scan_in_array_parents[thread_id]];
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ListArray_combinations_c(
    T** tocarry,
    C* toindex,
    U* fromindex,
    int64_t n,
    bool replacement,
    const V* starts,
    const W* stops,
    int64_t length,
    int64_t* scan_in_array_offsets,
    int64_t* scan_in_array_parents,
    int64_t* scan_in_array_local_indices,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t offsetslength = scan_in_array_offsets[length];
    int64_t i = 0;
    int64_t j = 0;

    if (thread_id < offsetslength) {
      if (n != 2) {
        RAISE_ERROR(LISTARRAY_COMBINATIONS_ERRORS::N_NOT_IMPLEMENTED)
      }

      int64_t n = stops[scan_in_array_parents[thread_id]] - starts[scan_in_array_parents[thread_id]];

      if (replacement) {
        int64_t b = 2 * n + 1;
        float discriminant = sqrtf(b * b - 8 * scan_in_array_local_indices[thread_id]);
        i = (int64_t)((b - discriminant) / 2);
        j = scan_in_array_local_indices[thread_id] + i * (i - b + 2) / 2;
      } else {
        int64_t b = 2 * n - 1;
        float discriminant = sqrtf(b * b - 8 * scan_in_array_local_indices[thread_id]);
        i = (int64_t)((b - discriminant) / 2);
        j = scan_in_array_local_indices[thread_id] + i * (i - b + 2) / 2 + 1;
      }

      i += starts[scan_in_array_parents[thread_id]];
      j += starts[scan_in_array_parents[thread_id]];

      tocarry[0][thread_id] = i;
      tocarry[1][thread_id] = j;
      toindex[0] = offsetslength;
      toindex[1] = offsetslength;
    }
  }
}
