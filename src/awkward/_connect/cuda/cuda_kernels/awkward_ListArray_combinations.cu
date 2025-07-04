// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, toindex, fromindex, n, replacement, starts, stops, length, invocation_index, err_code) = args
//     scan_in_array_offsets = cupy.zeros(length + 1, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_a", tocarry[0].dtype, toindex.dtype, fromindex.dtype, starts.dtype, stops.dtype]))(grid, block, (tocarry, toindex, fromindex, n, replacement, starts, stops, length, scan_in_array_offsets, invocation_index, err_code))
//     cupy.cumsum(scan_in_array_offsets, out = scan_in_array_offsets)
//     totallen=int(scan_in_array_offsets[length])
//     if totallen == 0:
//         return  # Nothing to do if no combinations, skip the rest
//     block_size = min(totallen, 1024)
//     grid_size = max(1, math.ceil(totallen / block_size))
//     scan_in_array_parents = cupy.zeros(totallen, dtype=cupy.int64)
//     scan_in_array_local_indices = cupy.zeros(totallen, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_b", tocarry[0].dtype, toindex.dtype, fromindex.dtype, starts.dtype, stops.dtype]))((grid_size,), (block_size,), (tocarry, toindex, fromindex, n, replacement, starts, stops, length, scan_in_array_offsets, scan_in_array_parents, scan_in_array_local_indices, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_c", tocarry[0].dtype, toindex.dtype, fromindex.dtype, starts.dtype, stops.dtype]))((grid_size,), (block_size,), (tocarry, toindex, fromindex, n, replacement, starts, stops, length, scan_in_array_offsets, scan_in_array_parents, scan_in_array_local_indices, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_d", tocarry[0].dtype, toindex.dtype, fromindex.dtype, starts.dtype, stops.dtype]))((grid_size,), (block_size,), (tocarry, toindex, fromindex, n, replacement, starts, stops, length, scan_in_array_offsets, scan_in_array_parents, scan_in_array_local_indices, invocation_index, err_code))
// out["awkward_ListArray_combinations_a", {dtype_specializations}] = None
// out["awkward_ListArray_combinations_b", {dtype_specializations}] = None
// out["awkward_ListArray_combinations_c", {dtype_specializations}] = None
// out["awkward_ListArray_combinations_d", {dtype_specializations}] = None
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
  if (err_code[0] != NO_ERROR) {
    return;
  }

  // For now only n==2 supported
  if (n != 2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      RAISE_ERROR(LISTARRAY_COMBINATIONS_ERRORS::N_NOT_IMPLEMENTED)
    }
    return;
  }

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < length) {
    int64_t counts = stops[thread_id] - starts[thread_id];
    int64_t result = replacement
                        ? counts * (counts + 1) / 2
                        : counts * (counts - 1) / 2;
    scan_in_array_offsets[thread_id + 1] = result;
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
    const int64_t* __restrict__ scan_in_array_offsets,
    int64_t* __restrict__ scan_in_array_parents,
    int64_t* scan_in_array_local_indices,
    uint64_t invocation_index,
    uint64_t* err_code) {
  long thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  long offsetslength = scan_in_array_offsets[length];

  if (thread_id >= offsetslength) return;

  // Binary search for parent index i such that
  // scan_in_array_offsets[i] <= thread_id < scan_in_array_offsets[i+1]
  long left = 0;
  long right = length;

  while (left < right) {
    long mid = (left + right) / 2;
    if (thread_id >= scan_in_array_offsets[mid + 1]) {
      left = mid + 1;
    } else if (thread_id < scan_in_array_offsets[mid]) {
      right = mid;
    } else {
      left = mid;
      break;
    }
  }

  scan_in_array_parents[thread_id] = left;
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
    const int64_t* __restrict__ scan_in_array_offsets,
    int64_t* __restrict__ scan_in_array_parents,
    int64_t* scan_in_array_local_indices,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] != NO_ERROR) {
    return;
  }

  // For now only n==2 supported
  if (n != 2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      RAISE_ERROR(LISTARRAY_COMBINATIONS_ERRORS::N_NOT_IMPLEMENTED)
    }
    return;
  }

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t offsetslength = scan_in_array_offsets[length];

  if (thread_id < offsetslength) {
    scan_in_array_local_indices[thread_id] = thread_id - scan_in_array_offsets[scan_in_array_parents[thread_id]];
  }
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ListArray_combinations_d(
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
  if (err_code[0] != NO_ERROR) {
    return;
  }

  // For now only n==2 supported
  if (n != 2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      RAISE_ERROR(LISTARRAY_COMBINATIONS_ERRORS::N_NOT_IMPLEMENTED)
    }
    return;
  }

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t offsetslength = scan_in_array_offsets[length];
  int64_t i = 0;
  int64_t j = 0;

  if (thread_id < offsetslength) {

    int64_t parent = scan_in_array_parents[thread_id];
    int64_t count = stops[parent] - starts[parent];
    int64_t local_index = scan_in_array_local_indices[thread_id];

    if (replacement) {
      int64_t b = 2 * count + 1;
      double discriminant = sqrt((double)(b * b - 8 * local_index));
      i = (int64_t)((b - discriminant) / 2);
      j = local_index + i * (i - b + 2) / 2;
    } else {
      int64_t b = 2 * count - 1;
      double discriminant = sqrt((double)(b * b - 8 * local_index));
      i = (int64_t)((b - discriminant) / 2);
      j = local_index + i * (i - b + 2) / 2 + 1;
    }

    i += starts[parent];
    j += starts[parent];

    tocarry[0][thread_id] = i;
    tocarry[1][thread_id] = j;
  }

  // Write toindex once
  if (thread_id == 0) {
    toindex[0] = offsetslength;
    toindex[1] = offsetslength;
  }
}
