// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     """
//     (tocarry, toindex, fromindex, n, replacement, starts, stops, length, invocation_index, err_code)
//     - tocarry: sequence/array of device arrays (length >= n), each 1D of total output size
//     - toindex: device array of length >= n (receives total output length per field)
//     - fromindex: kept for signature parity
//     """
//     import math
//     (tocarry, toindex, fromindex, n, replacement, starts, stops, length, invocation_index, err_code) = args
//     # Pass A: per-list counts (offsets[0] must be 0)
//     scan_in_array_offsets = cupy.zeros(length + 1, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization([
//         "awkward_ListArray_combinations_a",
//         tocarry[0].dtype, toindex.dtype, fromindex.dtype, starts.dtype, stops.dtype
//     ]))(
//         grid, block,
//         (tocarry, toindex, fromindex, n, bool(replacement), starts, stops, length,
//          scan_in_array_offsets, invocation_index, err_code)
//     )
//     # Inclusive scan (device-only)
//     scan_in_array_offsets = cupy.cumsum(scan_in_array_offsets)
//     # Allocate parents/local_indices (device-only), sized to total outputs
//     total = int(scan_in_array_offsets[length])
//     scan_in_array_parents = cupy.zeros(total, dtype=cupy.int64)
//     scan_in_array_local_indices = cupy.zeros(total, dtype=cupy.int64)
//     # Fill parents as a run-length expansion of [0..length-1]
//     # (pure device write in a trivial loop would be another kernel; your original loop is fine)
//     for i in range(1, length + 1):
//         scan_in_array_parents[scan_in_array_offsets[i - 1]:scan_in_array_offsets[i]] = i - 1
//     # Choose launch for passes B and C
//     block_size = min(1024, total) if total > 0 else 1
//     grid_size = (total + block_size - 1)//block_size if block_size > 0 else 1
//     # Pass B: compute local ranks
//     cuda_kernel_templates.get_function(fetch_specialization([
//         "awkward_ListArray_combinations_b",
//         tocarry[0].dtype, toindex.dtype, fromindex.dtype, starts.dtype, stops.dtype
//     ]))(
//         (grid_size,), (block_size,),
//         (tocarry, toindex, fromindex, n, bool(replacement), starts, stops, length,
//          scan_in_array_offsets, scan_in_array_parents, scan_in_array_local_indices,
//          invocation_index, err_code)
//     )
//     # Pass C: unrank and write carries
//     cuda_kernel_templates.get_function(fetch_specialization([
//         "awkward_ListArray_combinations_c",
//         tocarry[0].dtype, toindex.dtype, fromindex.dtype, starts.dtype, stops.dtype
//     ]))(
//         (grid_size,), (block_size,),
//         (tocarry, toindex, fromindex, n, bool(replacement), starts, stops, length,
//          scan_in_array_offsets, scan_in_array_parents, scan_in_array_local_indices,
//          invocation_index, err_code)
//     )
// out["awkward_ListArray_combinations_a", {dtype_specializations}] = None
// out["awkward_ListArray_combinations_b", {dtype_specializations}] = None
// out["awkward_ListArray_combinations_c", {dtype_specializations}] = None
// END PYTHON


template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ListArray_combinations_a(
    T** /*tocarry*/,                 // not used in pass A
    C* /*toindex*/,                  // not used in pass A
    U* /*fromindex*/,                // not used in pass A
    int64_t n,
    bool replacement,
    const V* starts,
    const W* stops,
    int64_t length,
    int64_t* scan_in_array_offsets,  // size length+1; [0] should be 0
    uint64_t invocation_index,
    uint64_t* err_code) {

  if (err_code[0] != NO_ERROR) return;

  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= length) return;

  if (n < 0) {
    RAISE_ERROR(ARRAY_COMBINATIONS_ERRORS::N_NEGATIVE)
  }

  int64_t m = (int64_t)(stops[tid] - starts[tid]);

  int64_t count = 0;
  if (n == 0) {
    // One empty combination for any m >= 0
    count = 1;
  } else if (!replacement) {
    if (n > m) {
      count = 0;
    } else {
      if (!binom_safe<int64_t>(m, n, count, err_code)) {
        RAISE_ERROR(ARRAY_COMBINATIONS_ERRORS::OVERFLOW_IN_COMBINATORICS)
      }
    }
  } else { // with replacement
    // C(m + n - 1, n)
    if (m == 0) {
      count = (n == 0) ? 1 : 0;
    } else {
      int64_t top = m + n - 1;
      if (!binom_safe<int64_t>(top, n, count, err_code)) {
        RAISE_ERROR(ARRAY_COMBINATIONS_ERRORS::OVERFLOW_IN_COMBINATORICS)
      }
    }
  }

  scan_in_array_offsets[tid + 1] = count;
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
    int64_t* scan_in_array_offsets,   // inclusive-scanned before pass C
    int64_t* scan_in_array_parents,   // size = scan_in_array_offsets[length]
    int64_t* scan_in_array_local_indices, // same size as parents
    uint64_t invocation_index,
    uint64_t* err_code) {

  if (err_code[0] != NO_ERROR) return;

  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = scan_in_array_offsets[length];
  if (tid >= total) return;

  // parent was pre-filled on host pass (Python loop), we just compute local idx
  int64_t parent = scan_in_array_parents[tid];
  int64_t local0 = scan_in_array_offsets[parent];
  scan_in_array_local_indices[tid] = tid - local0;
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ListArray_combinations_c(
    T** tocarry,   // tocarry[0..n-1] device pointers
    C* toindex,    // length >= n
    U* fromindex,  // (kept for signature parity)
    int64_t n,
    bool replacement,
    const V* starts,
    const W* stops,
    int64_t length,
    int64_t* scan_in_array_offsets,     // inclusive-scanned
    int64_t* scan_in_array_parents,     // per-output parent
    int64_t* scan_in_array_local_indices, // per-output local rank within parent
    uint64_t invocation_index,
    uint64_t* err_code) {

  if (err_code[0] != NO_ERROR) return;

  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = scan_in_array_offsets[length];
  if (tid >= total) return;

  int64_t parent = scan_in_array_parents[tid];
  int64_t start  = (int64_t)starts[parent];
  int64_t stop   = (int64_t)stops[parent];
  int64_t m      = stop - start;

  int64_t k = scan_in_array_local_indices[tid];

  if (n == 0) {
    // nothing to write to tocarry; just set toindex[0..n-1] if any (none)
  } else {
    // unrank indices in [0, m) then shift by start
    // temp buffer on stack; n is typically small
    // NOTE: if you expect large n, consider capped array or dynamic alloc policy.
    const int MAX_N = 64; // safety cap; adjust as needed
    int64_t idxbuf_local[MAX_N];
    int64_t* idxbuf = idxbuf_local;
    if (n > MAX_N) {
      // If you need arbitrarily large n, rework with heap or split passes.
      RAISE_ERROR(ARRAY_COMBINATIONS_ERRORS::OVERFLOW_IN_COMBINATORICS)
      return;
    }

    if (!unrank_lex_general(m, n, k, replacement, idxbuf, err_code)) {
      RAISE_ERROR(ARRAY_COMBINATIONS_ERRORS::OVERFLOW_IN_COMBINATORICS)
      return;
    }

    // write each component to the corresponding carry
    for (int64_t r = 0; r < n; ++r) {
      tocarry[r][tid] = start + idxbuf[r];
    }
  }

  // advertise the produced length on each output index buffer
  // (mirrors the original behavior that wrote two entries; generalize to n)
  for (int64_t r = 0; r < n; ++r) {
    toindex[r] = total;
  }
}
