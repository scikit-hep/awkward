// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON

// def f(grid, block, args):
//     """
//     Device-only two-pass launcher for combinations length with general n.

//     Parameters
//     ----------
//     grid, block : CUDA launch config
//     args : tuple
//         (totallen, tooffsets, n, replacement, starts, stops, length, invocation_index, err_code)
//         - totallen: cupy.ndarray shape (), dtype=int64 or compatible scalar buffer
//         - tooffsets: cupy.ndarray shape (length+1,), dtype matches C
//         - n: Python int
//         - replacement: Python bool
//         - starts, stops: cupy.ndarray 1D
//         - length: Python int
//         - invocation_index: Python int (uint64)
//         - err_code: cupy.ndarray shape (1,), dtype=uint64
//     """

//     (totallen, tooffsets, n, replacement, starts, stops, length, invocation_index, err_code) = args

//     scan_in_array_totallen = cupy.empty(length, dtype=cupy.int64)
//     scan_in_array_tooffsets = cupy.empty(length, dtype=cupy.int64)

//     # Pass A: compute per-list combination counts (unsummed)
//     cuda_kernel_templates.get_function(fetch_specialization([
//         "awkward_ListArray_combinations_length_a",
//         totallen.dtype, tooffsets.dtype, starts.dtype, stops.dtype
//     ]))(
//         grid, block,
//         (totallen, tooffsets, n, bool(replacement), starts, stops, length,
//          scan_in_array_totallen, scan_in_array_tooffsets,
//          invocation_index, err_code)
//     )

//     # Device-only inclusive scans, **in-place** (no new allocations):
//     # (CuPy supports 'out='; we use it to avoid extra temporaries.)
//     cupy.cumsum(scan_in_array_totallen, out=scan_in_array_totallen)
//     cupy.cumsum(scan_in_array_tooffsets, out=scan_in_array_tooffsets)

//     # Pass B: finalize totallen and tooffsets using the scans
//     cuda_kernel_templates.get_function(fetch_specialization([
//         "awkward_ListArray_combinations_length_b",
//         totallen.dtype, tooffsets.dtype, starts.dtype, stops.dtype
//     ]))(
//         grid, block,
//         (totallen, tooffsets, n, bool(replacement), starts, stops, length,
//          scan_in_array_totallen, scan_in_array_tooffsets,
//          invocation_index, err_code)
//     )

// # Register specializations (handled elsewhere by your template system):
// out["awkward_ListArray_combinations_length_a", {dtype_specializations}] = None
// out["awkward_ListArray_combinations_length_b", {dtype_specializations}] = None
// END PYTHON


// template <typename I>
// __device__ __forceinline__ I dgcd(I a, I b) {
//   // Euclidean algorithm; inputs non-negative
//   while (b != 0) {
//     I t = a % b;
//     a = b;
//     b = t;
//   }
//   return a;
// }

// template <typename I>
// __device__ __forceinline__ bool mul_will_overflow(I a, I b) {
//   // check a * b > INT64_MAX (for signed I=int64_t)
//   if (a == 0 || b == 0) return false;
//   if (a < 0 || b < 0) return true; // not expected here
//   const unsigned long long ua = (unsigned long long)a;
//   const unsigned long long ub = (unsigned long long)b;
//   const unsigned long long umax = 0x7fffffffffffffffULL;
//   return ua > umax / ub;
// }

// template <typename I>
// __device__ __forceinline__ bool binom_safe(
//     I n, I k, I& out, uint64_t* err_code) {
//   // Compute C(n,k) exactly using gcd reductions to keep intermediates small.
//   // Returns false on overflow (err_code set) or invalid inputs.
//   if (k < 0 || n < 0 || k > n) { out = 0; return true; }
//   if (k == 0 || k == n) { out = 1; return true; }
//   if (k + k > n) k = n - k;

//   I result = 1;
//   I denom = 1;

//   for (I j = 1; j <= k; ++j) {
//     I a = n - j + 1;   // numerator factor
//     I b = j;           // denominator factor

//     // Reduce with current numerator/denominator
//     I g1 = dgcd(a, b);
//     a /= g1; b /= g1;

//     // Further reduce denominator against current result
//     I g2 = dgcd(result, b);
//     result /= g2; b /= g2;

//     // b should now be 1; multiply result *= a with overflow check
//     if (mul_will_overflow<I>(result, a)) {
//         // signal overflow
//         err_code[0] = (err_code[0] < static_cast<uint64_t>(
//             ARRAY_COMBINATIONS_ERRORS::OVERFLOW_IN_COMBINATORICS))
//             ? static_cast<uint64_t>(
//                 ARRAY_COMBINATIONS_ERRORS::OVERFLOW_IN_COMBINATORICS)
//             : err_code[0];
//         out = 0;
//         return false;
//     }
//     result *= a;
//     // b must be 1 now if inputs were valid
//   }

//   out = result;
//   return true;
// }

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_ListArray_combinations_length_a(
    T* totallen,                   // unused in pass A; kept for signature parity
    C* tooffsets,                  // unused in pass A; kept for signature parity
    int64_t n,
    bool replacement,
    const U* starts,
    const V* stops,
    int64_t length,
    int64_t* scan_in_array_totallen,   // per-list lengths (pre-scan)
    int64_t* scan_in_array_tooffsets,  // per-list lengths (pre-scan)
    uint64_t invocation_index,
    uint64_t* err_code) {

  if (err_code[0] != NO_ERROR) return;

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= length) return;

  int64_t size = (int64_t)(stops[thread_id] - starts[thread_id]);
  int64_t thisn = n;

  // Handle combinations with replacement by inflating n using stars-and-bars:
  // C(size + n - 1, n)
  int64_t top_n;     // n' in C(top_n, k')
  int64_t choose_k;  // k' in C(top_n, k')
  if (replacement) {
    // if size == 0: C(n-1, n) == 0 for n>0; C(-1,0) undefined but n>=0 here
    if (size == 0) {
      // Only n==0 yields 1 combination (empty multiset)
      int64_t comb = (n == 0) ? 1 : 0;
      scan_in_array_totallen[thread_id]  = comb;
      scan_in_array_tooffsets[thread_id] = comb;
      return;
    }
    top_n   = size + n - 1;
    choose_k = n;
  } else {
    top_n   = size;
    choose_k = n;
  }

  int64_t combinationslen = 0;
  if (choose_k > top_n) {
    combinationslen = 0;
  } else {
    // binomial(top_n, choose_k)
    (void)binom_safe<int64_t>(top_n, choose_k, combinationslen, err_code);
    // if overflow flagged, combinationslen is 0
  }

  scan_in_array_totallen[thread_id]  = combinationslen;
  scan_in_array_tooffsets[thread_id] = combinationslen;
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_ListArray_combinations_length_b(
    T* totallen,
    C* tooffsets,
    int64_t n,
    bool replacement,
    const U* starts,
    const V* stops,
    int64_t length,
    const int64_t* scan_in_array_totallen,   // inclusive scan buffer
    const int64_t* scan_in_array_tooffsets,  // inclusive scan buffer
    uint64_t invocation_index,
    uint64_t* err_code) {

  if (err_code[0] != NO_ERROR) return;

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  // Total number of emitted combinations across all lists:
  if (thread_id == 0) {
    *totallen = (length > 0) ? scan_in_array_totallen[length - 1] : 0;
    tooffsets[0] = 0;
  }

  if (thread_id < length) {
    tooffsets[thread_id + 1] = scan_in_array_tooffsets[thread_id];
  }
}
