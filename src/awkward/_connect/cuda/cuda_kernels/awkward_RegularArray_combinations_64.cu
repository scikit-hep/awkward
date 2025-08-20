// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, toindex, fromindex, n, replacement, size, length, invocation_index, err_code) = args
//
//     # Allocate device arrays
//     scan_in_array_offsets = cupy.zeros(length + 1, dtype=cupy.int64)
// 
//     print("length", length)
//     # --- PASS A: compute offsets ---
//     kernel_a = cuda_kernel_templates.get_function(
//         fetch_specialization([
//             "awkward_RegularArray_combinations_64_a",
//             tocarry[0].dtype, toindex.dtype, fromindex.dtype
//         ])
//     )
//     kernel_a(grid, block, (tocarry, toindex, fromindex, n, replacement,
//                             size, length, scan_in_array_offsets, invocation_index, err_code))
//
//     cupy.cuda.Stream.null.synchronize()
//     print("kernel_a finished")
//     # Exclusive scan of offsets (device-only)
//     scan_in_array_offsets = cupy.cumsum(scan_in_array_offsets)
//
//     total_combinations = int(scan_in_array_offsets[length])
//     if total_combinations == 0:
//         return
//
//     # Allocate parent and local index arrays on device
//     scan_in_array_parents = cupy.zeros(total_combinations, dtype=cupy.int64)
//     scan_in_array_local_indices = cupy.zeros(total_combinations, dtype=cupy.int64)
//
//     # --- PASS B: fill parents and local indices entirely on GPU ---
//     kernel_b = cuda_kernel_templates.get_function(
//         fetch_specialization([
//             "awkward_RegularArray_combinations_64_b",
//             tocarry[0].dtype, toindex.dtype, fromindex.dtype
//         ])
//     )
//
//     block_size = min(1024, total_combinations)
//     grid_size = (total_combinations + block_size - 1)//block_size
//
//     kernel_b((grid_size,), (block_size,), (tocarry, toindex, fromindex, n, replacement,
//                                            size, length, scan_in_array_offsets,
//                                            scan_in_array_parents, scan_in_array_local_indices,
//                                            invocation_index, err_code))
// 
//     # --- PASS C: compute combinations entirely on GPU ---
//     kernel_c = cuda_kernel_templates.get_function(
//         fetch_specialization([
//             "awkward_RegularArray_combinations_64_c",
//             tocarry[0].dtype, toindex.dtype, fromindex.dtype
//         ])
//     )
//     kernel_c((grid_size,), (block_size,), (tocarry, toindex, fromindex, n, replacement,
//                                            size, length, scan_in_array_offsets,
//                                            scan_in_array_parents, scan_in_array_local_indices,
//                                            invocation_index, err_code))
// 
// # Register kernel specializations as None
// out["awkward_RegularArray_combinations_64_a", {dtype_specializations}] = None
// out["awkward_RegularArray_combinations_64_b", {dtype_specializations}] = None
// out["awkward_RegularArray_combinations_64_c", {dtype_specializations}] = None
// END PYTHON

enum class REGULARARRAY_COMBINATIONS_ERRORS {
  N_NOT_IMPLEMENTED,  // message: "not implemented for given n"
};

template <typename T, typename C, typename U>
__global__ void
awkward_RegularArray_combinations_64_a(
    T** tocarry,
    C* toindex,
    U* fromindex,
    int64_t n,
    bool replacement,
    int64_t size,
    int64_t length,
    int64_t* scan_in_array_offsets,
    uint64_t invocation_index,
    uint64_t* err_code) {

    if (err_code[0] != NO_ERROR) return;
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= length) return;

    // Compute number of combinations for arbitrary n
    int64_t counts = size;
    int64_t combos = 0;

    if (replacement) {
        // n-combinations with replacement: C(n+r-1, r)
        combos = 1;
        for (int64_t k = 1; k <= n; k++)
            combos = combos * (counts + k - 1) / k;
    } else {
        // n-combinations without replacement: C(n, r)
        if (counts < n) combos = 0;
        else {
            combos = 1;
            for (int64_t k = 1; k <= n; k++)
                combos = combos * (counts - k + 1) / k;
        }
    }

    scan_in_array_offsets[thread_id + 1] = combos;
}


template <typename T, typename C, typename U>
__global__ void
awkward_RegularArray_combinations_64_b(
    T** tocarry,
    C* toindex,
    U* fromindex,
    int64_t n,
    bool replacement,
    int64_t size,
    int64_t length,
    int64_t* scan_in_array_offsets,
    int64_t* scan_in_array_parents,
    int64_t* scan_in_array_local_indices,
    uint64_t invocation_index,
    uint64_t* err_code)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= length) return;

    int64_t row_size = size; // size of this row
    int64_t offset = scan_in_array_offsets[tid];
    int64_t num_combinations = scan_in_array_offsets[tid + 1] - offset;

    // Assign parent row index for each combination
    for (int64_t i = 0; i < num_combinations; i++) {
        scan_in_array_parents[offset + i] = tid;   // this combination belongs to row `tid`
        scan_in_array_local_indices[offset + i] = i; // local index within the row
    }
}

// template <typename T, typename C, typename U>
// __global__ void
// awkward_RegularArray_combinations_64_b(
//     T** tocarry,
//     C* toindex,
//     U* fromindex,
//     int64_t n,
//     bool replacement,
//     int64_t size,
//     int64_t length,
//     int64_t* scan_in_array_offsets,
//     int64_t* scan_in_array_parents,
//     int64_t* scan_in_array_local_indices,
//     uint64_t invocation_index,
//     uint64_t* err_code) {

//     if (err_code[0] != NO_ERROR) return;
//     int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//     int64_t total_combos = scan_in_array_offsets[length];
//     if (thread_id >= total_combos) return;

//     // Find parent index via binary search in offsets
//     int64_t parent = 0;
//     int64_t low = 0, high = length;
//     while (low < high) {
//         int64_t mid = (low + high) / 2;
//         if (scan_in_array_offsets[mid] <= thread_id)
//             low = mid + 1;
//         else
//             high = mid;
//     }
//     parent = low - 1;

//     scan_in_array_parents[thread_id] = parent;
//     scan_in_array_local_indices[thread_id] = thread_id - scan_in_array_offsets[parent];
// }

// Safe unranking kernel C for RegularArray combinations (device-only).
// Assumes scan_in_array_offsets is an exclusive device scan (offsets[0]==0).
// - tocarry: array of pointers for each field (length = n)
// - toindex: index buffer (we set toindex[0] = total_combinations from thread 0)
// - scan_in_array_parents/local_indices: filled by kernel B
// - err_code: set to non-zero on error

// returns C(n,k) or cap+1 if value > cap (avoid overflow)
// __device__ inline unsigned __int128 comb_capped_u128(int64_t n, int64_t k, unsigned __int128 cap) {
//   if (k < 0 || n < 0 || k > n) return 0;
//   if (k == 0 || k == n) return 1;
//   if (k > n - k) k = n - k;
//   unsigned __int128 res = 1;
//   for (int64_t i = 1; i <= k; ++i) {
//     // numerator = n - k + i, denominator = i
//     res = (res * (unsigned __int128)(n - k + i)) / (unsigned __int128)i;
//     if (res > cap) return cap + 1;
//   }
//   return res;
// }

__device__ inline int64_t comb_capped(int64_t n, int64_t k, int64_t cap) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    int64_t res = 1;
    for (int64_t i = 1; i <= k; i++) {
        if (res > cap / (n - k + i)) {
            return cap;  // overflow capped
        }
        res = res * (n - k + i) / i;
    }
    return res;
}

// template <typename T, typename C, typename U>
// __global__ void
// awkward_RegularArray_combinations_64_c(
//     T** tocarry,
//     C* toindex,
//     U* fromindex,
//     int64_t n,
//     bool replacement,
//     int64_t size,
//     int64_t length,
//     int64_t* scan_in_array_offsets,   // exclusive scan: offsets[0]=0 ... offsets[length]=total
//     int64_t* scan_in_array_parents,   // parent for each global combination index
//     int64_t* scan_in_array_local_indices,
//     uint64_t invocation_index,
//     uint64_t* err_code) {

//   if (err_code[0] != NO_ERROR) {
//     return;
//   }

//   const int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//   const int64_t total_combos = scan_in_array_offsets[length];
//   if (thread_id >= total_combos) {
//     return;
//   }

//   // Basic guards
//   if (n <= 0) {
//     // nothing to write
//     if (thread_id == 0) {
//       toindex[0] = total_combos;
//     }
//     return;
//   }
//   const int MAX_N = 64;
//   if (n > MAX_N) {
//     // unsupported n
//     err_code[0] = (uint64_t)1; // error code 1 = N too large
//     return;
//   }

//   const int64_t idx = scan_in_array_local_indices[thread_id];
//   const int64_t parent = scan_in_array_parents[thread_id];

//   // local variables for unranking
//   int64_t combo_local[MAX_N]; // local indices within [0,size)
//   for (int64_t i = 0; i < n; ++i) combo_local[i] = 0;

//   // remaining rank within this parent's combinations
//   unsigned long long remaining = (unsigned long long) idx;

//   // r = number of picks left (remaining to choose)
//   int64_t r = n;
//   // m = effective pool size for each step: number of available distinct elements
//   // For RegularArray, each parent's block has 'size' entries, so we start with m = size.
//   int64_t m = size;

//   // To avoid repeated expensive cap values, set cap = remaining (we only need to know if count > remaining)
//   int64_t cap = remaining;

//   if (!replacement) {
//     // Unranking combinations without replacement (lexicographic).
//     int64_t r = n;          // number of picks left
//     int64_t m = size;       // remaining elements in the pool
//     int64_t offset = 0;     // offset into parent block to map to global index

//     for (int64_t i = 0; i < n; ++i) {
//         int64_t lo = 0;
//         int64_t hi = m - r; // inclusive
//         int64_t found = -1;

//         while (lo <= hi) {
//             int64_t mid = (lo + hi) >> 1;
//             int64_t choose_n = m - mid - 1;
//             int64_t choose_k = r - 1;
//             int64_t c = (choose_k <= 0) ? 1 : comb_capped(choose_n, choose_k, cap);

//             if (c > cap) {
//                 found = mid;
//                 hi = mid - 1;
//             } else {
//                 remaining = (unsigned long long)((int64_t)remaining - c);
//                 cap = (int64_t)remaining;
//                 lo = mid + 1;
//             }
//         }

//         if (found == -1) {
//             err_code[0] = (uint64_t)2; // unranking failure
//             return;
//         }

//         combo_local[i] = offset + found; // add offset to map to original pool

//         // update pool for next step
//         offset += found + 1;
//         m -= (found + 1);
//         r -= 1;
//         cap = (int64_t)remaining;
//     }
// }

//   // if (!replacement) {
//   //   // Unranking combinations without replacement (lexicographic).
//   //   // At step i: choose value x in [0 .. m - r] (the local index)
//   //   // count for choosing a particular x is C(m - x - 1, r - 1)
//   //   for (int64_t i = 0; i < n; ++i) {
//   //     // valid x range
//   //     int64_t lo = 0;
//   //     int64_t hi = m - r;         // inclusive
//   //     int64_t found = -1;

//   //     // Binary search for smallest x such that count > remaining
//   //     while (lo <= hi) {
//   //       int64_t mid = (lo + hi) >> 1;
//   //       int64_t choose_n = m - mid - 1;
//   //       int64_t choose_k = r - 1;
//   //       // if choose_k == 0: count = 1
//   //       int64_t c;
//   //       if (choose_k <= 0) {
//   //         c = 1;
//   //       } else {
//   //         c = comb_capped(choose_n, choose_k, cap);
//   //       }
//   //       if (c > cap) {
//   //         // c > remaining (cap==remaining), mid is candidate
//   //         found = mid;
//   //         hi = mid - 1;
//   //       }
//   //       else {
//   //         // subtract and continue
//   //         remaining = (unsigned long long)((int64_t)remaining - c);
//   //         cap = (int64_t)remaining;
//   //         lo = mid + 1;
//   //       }
//   //     }

//   //     if (found == -1) {
//   //       // something went wrong (shouldn't happen if offsets correct)
//   //       err_code[0] = (uint64_t)2; // unranking failure
//   //       return;
//   //     }

//   //     combo_local[i] = found;
//   //     // update m and r for next step:
//   //     // we consumed 'found' skipped elements plus the chosen one
//   //     m = m - (found + 1);
//   //     r = r - 1;
//   //     // update cap to new remaining
//   //     cap = (int64_t)remaining;
//   //   }
//   // }
//   else {
//     // Unranking combinations with replacement (non-decreasing sequences).
//     // Model as stars-and-bars: after choosing x, next choices must be >= x.
//     // At step i: choose x in [0 .. m-1] (local index relative to current origin)
//     // count for selecting mid is C((m - mid) + (r - 1) - 1, r - 1) = C(m - mid + r - 2, r - 1)
//     // We'll keep an "origin shift" so chosen x is interpreted relative to parent's block.
//     int64_t origin = 0;   // offset into the original block of size 'size'
//     int64_t cur_m = size; // remaining distinct choices from origin..size-1

//     for (int64_t i = 0; i < n; ++i) {
//       int64_t lo = 0;
//       int64_t hi = cur_m - 1;  // inclusive
//       int64_t found = -1;

//       while (lo <= hi) {
//         int64_t mid = (lo + hi) >> 1;
//         // after choosing value at origin+mid, remaining choices for r-1 picks =
//         // combinations of (cur_m - mid + r - 2) choose (r - 1)
//         int64_t comb_n = (cur_m - mid) + (r - 1) - 1; // = cur_m - mid + r - 2
//         int64_t comb_k = r - 1;
//         int64_t c;
//         if (comb_k <= 0) {
//           c = 1;
//         } else {
//           c = comb_capped(comb_n, comb_k, cap);
//         }
//         if (c > cap) {
//           found = mid;
//           hi = mid - 1;
//         } else {
//           remaining = (unsigned long long)((int64_t)remaining - c);
//           cap = (int64_t)remaining;
//           lo = mid + 1;
//         }
//       }

//       if (found == -1) {
//         err_code[0] = (uint64_t)3; // unranking-with-replacement failure
//         return;
//       }

//       // local index relative to original parent's block
//       combo_local[i] = origin + found;

//       // next picks must be >= chosen value => move origin forward, cur_m reduces
//       origin = origin + found;
//       cur_m = size - origin; // remaining distinct choices from new origin
//       r = r - 1;
//       cap = (int64_t)remaining;
//     }
//   }

//   // Write result to output arrays: combine parent base and local indices
//   // Note: tocarry[k] is expected to be preallocated with size >= total_combos
//   for (int64_t k = 0; k < n; ++k) {
//     int64_t global_value_index = parent * size + combo_local[k];
//     // write into tocarry[k][thread_id]
//     tocarry[k][thread_id] = (T)global_value_index;
//   }

//   // only thread 0 writes toindex metadata to avoid race
//   if (thread_id == 0) {
//     toindex[0] = total_combos;
//     // keep second slot for parity with other code, if used:
//     if (1) toindex[1] = total_combos;
//   }
// }
// template <typename T, typename C, typename U>
// __global__ void
// awkward_RegularArray_combinations_64_c(
//     T** tocarry,
//     C* toindex,
//     U* fromindex,
//     int64_t n,
//     bool replacement,
//     int64_t size,
//     int64_t length,
//     int64_t* scan_in_array_offsets,
//     int64_t* scan_in_array_parents,
//     int64_t* scan_in_array_local_indices,
//     uint64_t invocation_index,
//     uint64_t* err_code) 
// {
//     if (err_code[0] != NO_ERROR) return;

//     const int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//     const int64_t total_combos = scan_in_array_offsets[length];
//     if (thread_id >= total_combos) return;

//     if (n <= 0) {
//         if (thread_id == 0) toindex[0] = total_combos;
//         return;
//     }

//     const int MAX_N = 64;
//     if (n > MAX_N) {
//         err_code[0] = 1; // N too large
//         return;
//     }

//     const int64_t idx = scan_in_array_local_indices[thread_id];
//     const int64_t parent = scan_in_array_parents[thread_id];

//     // Local buffer for combination indices
//     int64_t combo_local[MAX_N] = {0};

//     unsigned long long remaining = (unsigned long long) idx;

//     if (!replacement) {
//         // Combinations without replacement (lexicographic)
//         int64_t r = n;
//         int64_t m = size;
//         int64_t offset = 0;

//         for (int64_t i = 0; i < n; ++i) {
//             int64_t lo = 0, hi = m - r, found = -1;

//             while (lo <= hi) {
//                 int64_t mid = (lo + hi) >> 1;
//                 int64_t choose_n = m - mid - 1;
//                 int64_t choose_k = r - 1;
//                 int64_t c = (choose_k <= 0) ? 1 : comb_capped(choose_n, choose_k, (int64_t)remaining);

//                 if (c > (int64_t)remaining) {
//                     found = mid;
//                     hi = mid - 1;
//                 } else {
//                     remaining -= c;
//                     lo = mid + 1;
//                 }
//             }

//             if (found == -1) {
//                 err_code[0] = 2; // unranking failure
//                 return;
//             }

//             combo_local[i] = offset + found;
//             offset += found + 1;
//             m -= (found + 1);
//             r -= 1;
//         }
//     } 
//     else {
//         // Combinations with replacement (non-decreasing)
//         int64_t origin = 0;
//         int64_t cur_m = size;
//         int64_t r = n;

//         for (int64_t i = 0; i < n; ++i) {
//             int64_t lo = 0, hi = cur_m - 1, found = -1;

//             while (lo <= hi) {
//                 int64_t mid = (lo + hi) >> 1;
//                 int64_t comb_n = cur_m - mid + r - 2;
//                 int64_t comb_k = r - 1;
//                 int64_t c = (comb_k <= 0) ? 1 : comb_capped(comb_n, comb_k, (int64_t)remaining);

//                 if (c > (int64_t)remaining) {
//                     found = mid;
//                     hi = mid - 1;
//                 } else {
//                     remaining -= c;
//                     lo = mid + 1;
//                 }
//             }

//             if (found == -1) {
//                 err_code[0] = 3; // unranking-with-replacement failure
//                 return;
//             }

//             combo_local[i] = origin + found;
//             origin += found;
//             cur_m = size - origin;
//             r -= 1;
//         }
//     }

//     // Write global combination indices
//     for (int64_t k = 0; k < n; ++k) {
//         tocarry[k][thread_id] = (T)(parent * size + combo_local[k]);
//     }

//     // Thread 0 writes metadata
//     if (thread_id == 0) {
//         toindex[0] = total_combos;
//         toindex[1] = total_combos; // keep parity with original code
//     }
//     if (thread_id < 8) {
//         printf("thread %lld combo_local:", thread_id);
//         for (int j = 0; j < n; ++j) printf(" %lld", combo_local[j]);
//         printf("\n");
//     }
// }
// template <typename T, typename C, typename U>
// __global__ void
// awkward_RegularArray_combinations_64_c(
//     T** tocarry,
//     C* toindex,
//     U* fromindex,
//     int64_t n,
//     bool replacement,
//     int64_t size,
//     int64_t length,
//     int64_t* scan_in_array_offsets,
//     int64_t* scan_in_array_parents,
//     int64_t* scan_in_array_local_indices,
//     uint64_t invocation_index,
//     uint64_t* err_code) 
// {
//     if (err_code[0] != NO_ERROR) return;

//     const int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//     const int64_t total_combos = scan_in_array_offsets[length];
//     if (thread_id >= total_combos) return;

//     if (n <= 0) {
//         if (thread_id == 0) toindex[0] = total_combos;
//         return;
//     }

//     const int MAX_N = 64;
//     if (n > MAX_N) {
//         err_code[0] = 1; // N too large
//         return;
//     }

//     const int64_t idx = scan_in_array_local_indices[thread_id];
//     const int64_t parent = scan_in_array_parents[thread_id];

//     // Local buffer for combination indices
//     int64_t combo_local[MAX_N] = {0};

//     unsigned long long remaining = (unsigned long long) idx;

//     if (!replacement) {
//         // Combinations without replacement (lexicographic)
//         int64_t r = n;
//         int64_t m = size;
//         int64_t offset = 0;

//         for (int64_t i = 0; i < n; ++i) {
//             int64_t lo = 0, hi = m - r, found = -1;

//             while (lo <= hi) {
//                 int64_t mid = (lo + hi) >> 1;
//                 int64_t choose_n = m - mid - 1;
//                 int64_t choose_k = r - 1;
//                 int64_t c = (choose_k <= 0) ? 1 : comb_capped(choose_n, choose_k, (int64_t)remaining);

//                 if (c > (int64_t)remaining) {
//                     found = mid;
//                     hi = mid - 1;
//                 } else {
//                     remaining -= c;
//                     lo = mid + 1;
//                 }
//             }

//             if (found == -1) {
//                 err_code[0] = 2; // unranking failure
//                 return;
//             }

//             // ✅ Use zero-based index
//             combo_local[i] = offset + found;  
//             offset += found;      // remove the +1 to avoid skipping 0
//             m -= (found + 1);     // keep remaining pool correct
//             r -= 1;
//         }
//     } 
//     else {
//         // Combinations with replacement (non-decreasing)
//         int64_t origin = 0;
//         int64_t cur_m = size;
//         int64_t r = n;

//         for (int64_t i = 0; i < n; ++i) {
//             int64_t lo = 0, hi = cur_m - 1, found = -1;

//             while (lo <= hi) {
//                 int64_t mid = (lo + hi) >> 1;
//                 int64_t comb_n = cur_m - mid + r - 2;
//                 int64_t comb_k = r - 1;
//                 int64_t c = (comb_k <= 0) ? 1 : comb_capped(comb_n, comb_k, (int64_t)remaining);

//                 if (c > (int64_t)remaining) {
//                     found = mid;
//                     hi = mid - 1;
//                 } else {
//                     remaining -= c;
//                     lo = mid + 1;
//                 }
//             }

//             if (found == -1) {
//                 err_code[0] = 3; // unranking-with-replacement failure
//                 return;
//             }

//             // ✅ zero-based index
//             combo_local[i] = origin + found;  
//             origin += found;  // remove +found+1
//             cur_m = size - origin;
//             r -= 1;
//         }
//     }

//     // Write global combination indices
//     for (int64_t k = 0; k < n; ++k) {
//         tocarry[k][thread_id] = (T)(parent * size + combo_local[k]);
//     }

//     // Thread 0 writes metadata
//     if (thread_id == 0) {
//         toindex[0] = total_combos;
//         toindex[1] = total_combos;
//     }

//     // Debug first 8 threads
//     if (thread_id < 8) {
//         printf("thread %lld combo_local:", thread_id);
//         for (int j = 0; j < n; ++j) printf(" %lld", combo_local[j]);
//         printf("\n");
//     }
// }
template <typename T, typename C, typename U>
__global__ void
awkward_RegularArray_combinations_64_c(
    T** tocarry,
    C* toindex,
    U* fromindex,
    int64_t n,
    bool replacement,
    int64_t size,
    int64_t length,
    int64_t* scan_in_array_offsets,
    int64_t* scan_in_array_parents,
    int64_t* scan_in_array_local_indices,
    uint64_t invocation_index,
    uint64_t* err_code) 
{
    if (err_code[0] != NO_ERROR) return;

    const int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_combos = scan_in_array_offsets[length];
    if (thread_id >= total_combos) return;

    if (n <= 0) {
        if (thread_id == 0) toindex[0] = total_combos;
        return;
    }

    const int MAX_N = 64;
    if (n > MAX_N) {
        err_code[0] = 1; // N too large
        return;
    }

    const int64_t idx = scan_in_array_local_indices[thread_id];
    const int64_t parent = scan_in_array_parents[thread_id];

    // Local buffer for combination indices
    int64_t combo_local[MAX_N] = {0};
    unsigned long long remaining = (unsigned long long) idx;

    if (!replacement) {
        // Combinations without replacement (corrected unranking)
        int64_t r = n;
        int64_t offset = 0;

        int64_t x = 0;   // current element candidate
        for (int64_t i = 0; i < n; ++i) {
            for (; x < size; ++x) {
                int64_t c = comb_capped(size - x - 1, n - i - 1, remaining);
                if (c <= remaining) {
                    remaining -= c;  // skip this element
                } else {
                    combo_local[i] = x;
                    x++;             // next candidate starts after chosen
                    break;
                }
            }
        }
        // for (int64_t i = 0; i < n; ++i) {
        //     for (int64_t j = 0; j < size - offset; ++j) {
        //         int64_t c = (r - 1 <= 0) ? 1 : comb_capped(size - offset - j - 1, r - 1, (int64_t)remaining);
        //         if (c <= remaining) {
        //             remaining -= c;
        //         } else {
        //             combo_local[i] = offset + j;
        //             offset += j + 1;
        //             r -= 1;
        //             break;
        //         }
        //     }
        // }
    } 
    else {
        // Combinations with replacement (unchanged)
        int64_t origin = 0;
        int64_t cur_m = size;
        int64_t r = n;

        for (int64_t i = 0; i < n; ++i) {
            int64_t lo = 0, hi = cur_m - 1, found = -1;

            while (lo <= hi) {
                int64_t mid = (lo + hi) >> 1;
                int64_t comb_n = cur_m - mid + r - 2;
                int64_t comb_k = r - 1;
                int64_t c = (comb_k <= 0) ? 1 : comb_capped(comb_n, comb_k, (int64_t)remaining);

                if (c > (int64_t)remaining) {
                    found = mid;
                    hi = mid - 1;
                } else {
                    remaining -= c;
                    lo = mid + 1;
                }
            }

            if (found == -1) {
                err_code[0] = 3; // unranking-with-replacement failure
                return;
            }

            combo_local[i] = origin + found;
            origin += found;
            cur_m = size - origin;
            r -= 1;
        }
    }

    // Write global combination indices
    for (int64_t k = 0; k < n; ++k) {
        tocarry[k][thread_id] = (T)(parent * size + combo_local[k]);
        printf("tocarry[k][thread_id] %lld:", parent * size + combo_local[k]);
    }

    // Thread 0 writes metadata
    if (thread_id == 0) {
        toindex[0] = total_combos;
        toindex[1] = total_combos; // keep parity with original code
    }

    // Debug printing for first 8 threads
    if (thread_id < 1) {
        printf("thread %lld combo_local:", thread_id);
        for (int j = 0; j < n; ++j) printf(" %lld", combo_local[j]);
        printf("\n");
    }
}

// template <typename T, typename C, typename U>
// __global__ void
// awkward_RegularArray_combinations_64_c(
//     T** tocarry,
//     C* toindex,
//     U* fromindex,
//     int64_t n,
//     bool replacement,
//     int64_t size,
//     int64_t length,
//     int64_t* scan_in_array_offsets,
//     int64_t* scan_in_array_parents,
//     int64_t* scan_in_array_local_indices,
//     uint64_t invocation_index,
//     uint64_t* err_code) {

//     if (err_code[0] != NO_ERROR) return;
//     int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//     int64_t offsetslength = scan_in_array_offsets[length];
//     if (thread_id >= offsetslength) return;

//     int64_t idx = scan_in_array_local_indices[thread_id];
//     int64_t parent = scan_in_array_parents[thread_id];

//     // Allocate combination array
//     int64_t combo[16];  // max n = 16; adjust if needed

//     int64_t remaining = idx;
//     int64_t r = n;
//     int64_t m = size;

//     if (replacement) {
//         // Unranking combinations with replacement
//         for (int64_t i = 0; i < n; i++) {
//             int64_t x = 0;
//             while (true) {
//                 int64_t c = 1;
//                 for (int64_t k = 1; k <= r - 1; k++)
//                     c = c * (m - x + k - 1) / k;
//                 if (c > remaining) break;
//                 remaining -= c;
//                 x++;
//             }
//             combo[i] = x + parent * size;
//             m -= 0;  // no decrement for replacement
//             r--;
//         }
//     } else {
//         // Unranking combinations without replacement
//         for (int64_t i = 0; i < n; i++) {
//             int64_t x = 0;
//             while (true) {
//                 int64_t c = 1;
//                 for (int64_t k = 1; k <= r - 1; k++)
//                     c = c * (m - x - 1) / k;
//                 if (c > remaining) break;
//                 remaining -= c;
//                 x++;
//             }
//             combo[i] = x + parent * size;
//             m -= x + 1;
//             r--;
//         }
//     }

//     // Copy to output
//     for (int64_t i = 0; i < n; i++)
//         tocarry[i][thread_id] = combo[i];

//     toindex[0] = offsetslength;
//     toindex[1] = offsetslength;
// }
