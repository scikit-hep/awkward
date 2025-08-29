// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, toindex, fromindex, n, replacement, size, length, invocation_index, err_code) = args
//     scan_in_array_offsets = cupy.zeros(length + 1, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_RegularArray_combinations_64_a", tocarry[0].dtype, toindex.dtype, fromindex.dtype]))(grid, block, (tocarry, toindex, fromindex, n, replacement, size, length, scan_in_array_offsets, invocation_index, err_code))
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
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_RegularArray_combinations_64_b", tocarry[0].dtype, toindex.dtype, fromindex.dtype]))((grid_size,), (block_size,), (tocarry, toindex, fromindex, n, replacement, size, length, scan_in_array_offsets, scan_in_array_parents, scan_in_array_local_indices, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_RegularArray_combinations_64_c", tocarry[0].dtype, toindex.dtype, fromindex.dtype]))((grid_size,), (block_size,), (tocarry, toindex, fromindex, n, replacement, size, length, scan_in_array_offsets, scan_in_array_parents, scan_in_array_local_indices, invocation_index, err_code))
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
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if (n != 2) {
        RAISE_ERROR(REGULARARRAY_COMBINATIONS_ERRORS::N_NOT_IMPLEMENTED)
      }
      int64_t counts = size;
      if (replacement) {
        scan_in_array_offsets[thread_id + 1] = counts * (counts + 1) / 2;
      } else {
        scan_in_array_offsets[thread_id + 1] = counts * (counts - 1) / 2;
      }
    }
  }
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
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t offsetslength = scan_in_array_offsets[length];

    if (thread_id < offsetslength) {
      if (n != 2) {
        RAISE_ERROR(REGULARARRAY_COMBINATIONS_ERRORS::N_NOT_IMPLEMENTED)
      }
      scan_in_array_local_indices[thread_id] = thread_id - scan_in_array_offsets[scan_in_array_parents[thread_id]];
    }
  }
}

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
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t offsetslength = scan_in_array_offsets[length];
    int64_t i = 0;
    int64_t j = 0;

    if (thread_id < offsetslength) {
      if (n != 2) {
        RAISE_ERROR(REGULARARRAY_COMBINATIONS_ERRORS::N_NOT_IMPLEMENTED)
      }

      if (replacement) {
        int64_t b = 2 * size + 1;
        float discriminant = sqrtf(b * b - 8 * scan_in_array_local_indices[thread_id]);
        i = (int64_t)((b - discriminant) / 2);
        j = scan_in_array_local_indices[thread_id] + i * (i - b + 2) / 2;
      } else {
        int64_t b = 2 * size - 1;
        float discriminant = sqrtf(b * b - 8 * scan_in_array_local_indices[thread_id]);
        i = (int64_t)((b - discriminant) / 2);
        j = scan_in_array_local_indices[thread_id] + i * (i - b + 2) / 2 + 1;
      }

      i += size * scan_in_array_parents[thread_id];
      j += size * scan_in_array_parents[thread_id];

      tocarry[0][thread_id] = i;
      tocarry[1][thread_id] = j;
      toindex[0] = offsetslength;
      toindex[1] = offsetslength;
    }
  }
}

// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// #BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, toindex, fromindex, n, replacement, size, length, invocation_index, err_code) = args

//     # Allocate device arrays
//     scan_in_array_offsets = cupy.zeros(length + 1, dtype=cupy.int64)

//     # --- PASS A: compute offsets ---
//     kernel_a = cuda_kernel_templates.get_function(
//         fetch_specialization([
//             "awkward_RegularArray_combinations_64_a",
//             tocarry[0].dtype, toindex.dtype, fromindex.dtype
//         ])
//     )
//     kernel_a(grid, block, (tocarry, toindex, fromindex, n, replacement,
//                             size, length, scan_in_array_offsets, invocation_index, err_code))

//     cupy.cuda.Stream.null.synchronize()
//     # Exclusive scan of offsets (device-only)
//     scan_in_array_offsets = cupy.cumsum(scan_in_array_offsets)

//     total_combinations = int(scan_in_array_offsets[length])
//     if total_combinations == 0:
//         return

//     # Allocate parent and local index arrays on device
//     scan_in_array_parents = cupy.zeros(total_combinations, dtype=cupy.int64)
//     scan_in_array_local_indices = cupy.zeros(total_combinations, dtype=cupy.int64)

//     # --- PASS B: fill parents and local indices entirely on GPU ---
//     kernel_b = cuda_kernel_templates.get_function(
//         fetch_specialization([
//             "awkward_RegularArray_combinations_64_b",
//             tocarry[0].dtype, toindex.dtype, fromindex.dtype
//         ])
//     )

//     block_size = min(1024, total_combinations)
//     grid_size = (total_combinations + block_size - 1)//block_size

//     kernel_b((grid_size,), (block_size,), (tocarry, toindex, fromindex, n, replacement,
//                                            size, length, scan_in_array_offsets,
//                                            scan_in_array_parents, scan_in_array_local_indices,
//                                            invocation_index, err_code))

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
// # Register kernel specializations as None
// out["awkward_RegularArray_combinations_64_a", {dtype_specializations}] = None
// out["awkward_RegularArray_combinations_64_b", {dtype_specializations}] = None
// out["awkward_RegularArray_combinations_64_c", {dtype_specializations}] = None
// #END PYTHON

// ================= RegularArray passes =================

// enum class ARRAY_COMBINATIONS_ERRORS {
//   N_NEGATIVE = 1,
//   OVERFLOW_IN_COMBINATORICS = 2
// };

// // Pass A: per-row counts, offsets[0] expected to be 0 on entry
// template <typename T, typename C, typename U>
// __global__ void
// awkward_RegularArray_combinations_64_a(
//     T** /*tocarry*/,           // unused in A
//     C*  /*toindex*/,           // unused in A
//     U*  /*fromindex*/,         // unused in A
//     int64_t n,
//     bool replacement,
//     int64_t size,              // fixed length per row
//     int64_t length,            // number of rows
//     int64_t* scan_in_array_offsets, // length+1; write counts at [i+1]
//     uint64_t invocation_index,
//     uint64_t* err_code) {

//   if (err_code[0] != NO_ERROR) return;

//   int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (tid >= length) return;

//   if (n < 0) {
//     RAISE_ERROR(ARRAY_COMBINATIONS_ERRORS::N_NEGATIVE)
//   }

//   const int64_t m = size;
//   int64_t count = 0;

//   if (n == 0) {
//     count = 1;
//   } else if (!replacement) {
//     if (n > m) {
//       count = 0;
//     } else {
//       (void)binom_safe<int64_t>(m, n, count, err_code);
//     }
//   } else {
//     // with replacement: C(m + n - 1, n)
//     if (m == 0) {
//       count = (n == 0) ? 1 : 0;
//     } else {
//       int64_t top = m + n - 1;
//       (void)binom_safe<int64_t>(top, n, count, err_code);
//     }
//   }

//   scan_in_array_offsets[tid + 1] = count;
// }

// // Pass B: compute local ranks; parents are prefilled on host using offsets
// template <typename T, typename C, typename U>
// __global__ void
// awkward_RegularArray_combinations_64_b(
//     T** /*tocarry*/,
//     C*  /*toindex*/,
//     U*  /*fromindex*/,
//     int64_t n,
//     bool replacement,
//     int64_t size,
//     int64_t length,
//     int64_t* scan_in_array_offsets,      // inclusive-scanned
//     int64_t* scan_in_array_parents,      // size = total outputs
//     int64_t* scan_in_array_local_indices,// size = total outputs
//     uint64_t /*invocation_index*/,
//     uint64_t* err_code) {

//   if (err_code[0] != NO_ERROR) return;

//   int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int64_t total = scan_in_array_offsets[length];
//   if (tid >= total) return;

//   int64_t parent = scan_in_array_parents[tid];
//   int64_t local0 = scan_in_array_offsets[parent];
//   scan_in_array_local_indices[tid] = tid - local0;
// }

// // Pass C: unrank and write carries
// template <typename T, typename C, typename U>
// __global__ void
// awkward_RegularArray_combinations_64_c(
//     T** tocarry,                       // tocarry[0..n-1][total]
//     C*  toindex,                       // length >= n
//     U*  /*fromindex*/,
//     int64_t n,
//     bool replacement,
//     int64_t size,                      // m
//     int64_t length,                    // rows
//     int64_t* scan_in_array_offsets,    // inclusive-scanned
//     int64_t* scan_in_array_parents,    // per-output parent row
//     int64_t* scan_in_array_local_indices, // per-output local rank in row
//     uint64_t /*invocation_index*/,
//     uint64_t* err_code) {

//   if (err_code[0] != NO_ERROR) return;

//   int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int64_t total = scan_in_array_offsets[length];
//   if (tid >= total) return;

//   const int64_t m = size;
//   int64_t k = scan_in_array_local_indices[tid];
//   int64_t parent = scan_in_array_parents[tid];

//   if (n > 0) {
//     const int MAX_N = 64;  // adjust if you expect larger n
//     if (n > MAX_N) {
//       err_code[0] = (uint64_t)ARRAY_COMBINATIONS_ERRORS::OVERFLOW_IN_COMBINATORICS;
//       return;
//     }
//     int64_t idxbuf[MAX_N];

//     if (!unrank_lex_general(m, n, k, replacement, idxbuf, err_code)) {
//       err_code[0] = (uint64_t)ARRAY_COMBINATIONS_ERRORS::OVERFLOW_IN_COMBINATORICS;
//       return;
//     }

//     // Write global flat indices: row-offset + local index
//     const int64_t base = parent * m;
//     for (int64_t r = 0; r < n; ++r) {
//       tocarry[r][tid] = (T)(base + idxbuf[r]);
//     }
//   }

//   // Publish produced length for each output index buffer (parity with original)
//   for (int64_t r = 0; r < n; ++r) {
//     toindex[r] = total;
//   }
// }
