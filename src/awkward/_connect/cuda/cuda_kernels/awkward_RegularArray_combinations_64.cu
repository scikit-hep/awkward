// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, toindex, fromindex, n, replacement, size, length, invocation_index, err_code) = args
// 
//     # Allocate device arrays
//     scan_in_array_offsets = cupy.zeros(length + 1, dtype=cupy.int64)
// 
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
    uint64_t* err_code) {

    if (err_code[0] != NO_ERROR) return;
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_combos = scan_in_array_offsets[length];
    if (thread_id >= total_combos) return;

    // Find parent index via binary search in offsets
    int64_t parent = 0;
    int64_t low = 0, high = length;
    while (low < high) {
        int64_t mid = (low + high) / 2;
        if (scan_in_array_offsets[mid] <= thread_id)
            low = mid + 1;
        else
            high = mid;
    }
    parent = low - 1;

    scan_in_array_parents[thread_id] = parent;
    scan_in_array_local_indices[thread_id] = thread_id - scan_in_array_offsets[parent];
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

    if (err_code[0] != NO_ERROR) return;
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t offsetslength = scan_in_array_offsets[length];
    if (thread_id >= offsetslength) return;

    int64_t idx = scan_in_array_local_indices[thread_id];
    int64_t parent = scan_in_array_parents[thread_id];

    // Allocate combination array
    int64_t combo[16];  // max n = 16; adjust if needed

    int64_t remaining = idx;
    int64_t r = n;
    int64_t m = size;

    if (replacement) {
        // Unranking combinations with replacement
        for (int64_t i = 0; i < n; i++) {
            int64_t x = 0;
            while (true) {
                int64_t c = 1;
                for (int64_t k = 1; k <= r - 1; k++)
                    c = c * (m - x + k - 1) / k;
                if (c > remaining) break;
                remaining -= c;
                x++;
            }
            combo[i] = x + parent * size;
            m -= 0;  // no decrement for replacement
            r--;
        }
    } else {
        // Unranking combinations without replacement
        for (int64_t i = 0; i < n; i++) {
            int64_t x = 0;
            while (true) {
                int64_t c = 1;
                for (int64_t k = 1; k <= r - 1; k++)
                    c = c * (m - x - 1) / k;
                if (c > remaining) break;
                remaining -= c;
                x++;
            }
            combo[i] = x + parent * size;
            m -= x + 1;
            r--;
        }
    }

    // Copy to output
    for (int64_t i = 0; i < n; i++)
        tocarry[i][thread_id] = combo[i];

    toindex[0] = offsetslength;
    toindex[1] = offsetslength;
}
