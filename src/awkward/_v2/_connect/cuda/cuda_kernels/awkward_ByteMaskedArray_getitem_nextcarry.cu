// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, tocarry, mask, length, validwhen, invocation_index, err_code):
//     scan_in_array = cupy.empty(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_getitem_nextcarry_filter_mask']))(grid, block, (mask, scan_in_array, tocarry, mask, validwhen, invocation_index, err_code))
//     scan_in_array = (grid, block, scan_in_array, length, invocation_index, err_code)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_getitem_nextcarry_kernel']))(grid, block, (scan_in_array, tocarry, mask, length, invocation_index, err_code))
// END PYTHON

__global__ void
awkward_ByteMaskedArray_getitem_nextcarry_filter_mask(const int8_t* mask,
                                                      int64_t* scan_in_array,
                                                      bool validwhen,
                                                      int64_t length,
                                                      uint64_t* invocation_index,
                                                      uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == validwhen) {
        scan_in_array[thread_id] = 1;
      } else {
        scan_in_array[thread_id] = 0;
      }
    }
  }
}

__global__ void
awkward_ByteMaskedArray_getitem_nextcarry_kernel(int64_t* scan_in_array,
                                                 int64_t* to_carry,
                                                 const int8_t* mask,
                                                 bool validwhen,
                                                 int64_t length,
                                                 uint64_t* invocation_index,
                                                 uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == validwhen) {
        to_carry[scan_in_array[thread_id] - 1] = thread_id;
      }
    }
  }
}

// __device__ void
// awkward_ByteMaskedArray_getitem_nextcarry_kernel(int64_t* block_sum,
//                                                  int64_t* to_carry,
//                                                  int8_t* mask,
//                                                  bool validwhen,
//                                                  int64_t length) {
//   int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//   __shared__ int64_t mask_block[blockDim.x];
//
//   if (thread_id < length) {
//     if ((mask[thread_id] != 0) == validwhen) {
//       mask_block[thread_id] = 1;
//     }
//     __syncthreads();  // Block level synchronization
//
//     if (!threadIdx.x) {
//       void     *d_temp_storage = nullptr;
//       size_t   temp_storage_bytes = 0;
//       cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, mask_block, mask_block, blockDim.x);
//       cudaMalloc(&d_temp_storage, temp_storage_bytes);
//       cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, mask_block, mask_block, blockDim.x);
//       cudaFree(d_temp_storage);
//     }
//     __syncthreads();  // Block level synchronization
//
//     if (mask[thread_id] != 0) {
//       to_carry[(mask_block[thread_id] + block_sum[blockIdx.x]) - 1] = thread_id;
//     }
//   }
// }
//
// __device__ void
// awkward_ByteMaskedArray_getitem_nextcarry_block_sum(int64_t* block_sum,
//                                                     int8_t* mask,
//                                                     bool validwhen,
//                                                     int length, ) {
//   int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//   __shared__ int64_t mask_block[blockDim.x];
//
//   if (thread_id < length) {
//     if ((mask[thread_id] != 0) == validwhen) {
//       mask_block[thread_id] = 1;
//     }
//     __syncthreads();  // Block level synchronization
//
//     if (!threadIdx.x) {
//       void     *d_temp_storage = nullptr;
//       size_t   temp_storage_bytes = 0;
//       cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, mask_block, block_sum + blockIdx.x, blockDim.x, PlusOp, 0);
//       // Allocate temporary storage
//       cudaMalloc(&d_temp_storage, temp_storage_bytes);
//       // Run reduction
//       cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, mask_block, block_sum + blockIdx.x, blockDim.x, PlusOp, 0);
//       cudaFree(d_temp_storage);
//     }
//   }
// }
//
// __device__ void
// awkward_ByteMaskedArray_getitem_nextcarry_block_sum_scan(int64_t* block_sum) {
//   if (!blockIdx.x && !threadIdx.x) {
//     void     *d_temp_storage = nullptr;
//     size_t   temp_storage_bytes = 0;
//     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, block_sum, block_sum, gridDim.x);
//     cudaMalloc(&d_temp_storage, temp_storage_bytes);
//     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, block_sum, block_sum, gridDim.x);
//     cudaFree(d_temp_storage);
//   }
// }
//
// __device__ int64_t* get_buffer(int64_t size) {
//   __device__ int64_t block_sum[size];
//   return block_sum;
// }
//
// template <typename T, typename C>
// __global__ void
// awkward_ByteMaskedArray_getitem_nextcarry(T* tocarry,
//                                           const C* mask,
//                                           int64_t length,
//                                           bool validwhen) {
//   if (!blockIdx.x && !threadIdx.x) {
//     int64_t* block_sum = get_buffer(gridDim.x);
//     awkward_ByteMaskedArray_getitem_nextcarry_block_sum<<<
//         (gridDim.x, gridDim.y, gridDim.z),
//         (blockDim.x, blockDim.y, blockDim.z)>>>(
//         block_sum, mask, validwhen, length);
//     awkward_ByteMaskedArray_getitem_nextcarry_block_sum_scan<<<1, 1>>>(
//         block_sum);
//     awkward_ByteMaskedArray_getitem_nextcarry_kernel<<<
//         (gridDim.x, gridDim.y, gridDim.z),
//         (blockDim.x, blockDim.y, blockDim.z)>>>(
//         block_sum, to_carry, mask, validwhen, length);
//   }
// }
