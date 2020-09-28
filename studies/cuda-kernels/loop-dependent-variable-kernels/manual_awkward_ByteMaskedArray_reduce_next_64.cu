#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/manual_awkward_ByteMaskedArray_reduce_next.cu", line)

#include "standard_parallel_algorithms.h"
#include "awkward/kernels/reducers.h"

__global__ void
awkward_ByteMaskedArray_reduce_next_64_filter_mask(
	int8_t* filtered_mask,
	const int8_t* mask,
  bool validwhen,
  int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    if ((mask[thread_id] != 0) == validwhen) {
      filtered_mask[thread_id] = 1;
    }
  }
}

__global__
void awkward_ByteMaskedArray_reduce_next_64_kernel(
	int64_t* nextcarry,
  int64_t* nextparents,
  int64_t* outindex,
  const int8_t* mask,
  const int64_t* parents,
  int64_t length,
	int8_t* filtered_mask,
	int64_t* prefixed_mask) {
	
	int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if(thread_id < length) {
		if(filtered_mask[thread_id] == 1) {
			nextcarry[prefixed_mask[thread_id] - 1] = thread_id;
			nextparents[prefixed_mask[thread_id] - 1] = parents[thread_id];
			outindex[thread_id] = prefixed_mask[thread_id] - 1;
		} else {
			outindex[thread_id] = -1;
		}
	}
}
						
	

ERROR awkward_ByteMaskedArray_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t* outindex,
  const int8_t* mask,
  const int64_t* parents,
  int64_t length,
  bool validwhen) {

	dim3 blocks_per_grid = blocks(length);
	dim3 threads_per_block = threads(length);

	int8_t* filtered_mask;
	HANDLE_ERROR(cudaMalloc((void**)&filtered_mask, sizeof(int8_t) * length));
	HANDLE_ERROR(cudaMemcpy(filtered_mask, mask, sizeof(int8_t) * length, cudaMemcpyDeviceToDevice));

	awkward_ByteMaskedArray_reduce_next_64_filter_mask<<<blocks_per_grid, threads_per_block>>>(
		filtered_mask,
		mask,
		validwhen,
		length);

	int64_t* prefixed_mask;
	HANDLE_ERROR(cudaMalloc((void**)&prefixed_mask, sizeof(int64_t) * length));

	exclusive_scan(prefixed_mask, filtered_mask, length);

	awkward_ByteMaskedArray_reduce_next_64_kernel<<<blocks_per_grid, threads_per_block>>>(
		nextcarry,
		nextparents,
		outindex,
		filtered_mask,
		parents,
		length,
		filtered_mask,
		prefixed_mask);

	cudaDeviceSynchronize();
	
  return success();
}
