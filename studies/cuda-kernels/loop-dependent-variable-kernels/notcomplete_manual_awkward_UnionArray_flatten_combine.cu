#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/manual_awkward_Content_getitem_next_missing_jagged_getmaskstartstop.cu", line)

#include "awkward/kernels.h"
#include "standard_parallel_algorithms.h"
 
template <typename FROMTAGS,
          typename FROMINDEX,
          typename T>
__global__
void awkward_UnionArray_flatten_combine_offsets(
  TOINDEX* toindex,
  T* tooffsets,
  const FROMTAGS* fromtags,
  const FROMINDEX* fromindex,
  int64_t length,
  T** offsetsraws) {

 int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < length) {
		FROMTAGS tag = fromtags[thread_id];
    FROMINDEX idx = fromindex[thread_id];
    T start = offsetsraws[tag][idx];
    T stop = offsetsraws[tag][idx + 1];
		tooffsets[thread_id] = stop - start;
	}
}


template <typename FROMTAGS,
          typename FROMINDEX,
          typename TOTAGS,
          typename TOINDEX,
          typename T>
__global__
ERROR awkward_UnionArray_flatten_combine(
  TOTAGS* totags,
  TOINDEX* toindex,
  T* tooffsets,
  const FROMTAGS* fromtags,
  const FROMINDEX* fromindex,
  int64_t length) {
		int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

		if(thread_id < length) {
			FROMTAGS tag = fromtags[thread_id];
    	FROMINDEX idx = fromindex[thread_id];
      T start = offsetsraws[tag][idx];
      T stop = offsetsraws[tag][idx + 1];

			if(


  int64_t* h_mask = new int64_t[length];

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMalloc((void**)&filtered_index, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMemcpy(
      filtered_index, index_in, sizeof(int64_t) * length, cudaMemcpyDeviceToDevice));

  awkward_Content_getitem_next_missing_jagged_getmaskstartstop_filter_mask<<<
      blocks_per_grid,
      threads_per_block>>>(index_in, filtered_index, length);


  exclusive_scan<int64_t, int64_t>(res_temp, filtered_index, length);


  awkward_Content_getitem_next_missing_jagged_getmaskstartstop_kernel<<<blocks_per_grid,
                                                                        threads_per_block>>>(
      res_temp, index_in, offsets_in, mask_out, starts_out, stops_out, length);

  cudaDeviceSynchronize();

  return success();
}
template <typename FROMTAGS,
          typename FROMINDEX,
          typename TOTAGS,
          typename TOINDEX,
          typename T>
ERROR awkward_UnionArray_flatten_combine(
  TOTAGS* totags,
  TOINDEX* toindex,
  T* tooffsets,
  const FROMTAGS* fromtags,
  const FROMINDEX* fromindex,
  int64_t length,
  T** offsetsraws) {
	dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

	awkward_UnionArray_flatten_combine_offsets<FROMTAGS, FROMINDEX, T><<<blocks_per_grid, threads_per_block>>>(
    tooffsets,
    fromtags,
    fromindex,
    length,
    offsetsraws);

	exclusive_scan<T, T>(tooffsets, tooffsets, length);

	awkward_UnionArray_flatten_combine_kernel<<<blocks_per_grid, threads_per_block>>>(
			tooffsets,
			fromtags,
			fromindex,
			length);
  for (int64_t i = 0;  i < length;  i++) {
    FROMTAGS tag = fromtags[i];
    FROMINDEX idx = fromindex[i];
    T start = offsetsraws[tag][idx];
    T stop = offsetsraws[tag][idx + 1];
    tooffsets[i + 1] = tooffsets[i] + (stop - start);
    for (int64_t j = start;  j < stop;  j++) {
      totags[k] = tag;
      toindex[k] = j;
      k++;
    }
  }
  return success();
}
ERROR awkward_UnionArray32_flatten_combine_64(
  int8_t* totags,
  int64_t* toindex,
  int64_t* tooffsets,
  const int8_t* fromtags,
  const int32_t* fromindex,
  int64_t length,
  int64_t** offsetsraws) {
  return awkward_UnionArray_flatten_combine<int8_t, int32_t, int8_t, int64_t, int64_t>(
    totags,
    toindex,
    tooffsets,
    fromtags,
    fromindex,
    length,
    offsetsraws);
}
ERROR awkward_UnionArrayU32_flatten_combine_64(
  int8_t* totags,
  int64_t* toindex,
  int64_t* tooffsets,
  const int8_t* fromtags,
  const uint32_t* fromindex,
  int64_t length,
  int64_t** offsetsraws) {
  return awkward_UnionArray_flatten_combine<int8_t, uint32_t, int8_t, int64_t, int64_t>(
    totags,
    toindex,
    tooffsets,
    fromtags,
    fromindex,
    length,
    offsetsraws);
}
ERROR awkward_UnionArray64_flatten_combine_64(
  int8_t* totags,
  int64_t* toindex,
  int64_t* tooffsets,
  const int8_t* fromtags,
  const int64_t* fromindex,
  int64_t length,
  int64_t** offsetsraws) {
  return awkward_UnionArray_flatten_combine<int8_t, int64_t, int8_t, int64_t, int64_t>(
    totags,
    toindex,
    tooffsets,
    fromtags,
    fromindex,
    length,
    offsetsraws);
}

