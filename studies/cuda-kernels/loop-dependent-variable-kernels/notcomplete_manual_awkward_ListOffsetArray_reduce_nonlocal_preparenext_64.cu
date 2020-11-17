// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/awkward_ListOffsetArray_reduce_nonlocal_preparenext_64.cpp", line)

#include "standard_parallel_algorithms.h"
#include "awkward/kernels.h"

__global__
void awkward_ListOffsetArray_reduce_nonlocal_preparenext_64_initialize_distincts(
  int64_t* distincts,
  int64_t distinctlen) {

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < distinctlen) {
    distincts[thread_id] = -1;
  }
}

__global__
void awkward_ListOffsetArray_reduce_nonlocal_preparenext_64_filter_k(
  int64_t* offsetscopy,
  const int64_t* offsets,
  int8_t* filter_k,
  int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    if (offsetscopy[thread_id] < offsets[thread_id + 1]) {
      filter_k[thread_id] = 1;
    }
  }
}

ERROR awkward_ListOffsetArray_reduce_nonlocal_preparenext_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t nextlen,
  int64_t* maxnextparents,
  int64_t* distincts,
  int64_t distinctslen,
  int64_t* offsetscopy,
  const int64_t* offsets,
  int64_t length,
  const int64_t* parents,
  int64_t maxcount) {
  *maxnextparents = 0;

  dim3 blocks_per_grid = blocks(distinctslen);
  dim3 threads_per_block = threads(distinctslen);

  awkward_ListOffsetArray_reduce_nonlocal_preparenext_64_initialize_distincts<<<blocks_per_grid, threads_per_block>>>(
    distincts,
    distinctslen);

  blocks_per_grid = blocks(nextlen);
  threads_per_block = threads(nextlen);

  int8_t* f;

  HANDLE_ERROR(cudaMalloc(k_mask, sizeof(int8_t) * length * nextlen));
  HANDLE_ERROR(cudaMemset(k_mask, 0, sizeof(int8_t) * length * nextlen));
  awkward_ListOffsetArray_reduce_nonlocal_preparenext_64_k_mask(
      k_mask,
      )
  int64_t k = 0;
  while (k < nextlen) {
    int64_t j = 0;
    for (int64_t i = 0;  i < length;  i++) {
      if (offsetscopy[i] < offsets[i + 1]) {
        int64_t diff = offsetscopy[i] - offsets[i];
        int64_t parent = parents[i];

        nextcarry[k] = offsetscopy[i];
        nextparents[k] = parent*maxcount + diff;

        if (*maxnextparents < nextparents[k]) {
          *maxnextparents = nextparents[k];
        }

        if (distincts[nextparents[k]] == -1) {
          distincts[nextparents[k]] = j;
          j++;
        }

        k++;
        offsetscopy[i]++;
      } 
    }
  }
  return success();
}
