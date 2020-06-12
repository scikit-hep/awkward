// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cuda-kernels/operations.cuh"
#include <stdio.h>

extern "C" {
  __global__
  void cuda_listarray8_num32(
    int32_t* tonum,
    const int8_t* fromstarts,
    int32_t startsoffset,
    const int8_t* fromstops,
    int32_t stopsoffset
    ) {
    int thread_id = threadIdx.x;
    int32_t start = fromstarts[startsoffset + thread_id];
    int32_t stop = fromstops[stopsoffset + thread_id];
    tonum[thread_id] = (int64_t)(stop - start);
  }

  void
  awkward_cuda_listarray8_num_32(
    int32_t* tonum,
    const int8_t* fromstarts,
    int32_t startsoffset,
    const int8_t* fromstops,
    int32_t stopsoffset,
    int32_t length) {
    cuda_listarray8_num32<<<1, length>>>(tonum, fromstarts, startsoffset, fromstops, stopsoffset);
  }
}
