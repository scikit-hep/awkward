// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cuda-kernels/operations.cuh"

extern "C" {
  __global__
  void cuda_listarray32_num64(
    int64_t* tonum,
    const int32_t* fromstarts,
    int64_t startsoffset,
    const int32_t* fromstops,
    int64_t stopsoffset
    ) {
    int thread_id = threadIdx.x;
    int32_t start = fromstarts[startsoffset + thread_id];
    int32_t stop = fromstops[stopsoffset + thread_id];
    tonum[thread_id] = (int64_t)(stop - start);
  }

  EXPORT_SYMBOL struct Error
  awkward_cuda_listarray32_num_64(
    int64_t* tonum,
    const int32_t* fromstarts,
    int64_t startsoffset,
    const int32_t* fromstops,
    int64_t stopsoffset,
    int64_t length) {
    cuda_listarray32_num64<<<1, length>>>(tonum, fromstarts, startsoffset, fromstops, stopsoffset);
    return success();
  }
}
