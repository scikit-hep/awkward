// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line)          \
  FILENAME_FOR_EXCEPTIONS_CUDA( \
      "src/cuda-kernels/awkward_reduce_countnonzero.cu", line)

#include "standard_parallel_algorithms.h"
#include "awkward/kernels.h"

template <typename IN>
__global__ void
awkward_reduce_countnonzero_kernel(int64_t* toptr,
                                   const IN* fromptr,
                                   const int64_t* parents,
                                   int64_t lenparents) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < lenparents) {
    toptr[parents[thread_id]] += (fromptr[thread_id] != 0);
  }
}

template <typename IN>
ERROR
awkward_reduce_countnonzero(int64_t* toptr,
                            const IN* fromptr,
                            const int64_t* parents,
                            int64_t lenparents,
                            int64_t outlength) {
  HANDLE_ERROR(cudaMemset(toptr, 0, sizeof(int64_t) * outlength));

  dim3 blocks_per_grid = blocks(lenparents);
  dim3 threads_per_block = threads(lenparents);

  awkward_reduce_countnonzero_kernel<<<blocks_per_grid, threads_per_block>>>(
      toptr, fromptr, parents, lenparents);

  return success();
}
ERROR
awkward_reduce_countnonzero_bool_64(int64_t* toptr,
                                    const bool* fromptr,
                                    const int64_t* parents,
                                    int64_t lenparents,
                                    int64_t outlength) {
  return awkward_reduce_countnonzero<bool>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_countnonzero_int8_64(int64_t* toptr,
                                    const int8_t* fromptr,
                                    const int64_t* parents,
                                    int64_t lenparents,
                                    int64_t outlength) {
  return awkward_reduce_countnonzero<int8_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_countnonzero_uint8_64(int64_t* toptr,
                                     const uint8_t* fromptr,
                                     const int64_t* parents,
                                     int64_t lenparents,
                                     int64_t outlength) {
  return awkward_reduce_countnonzero<uint8_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_countnonzero_int16_64(int64_t* toptr,
                                     const int16_t* fromptr,
                                     const int64_t* parents,
                                     int64_t lenparents,
                                     int64_t outlength) {
  return awkward_reduce_countnonzero<int16_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_countnonzero_uint16_64(int64_t* toptr,
                                      const uint16_t* fromptr,
                                      const int64_t* parents,
                                      int64_t lenparents,
                                      int64_t outlength) {
  return awkward_reduce_countnonzero<uint16_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_countnonzero_int32_64(int64_t* toptr,
                                     const int32_t* fromptr,
                                     const int64_t* parents,
                                     int64_t lenparents,
                                     int64_t outlength) {
  return awkward_reduce_countnonzero<int32_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_countnonzero_uint32_64(int64_t* toptr,
                                      const uint32_t* fromptr,
                                      const int64_t* parents,
                                      int64_t lenparents,
                                      int64_t outlength) {
  return awkward_reduce_countnonzero<uint32_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_countnonzero_int64_64(int64_t* toptr,
                                     const int64_t* fromptr,
                                     const int64_t* parents,
                                     int64_t lenparents,
                                     int64_t outlength) {
  return awkward_reduce_countnonzero<int64_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_countnonzero_uint64_64(int64_t* toptr,
                                      const uint64_t* fromptr,
                                      const int64_t* parents,
                                      int64_t lenparents,
                                      int64_t outlength) {
  return awkward_reduce_countnonzero<uint64_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_countnonzero_float32_64(int64_t* toptr,
                                       const float* fromptr,
                                       const int64_t* parents,
                                       int64_t lenparents,
                                       int64_t outlength) {
  return awkward_reduce_countnonzero<float>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_countnonzero_float64_64(int64_t* toptr,
                                       const double* fromptr,
                                       const int64_t* parents,
                                       int64_t lenparents,
                                       int64_t outlength) {
  return awkward_reduce_countnonzero<double>(
      toptr, fromptr, parents, lenparents, outlength);
}
