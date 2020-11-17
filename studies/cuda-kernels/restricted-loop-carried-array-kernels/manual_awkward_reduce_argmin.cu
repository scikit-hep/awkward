// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line)                                                      \
  FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/awkward_reduce_argmin.cu", \
                               line)

#include "standard_parallel_algorithms.h"
#include "awkward/kernels.h"

template <typename OUT, typename IN>
__global__ void
awkward_reduce_argmin_kernel(OUT* toptr,
                             const IN* fromptr,
                             const int64_t* parents,
                             int64_t lenparents) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < lenparents) {
    int64_t parent = parents[thread_id];
    if (toptr[parent] == -1  ||  (fromptr[thread_id] < (fromptr[toptr[parent]]))) {
      toptr[parent] = thread_id;
    }
  }
}

template <typename OUT, typename IN>
ERROR
awkward_reduce_argmin(OUT* toptr,
                      const IN* fromptr,
                      const int64_t* parents,
                      int64_t lenparents,
                      int64_t outlength) {
  HANDLE_ERROR(cudaMemset(toptr, -1, sizeof(OUT) * outlength));

  dim3 blocks_per_grid = blocks(lenparents);
  dim3 threads_per_block = threads(lenparents);

  awkward_reduce_argmin_kernel<<<blocks_per_grid, threads_per_block>>>(
      toptr, fromptr, parents, lenparents);

  return success();
}
ERROR awkward_reduce_argmin_int8_64(
    int64_t* toptr,
    const int8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength) {
  return awkward_reduce_argmin<int64_t, int8_t>(
      toptr,
      fromptr,
      parents,
      lenparents,
      outlength);
}
ERROR awkward_reduce_argmin_uint8_64(
    int64_t* toptr,
    const uint8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength) {
  return awkward_reduce_argmin<int64_t, uint8_t>(
      toptr,
      fromptr,
      parents,
      lenparents,
      outlength);
}
ERROR awkward_reduce_argmin_int16_64(
    int64_t* toptr,
    const int16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength) {
  return awkward_reduce_argmin<int64_t, int16_t>(
      toptr,
      fromptr,
      parents,
      lenparents,
      outlength);
}
ERROR awkward_reduce_argmin_uint16_64(
    int64_t* toptr,
    const uint16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength) {
  return awkward_reduce_argmin<int64_t, uint16_t>(
      toptr,
      fromptr,
      parents,
      lenparents,
      outlength);
}
ERROR awkward_reduce_argmin_int32_64(
    int64_t* toptr,
    const int32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength) {
  return awkward_reduce_argmin<int64_t, int32_t>(
      toptr,
      fromptr,
      parents,
      lenparents,
      outlength);
}
ERROR awkward_reduce_argmin_uint32_64(
    int64_t* toptr,
    const uint32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength) {
  return awkward_reduce_argmin<int64_t, uint32_t>(
      toptr,
      fromptr,
      parents,
      lenparents,
      outlength);
}
ERROR awkward_reduce_argmin_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength) {
  return awkward_reduce_argmin<int64_t, int64_t>(
      toptr,
      fromptr,
      parents,
      lenparents,
      outlength);
}
ERROR awkward_reduce_argmin_uint64_64(
    int64_t* toptr,
    const uint64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength) {
  return awkward_reduce_argmin<int64_t, uint64_t>(
      toptr,
      fromptr,
      parents,
      lenparents,
      outlength);
}
ERROR awkward_reduce_argmin_float32_64(
    int64_t* toptr,
    const float* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength) {
  return awkward_reduce_argmin<int64_t, float>(
      toptr,
      fromptr,
      parents,
      lenparents,
      outlength);
}
ERROR awkward_reduce_argmin_float64_64(
    int64_t* toptr,
    const double* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength) {
  return awkward_reduce_argmin<int64_t, double>(
      toptr,
      fromptr,
      parents,
      lenparents,
      outlength);
}
