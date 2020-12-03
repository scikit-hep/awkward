// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line)                                                         \
  FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/awkward_reduce_prod_bool.cu", \
                               line)

#include "standard_parallel_algorithms.h"
#include "awkward/kernels.h"

template <typename IN>
__global__ void
awkward_reduce_prod_bool_kernel(bool* toptr,
                          const IN* fromptr,
                          const int64_t* parents,
                          int64_t lenparents) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < lenparents) {
    toptr[parents[thread_id]] &= (fromptr[thread_id] != 0);
  }
}

__global__ void
awkward_reduce_prod_bool_initialize_toptr(bool* toptr, int64_t outlength) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < outlength) {
    toptr[thread_id] = true;
  }
}
template <typename IN>
ERROR
awkward_reduce_prod_bool(bool* toptr,
                         const IN* fromptr,
                         const int64_t* parents,
                         int64_t lenparents,
                         int64_t outlength) {
  dim3 blocks_per_grid = blocks(outlength);
  dim3 threads_per_block = threads(outlength);

  awkward_reduce_prod_bool_initialize_toptr<<<blocks_per_grid,
                                              threads_per_block>>>(toptr,
                                                                   outlength);
  blocks_per_grid = blocks(lenparents);
  threads_per_block = threads(lenparents);

  awkward_reduce_prod_bool_kernel<<<blocks_per_grid, threads_per_block>>>(
      toptr, fromptr, parents, lenparents);
    return success();
}
ERROR
awkward_reduce_prod_bool_bool_64(bool* toptr,
                                 const bool* fromptr,
                                 const int64_t* parents,
                                 int64_t lenparents,
                                 int64_t outlength) {
  return awkward_reduce_prod_bool<bool>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_bool_int8_64(bool* toptr,
                                 const int8_t* fromptr,
                                 const int64_t* parents,
                                 int64_t lenparents,
                                 int64_t outlength) {
  return awkward_reduce_prod_bool<int8_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_bool_uint8_64(bool* toptr,
                                  const uint8_t* fromptr,
                                  const int64_t* parents,
                                  int64_t lenparents,
                                  int64_t outlength) {
  return awkward_reduce_prod_bool<uint8_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_bool_int16_64(bool* toptr,
                                  const int16_t* fromptr,
                                  const int64_t* parents,
                                  int64_t lenparents,
                                  int64_t outlength) {
  return awkward_reduce_prod_bool<int16_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_bool_uint16_64(bool* toptr,
                                   const uint16_t* fromptr,
                                   const int64_t* parents,
                                   int64_t lenparents,
                                   int64_t outlength) {
  return awkward_reduce_prod_bool<uint16_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_bool_int32_64(bool* toptr,
                                  const int32_t* fromptr,
                                  const int64_t* parents,
                                  int64_t lenparents,
                                  int64_t outlength) {
  return awkward_reduce_prod_bool<int32_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_bool_uint32_64(bool* toptr,
                                   const uint32_t* fromptr,
                                   const int64_t* parents,
                                   int64_t lenparents,
                                   int64_t outlength) {
  return awkward_reduce_prod_bool<uint32_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_bool_int64_64(bool* toptr,
                                  const int64_t* fromptr,
                                  const int64_t* parents,
                                  int64_t lenparents,
                                  int64_t outlength) {
  return awkward_reduce_prod_bool<int64_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_bool_uint64_64(bool* toptr,
                                   const uint64_t* fromptr,
                                   const int64_t* parents,
                                   int64_t lenparents,
                                   int64_t outlength) {
  return awkward_reduce_prod_bool<uint64_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_bool_float32_64(bool* toptr,
                                    const float* fromptr,
                                    const int64_t* parents,
                                    int64_t lenparents,
                                    int64_t outlength) {
  return awkward_reduce_prod_bool<float>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_bool_float64_64(bool* toptr,
                                    const double* fromptr,
                                    const int64_t* parents,
                                    int64_t lenparents,
                                    int64_t outlength) {
  return awkward_reduce_prod_bool<double>(
      toptr, fromptr, parents, lenparents, outlength);
}
