// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) \
  FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/awkward_reduce_prod.cu", line)

#include "standard_parallel_algorithms.h"
#include "awkward/kernels.h"

template <typename OUT, typename IN>
__global__ void
awkward_reduce_prod_kernel(OUT* toptr,
                          const IN* fromptr,
                          const int64_t* parents,
                          int64_t lenparents) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < lenparents) {
    toptr[parents[thread_id]] *= (OUT)fromptr[thread_id];
  }
}

template <typename OUT, typename IN>
ERROR
awkward_reduce_prod(OUT* toptr,
                   const IN* fromptr,
                   const int64_t* parents,
                   int64_t lenparents,
                   int64_t outlength) {
  HANDLE_ERROR(cudaMemset(toptr, 1, sizeof(OUT) * outlength));

  dim3 blocks_per_grid = blocks(lenparents);
  dim3 threads_per_block = threads(lenparents);

  awkward_reduce_prod_kernel<<<blocks_per_grid, threads_per_block>>>(
      toptr, fromptr, parents, lenparents);

  return success();
}
ERROR
awkward_reduce_prod_int64_int8_64(int64_t* toptr,
                                 const int8_t* fromptr,
                                 const int64_t* parents,
                                 int64_t lenparents,
                                 int64_t outlength) {
  return awkward_reduce_prod<int64_t, int8_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_uint64_uint8_64(uint64_t* toptr,
                                   const uint8_t* fromptr,
                                   const int64_t* parents,
                                   int64_t lenparents,
                                   int64_t outlength) {
  return awkward_reduce_prod<uint64_t, uint8_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_int64_int16_64(int64_t* toptr,
                                  const int16_t* fromptr,
                                  const int64_t* parents,
                                  int64_t lenparents,
                                  int64_t outlength) {
  return awkward_reduce_prod<int64_t, int16_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_uint64_uint16_64(uint64_t* toptr,
                                    const uint16_t* fromptr,
                                    const int64_t* parents,
                                    int64_t lenparents,
                                    int64_t outlength) {
  return awkward_reduce_prod<uint64_t, uint16_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_int64_int32_64(int64_t* toptr,
                                  const int32_t* fromptr,
                                  const int64_t* parents,
                                  int64_t lenparents,
                                  int64_t outlength) {
  return awkward_reduce_prod<int64_t, int32_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_uint64_uint32_64(uint64_t* toptr,
                                    const uint32_t* fromptr,
                                    const int64_t* parents,
                                    int64_t lenparents,
                                    int64_t outlength) {
  return awkward_reduce_prod<uint64_t, uint32_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_int64_int64_64(int64_t* toptr,
                                  const int64_t* fromptr,
                                  const int64_t* parents,
                                  int64_t lenparents,
                                  int64_t outlength) {
  return awkward_reduce_prod<int64_t, int64_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_uint64_uint64_64(uint64_t* toptr,
                                    const uint64_t* fromptr,
                                    const int64_t* parents,
                                    int64_t lenparents,
                                    int64_t outlength) {
  return awkward_reduce_prod<uint64_t, uint64_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_float32_float32_64(float* toptr,
                                      const float* fromptr,
                                      const int64_t* parents,
                                      int64_t lenparents,
                                      int64_t outlength) {
  return awkward_reduce_prod<float, float>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_float64_float64_64(double* toptr,
                                      const double* fromptr,
                                      const int64_t* parents,
                                      int64_t lenparents,
                                      int64_t outlength) {
  return awkward_reduce_prod<double, double>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_int32_int8_64(int32_t* toptr,
                                 const int8_t* fromptr,
                                 const int64_t* parents,
                                 int64_t lenparents,
                                 int64_t outlength) {
  return awkward_reduce_prod<int32_t, int8_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_uint32_uint8_64(uint32_t* toptr,
                                   const uint8_t* fromptr,
                                   const int64_t* parents,
                                   int64_t lenparents,
                                   int64_t outlength) {
  return awkward_reduce_prod<uint32_t, uint8_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_int32_int16_64(int32_t* toptr,
                                  const int16_t* fromptr,
                                  const int64_t* parents,
                                  int64_t lenparents,
                                  int64_t outlength) {
  return awkward_reduce_prod<int32_t, int16_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_uint32_uint16_64(uint32_t* toptr,
                                    const uint16_t* fromptr,
                                    const int64_t* parents,
                                    int64_t lenparents,
                                    int64_t outlength) {
  return awkward_reduce_prod<uint32_t, uint16_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_int32_int32_64(int32_t* toptr,
                                  const int32_t* fromptr,
                                  const int64_t* parents,
                                  int64_t lenparents,
                                  int64_t outlength) {
  return awkward_reduce_prod<int32_t, int32_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
ERROR
awkward_reduce_prod_uint32_uint32_64(uint32_t* toptr,
                                    const uint32_t* fromptr,
                                    const int64_t* parents,
                                    int64_t lenparents,
                                    int64_t outlength) {
  return awkward_reduce_prod<uint32_t, uint32_t>(
      toptr, fromptr, parents, lenparents, outlength);
}
