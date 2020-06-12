// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cuda-kernels/getitem.cuh"
#include <iostream>

extern "C" {

  int8_t awkward_cuda_index8_getitem_at_nowrap(const int8_t* ptr, int64_t offset, int64_t at) {
    int8_t item;
    cudaMemcpy(&item, &ptr[(int64_t) (offset + at)], sizeof(int8_t), cudaMemcpyDeviceToHost);
    return item;
  }
  uint8_t awkward_cuda_indexU8_getitem_at_nowrap(const uint8_t* ptr, int64_t offset, int64_t at) {
    uint8_t item;
    cudaMemcpy(&item, &ptr[(int64_t) (offset + at)], sizeof(uint8_t), cudaMemcpyDeviceToHost);
    return item;
  }
  int32_t awkward_cuda_index32_getitem_at_nowrap(const int32_t* ptr, int64_t offset, int64_t at) {
    int32_t item;
    cudaMemcpy(&item, &ptr[(int64_t) (offset + at)], sizeof(int32_t), cudaMemcpyDeviceToHost);
    return item;
  }
  uint32_t awkward_cuda_indexU32_getitem_at_nowrap(const uint32_t* ptr, int64_t offset, int64_t at) {
    uint32_t item;
    cudaMemcpy(&item, &ptr[(int64_t) (offset + at)], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return item;
  }
  int64_t awkward_cuda_index64_getitem_at_nowrap(const int64_t * ptr, int64_t offset, int64_t at) {
    int64_t item;
    cudaMemcpy(&item, &ptr[(int64_t) (offset + at)], sizeof(int64_t), cudaMemcpyDeviceToHost);
    return item;
  }
  void awkward_cuda_index8_setitem_at_nowrap(const int8_t* ptr, int64_t offset, int64_t at, int8_t value) {
    cudaMemcpy((void *) &ptr[(int64_t) (offset + at)], &value, sizeof(int8_t), cudaMemcpyHostToDevice);
  }
  void awkward_cuda_indexU8_setitem_at_nowrap(const uint8_t* ptr, int64_t offset, int64_t at, uint8_t value) {
    cudaMemcpy((void *) &ptr[(int64_t) (offset + at)], &value, sizeof(uint8_t), cudaMemcpyHostToDevice);
  }
  void awkward_cuda_index32_setitem_at_nowrap(const int32_t* ptr, int64_t offset, int64_t at, int32_t value) {
    cudaMemcpy((void *) &ptr[(int64_t) (offset + at)], &value, sizeof(int32_t), cudaMemcpyHostToDevice);
  }
  void awkward_cuda_indexU32_setitem_at_nowrap(const uint32_t* ptr, int64_t offset, int64_t at, uint32_t value) {
    cudaMemcpy((void *) &ptr[(int64_t) (offset + at)], &value, sizeof(uint32_t), cudaMemcpyHostToDevice);
  }
  void awkward_cuda_index64_setitem_at_nowrap(const int64_t* ptr, int64_t offset, int64_t at, int64_t value) {
    cudaMemcpy((void *) &ptr[(int64_t) (offset + at)], &value, sizeof(int64_t), cudaMemcpyHostToDevice);
  }
}