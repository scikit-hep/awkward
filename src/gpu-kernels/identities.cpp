#include "awkward/gpu-kernels/identities.h"

template<typename T>
ERROR awkward_gpu_new_identities(uint8_t memory_loc,
                                 T *toptr,
                                 int64_t length) {
  for (T i = 0; i < length; i++) {
    toptr[i] = i;
  }
  return success();
}
ERROR awkward_gpu_new_identities32(uint8_t memory_loc,
                                   int32_t *toptr,
                                   int64_t length) {
  return awkward_gpu_new_identities<int32_t>(memory_loc,
                                             toptr,
                                             length);
}
ERROR awkward_gpu_new_identities64(uint8_t memory_loc,
                                   int64_t *toptr,
                                   int64_t length) {
  return awkward_gpu_new_identities<int64_t>(memory_loc,
                                             toptr,
                                             length);
}
