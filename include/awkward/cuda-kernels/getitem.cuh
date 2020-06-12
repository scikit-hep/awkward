// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_GETITEM_CUH_
#define AWKWARD_GETITEM_CUH_

#include <stdint.h>

extern "C" {
  int8_t awkward_cuda_index8_getitem_at_nowrap(const int8_t* ptr, int64_t offset, int64_t at);
  uint8_t awkward_cuda_indexU8_getitem_at_nowrap(const uint8_t* ptr, int64_t offset, int64_t at);
  int32_t awkward_cuda_index32_getitem_at_nowrap(const int32_t* ptr, int64_t offset, int64_t at);
  uint32_t awkward_cuda_indexU32_getitem_at_nowrap(const uint32_t* ptr, int64_t offset, int64_t at);
  int64_t awkward_cuda_index64_getitem_at_nowrap(const int64_t * ptr, int64_t offset, int64_t at);
};

#endif //AWKWARD_GETITEM_CUH_
