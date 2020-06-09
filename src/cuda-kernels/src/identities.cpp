#include "awkward/cuda-kernels/identities.h"

template <typename T>
ERROR awkward_cuda_new_identities(
    T *toptr,
    int64_t length)
{
  for (T i = 0; i < length; i++)
  {
    toptr[i] = i;
  }
  return success();
}
ERROR awkward_cuda_new_identities32(
    int32_t *toptr,
    int64_t length)
{
  return awkward_cuda_new_identities<int32_t>(
      toptr,
      length);
}
ERROR awkward_cuda_new_identities64(
    int64_t *toptr,
    int64_t length)
{
  return awkward_cuda_new_identities<int64_t>(
      toptr,
      length);
}
