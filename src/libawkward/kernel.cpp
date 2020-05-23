#include "awkward/kernel.h"
#include "awkward/util.h"
#include "awkward/cpu-kernels/operations.h"
#include "awkward/common_utils.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/reducers.h"

#ifdef CUDA_KERNELS
#include "awkward/cuda-kernels/identities.h"
#endif

namespace kernel
{
  template <>
  ERROR new_identities<int32_t>(int32_t *toptr,
                                int64_t length)
  {
#ifdef CUDA_KERNELS
    return awkward_cuda_new_identities32(toptr, length);
#endif
    return awkward_new_identities32(toptr, length);
  }

  template <>
  ERROR new_identities<int64_t>(int64_t *toptr,
                                int64_t length)
  {
#ifdef CUDA_KERNELS
    return awkward_cuda_new_identities64(toptr, length);
#endif
    return awkward_new_identities64(toptr, length);
  }
} // namespace kernel
