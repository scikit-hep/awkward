// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/cpu-kernels/operations.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/reducers.h"

#ifdef BUILD_CUDA_KERNELS
#include "awkward/cuda-kernels/identities.h"
#endif

#include "awkward/kernel.h"

namespace kernel {
  template <>
  int8_t* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptri8_alloc(length);
    else if(ptr_lib == cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_NOW);
      std::cout << handle << "\n";
      if (!handle) {
        fputs (dlerror(), stderr);
        return nullptr;
      }
      typedef int8_t* (func_awkward_cuda_ptri8_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(int8_t* ptr);
      func_awkward_cuda_ptri8_alloc_t *func_awkward_cuda_ptri8_alloc = reinterpret_cast<func_awkward_cuda_ptri8_alloc_t *>
                                                                                        (dlsym(handle, "awkward_cuda_ptri8_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptri8_alloc)(length);
      std::cout << "Device " << (*func_awkward_cuda_ptr_loc)(ptr) << "\n";
      return ptr;
    }
  }


  template <>
  uint8_t* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptriU8_alloc(length);
  }

  template <>
  int32_t* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptri32_alloc(length);
  }

  template <>
  uint32_t* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptriU32_alloc(length);
  }

  template <>
  int64_t* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptri64_alloc(length);
  }

  template <>
  float* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptrf_alloc(length);
  }

  template <>
  double* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptrd_alloc(length);
  }

  template <>
  bool* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptrb_alloc(length);
  }

  template <>
  ERROR new_identities<int32_t>(int32_t *toptr,
                                int64_t length) {
#ifdef BUILD_CUDA_KERNELS
    return awkward_cuda_new_identities32(toptr, length);
#endif
    return awkward_new_identities32(toptr, length);
  }

  template <>
  ERROR new_identities<int64_t>(int64_t *toptr,
                                int64_t length) {
#ifdef BUILD_CUDA_KERNELS
    return awkward_cuda_new_identities64(toptr, length);
#endif
    return awkward_new_identities64(toptr, length);
  }
}
