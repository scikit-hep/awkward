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
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_NOW);
      
      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef int8_t* (func_awkward_cuda_ptri8_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void* ptr);
      func_awkward_cuda_ptri8_alloc_t *func_awkward_cuda_ptri8_alloc = reinterpret_cast<func_awkward_cuda_ptri8_alloc_t *>
                                                                                        (dlsym(handle, "awkward_cuda_ptri8_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptri8_alloc)(length);
      std::cout << "Device " << (*func_awkward_cuda_ptr_loc)(static_cast<void*>(ptr)) << "\n";
      return ptr;
    }
  }


  template <>
  uint8_t* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptriU8_alloc(length);
    else if(ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_NOW);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef uint8_t* (func_awkward_cuda_ptriU8_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void* ptr);
      func_awkward_cuda_ptriU8_alloc_t *func_awkward_cuda_ptriU8_alloc = reinterpret_cast<func_awkward_cuda_ptriU8_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptriU8_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptriU8_alloc)(length);
      std::cout << "Device " << (*func_awkward_cuda_ptr_loc)(static_cast<void*>(ptr)) << "\n";
      return ptr;
    }
  }

  template <>
  int32_t* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptri32_alloc(length);
    else if(ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_NOW);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef int32_t* (func_awkward_cuda_ptri32_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void* ptr);
      func_awkward_cuda_ptri32_alloc_t *func_awkward_cuda_ptri32_alloc = reinterpret_cast<func_awkward_cuda_ptri32_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptri32_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptri32_alloc)(length);
      std::cout << "Device " << (*func_awkward_cuda_ptr_loc)(static_cast<void*>(ptr)) << "\n";
      return ptr;
    }
  }

  template <>
  uint32_t* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptriU32_alloc(length);
    else if(ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_NOW);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef uint32_t* (func_awkward_cuda_ptriU32_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void* ptr);
      func_awkward_cuda_ptriU32_alloc_t *func_awkward_cuda_ptriU32_alloc = reinterpret_cast<func_awkward_cuda_ptriU32_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptriU32_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptriU32_alloc)(length);
      std::cout << "Device " << (*func_awkward_cuda_ptr_loc)(static_cast<void*>(ptr)) << "\n";
      return ptr;
    }
  }

  template <>
  int64_t* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptri64_alloc(length);
    else if(ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_NOW);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef int64_t* (func_awkward_cuda_ptri64_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void* ptr);
      func_awkward_cuda_ptri64_alloc_t *func_awkward_cuda_ptri64_alloc = reinterpret_cast<func_awkward_cuda_ptri64_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptri64_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptri64_alloc)(length);
      std::cout << "Device " << (*func_awkward_cuda_ptr_loc)(static_cast<void*>(ptr)) << "\n";
      return ptr;
    }
  }

  template <>
  float* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptrf_alloc(length);
    else if(ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_NOW);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef float* (func_awkward_cuda_ptrf_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void* ptr);
      func_awkward_cuda_ptrf_alloc_t *func_awkward_cuda_ptrf_alloc = reinterpret_cast<func_awkward_cuda_ptrf_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptrf_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptrf_alloc)(length);
      std::cout << "Device " << (*func_awkward_cuda_ptr_loc)(static_cast<void*>(ptr)) << "\n";
      return ptr;
    }
  }

  template <>
  double* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptrd_alloc(length);
    else if(ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_NOW);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef double* (func_awkward_cuda_ptrd_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void* ptr);
      func_awkward_cuda_ptrd_alloc_t *func_awkward_cuda_ptrd_alloc = reinterpret_cast<func_awkward_cuda_ptrd_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptri64_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptrd_alloc)(length);
      std::cout << "Device " << (*func_awkward_cuda_ptr_loc)(static_cast<void*>(ptr)) << "\n";
      return ptr;
    }
  }

  template <>
  bool* ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if(ptr_lib == cpu_kernels)
      return awkward_cpu_ptrb_alloc(length);
    else if(ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_NOW);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef bool* (func_awkward_cuda_ptrb_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void* ptr);
      func_awkward_cuda_ptrb_alloc_t *func_awkward_cuda_ptrb_alloc = reinterpret_cast<func_awkward_cuda_ptrb_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptrb_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptrb_alloc)(length);
      std::cout << "Device " << (*func_awkward_cuda_ptr_loc)(static_cast<void*>(ptr)) << "\n";
      return ptr;
    }
  }
}
