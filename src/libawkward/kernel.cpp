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
  template<>
  int8_t *ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_cpu_ptri8_alloc(length);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef int8_t *(func_awkward_cuda_ptri8_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void *ptr);
      func_awkward_cuda_ptri8_alloc_t *func_awkward_cuda_ptri8_alloc = reinterpret_cast<func_awkward_cuda_ptri8_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptri8_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(
        handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptri8_alloc)(length);
      return ptr;
    }
  }

  template<>
  int8_t *host_to_device_buff_transfer(int8_t *ptr, int64_t length, KernelsLib ptr_lib) {
    if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef int8_t *(func_awkward_cuda_host_to_device_buffi8_transfer_t)(int8_t *ptr, int64_t length);
      func_awkward_cuda_host_to_device_buffi8_transfer_t *func_awkward_cuda_host_to_device_buffi8_transfer = reinterpret_cast<func_awkward_cuda_host_to_device_buffi8_transfer_t *>
      (dlsym(handle, "awkward_cuda_host_to_device_buffi8_transfer"));

      auto cuda_ptr = (*func_awkward_cuda_host_to_device_buffi8_transfer)(ptr, length);

      return cuda_ptr;
    }
  }


  template<>
  uint8_t *ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_cpu_ptriU8_alloc(length);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef uint8_t *(func_awkward_cuda_ptriU8_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void *ptr);
      func_awkward_cuda_ptriU8_alloc_t *func_awkward_cuda_ptriU8_alloc = reinterpret_cast<func_awkward_cuda_ptriU8_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptriU8_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(
        handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptriU8_alloc)(length);

      return ptr;
    }
  }

  template<>
  uint8_t *host_to_device_buff_transfer(uint8_t *ptr, int64_t length, KernelsLib ptr_lib) {
    if (ptr_lib == cuda_kernels) {
//      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
//
//      if (!handle) {
//        fprintf(stderr, "dlopen failed: %s\n", dlerror());
//        return nullptr;
//      }
//      typedef uint8_t* (func_awkward_cuda_host_to_device_buffi8_transfer_t)(int8_t* ptr, int64_t length);
//      func_awkward_cuda_host_to_device_buffi8_transfer_t *func_awkward_cuda_host_to_device_buffi8_transfer = reinterpret_cast<func_awkward_cuda_host_to_device_buffi8_transfer_t *>
//      (dlsym(handle, "awkward_cuda_host_to_device_buffi8_transfer"));
//
//      auto cuda_ptr = (*func_awkward_cuda_host_to_device_buffi8_transfer)(ptr, length);
//
//      return cuda_ptr;
      return nullptr;
    }
  }

  template<>
  int32_t *ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_cpu_ptri32_alloc(length);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef int32_t *(func_awkward_cuda_ptri32_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void *ptr);
      func_awkward_cuda_ptri32_alloc_t *func_awkward_cuda_ptri32_alloc = reinterpret_cast<func_awkward_cuda_ptri32_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptri32_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(
        handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptri32_alloc)(length);

      return ptr;
    }
  }

  template<>
  int32_t *host_to_device_buff_transfer(int32_t *ptr, int64_t length, KernelsLib ptr_lib) {
    //      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
//
//      if (!handle) {
//        fprintf(stderr, "dlopen failed: %s\n", dlerror());
//        return nullptr;
//      }
//      typedef uint8_t* (func_awkward_cuda_host_to_device_buffi8_transfer_t)(int8_t* ptr, int64_t length);
//      func_awkward_cuda_host_to_device_buffi8_transfer_t *func_awkward_cuda_host_to_device_buffi8_transfer = reinterpret_cast<func_awkward_cuda_host_to_device_buffi8_transfer_t *>
//      (dlsym(handle, "awkward_cuda_host_to_device_buffi8_transfer"));
//
//      auto cuda_ptr = (*func_awkward_cuda_host_to_device_buffi8_transfer)(ptr, length);
//
//      return cuda_ptr;
    return nullptr;
  }

  template<>
  uint32_t *ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_cpu_ptriU32_alloc(length);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef uint32_t *(func_awkward_cuda_ptriU32_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void *ptr);
      func_awkward_cuda_ptriU32_alloc_t *func_awkward_cuda_ptriU32_alloc = reinterpret_cast<func_awkward_cuda_ptriU32_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptriU32_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(
        handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptriU32_alloc)(length);

      return ptr;
    }
  }

  template<>
  uint32_t *host_to_device_buff_transfer(uint32_t *ptr, int64_t length, KernelsLib ptr_lib) {
    //      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
//
//      if (!handle) {
//        fprintf(stderr, "dlopen failed: %s\n", dlerror());
//        return nullptr;
//      }
//      typedef uint8_t* (func_awkward_cuda_host_to_device_buffi8_transfer_t)(int8_t* ptr, int64_t length);
//      func_awkward_cuda_host_to_device_buffi8_transfer_t *func_awkward_cuda_host_to_device_buffi8_transfer = reinterpret_cast<func_awkward_cuda_host_to_device_buffi8_transfer_t *>
//      (dlsym(handle, "awkward_cuda_host_to_device_buffi8_transfer"));
//
//      auto cuda_ptr = (*func_awkward_cuda_host_to_device_buffi8_transfer)(ptr, length);
//
//      return cuda_ptr;
    return nullptr;
  }

  template<>
  int64_t *ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_cpu_ptri64_alloc(length);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef int64_t *(func_awkward_cuda_ptri64_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void *ptr);
      func_awkward_cuda_ptri64_alloc_t *func_awkward_cuda_ptri64_alloc = reinterpret_cast<func_awkward_cuda_ptri64_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptri64_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(
        handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptri64_alloc)(length);

      return ptr;
    }
  }

  template<>
  int64_t *host_to_device_buff_transfer(int64_t *ptr, int64_t length, KernelsLib ptr_lib) {
    //      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
//
//      if (!handle) {
//        fprintf(stderr, "dlopen failed: %s\n", dlerror());
//        return nullptr;
//      }
//      typedef uint8_t* (func_awkward_cuda_host_to_device_buffi8_transfer_t)(int8_t* ptr, int64_t length);
//      func_awkward_cuda_host_to_device_buffi8_transfer_t *func_awkward_cuda_host_to_device_buffi8_transfer = reinterpret_cast<func_awkward_cuda_host_to_device_buffi8_transfer_t *>
//      (dlsym(handle, "awkward_cuda_host_to_device_buffi8_transfer"));
//
//      auto cuda_ptr = (*func_awkward_cuda_host_to_device_buffi8_transfer)(ptr, length);
//
//      return cuda_ptr;
    return nullptr;
  }

  template<>
  float *ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_cpu_ptrf_alloc(length);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef float *(func_awkward_cuda_ptrf_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void *ptr);
      func_awkward_cuda_ptrf_alloc_t *func_awkward_cuda_ptrf_alloc = reinterpret_cast<func_awkward_cuda_ptrf_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptrf_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(
        handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptrf_alloc)(length);

      return ptr;
    }
  }

  template<>
  float *host_to_device_buff_transfer(float *ptr, int64_t length, KernelsLib ptr_lib) {
    //      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
//
//      if (!handle) {
//        fprintf(stderr, "dlopen failed: %s\n", dlerror());
//        return nullptr;
//      }
//      typedef uint8_t* (func_awkward_cuda_host_to_device_buffi8_transfer_t)(int8_t* ptr, int64_t length);
//      func_awkward_cuda_host_to_device_buffi8_transfer_t *func_awkward_cuda_host_to_device_buffi8_transfer = reinterpret_cast<func_awkward_cuda_host_to_device_buffi8_transfer_t *>
//      (dlsym(handle, "awkward_cuda_host_to_device_buffi8_transfer"));
//
//      auto cuda_ptr = (*func_awkward_cuda_host_to_device_buffi8_transfer)(ptr, length);
//
//      return cuda_ptr;
    return nullptr;
  }

  template<>
  double *ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_cpu_ptrd_alloc(length);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef double *(func_awkward_cuda_ptrd_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void *ptr);
      func_awkward_cuda_ptrd_alloc_t *func_awkward_cuda_ptrd_alloc = reinterpret_cast<func_awkward_cuda_ptrd_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptri64_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(
        handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptrd_alloc)(length);

      return ptr;
    }
  }

  template<>
  double *host_to_device_buff_transfer(double *ptr, int64_t length, KernelsLib ptr_lib) {
    //      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
//
//      if (!handle) {
//        fprintf(stderr, "dlopen failed: %s\n", dlerror());
//        return nullptr;
//      }
//      typedef uint8_t* (func_awkward_cuda_host_to_device_buffi8_transfer_t)(int8_t* ptr, int64_t length);
//      func_awkward_cuda_host_to_device_buffi8_transfer_t *func_awkward_cuda_host_to_device_buffi8_transfer = reinterpret_cast<func_awkward_cuda_host_to_device_buffi8_transfer_t *>
//      (dlsym(handle, "awkward_cuda_host_to_device_buffi8_transfer"));
//
//      auto cuda_ptr = (*func_awkward_cuda_host_to_device_buffi8_transfer)(ptr, length);
//
//      return cuda_ptr;
    return nullptr;
  }

  template<>
  bool *ptr_alloc(int64_t length, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_cpu_ptrb_alloc(length);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return nullptr;
      }
      typedef bool *(func_awkward_cuda_ptrb_alloc_t)(int64_t length);
      typedef int (func_awkward_cuda_ptr_loc_t)(void *ptr);
      func_awkward_cuda_ptrb_alloc_t *func_awkward_cuda_ptrb_alloc = reinterpret_cast<func_awkward_cuda_ptrb_alloc_t *>
      (dlsym(handle, "awkward_cuda_ptrb_alloc"));

      func_awkward_cuda_ptr_loc_t *func_awkward_cuda_ptr_loc = reinterpret_cast<func_awkward_cuda_ptr_loc_t *>(dlsym(
        handle, "awkward_cuda_ptr_loc"));
      auto ptr = (*func_awkward_cuda_ptrb_alloc)(length);

      return ptr;
    }
  }

  template<>
  bool *host_to_device_buff_transfer(bool *ptr, int64_t length, KernelsLib ptr_lib) {
    //      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
//
//      if (!handle) {
//        fprintf(stderr, "dlopen failed: %s\n", dlerror());
//        return nullptr;
//      }
//      typedef uint8_t* (func_awkward_cuda_host_to_device_buffi8_transfer_t)(int8_t* ptr, int64_t length);
//      func_awkward_cuda_host_to_device_buffi8_transfer_t *func_awkward_cuda_host_to_device_buffi8_transfer = reinterpret_cast<func_awkward_cuda_host_to_device_buffi8_transfer_t *>
//      (dlsym(handle, "awkward_cuda_host_to_device_buffi8_transfer"));
//
//      auto cuda_ptr = (*func_awkward_cuda_host_to_device_buffi8_transfer)(ptr, length);
//
//      return cuda_ptr;
    return nullptr;
  }

  template<>
  int8_t index_getitem_at_nowrap(int8_t *ptr, int64_t offset, int64_t at, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_index8_getitem_at_nowrap(ptr, offset, at);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return -1;
      }
      typedef int8_t (func_awkward_cuda_index8_getitem_at_nowrap_t)(const int8_t *ptr, int64_t offset, int64_t at);
      func_awkward_cuda_index8_getitem_at_nowrap_t *func_awkward_cuda_index8_getitem_at_nowrap = reinterpret_cast<func_awkward_cuda_index8_getitem_at_nowrap_t *>
      (dlsym(handle, "awkward_cuda_index8_getitem_at_nowrap"));

      int8_t item = (*func_awkward_cuda_index8_getitem_at_nowrap)(ptr, offset, at);

      return item;
    }
  }

  template<>
  uint8_t index_getitem_at_nowrap(uint8_t *ptr, int64_t offset, int64_t at, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_indexU8_getitem_at_nowrap(ptr, offset, at);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return -1;
      }
      typedef uint8_t (func_awkward_cuda_indexU8_getitem_at_nowrap_t)(const uint8_t *ptr, int64_t offset, int64_t at);
      func_awkward_cuda_indexU8_getitem_at_nowrap_t *func_awkward_cuda_indexU8_getitem_at_nowrap = reinterpret_cast<func_awkward_cuda_indexU8_getitem_at_nowrap_t *>
      (dlsym(handle, "awkward_cuda_indexU8_getitem_at_nowrap"));

      uint8_t item = (*func_awkward_cuda_indexU8_getitem_at_nowrap)(ptr, offset, at);

      return item;
    }
  }

  template<>
  int32_t index_getitem_at_nowrap(int32_t *ptr, int64_t offset, int64_t at, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_index32_getitem_at_nowrap(ptr, offset, at);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return -1;
      }
      typedef int32_t (func_awkward_cuda_index32_getitem_at_nowrap_t)(const int32_t *ptr, int64_t offset, int64_t at);
      func_awkward_cuda_index32_getitem_at_nowrap_t *func_awkward_cuda_index32_getitem_at_nowrap = reinterpret_cast<func_awkward_cuda_index32_getitem_at_nowrap_t *>
      (dlsym(handle, "awkward_cuda_index32_getitem_at_nowrap"));

      int32_t item = (*func_awkward_cuda_index32_getitem_at_nowrap)(ptr, offset, at);

      return item;
    }
  }

  template<>
  uint32_t index_getitem_at_nowrap(uint32_t *ptr, int64_t offset, int64_t at, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_indexU32_getitem_at_nowrap(ptr, offset, at);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return -1;
      }
      typedef uint32_t (func_awkward_cuda_indexU32_getitem_at_nowrap_t)(const uint32_t *ptr, int64_t offset,
                                                                        int64_t at);
      func_awkward_cuda_indexU32_getitem_at_nowrap_t *func_awkward_cuda_indexU32_getitem_at_nowrap = reinterpret_cast<func_awkward_cuda_indexU32_getitem_at_nowrap_t *>
      (dlsym(handle, "awkward_cuda_indexU32_getitem_at_nowrap"));

      uint32_t item = (*func_awkward_cuda_indexU32_getitem_at_nowrap)(ptr, offset, at);

      return item;
    }
  }

  template<>
  int64_t index_getitem_at_nowrap(int64_t *ptr, int64_t offset, int64_t at, KernelsLib ptr_lib) {
    if (ptr_lib == cpu_kernels)
      return awkward_index64_getitem_at_nowrap(ptr, offset, at);
    else if (ptr_lib == cuda_kernels) {
      auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);

      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return -1;
      }
      typedef int64_t (func_awkward_cuda_index64_getitem_at_nowrap_t)(const int64_t *ptr, int64_t offset, int64_t at);
      func_awkward_cuda_index64_getitem_at_nowrap_t *func_awkward_cuda_index64_getitem_at_nowrap = reinterpret_cast<func_awkward_cuda_index64_getitem_at_nowrap_t *>
      (dlsym(handle, "awkward_cuda_index64_getitem_at_nowrap"));

      int64_t item = (*func_awkward_cuda_index64_getitem_at_nowrap)(ptr, offset, at);

      return item;
    }
  }
}



