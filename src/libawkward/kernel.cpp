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

using namespace awkward;

namespace kernel {
  class LibraryPathCallback {
    public:
    const std::string library_path() {
      return std::string("/home/trickarcher/gsoc_2020/awkward-1"
                         ".0/src/cuda-kernels/build/libawkward-cuda-kernels.so");
    }
  };

  class LibraryCallback {
    public:
    void add_cuda_library_path_callback(
      const std::shared_ptr<LibraryPathCallback> &callback) {
      std::lock_guard<std::mutex> lock(lib_path_callbacks_mutex);
      lib_path_callbacks.at(kernel::Lib::cuda_kernels).push_back(*callback);
    }

    private:
    std::map<kernel::Lib, std::vector<LibraryPathCallback>> lib_path_callbacks =
      {{kernel::Lib::cuda_kernels, std::vector<LibraryPathCallback>()}};

    std::mutex lib_path_callbacks_mutex;
  };

  template <>
  Error H2D(
    kernel::Lib ptr_lib,
    bool** to_ptr,
    bool* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_H2Dbool_t)
        (bool **to_ptr, bool *from_ptr, int64_t length);
      func_awkward_cuda_H2Dbool_t
        *func_awkward_cuda_H2Dbool =
        reinterpret_cast<func_awkward_cuda_H2Dbool_t *>
        (dlsym(handle, "awkward_cuda_H2Dbool"));

      return (*func_awkward_cuda_H2Dbool)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error H2D(
    kernel::Lib ptr_lib,
    int8_t** to_ptr,
    int8_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);
      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }

      typedef Error (func_awkward_cuda_H2D8_t)
        (int8_t **to_ptr, int8_t *from_ptr, int8_t length);
      func_awkward_cuda_H2D8_t
        *func_awkward_cuda_H2D8 =
        reinterpret_cast<func_awkward_cuda_H2D8_t *>
        (dlsym(handle, "awkward_cuda_H2D8"));

      return (*func_awkward_cuda_H2D8)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error H2D(
    kernel::Lib ptr_lib,
    uint8_t** to_ptr,
    uint8_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_H2DU8_t)
        (uint8_t **to_ptr, uint8_t *from_ptr, int64_t length);
      func_awkward_cuda_H2DU8_t
        *func_awkward_cuda_H2DU8 =
        reinterpret_cast<func_awkward_cuda_H2DU8_t *>
        (dlsym(handle, "awkward_cuda_H2DU8"));

      return (*func_awkward_cuda_H2DU8)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error H2D(
    kernel::Lib ptr_lib,
    int16_t** to_ptr,
    int16_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_H2D16_t)
        (int16_t **to_ptr, int16_t *from_ptr, int64_t length);
      func_awkward_cuda_H2D16_t
        *func_awkward_cuda_H2D16 =
        reinterpret_cast<func_awkward_cuda_H2D16_t *>
        (dlsym(handle, "awkward_cuda_H2D16"));

      return (*func_awkward_cuda_H2D16)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error H2D(
    kernel::Lib ptr_lib,
    uint16_t** to_ptr,
    uint16_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_H2DU16_t)
        (uint16_t **to_ptr, uint16_t *from_ptr, int64_t length);
      func_awkward_cuda_H2DU16_t
        *func_awkward_cuda_H2DU16 =
        reinterpret_cast<func_awkward_cuda_H2DU16_t *>
        (dlsym(handle, "awkward_cuda_H2DU16"));

      return (*func_awkward_cuda_H2DU16)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error H2D(
    kernel::Lib ptr_lib,
    int32_t** to_ptr,
    int32_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_H2D32_t)
        (int32_t **to_ptr, int32_t *from_ptr, int64_t length);
      func_awkward_cuda_H2D32_t
        *func_awkward_cuda_H2D32 =
        reinterpret_cast<func_awkward_cuda_H2D32_t *>
        (dlsym(handle, "awkward_cuda_H2D32"));

      return (*func_awkward_cuda_H2D32)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error H2D(
    kernel::Lib ptr_lib,
    uint32_t** to_ptr,
    uint32_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_H2DU32_t)
        (uint32_t **to_ptr, uint32_t *from_ptr, int64_t length);
      func_awkward_cuda_H2DU32_t
        *func_awkward_cuda_H2DU32 =
        reinterpret_cast<func_awkward_cuda_H2DU32_t *>
        (dlsym(handle, "awkward_cuda_H2DU32"));

      return (*func_awkward_cuda_H2DU32)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error H2D(
    kernel::Lib ptr_lib,
    int64_t** to_ptr,
    int64_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_H2D64_t)
        (int64_t **to_ptr, int64_t *from_ptr, int64_t length);
      func_awkward_cuda_H2D64_t
        *func_awkward_cuda_H2D64 =
        reinterpret_cast<func_awkward_cuda_H2D64_t *>
        (dlsym(handle, "awkward_cuda_H2D64"));

      return (*func_awkward_cuda_H2D64)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error H2D(
    kernel::Lib ptr_lib,
    uint64_t** to_ptr,
    uint64_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_H2DU64_t)
        (uint64_t **to_ptr, uint64_t *from_ptr, int64_t length);
      func_awkward_cuda_H2DU64_t
        *func_awkward_cuda_H2DU64 =
        reinterpret_cast<func_awkward_cuda_H2DU64_t *>
        (dlsym(handle, "awkward_cuda_H2DU64"));

      return (*func_awkward_cuda_H2DU64)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error H2D(
    kernel::Lib ptr_lib,
    float** to_ptr,
    float* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_H2Dfloat32_t)
        (float **to_ptr, float *from_ptr, int64_t length);
      func_awkward_cuda_H2Dfloat32_t
        *func_awkward_cuda_H2Dfloat32 =
        reinterpret_cast<func_awkward_cuda_H2Dfloat32_t *>
        (dlsym(handle, "awkward_cuda_H2Dfloat32"));

      return (*func_awkward_cuda_H2Dfloat32)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error H2D(
    kernel::Lib ptr_lib,
    double** to_ptr,
    double* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_H2Dfloat64_t)
        (double **to_ptr, double *from_ptr, int64_t length);
      func_awkward_cuda_H2Dfloat64_t
        *func_awkward_cuda_H2Dfloat64 =
        reinterpret_cast<func_awkward_cuda_H2Dfloat64_t *>
        (dlsym(handle, "awkward_cuda_H2Dfloat64"));

      return (*func_awkward_cuda_H2Dfloat64)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }


  template<>
  std::shared_ptr<bool> ptr_alloc(kernel::Lib ptr_lib, int64_t length) {
    #ifndef _MSC_VER
      if (ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return nullptr;
        }

        typedef bool *(func_awkward_cuda_ptrbool_alloc_t)(int64_t length);

        func_awkward_cuda_ptrbool_alloc_t *func_awkward_cuda_ptrbool_alloc =
          reinterpret_cast<func_awkward_cuda_ptrbool_alloc_t *>
          (dlsym(handle, "awkward_cuda_ptrbool_alloc"));

        return std::shared_ptr<bool>((*func_awkward_cuda_ptrbool_alloc)(length),
                                     kernel::cuda_array_deleter<bool>());
      }
    #endif
    return std::shared_ptr<bool>(awkward_cpu_ptrbool_alloc(length),
                                 kernel::array_deleter<bool>());
  }
  template<>
  std::shared_ptr<int8_t> ptr_alloc(kernel::Lib ptr_lib, int64_t length) {
    #ifndef _MSC_VER
      if (ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);
          awkward::util::handle_cuda_error(err);
          return nullptr;
        }
        typedef int8_t *(func_awkward_cuda_ptr8_alloc_t)(int64_t length);

        func_awkward_cuda_ptr8_alloc_t *func_awkward_cuda_ptr8_alloc =
          reinterpret_cast<func_awkward_cuda_ptr8_alloc_t *>
            (dlsym(handle, "awkward_cuda_ptr8_alloc"));

        return std::shared_ptr<int8_t>((*func_awkward_cuda_ptr8_alloc)(length),
                                    kernel::cuda_array_deleter<int8_t>());
      }
    #endif
    return std::shared_ptr<int8_t>(awkward_cpu_ptr8_alloc(length),
                                   kernel::array_deleter<int8_t>());
  }
  template<>
  std::shared_ptr<uint8_t> ptr_alloc(kernel::Lib ptr_lib, int64_t length) {
    #ifndef _MSC_VER
      if (ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return nullptr;
        }

        typedef uint8_t *(func_awkward_cuda_ptrU8_alloc_t)(int64_t length);

        func_awkward_cuda_ptrU8_alloc_t *func_awkward_cuda_ptrU8_alloc =
          reinterpret_cast<func_awkward_cuda_ptrU8_alloc_t *>
          (dlsym(handle, "awkward_cuda_ptrU8_alloc"));

        return std::shared_ptr<uint8_t>((*func_awkward_cuda_ptrU8_alloc)(length),
                                        kernel::cuda_array_deleter<uint8_t>());
      }
    #endif
    return std::shared_ptr<uint8_t>(awkward_cpu_ptrU8_alloc(length),
                                   kernel::array_deleter<uint8_t>());
  }
  template<>
  std::shared_ptr<int16_t> ptr_alloc(kernel::Lib ptr_lib, int64_t length) {
    #ifndef _MSC_VER
      if (ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return nullptr;
        }

        typedef int16_t *(func_awkward_cuda_ptr16_alloc_t)(int64_t length);

        func_awkward_cuda_ptr16_alloc_t *func_awkward_cuda_ptr16_alloc =
          reinterpret_cast<func_awkward_cuda_ptr16_alloc_t *>
          (dlsym(handle, "awkward_cuda_ptr16_alloc"));

        return std::shared_ptr<int16_t>((*func_awkward_cuda_ptr16_alloc)(length),
                                        kernel::cuda_array_deleter<int16_t>());
      }
    #endif
    return std::shared_ptr<int16_t>(awkward_cpu_ptr16_alloc(length),
                                    kernel::array_deleter<int16_t>());
  }
  template<>
  std::shared_ptr<uint16_t> ptr_alloc(kernel::Lib ptr_lib, int64_t length) {
    #ifndef _MSC_VER
      if (ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return nullptr;
        }

        typedef uint16_t *(func_awkward_cuda_ptrU16_alloc_t)(int64_t length);

        func_awkward_cuda_ptrU16_alloc_t *func_awkward_cuda_ptrU16_alloc =
          reinterpret_cast<func_awkward_cuda_ptrU16_alloc_t *>
          (dlsym(handle, "awkward_cuda_ptrU16_alloc"));

        return std::shared_ptr<uint16_t>((*func_awkward_cuda_ptrU16_alloc)(length),
                                         kernel::cuda_array_deleter<uint16_t>());
      }
    #endif
    return std::shared_ptr<uint16_t>(awkward_cpu_ptrU16_alloc(length),
                                     kernel::array_deleter<uint16_t>());
  }
  template<>
  std::shared_ptr<int32_t> ptr_alloc(kernel::Lib ptr_lib, int64_t length) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return nullptr;
        }

        typedef int32_t *(func_awkward_cuda_ptr32_alloc_t)(int64_t length);

        func_awkward_cuda_ptr32_alloc_t *func_awkward_cuda_ptr32_alloc =
          reinterpret_cast<func_awkward_cuda_ptr32_alloc_t *>
          (dlsym(handle, "awkward_cuda_ptr32_alloc"));

        return std::shared_ptr<int32_t>((*func_awkward_cuda_ptr32_alloc)(length),
                                        kernel::cuda_array_deleter<int32_t>());
      }
    #endif
    return std::shared_ptr<int32_t>(awkward_cpu_ptr32_alloc(length),
                                    kernel::array_deleter<int32_t>());
  }
  template<>
  std::shared_ptr<uint32_t> ptr_alloc(kernel::Lib ptr_lib, int64_t length) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return nullptr;
        }

        typedef uint32_t *(func_awkward_cuda_ptrU32_alloc_t)(int64_t length);

        func_awkward_cuda_ptrU32_alloc_t *func_awkward_cuda_ptrU32_alloc =
          reinterpret_cast<func_awkward_cuda_ptrU32_alloc_t *>
          (dlsym(handle, "awkward_cuda_ptrU32_alloc"));

        return std::shared_ptr<uint32_t>((*func_awkward_cuda_ptrU32_alloc)(length),
                                         kernel::cuda_array_deleter<uint32_t>());
      }
    #endif
    return std::shared_ptr<uint32_t>(awkward_cpu_ptrU32_alloc(length),
                                     kernel::array_deleter<uint32_t>());
  }
  template<>
  std::shared_ptr<int64_t> ptr_alloc(kernel::Lib ptr_lib, int64_t length) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return nullptr;
        }

        typedef int64_t *(func_awkward_cuda_ptr64_alloc_t)(int64_t length);

        func_awkward_cuda_ptr64_alloc_t *func_awkward_cuda_ptr64_alloc =
          reinterpret_cast<func_awkward_cuda_ptr64_alloc_t *>
          (dlsym(handle, "awkward_cuda_ptr64_alloc"));

        return std::shared_ptr<int64_t>((*func_awkward_cuda_ptr64_alloc)(length),
                                        kernel::cuda_array_deleter<int64_t>());
      }
    #endif
    return std::shared_ptr<int64_t>(awkward_cpu_ptr64_alloc(length),
                                    kernel::array_deleter<int64_t>());
  }
  template<>
  std::shared_ptr<uint64_t> ptr_alloc(kernel::Lib ptr_lib, int64_t length) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return nullptr;
        }

        typedef uint64_t *(func_awkward_cuda_ptrU64_alloc_t)(int64_t length);

        func_awkward_cuda_ptrU64_alloc_t *func_awkward_cuda_ptrU64_alloc =
          reinterpret_cast<func_awkward_cuda_ptrU64_alloc_t *>
          (dlsym(handle, "awkward_cuda_ptrU64_alloc"));

        return std::shared_ptr<uint64_t>((*func_awkward_cuda_ptrU64_alloc)(length),
                                         kernel::cuda_array_deleter<uint64_t>());
      }
    #endif
    return std::shared_ptr<uint64_t>(awkward_cpu_ptrU64_alloc(length),
                                     kernel::array_deleter<uint64_t>());
  }
  template<>
  std::shared_ptr<float> ptr_alloc(kernel::Lib ptr_lib, int64_t length) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return nullptr;
        }

        typedef float *(func_awkward_cuda_ptrfloat32_alloc_t)(int64_t length);

        func_awkward_cuda_ptrfloat32_alloc_t *func_awkward_cuda_ptrfloat32_alloc =
          reinterpret_cast<func_awkward_cuda_ptrfloat32_alloc_t *>
          (dlsym(handle, "awkward_cuda_ptrfloat32_alloc"));

        return std::shared_ptr<float>((*func_awkward_cuda_ptrfloat32_alloc)(length),
                                      kernel::cuda_array_deleter<float>());
      }
    #endif
    return std::shared_ptr<float>(awkward_cpu_ptrfloat32_alloc(length),
                                  kernel::array_deleter<float>());
  }
  template<>
  std::shared_ptr<double> ptr_alloc(kernel::Lib ptr_lib, int64_t length) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return nullptr;
        }

        typedef double *(func_awkward_cuda_ptrfloat64_alloc_t)(int64_t length);

        func_awkward_cuda_ptrfloat64_alloc_t *func_awkward_cuda_ptrfloat64_alloc =
          reinterpret_cast<func_awkward_cuda_ptrfloat64_alloc_t *>
          (dlsym(handle, "awkward_cuda_ptrfloat64_alloc"));

        return std::shared_ptr<double>((*func_awkward_cuda_ptrfloat64_alloc)(length),
                                       kernel::cuda_array_deleter<double>());
      }
    #endif
    return std::shared_ptr<double>(awkward_cpu_ptrfloat64_alloc(length),
                                   kernel::array_deleter<double>());
  }

  template<>
  int8_t index_getitem_at_nowrap(kernel::Lib ptr_lib,
                                 int8_t *ptr,
                                 int64_t offset,
                                 int64_t at) {
    #ifndef _MSC_VER
      if (ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return -1;
        }
        typedef int8_t (func_awkward_cuda_index8_getitem_at_nowrap_t)
                                  (const int8_t *ptr, int64_t offset, int64_t at);
        func_awkward_cuda_index8_getitem_at_nowrap_t
          *func_awkward_cuda_index8_getitem_at_nowrap =
            reinterpret_cast<func_awkward_cuda_index8_getitem_at_nowrap_t *>
            (dlsym(handle, "awkward_cuda_index8_getitem_at_nowrap"));

        return (*func_awkward_cuda_index8_getitem_at_nowrap)(ptr, offset, at);
      }
    #endif
    return awkward_index8_getitem_at_nowrap(ptr, offset, at);
  }
  template<>
  uint8_t index_getitem_at_nowrap(kernel::Lib ptr_lib,
                                  uint8_t *ptr,
                                  int64_t offset,
                                  int64_t at) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return -1;
        }
        typedef uint8_t (func_awkward_cuda_indexU8_getitem_at_nowrap_t)
                                  (const uint8_t *ptr, int64_t offset, int64_t at);
        func_awkward_cuda_indexU8_getitem_at_nowrap_t
          *func_awkward_cuda_indexU8_getitem_at_nowrap =
            reinterpret_cast<func_awkward_cuda_indexU8_getitem_at_nowrap_t *>
            (dlsym(handle, "awkward_cuda_indexU8_getitem_at_nowrap"));

        return (*func_awkward_cuda_indexU8_getitem_at_nowrap)(ptr, offset, at);
      }
    #endif
    return awkward_indexU8_getitem_at_nowrap(ptr, offset, at);
  }

  template<>
  int32_t index_getitem_at_nowrap(kernel::Lib ptr_lib,
                                  int32_t *ptr,
                                  int64_t offset,
                                  int64_t at) {
    #ifndef _MSC_VER
      if (ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return -1;
        }
        typedef int32_t (func_awkward_cuda_index32_getitem_at_nowrap_t)
                                  (const int32_t *ptr, int64_t offset, int64_t at);
        func_awkward_cuda_index32_getitem_at_nowrap_t
          *func_awkward_cuda_index32_getitem_at_nowrap =
            reinterpret_cast<func_awkward_cuda_index32_getitem_at_nowrap_t *>
            (dlsym(handle, "awkward_cuda_index32_getitem_at_nowrap"));

        return (*func_awkward_cuda_index32_getitem_at_nowrap)(ptr, offset, at);
      }
    #endif
    return awkward_index32_getitem_at_nowrap(ptr, offset, at);
  }

  template<>
  uint32_t index_getitem_at_nowrap(kernel::Lib ptr_lib,
                                   uint32_t *ptr,
                                   int64_t offset,
                                   int64_t at) {
    #ifndef _MSC_VER
      if (ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return -1;
        }
        typedef uint32_t (func_awkward_cuda_indexU32_getitem_at_nowrap_t)
                                (const uint32_t *ptr, int64_t offset, int64_t at);
        func_awkward_cuda_indexU32_getitem_at_nowrap_t
          *func_awkward_cuda_indexU32_getitem_at_nowrap =
            reinterpret_cast<func_awkward_cuda_indexU32_getitem_at_nowrap_t *>
            (dlsym(handle, "awkward_cuda_indexU32_getitem_at_nowrap"));

        return (*func_awkward_cuda_indexU32_getitem_at_nowrap)(ptr, offset, at);
      }
    #endif
    return awkward_indexU32_getitem_at_nowrap(ptr, offset, at);
  }

  template<>
  int64_t index_getitem_at_nowrap(kernel::Lib ptr_lib,
                                  int64_t *ptr,
                                  int64_t offset,
                                  int64_t at) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
          return -1;
        }
        typedef int64_t (func_awkward_cuda_index64_getitem_at_nowrap_t)
                                  (const int64_t *ptr, int64_t offset, int64_t at);
        func_awkward_cuda_index64_getitem_at_nowrap_t
          *func_awkward_cuda_index64_getitem_at_nowrap =
            reinterpret_cast<func_awkward_cuda_index64_getitem_at_nowrap_t *>
            (dlsym(handle, "awkward_cuda_index64_getitem_at_nowrap"));

        return (*func_awkward_cuda_index64_getitem_at_nowrap)(ptr, offset, at);
      }
    #endif
    return awkward_index64_getitem_at_nowrap(ptr, offset, at);
  }

  template<>
  void index_setitem_at_nowrap(kernel::Lib ptr_lib,
                               int8_t* ptr,
                               int64_t offset,
                               int64_t at,
                               int8_t value) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
        }
        typedef void (func_awkward_cuda_index8_setitem_at_nowrap_t)
                    (const int8_t *ptr, int64_t offset, int64_t at, int8_t value);
        func_awkward_cuda_index8_setitem_at_nowrap_t
          *func_awkward_cuda_index8_setitem_at_nowrap =
           reinterpret_cast<func_awkward_cuda_index8_setitem_at_nowrap_t *>
          (dlsym(handle, "awkward_cuda_index8_setitem_at_nowrap"));

        (*func_awkward_cuda_index8_setitem_at_nowrap)(ptr, offset, at, value);
      }
    #endif
    awkward_index8_setitem_at_nowrap(ptr, offset, at, value);
  }

  template<>
  void index_setitem_at_nowrap(kernel::Lib ptr_lib,
                               uint8_t* ptr,
                               int64_t offset,
                               int64_t at,
                               uint8_t value) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
        }
        typedef void (func_awkward_cuda_indexU8_setitem_at_nowrap_t)
                  (const uint8_t *ptr, int64_t offset, int64_t at, uint8_t value);
        func_awkward_cuda_indexU8_setitem_at_nowrap_t
          *func_awkward_cuda_indexU8_setitem_at_nowrap =
            reinterpret_cast<func_awkward_cuda_indexU8_setitem_at_nowrap_t *>
            (dlsym(handle, "awkward_cuda_indexU8_setitem_at_nowrap"));

        (*func_awkward_cuda_indexU8_setitem_at_nowrap)(ptr, offset, at, value);
      }
    #endif
    return awkward_indexU8_setitem_at_nowrap(ptr, offset, at, value);
  }

  template<>
  void index_setitem_at_nowrap(kernel::Lib ptr_lib,
                               int32_t* ptr,
                               int64_t offset,
                               int64_t at,
                               int32_t value) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
        }
        typedef void (func_awkward_cuda_index32_setitem_at_nowrap_t)
                  (const int32_t *ptr, int64_t offset, int64_t at, int32_t value);
        func_awkward_cuda_index32_setitem_at_nowrap_t
          *func_awkward_cuda_index32_setitem_at_nowrap =
            reinterpret_cast<func_awkward_cuda_index32_setitem_at_nowrap_t *>
            (dlsym(handle, "awkward_cuda_index32_setitem_at_nowrap"));

        (*func_awkward_cuda_index32_setitem_at_nowrap)(ptr, offset, at, value);
      }
    #endif
    awkward_index32_setitem_at_nowrap(ptr, offset, at, value);
  }

  template<>
  void index_setitem_at_nowrap(kernel::Lib ptr_lib,
                               uint32_t* ptr,
                               int64_t offset,
                               int64_t at,
                               uint32_t value) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
        }
        typedef void (func_awkward_cuda_indexU32_setitem_at_nowrap_t)
                (const uint32_t *ptr, int64_t offset, int64_t at, uint32_t value);
        func_awkward_cuda_indexU32_setitem_at_nowrap_t
          *func_awkward_cuda_indexU32_setitem_at_nowrap =
            reinterpret_cast<func_awkward_cuda_indexU32_setitem_at_nowrap_t *>
            (dlsym(handle, "awkward_cuda_indexU32_setitem_at_nowrap"));

        (*func_awkward_cuda_indexU32_setitem_at_nowrap)(ptr, offset, at, value);
      }
    #endif
    awkward_indexU32_setitem_at_nowrap(ptr, offset, at, value);
  }

  template<>
  void index_setitem_at_nowrap(kernel::Lib ptr_lib,
                               int64_t* ptr,
                               int64_t offset,
                               int64_t at,
                               int64_t value) {
    #ifndef _MSC_VER
      if(ptr_lib == kernel::Lib::cuda_kernels) {
        auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

        if (!handle) {
          Error err = failure("Failed to find awkward1[cuda]",
                              0,
                              kSliceNone);

          awkward::util::handle_cuda_error(err);
        }
        typedef void (func_awkward_cuda_index64_setitem_at_nowrap_t)
                  (const int64_t *ptr, int64_t offset, int64_t at, int64_t value);
        func_awkward_cuda_index64_setitem_at_nowrap_t
          *func_awkward_cuda_index64_setitem_at_nowrap =
            reinterpret_cast<func_awkward_cuda_index64_setitem_at_nowrap_t *>
            (dlsym(handle, "awkward_cuda_index64_setitem_at_nowrap"));

        (*func_awkward_cuda_index64_setitem_at_nowrap)(ptr, offset, at, value);
      }
    #endif
    awkward_index64_setitem_at_nowrap(ptr, offset, at, value);
  }
}



