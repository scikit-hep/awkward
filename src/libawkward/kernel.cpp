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

// FIXME-PR293: utility for trying to load libawkward-cuda-kernels.so
// with exception-raising logic for "can't find library"
// up to you: you don't need a util::handle_cuda_error

// Error message:
// install the 'awkward1-cuda-kernels' package with:
//
//     pip install awkward1[cuda] --upgrade

namespace kernel {

  std::shared_ptr<LibraryCallback> lib_callback = std::make_shared<LibraryCallback>();

#ifndef _MSC_VER

//  const char* awkward_cuda_path() {
//    // Remove this when callback is implemented
//    lib_callback().add_cuda_library_path_callback(std::make_shared<LibraryPathCallback>());
//
//    for(auto i : lib_path_callbacks.at(kernel::Lib::cuda_kernels)) {
//      auto handle = dlopen(i.library_path().c_str(), RTLD_NOW);
//      if(handle != nullptr)
//        return i.library_path().c_str();
//    }
//  }

  template<>
  void cuda_array_deleter<bool>::operator()(bool const *p) {
    auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
    if (!handle) {
      Error err = failure("Failed to find awkward1[cuda]",
                          0,
                          kSliceNone);

      util::handle_cuda_error(err);
    }

    typedef Error (func_awkward_cuda_ptrbool_dealloc_t)(const bool *ptr);
    func_awkward_cuda_ptrbool_dealloc_t *func_awkward_cuda_ptrbool_dealloc =
      reinterpret_cast<func_awkward_cuda_ptrbool_dealloc_t *>
      (dlsym(handle, "awkward_cuda_ptrbool_dealloc"));

    util::handle_cuda_error((*func_awkward_cuda_ptrbool_dealloc)(p));
  }

  template<>
  void cuda_array_deleter<int8_t>::operator()(int8_t const *p) {
    auto handle = dlopen(kernel::lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
    if (!handle) {
      Error err = failure("Failed to find awkward1[cuda]",
                          0,
                          kSliceNone);

      awkward::util::handle_cuda_error(err);
    }

    typedef Error (func_awkward_cuda_ptr8_dealloc_t)(const int8_t *ptr);
    func_awkward_cuda_ptr8_dealloc_t *func_awkward_cuda_ptr8_dealloc =
      reinterpret_cast<func_awkward_cuda_ptr8_dealloc_t *>
      (dlsym(handle, "awkward_cuda_ptr8_dealloc"));

    util::handle_cuda_error((*func_awkward_cuda_ptr8_dealloc)(p));
  }


  template<>
  void cuda_array_deleter<uint8_t>::operator()(uint8_t const *p) {
    auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
    if (!handle) {
      Error err = failure("Failed to find awkward1[cuda]",
                          0,
                          kSliceNone);

      awkward::util::handle_cuda_error(err);
    }

    typedef Error (func_awkward_cuda_ptrU8_dealloc_t)(const uint8_t *ptr);
    func_awkward_cuda_ptrU8_dealloc_t *func_awkward_cuda_ptrU8_dealloc =
      reinterpret_cast<func_awkward_cuda_ptrU8_dealloc_t *>
      (dlsym(handle, "awkward_cuda_ptrU8_dealloc"));

    util::handle_cuda_error((*func_awkward_cuda_ptrU8_dealloc)(p));
  }


  template<>
  void cuda_array_deleter<int16_t>::operator()(int16_t const *p) {
    auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
    if (!handle) {
      Error err = failure("Failed to find awkward1[cuda]",
                          0,
                          kSliceNone);

      awkward::util::handle_cuda_error(err);
    }

    typedef Error (func_awkward_cuda_ptr16_dealloc_t)(const int16_t *ptr);
    func_awkward_cuda_ptr16_dealloc_t *func_awkward_cuda_ptr16_dealloc =
      reinterpret_cast<func_awkward_cuda_ptr16_dealloc_t *>
      (dlsym(handle, "awkward_cuda_ptr16_dealloc"));

    util::handle_cuda_error((*func_awkward_cuda_ptr16_dealloc)(p));
  }

  template<>
  void cuda_array_deleter<uint16_t>::operator()(uint16_t const *p) {
    auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
    if (!handle) {
      Error err = failure("Failed to find awkward1[cuda]",
                          0,
                          kSliceNone);

      awkward::util::handle_cuda_error(err);
    }

    typedef Error (func_awkward_cuda_ptrU16_dealloc_t)(const uint16_t *ptr);
    func_awkward_cuda_ptrU16_dealloc_t *func_awkward_cuda_ptrU16_dealloc =
      reinterpret_cast<func_awkward_cuda_ptrU16_dealloc_t *>
      (dlsym(handle, "awkward_cuda_ptrU16_dealloc"));

    util::handle_cuda_error((*func_awkward_cuda_ptrU16_dealloc)(p));
  }


  template<>
  void cuda_array_deleter<int32_t>::operator()(int32_t const *p) {
    auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
    if (!handle) {
      Error err = failure("Failed to find awkward1[cuda]",
                          0,
                          kSliceNone);

      awkward::util::handle_cuda_error(err);
    }

    typedef Error (func_awkward_cuda_ptr32_dealloc_t)(const int32_t *ptr);
    func_awkward_cuda_ptr32_dealloc_t *func_awkward_cuda_ptr32_dealloc =
      reinterpret_cast<func_awkward_cuda_ptr32_dealloc_t *>
      (dlsym(handle, "awkward_cuda_ptr32_dealloc"));

    util::handle_cuda_error((*func_awkward_cuda_ptr32_dealloc)(p));
  }

  template<>
  void cuda_array_deleter<uint32_t>::operator()(uint32_t const *p) {
    auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
    if (!handle) {
      Error err = failure("Failed to find awkward1[cuda]",
                          0,
                          kSliceNone);

      awkward::util::handle_cuda_error(err);
    }

    typedef Error (func_awkward_cuda_ptrU32_dealloc_t)(const uint32_t *ptr);
    func_awkward_cuda_ptrU32_dealloc_t *func_awkward_cuda_ptrU32_dealloc =
      reinterpret_cast<func_awkward_cuda_ptrU32_dealloc_t *>
      (dlsym(handle, "awkward_cuda_ptrU32_dealloc"));

    util::handle_cuda_error((*func_awkward_cuda_ptrU32_dealloc)(p));
  }

  template<>
  void cuda_array_deleter<int64_t>::operator()(int64_t const *p) {
    auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
    if (!handle) {
      Error err = failure("Failed to find awkward1[cuda]",
                          0,
                          kSliceNone);

      awkward::util::handle_cuda_error(err);
    }

    typedef Error (func_awkward_cuda_ptr64_dealloc_t)(const int64_t *ptr);
    func_awkward_cuda_ptr64_dealloc_t *func_awkward_cuda_ptr64_dealloc =
      reinterpret_cast<func_awkward_cuda_ptr64_dealloc_t *>
      (dlsym(handle, "awkward_cuda_ptr64_dealloc"));

    util::handle_cuda_error((*func_awkward_cuda_ptr64_dealloc)(p));
  }

  template<>
  void cuda_array_deleter<uint64_t>::operator()(uint64_t const *p) {
    auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
    if (!handle) {
      Error err = failure("Failed to find awkward1[cuda]",
                          0,
                          kSliceNone);

      awkward::util::handle_cuda_error(err);
    }

    typedef Error (func_awkward_cuda_ptrU64_dealloc_t)(const uint64_t *ptr);
    func_awkward_cuda_ptrU64_dealloc_t *func_awkward_cuda_ptrU64_dealloc =
      reinterpret_cast<func_awkward_cuda_ptrU64_dealloc_t *>
      (dlsym(handle, "awkward_cuda_ptrU64_dealloc"));

    util::handle_cuda_error((*func_awkward_cuda_ptrU64_dealloc)(p));
  }

  template<>
  void cuda_array_deleter<float>::operator()(float const *p) {

    auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
    if (!handle) {
      Error err = failure("Failed to find awkward1[cuda]",
                          0,
                          kSliceNone);

      awkward::util::handle_cuda_error(err);
    }

    typedef Error (func_awkward_cuda_ptrfloat32_dealloc_t)(const float *
    ptr);
    func_awkward_cuda_ptrfloat32_dealloc_t
      *func_awkward_cuda_ptrfloat32_dealloc =
      reinterpret_cast<func_awkward_cuda_ptrfloat32_dealloc_t *>
      (dlsym(handle, "awkward_cuda_ptrfloat32_dealloc"));

    util::handle_cuda_error((*func_awkward_cuda_ptrfloat32_dealloc)(p));
  }

  template<>
  void cuda_array_deleter<double>::operator()(double const *p) {
    auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
    if (!handle) {
      Error err = failure("Failed to find awkward1[cuda]",
                          0,
                          kSliceNone);

      awkward::util::handle_cuda_error(err);
    }

    typedef Error (func_awkward_cuda_ptrfloat64_dealloc_t)(const double *
    ptr);
    func_awkward_cuda_ptrfloat64_dealloc_t
      *func_awkward_cuda_ptrfloat64_dealloc =
      reinterpret_cast<func_awkward_cuda_ptrfloat64_dealloc_t *>
      (dlsym(handle, "awkward_cuda_ptrfloat64_dealloc"));

    awkward::util::handle_cuda_error((*func_awkward_cuda_ptrfloat64_dealloc)
                                       (p));
  }
#endif

  template<typename T>
  int get_ptr_device_num(kernel::Lib ptr_lib, T *ptr) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
        return -1;
      }

      int device_num = -1;

      typedef Error (func_awkward_cuda_ptr_device_num_t)
        (int &device_num, void *ptr);

      func_awkward_cuda_ptr_device_num_t
        *func_awkward_cuda_ptr_device_num =
        reinterpret_cast<func_awkward_cuda_ptr_device_num_t *>
        (dlsym(handle, "awkward_cuda_ptr_device_num"));

      Error err = (*func_awkward_cuda_ptr_device_num)(device_num, (void *) ptr);
      util::handle_cuda_error(err);
      return
        device_num;
    }
#endif
    return -1;
  }

  template int get_ptr_device_num(kernel::Lib ptr_lib, bool* ptr);
  template int get_ptr_device_num(kernel::Lib ptr_lib, int8_t* ptr);
  template int get_ptr_device_num(kernel::Lib ptr_lib, uint8_t* ptr);
  template int get_ptr_device_num(kernel::Lib ptr_lib, int16_t* ptr);
  template int get_ptr_device_num(kernel::Lib ptr_lib, uint16_t* ptr);
  template int get_ptr_device_num(kernel::Lib ptr_lib, int32_t* ptr);
  template int get_ptr_device_num(kernel::Lib ptr_lib, uint32_t* ptr);
  template int get_ptr_device_num(kernel::Lib ptr_lib, int64_t* ptr);
  template int get_ptr_device_num(kernel::Lib ptr_lib, uint64_t* ptr);
  template int get_ptr_device_num(kernel::Lib ptr_lib, float* ptr);
  template int get_ptr_device_num(kernel::Lib ptr_lib, double* ptr);

  template <typename T>
  std::string get_ptr_device_name(kernel::Lib ptr_lib, T* ptr){
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
        return std::string("");
      }

      std::string device_name = std::string("");

      typedef Error (func_awkward_cuda_ptr_device_name_t)
        (std::string& device_name, void* ptr);
      func_awkward_cuda_ptr_device_name_t
        *func_awkward_cuda_ptr_device_name =
        reinterpret_cast<func_awkward_cuda_ptr_device_name_t *>
        (dlsym(handle, "awkward_cuda_ptr_device_name"));

      Error err = (*func_awkward_cuda_ptr_device_name)(device_name,
                                                       (void*)ptr);
      util::handle_cuda_error(err);
      return device_name;
    }
#endif
    return std::string("");
  }

  template std::string get_ptr_device_name(kernel::Lib ptr_lib, bool* ptr);
  template std::string get_ptr_device_name(kernel::Lib ptr_lib, int8_t* ptr);
  template std::string get_ptr_device_name(kernel::Lib ptr_lib, uint8_t* ptr);
  template std::string get_ptr_device_name(kernel::Lib ptr_lib, int16_t* ptr);
  template std::string get_ptr_device_name(kernel::Lib ptr_lib, uint16_t* ptr);
  template std::string get_ptr_device_name(kernel::Lib ptr_lib, int32_t* ptr);
  template std::string get_ptr_device_name(kernel::Lib ptr_lib, uint32_t* ptr);
  template std::string get_ptr_device_name(kernel::Lib ptr_lib, int64_t* ptr);
  template std::string get_ptr_device_name(kernel::Lib ptr_lib, uint64_t* ptr);
  template std::string get_ptr_device_name(kernel::Lib ptr_lib, float* ptr);
  template std::string get_ptr_device_name(kernel::Lib ptr_lib, double* ptr);

  template <>
  Error H2D(
    kernel::Lib ptr_lib,
    bool** to_ptr,
    bool* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
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
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);
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
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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

  template <>
  Error D2H(
    kernel::Lib ptr_lib,
    bool** to_ptr,
    bool* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_D2Hbool_t)
        (bool **to_ptr, bool *from_ptr, int64_t length);
      func_awkward_cuda_D2Hbool_t
        *func_awkward_cuda_D2Hbool =
        reinterpret_cast<func_awkward_cuda_D2Hbool_t *>
        (dlsym(handle, "awkward_cuda_D2Hbool"));

      return (*func_awkward_cuda_D2Hbool)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error D2H(
    kernel::Lib ptr_lib,
    int8_t** to_ptr,
    int8_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }

      typedef Error (func_awkward_cuda_D2H8_t)
        (int8_t **to_ptr, int8_t *from_ptr, int8_t length);
      func_awkward_cuda_D2H8_t
        *func_awkward_cuda_D2H8 =
        reinterpret_cast<func_awkward_cuda_D2H8_t *>
        (dlsym(handle, "awkward_cuda_D2H8"));

      return (*func_awkward_cuda_D2H8)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error D2H(
    kernel::Lib ptr_lib,
    uint8_t** to_ptr,
    uint8_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_D2HU8_t)
        (uint8_t **to_ptr, uint8_t *from_ptr, int64_t length);
      func_awkward_cuda_D2HU8_t
        *func_awkward_cuda_D2HU8 =
        reinterpret_cast<func_awkward_cuda_D2HU8_t *>
        (dlsym(handle, "awkward_cuda_D2HU8"));

      return (*func_awkward_cuda_D2HU8)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error D2H(
    kernel::Lib ptr_lib,
    int16_t** to_ptr,
    int16_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_D2H16_t)
        (int16_t **to_ptr, int16_t *from_ptr, int64_t length);
      func_awkward_cuda_D2H16_t
        *func_awkward_cuda_D2H16 =
        reinterpret_cast<func_awkward_cuda_D2H16_t *>
        (dlsym(handle, "awkward_cuda_D2H16"));

      return (*func_awkward_cuda_D2H16)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error D2H(
    kernel::Lib ptr_lib,
    uint16_t** to_ptr,
    uint16_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_D2HU16_t)
        (uint16_t **to_ptr, uint16_t *from_ptr, int64_t length);
      func_awkward_cuda_D2HU16_t
        *func_awkward_cuda_D2HU16 =
        reinterpret_cast<func_awkward_cuda_D2HU16_t *>
        (dlsym(handle, "awkward_cuda_D2HU16"));

      return (*func_awkward_cuda_D2HU16)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error D2H(
    kernel::Lib ptr_lib,
    int32_t** to_ptr,
    int32_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_D2H32_t)
        (int32_t **to_ptr, int32_t *from_ptr, int64_t length);
      func_awkward_cuda_D2H32_t
        *func_awkward_cuda_D2H32 =
        reinterpret_cast<func_awkward_cuda_D2H32_t *>
        (dlsym(handle, "awkward_cuda_D2H32"));

      return (*func_awkward_cuda_D2H32)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error D2H(
    kernel::Lib ptr_lib,
    uint32_t** to_ptr,
    uint32_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_D2HU32_t)
        (uint32_t **to_ptr, uint32_t *from_ptr, int64_t length);
      func_awkward_cuda_D2HU32_t
        *func_awkward_cuda_D2HU32 =
        reinterpret_cast<func_awkward_cuda_D2HU32_t *>
        (dlsym(handle, "awkward_cuda_D2HU32"));

      return (*func_awkward_cuda_D2HU32)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error D2H(
    kernel::Lib ptr_lib,
    int64_t** to_ptr,
    int64_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_D2H64_t)
        (int64_t **to_ptr, int64_t *from_ptr, int64_t length);
      func_awkward_cuda_D2H64_t
        *func_awkward_cuda_D2H64 =
        reinterpret_cast<func_awkward_cuda_D2H64_t *>
        (dlsym(handle, "awkward_cuda_D2H64"));

      return (*func_awkward_cuda_D2H64)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error D2H(
    kernel::Lib ptr_lib,
    uint64_t** to_ptr,
    uint64_t* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_D2HU64_t)
        (uint64_t **to_ptr, uint64_t *from_ptr, int64_t length);
      func_awkward_cuda_D2HU64_t
        *func_awkward_cuda_D2HU64 =
        reinterpret_cast<func_awkward_cuda_D2HU64_t *>
        (dlsym(handle, "awkward_cuda_D2HU64"));

      return (*func_awkward_cuda_D2HU64)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error D2H(
    kernel::Lib ptr_lib,
    float** to_ptr,
    float* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_D2Hfloat32_t)
        (float **to_ptr, float *from_ptr, int64_t length);
      func_awkward_cuda_D2Hfloat32_t
        *func_awkward_cuda_D2Hfloat32 =
        reinterpret_cast<func_awkward_cuda_D2Hfloat32_t *>
        (dlsym(handle, "awkward_cuda_D2Hfloat32"));

      return (*func_awkward_cuda_D2Hfloat32)(to_ptr, from_ptr, length);
    }
#endif
    return failure("No suitable kernel for transfer",
                   0,
                   kSliceNone);
  }
  template <>
  Error D2H(
    kernel::Lib ptr_lib,
    double** to_ptr,
    double* from_ptr,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if(!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_D2Hfloat64_t)
        (double **to_ptr, double *from_ptr, int64_t length);
      func_awkward_cuda_D2Hfloat64_t
        *func_awkward_cuda_D2Hfloat64 =
        reinterpret_cast<func_awkward_cuda_D2Hfloat64_t *>
        (dlsym(handle, "awkward_cuda_D2Hfloat64"));

      return (*func_awkward_cuda_D2Hfloat64)(to_ptr, from_ptr, length);
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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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
        auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

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

  template <>
  Error ListArray_num_64<int32_t>(
    kernel::Lib ptr_lib,
    int64_t* tonum,
    const int32_t* fromstarts,
    int64_t startsoffset,
    const int32_t* fromstops,
    int64_t stopsoffset,
    int64_t length) {

#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if (!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        awkward::util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_ListArray32_num_64_t)
       (int64_t* tonum,
        const int32_t* fromstarts,
        int64_t startsoffset,
        const int32_t* fromstops,
        int64_t stopsoffset,
        int64_t length);
      func_awkward_cuda_ListArray32_num_64_t
        *func_awkward_cuda_ListArray32_num_64 =
        reinterpret_cast<func_awkward_cuda_ListArray32_num_64_t *>
        (dlsym(handle, "awkward_cuda_ListArray32_num_64"));

      return (*func_awkward_cuda_ListArray32_num_64)(
        tonum,
        fromstarts,
        startsoffset,
        fromstops,
        stopsoffset,
        length);
    }
#endif
    return awkward_listarray32_num_64(
      tonum,
      fromstarts,
      startsoffset,
      fromstops,
      stopsoffset,
      length);
  }
  template <>
  Error ListArray_num_64<uint32_t>(
    kernel::Lib ptr_lib,
    int64_t* tonum,
    const uint32_t* fromstarts,
    int64_t startsoffset,
    const uint32_t* fromstops,
    int64_t stopsoffset,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if (!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        awkward::util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_ListArrayU32_num_64_t)
        (int64_t* tonum,
         const uint32_t* fromstarts,
         int64_t startsoffset,
         const uint32_t* fromstops,
         int64_t stopsoffset,
         int64_t length);
      func_awkward_cuda_ListArrayU32_num_64_t
        *func_awkward_cuda_ListArrayU32_num_64 =
        reinterpret_cast<func_awkward_cuda_ListArrayU32_num_64_t *>
        (dlsym(handle, "awkward_cuda_ListArrayU32_num_64"));

      return (*func_awkward_cuda_ListArrayU32_num_64)(
        tonum,
        fromstarts,
        startsoffset,
        fromstops,
        stopsoffset,
        length);
    }
#endif
    return awkward_listarrayU32_num_64(
      tonum,
      fromstarts,
      startsoffset,
      fromstops,
      stopsoffset,
      length);
  }
  template <>
  Error ListArray_num_64<int64_t>(
    kernel::Lib ptr_lib,
    int64_t* tonum,
    const int64_t* fromstarts,
    int64_t startsoffset,
    const int64_t* fromstops,
    int64_t stopsoffset,
    int64_t length) {
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen(lib_callback->awkward_cuda_path().c_str(), RTLD_LAZY);

      if (!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        awkward::util::handle_cuda_error(err);
      }
      typedef Error (func_awkward_cuda_ListArray64_num_64_t)
        (int64_t* tonum,
         const int64_t* fromstarts,
         int64_t startsoffset,
         const int64_t* fromstops,
         int64_t stopsoffset,
         int64_t length);
      func_awkward_cuda_ListArray64_num_64_t
        *func_awkward_cuda_ListArray64_num_64 =
        reinterpret_cast<func_awkward_cuda_ListArray64_num_64_t *>
        (dlsym(handle, "awkward_cuda_ListArray32_num_64"));

      return (*func_awkward_cuda_ListArray64_num_64)(
        tonum,
        fromstarts,
        startsoffset,
        fromstops,
        stopsoffset,
        length);
    }
#endif
    return awkward_listarray64_num_64(
      tonum,
      fromstarts,
      startsoffset,
      fromstops,
      stopsoffset,
      length);
  }
}



