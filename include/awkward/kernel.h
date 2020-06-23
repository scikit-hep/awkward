// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNEL_H
#define AWKWARD_KERNEL_H

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/cpu-kernels/allocators.h"
#include "awkward/cpu-kernels/getitem.h"

#ifndef _MSC_VER
  #include "dlfcn.h"
#endif

using namespace awkward;

namespace kernel {
  enum Lib {
    cpu_kernels,
    cuda_kernels
  };


  /// @class array_deleter
  ///
  /// @brief Used as a `std::shared_ptr` deleter (second argument) to
  /// overload `delete ptr` with `delete[] ptr`.
  ///
  /// This is necessary for `std::shared_ptr` to contain array buffers.
  ///
  /// See also
  ///   - no_deleter, which does not free memory at all (for borrowed
  ///     references).
  ///   - pyobject_deleter, which reduces the reference count of a
  ///     Python object when there are no more C++ shared pointers
  ///     referencing it.
  template<typename T>
  class EXPORT_SYMBOL array_deleter {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(T const *p) {
      delete[] p;
    }
  };

#ifndef _MSC_VER
  template<typename T>
  class EXPORT_SYMBOL cuda_array_deleter;

  template<>
  class cuda_array_deleter<bool> {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(bool const *p) {

      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);
      if (!handle) {
        Error err = failure("Failed to find awkward1[cuda]",
                            0,
                            kSliceNone);

        awkward::util::handle_cuda_error(err);
      }

      typedef Error (func_awkward_cuda_ptrbool_dealloc_t)(const bool *ptr);
      func_awkward_cuda_ptrbool_dealloc_t *func_awkward_cuda_ptrbool_dealloc =
        reinterpret_cast<func_awkward_cuda_ptrbool_dealloc_t *>
        (dlsym(handle, "awkward_cuda_ptrbool_dealloc"));

      util::handle_cuda_error((*func_awkward_cuda_ptrbool_dealloc)(p));
    }
  };

  template<>
  class cuda_array_deleter<int8_t> {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(int8_t const *p) {

      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);
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
  };

  template<>
  class cuda_array_deleter<uint8_t> {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(uint8_t const *p) {

      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);
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
  };

  template<>
  class cuda_array_deleter<int16_t> {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(int16_t const *p) {

      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);
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
  };

  template<>
  class cuda_array_deleter<uint16_t> {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(uint16_t const *p) {

      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);
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
  };

  template<>
  class cuda_array_deleter<int32_t> {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(int32_t const *p) {

      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);
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
  };

  template<>
  class cuda_array_deleter<uint32_t> {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(uint32_t const *p) {

      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);
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
  };

  template<>
  class cuda_array_deleter<int64_t> {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(int64_t const *p) {

      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);
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
  };

  template<>
  class cuda_array_deleter<uint64_t> {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(uint64_t const *p) {

      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);
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
  };

  template<>
  class cuda_array_deleter<float> {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(float const *p) {

      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);
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
  };

  template<>
  class cuda_array_deleter<double> {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(double const *p) {

      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1"
                           ".0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);
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
  };
#endif

  template<typename T>
  Error
  H2D(
    kernel::Lib ptr_lib,
    T **to_ptr,
    T *from_ptr,
    int64_t length);

  template<typename T>
  int get_ptr_device_num(kernel::Lib ptr_lib, T *ptr) {

#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

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

  template <typename T>
  std::string get_ptr_device_name(kernel::Lib ptr_lib, T* ptr){
#ifndef _MSC_VER
    if(ptr_lib == kernel::Lib::cuda_kernels) {
      auto handle = dlopen("/home/trickarcher/gsoc_2020/awkward-1.0/src/cuda-kernels/build/libawkward-cuda-kernels.so", RTLD_LAZY);

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

  /// @class no_deleter
  ///
  /// @brief Used as a `std::shared_ptr` deleter (second argument) to
  /// overload `delete ptr` with nothing (no dereferencing).
  ///
  /// This could be used to pass borrowed references with the same
  /// C++ type as owned references.
  ///
  /// See also
  ///   - array_deleter, which frees array buffers, rather than objects.
  ///   - pyobject_deleter, which reduces the reference count of a
  ///     Python object when there are no more C++ shared pointers
  ///     referencing it.
  template<typename T>
  class EXPORT_SYMBOL no_deleter {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(T const *p) { }
  };


  template<typename T>
  std::shared_ptr<T>
    ptr_alloc(kernel::Lib ptr_lib,
              int64_t length);

  template <typename T>
  T
    index_getitem_at_nowrap(kernel::Lib ptr_lib,
                            T* ptr,
                            int64_t offset,
                            int64_t at);

  template <typename T>
  void
    index_setitem_at_nowrap(kernel::Lib ptr_lib,
                            T* ptr,
                            int64_t offset,
                            int64_t at,
                            T value);

};

#endif //AWKWARD_KERNEL_H
