// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNEL_H
#define AWKWARD_KERNEL_H

#include "awkward/common.h"
#include "awkward/cpu-kernels/allocators.h"
#include "awkward/cpu-kernels/getitem.h"
#include <dlfcn.h>


namespace kernel {
    template<typename T>
    T* ptr_alloc(int64_t length, KernelsLib ptr_lib);

    template <typename T>
    T* host_to_device_buff_transfer(T* ptr, int64_t length, KernelsLib ptr_lib);

    template <typename T>
    T index_getitem_at_nowrap(T* ptr, int64_t offset, int64_t at, KernelsLib ptr_lib);

    template<typename T>
    class EXPORT_SYMBOL array_deleter;

    template <>
    class EXPORT_SYMBOL array_deleter<int8_t> {
      public:

      array_deleter(KernelsLib ptr_lib) : ptr_lib_(ptr_lib) { }
      /// @brief Called by `std::shared_ptr` when its reference count reaches
      /// zero.
      void operator()(int8_t const *p) {
        if(ptr_lib_ == cpu_kernels)
          delete[] p;
        else if(ptr_lib_ == cuda_kernels) {
          auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
          if (!handle) {
            fputs (dlerror(), stderr);
          }
          typedef void* (func_awkward_cuda_ptri8_dealloc_t)(int8_t* ptr);
          func_awkward_cuda_ptri8_dealloc_t *func_awkward_cuda_ptri8_dealloc = reinterpret_cast<func_awkward_cuda_ptri8_dealloc_t *>
          (dlsym(handle, "awkward_cuda_ptri8_dealloc"));
        }
      }
      private:
      KernelsLib ptr_lib_;
    };

  template<>
  class EXPORT_SYMBOL array_deleter<uint8_t> {
    public:

    array_deleter(KernelsLib ptr_lib) : ptr_lib_(ptr_lib) { }
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(uint8_t const *p) {
      if(ptr_lib_ == cpu_kernels)
        delete[] p;
      else if(ptr_lib_ == cuda_kernels) {
        auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
        if (!handle) {
          fputs (dlerror(), stderr);
        }
        typedef void* (func_awkward_cuda_ptriU8_dealloc_t)(int8_t* ptr);
        func_awkward_cuda_ptriU8_dealloc_t *func_awkward_cuda_ptriU8_dealloc = reinterpret_cast<func_awkward_cuda_ptriU8_dealloc_t *>
        (dlsym(handle, "awkward_cuda_ptriU8_dealloc"));
      }
    }
    private:
    KernelsLib ptr_lib_;
  };

  template<>
  class EXPORT_SYMBOL array_deleter<int32_t> {
    public:

    array_deleter(KernelsLib ptr_lib) : ptr_lib_(ptr_lib) { }
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(int32_t const *p) {
      if(ptr_lib_ == cpu_kernels)
        delete[] p;
      else if(ptr_lib_ == cuda_kernels) {
        auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
        if (!handle) {
          fputs (dlerror(), stderr);
        }
        typedef void* (func_awkward_cuda_ptri32_dealloc_t)(int32_t* ptr);
        func_awkward_cuda_ptri32_dealloc_t *func_awkward_cuda_ptri32_dealloc = reinterpret_cast<func_awkward_cuda_ptri32_dealloc_t *>
        (dlsym(handle, "awkward_cuda_ptri32_dealloc"));
      }
    }
    private:
    KernelsLib ptr_lib_;
  };

  template<>
  class EXPORT_SYMBOL array_deleter<uint32_t> {
    public:

    array_deleter(KernelsLib ptr_lib) : ptr_lib_(ptr_lib) { }
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(uint32_t const *p) {
      if(ptr_lib_ == cpu_kernels)
        delete[] p;
      else if(ptr_lib_ == cuda_kernels) {
        auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
        if (!handle) {
          fputs (dlerror(), stderr);
        }
        typedef void* (func_awkward_cuda_ptriU32_dealloc_t)(uint32_t* ptr);
        func_awkward_cuda_ptriU32_dealloc_t *func_awkward_cuda_ptriU32_dealloc = reinterpret_cast<func_awkward_cuda_ptriU32_dealloc_t *>
        (dlsym(handle, "awkward_cuda_ptriU32_dealloc"));
      }
    }
    private:
    KernelsLib ptr_lib_;
  };

  template<>
  class EXPORT_SYMBOL array_deleter<int64_t> {
    public:

    array_deleter(KernelsLib ptr_lib) : ptr_lib_(ptr_lib) { }
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(int64_t const *p) {
      if(ptr_lib_ == cpu_kernels)
        delete[] p;
      else if(ptr_lib_ == cuda_kernels) {
        auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
        if (!handle) {
          fputs (dlerror(), stderr);
        }
        typedef void* (func_awkward_cuda_ptri64_dealloc_t)(int8_t* ptr);
        func_awkward_cuda_ptri64_dealloc_t *func_awkward_cuda_ptri64_dealloc = reinterpret_cast<func_awkward_cuda_ptri64_dealloc_t *>
        (dlsym(handle, "awkward_cuda_ptri64_dealloc"));
      }
    }
    private:
    KernelsLib ptr_lib_;
  };

  template<>
  class EXPORT_SYMBOL array_deleter<float> {
    public:

    array_deleter(KernelsLib ptr_lib) : ptr_lib_(ptr_lib) { }
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(float const *p) {
      if(ptr_lib_ == cpu_kernels)
        delete[] p;
      else if(ptr_lib_ == cuda_kernels) {
        auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
        if (!handle) {
          fputs (dlerror(), stderr);
        }
        typedef void* (func_awkward_cuda_ptrf_dealloc_t)(float* ptr);
        func_awkward_cuda_ptrf_dealloc_t *func_awkward_cuda_ptrf_dealloc = reinterpret_cast<func_awkward_cuda_ptrf_dealloc_t *>
        (dlsym(handle, "awkward_cuda_ptrf_dealloc"));
      }
    }
    private:
    KernelsLib ptr_lib_;
  };

  template<>
  class EXPORT_SYMBOL array_deleter<double> {
    public:

    array_deleter(KernelsLib ptr_lib) : ptr_lib_(ptr_lib) { }
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(double const *p) {
      if(ptr_lib_ == cpu_kernels)
        delete[] p;
      else if(ptr_lib_ == cuda_kernels) {
        auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
        if (!handle) {
          fputs (dlerror(), stderr);
        }
        typedef void* (func_awkward_cuda_ptrd_dealloc_t)(double* ptr);
        func_awkward_cuda_ptrd_dealloc_t *func_awkward_cuda_ptrd_dealloc = reinterpret_cast<func_awkward_cuda_ptrd_dealloc_t *>
        (dlsym(handle, "awkward_cuda_ptrd_dealloc"));
      }
    }
    private:
    KernelsLib ptr_lib_;
  };

  template<>
  class EXPORT_SYMBOL array_deleter<bool> {
    public:

    array_deleter(KernelsLib ptr_lib) : ptr_lib_(ptr_lib) { }
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(bool const *p) {
      if(ptr_lib_ == cpu_kernels)
        delete[] p;
      else if(ptr_lib_ == cuda_kernels) {
        auto handle = dlopen("libawkward-cuda-kernels.so", RTLD_LAZY);
        if (!handle) {
          fputs (dlerror(), stderr);
        }
        typedef void* (func_awkward_cuda_ptrb_dealloc_t)(bool* ptr);
        func_awkward_cuda_ptrb_dealloc_t *func_awkward_cuda_ptrb_dealloc = reinterpret_cast<func_awkward_cuda_ptrb_dealloc_t *>
        (dlsym(handle, "awkward_cuda_ptrb_dealloc"));
      }
    }
    private:
    KernelsLib ptr_lib_;
  };

  template <>
  int8_t* ptr_alloc(int64_t length, KernelsLib ptr_lib);

  template<>
  int8_t* host_to_device_buff_transfer(int8_t* ptr, int64_t length, KernelsLib ptr_lib);

  template <>
  uint8_t* ptr_alloc(int64_t length, KernelsLib ptr_lib);

  template<>
  uint8_t* host_to_device_buff_transfer(uint8_t* ptr, int64_t length, KernelsLib ptr_lib);

  template <>
  int32_t* ptr_alloc(int64_t length, KernelsLib ptr_lib);

  template<>
  int32_t * host_to_device_buff_transfer(int32_t * ptr, int64_t length, KernelsLib ptr_lib);

  template <>
  uint32_t* ptr_alloc(int64_t length, KernelsLib ptr_lib);

  template<>
  uint32_t* host_to_device_buff_transfer(uint32_t* ptr, int64_t length, KernelsLib ptr_lib);

  template <>
  int64_t* ptr_alloc(int64_t length, KernelsLib ptr_lib);

  template<>
  int64_t* host_to_device_buff_transfer(int64_t* ptr, int64_t length, KernelsLib ptr_lib);

  template <>
  float *ptr_alloc(int64_t length, KernelsLib ptr_lib);

  template<>
  float* host_to_device_buff_transfer(float* ptr, int64_t length, KernelsLib ptr_lib);

  template <>
  double* ptr_alloc(int64_t length, KernelsLib ptr_lib);

  template<>
  double* host_to_device_buff_transfer(double* ptr, int64_t length, KernelsLib ptr_lib);

  template <>
  bool *ptr_alloc(int64_t length, KernelsLib ptr_lib);

  template<>
  bool* host_to_device_buff_transfer(bool* ptr, int64_t length, KernelsLib ptr_lib);

  template<>
  int8_t index_getitem_at_nowrap(int8_t* ptr, int64_t offset, int64_t at, KernelsLib ptr_lib);

  template<>
  uint8_t index_getitem_at_nowrap(uint8_t* ptr, int64_t offset, int64_t at, KernelsLib ptr_lib);

  template<>
  int32_t index_getitem_at_nowrap(int32_t* ptr, int64_t offset, int64_t at, KernelsLib ptr_lib);

  template<>
  uint32_t index_getitem_at_nowrap(uint32_t* ptr, int64_t offset, int64_t at, KernelsLib ptr_lib);

  template<>
  int64_t index_getitem_at_nowrap(int64_t * ptr, int64_t offset, int64_t at, KernelsLib ptr_lib);
};

#endif //AWKWARD_KERNEL_H
