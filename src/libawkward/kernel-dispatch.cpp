// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/kernels/operations.h"
#include "awkward/kernels/getitem.h"
#include "awkward/kernels/identities.h"
#include "awkward/kernels/reducers.h"
#include "awkward/kernels/cuda-utils.h"

#include "awkward/kernel-dispatch.h"

#define FORM_KERNEL(libFnName, ptr_lib)          \
  auto handle = acquire_handle(ptr_lib);         \
  typedef decltype(libFnName) functor_type;      \
  auto* libFnName##_fcn =                        \
    reinterpret_cast<functor_type*>(acquire_symbol(handle, #libFnName));

namespace awkward {
  namespace kernel {

    std::shared_ptr<LibraryCallback> lib_callback =
      std::make_shared<LibraryCallback>();

    LibraryCallback::LibraryCallback() {
      lib_path_callbacks[kernel::lib::cuda] =
        std::vector<std::shared_ptr<LibraryPathCallback>>();
    }

    void LibraryCallback::add_library_path_callback(
      kernel::lib ptr_lib,
      const std::shared_ptr<LibraryPathCallback> &callback) {
      std::lock_guard<std::mutex> lock(lib_path_callbacks_mutex);
      lib_path_callbacks.at(ptr_lib).push_back(callback);
    }

    std::string LibraryCallback::awkward_library_path(kernel::lib ptr_lib) {
#ifndef _MSC_VER
      for (const auto& i : lib_path_callbacks.at(ptr_lib)) {
        auto handle = dlopen(i->library_path().c_str(), RTLD_LAZY);

        if (handle) {
          return i->library_path();
        }
      }
#endif
      return std::string("");
    }

    void *acquire_handle(kernel::lib ptr_lib) {
      void *handle = nullptr;
#ifndef _MSC_VER
      std::string path = lib_callback->awkward_library_path(ptr_lib);
      if (!path.empty()) {
        handle = dlopen(path.c_str(), RTLD_LAZY);
      }
      if (!handle) {
        if (ptr_lib == kernel::lib::cuda) {
          throw std::invalid_argument(
            "array resides on a GPU, but 'awkward1-cuda-kernels' is not "
            "installed; install it with:\n\n    "
            "pip install awkward1[cuda] --upgrade");
        }
      }
#endif
      return handle;
    }

    void *acquire_symbol(void* handle, std::string symbol_name) {
      void *symbol_ptr = nullptr;
#ifndef _MSC_VER
      symbol_ptr = dlsym(handle, symbol_name.c_str());

      if (!symbol_ptr) {
        std::stringstream out;
        out << symbol_name;
        out << " not found in .so";
        throw std::runtime_error(out.str());
      }
#endif
      return symbol_ptr;
    }

    template<typename T>
    int get_ptr_device_num(kernel::lib ptr_lib, T *ptr) {
      if (ptr_lib == kernel::lib::cuda) {
        auto handle = acquire_handle(kernel::lib::cuda);

        int device_num = -1;

        typedef Error (func_awkward_cuda_ptr_device_num_t)
          (int &device_num, void *ptr);

        func_awkward_cuda_ptr_device_num_t
          *func_awkward_cuda_ptr_device_num =
          reinterpret_cast<func_awkward_cuda_ptr_device_num_t *>
          (acquire_symbol(handle, "awkward_cuda_ptr_device_num"));

        Error err = (*func_awkward_cuda_ptr_device_num)(device_num,
                                                        (void *) ptr);
        util::handle_error(err);
        return
          device_num;
      }
      return -1;
    }

    template int get_ptr_device_num(kernel::lib ptr_lib, void *ptr);

    template int get_ptr_device_num(kernel::lib ptr_lib, bool *ptr);

    template int get_ptr_device_num(kernel::lib ptr_lib, int8_t *ptr);

    template int get_ptr_device_num(kernel::lib ptr_lib, uint8_t *ptr);

    template int get_ptr_device_num(kernel::lib ptr_lib, int16_t *ptr);

    template int get_ptr_device_num(kernel::lib ptr_lib, uint16_t *ptr);

    template int get_ptr_device_num(kernel::lib ptr_lib, int32_t *ptr);

    template int get_ptr_device_num(kernel::lib ptr_lib, uint32_t *ptr);

    template int get_ptr_device_num(kernel::lib ptr_lib, int64_t *ptr);

    template int get_ptr_device_num(kernel::lib ptr_lib, uint64_t *ptr);

    template int get_ptr_device_num(kernel::lib ptr_lib, float *ptr);

    template int get_ptr_device_num(kernel::lib ptr_lib, double *ptr);

    template<typename T>
    std::string get_ptr_device_name(kernel::lib ptr_lib, T *ptr) {
      if (ptr_lib == kernel::lib::cuda) {
        auto handle = acquire_handle(kernel::lib::cuda);

        std::string device_name = std::string("");

        typedef Error (func_awkward_cuda_ptr_device_name_t)
          (std::string &device_name, void *ptr);
        func_awkward_cuda_ptr_device_name_t
          *func_awkward_cuda_ptr_device_name =
          reinterpret_cast<func_awkward_cuda_ptr_device_name_t *>
          (acquire_symbol(handle, "awkward_cuda_ptr_device_name"));

        Error err = (*func_awkward_cuda_ptr_device_name)(device_name,
                                                         (void *) ptr);
        util::handle_error(err);
        return device_name;
      }
      return std::string("");
    }

    template std::string get_ptr_device_name(kernel::lib ptr_lib, void *ptr);

    template std::string get_ptr_device_name(kernel::lib ptr_lib, bool *ptr);

    template std::string get_ptr_device_name(kernel::lib ptr_lib, int8_t *ptr);

    template std::string get_ptr_device_name(kernel::lib ptr_lib, uint8_t *ptr);

    template std::string get_ptr_device_name(kernel::lib ptr_lib, int16_t *ptr);

    template std::string
    get_ptr_device_name(kernel::lib ptr_lib, uint16_t *ptr);

    template std::string get_ptr_device_name(kernel::lib ptr_lib, int32_t *ptr);

    template std::string
    get_ptr_device_name(kernel::lib ptr_lib, uint32_t *ptr);

    template std::string get_ptr_device_name(kernel::lib ptr_lib, int64_t *ptr);

    template std::string
    get_ptr_device_name(kernel::lib ptr_lib, uint64_t *ptr);

    template std::string get_ptr_device_name(kernel::lib ptr_lib, float *ptr);

    template std::string get_ptr_device_name(kernel::lib ptr_lib, double *ptr);

    ERROR copy_to(
      kernel::lib to_lib,
      kernel::lib from_lib,
      void* to_ptr,
      void* from_ptr,
      int64_t bytelength) {
      if (from_lib == lib::cpu  &&  to_lib == lib::cuda) {
        FORM_KERNEL(awkward_cuda_host_to_device, kernel::lib::cuda);
        return (*awkward_cuda_host_to_device_fcn)(to_ptr, from_ptr, bytelength);
      }
      else if (from_lib == lib::cuda  &&  to_lib == lib::cpu) {
        FORM_KERNEL(awkward_cuda_device_to_host, kernel::lib::cuda);
        return (*awkward_cuda_device_to_host_fcn)(to_ptr, from_ptr, bytelength);
      }
      else {
        throw std::runtime_error("unrecognized combination of from_lib and to_lib");
      }
    }

    const std::string
    fully_qualified_cache_key(const std::string& cache_key, kernel::lib ptr_lib) {
      switch (ptr_lib) {
        case kernel::lib::cuda:
          return cache_key + std::string(":cuda");
        default:
          return cache_key;
      }
    }

  /////////////////////////////////// awkward/kernels/getitem.h

    template<>
    bool NumpyArray_getitem_at0(
      kernel::lib ptr_lib,
      bool *ptr) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArraybool_getitem_at0(ptr);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_NumpyArraybool_getitem_at0, ptr_lib);
        return (*awkward_NumpyArraybool_getitem_at0_fcn)(ptr);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in bool NumpyArray_getitem_at0");
      }
    }

    template<>
    int8_t NumpyArray_getitem_at0(
      kernel::lib ptr_lib,
      int8_t *ptr) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray8_getitem_at0(ptr);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_NumpyArray8_getitem_at0, ptr_lib);
        return (*awkward_NumpyArray8_getitem_at0_fcn)(ptr);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in int8_t NumpyArray_getitem_at0");
      }
    }

    template<>
    uint8_t NumpyArray_getitem_at0(
      kernel::lib ptr_lib,
      uint8_t *ptr) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArrayU8_getitem_at0(ptr);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_NumpyArrayU8_getitem_at0, ptr_lib);
        return (*awkward_NumpyArrayU8_getitem_at0_fcn)(ptr);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in uint8_t NumpyArray_getitem_at0");
      }
    }

    template<>
    int16_t NumpyArray_getitem_at0(
      kernel::lib ptr_lib,
      int16_t *ptr) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray16_getitem_at0(ptr);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_NumpyArray16_getitem_at0, ptr_lib);
        return (*awkward_NumpyArray16_getitem_at0_fcn)(ptr);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in int16_t NumpyArray_getitem_at0");
      }
    }

    template<>
    uint16_t NumpyArray_getitem_at0(
      kernel::lib ptr_lib,
      uint16_t *ptr) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArrayU16_getitem_at0(ptr);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_NumpyArrayU16_getitem_at0, ptr_lib);
        return (*awkward_NumpyArrayU16_getitem_at0_fcn)(ptr);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in uint16_t NumpyArray_getitem_at0");
      }
    }

    template<>
    int32_t NumpyArray_getitem_at0(
      kernel::lib ptr_lib,
      int32_t *ptr) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray32_getitem_at0(ptr);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_NumpyArray32_getitem_at0, ptr_lib);
        return (*awkward_NumpyArray32_getitem_at0_fcn)(ptr);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in int32_t NumpyArray_getitem_at0");
      }
    }

    template<>
    uint32_t NumpyArray_getitem_at0(
      kernel::lib ptr_lib,
      uint32_t *ptr) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArrayU32_getitem_at0(ptr);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_NumpyArrayU32_getitem_at0, ptr_lib);
        return (*awkward_NumpyArrayU32_getitem_at0_fcn)(ptr);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in uint32_t NumpyArray_getitem_at0");
      }
    }

    template<>
    int64_t NumpyArray_getitem_at0(
        kernel::lib ptr_lib,
        int64_t *ptr) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray64_getitem_at0(ptr);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_NumpyArray64_getitem_at0, ptr_lib);
        return (*awkward_NumpyArray64_getitem_at0_fcn)(ptr);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in int64_t NumpyArray_getitem_at0");
      }
    }

    template<>
    uint64_t NumpyArray_getitem_at0(
        kernel::lib ptr_lib,
        uint64_t *ptr) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArrayU64_getitem_at0(ptr);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_NumpyArrayU64_getitem_at0, ptr_lib);
        return (*awkward_NumpyArrayU64_getitem_at0_fcn)(ptr);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in uint64_t NumpyArray_getitem_at0");
      }
    }

    template<>
    float NumpyArray_getitem_at0(
      kernel::lib ptr_lib,
      float *ptr) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArrayfloat32_getitem_at0(ptr);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_NumpyArrayfloat32_getitem_at0, ptr_lib);
        return (*awkward_NumpyArrayfloat32_getitem_at0_fcn)(ptr);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in float NumpyArray_getitem_at0");
      }
    }

    template<>
    double NumpyArray_getitem_at0(
      kernel::lib ptr_lib,
      double *ptr) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArrayfloat64_getitem_at0(ptr);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_NumpyArrayfloat64_getitem_at0, ptr_lib);
        return (*awkward_NumpyArrayfloat64_getitem_at0_fcn)(ptr);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in double NumpyArray_getitem_at0");
      }
    }

    // FIXME: move regularize_rangeslice to common.h; it's not a kernel.
    void regularize_rangeslice(
      int64_t *start,
      int64_t *stop,
      bool posstep,
      bool hasstart,
      bool hasstop,
      int64_t length) {
      return awkward_regularize_rangeslice(
        start,
        stop,
        posstep,
        hasstart,
        hasstop,
        length);
    }

    ERROR regularize_arrayslice_64(
      kernel::lib ptr_lib,
      int64_t *flatheadptr,
      int64_t lenflathead,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_regularize_arrayslice_64(
          flatheadptr,
          lenflathead,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for regularize_arrayslice_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for regularize_arrayslice_64");
      }
    }

    template<>
    ERROR Index_to_Index64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Index8_to_Index64(
          toptr,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_to_Index64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_to_Index64");
      }
    }

    template<>
    ERROR Index_to_Index64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexU8_to_Index64(
          toptr,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_to_Index64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_to_Index64");
      }
    }

    template<>
    ERROR Index_to_Index64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Index32_to_Index64(
          toptr,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_to_Index64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_to_Index64");
      }
    }

    template<>
    ERROR Index_to_Index64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexU32_to_Index64(
          toptr,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_to_Index64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_to_Index64");
      }
    }

    template<>
    ERROR Index_carry_64<int8_t>(
      kernel::lib ptr_lib,
      int8_t *toindex,
      const int8_t *fromindex,
      const int64_t *carry,
      int64_t lenfromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Index8_carry_64(
          toindex,
          fromindex,
          carry,
          lenfromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_carry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_carry_64");
      }
    }

    template<>
    ERROR Index_carry_64<uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toindex,
      const uint8_t *fromindex,
      const int64_t *carry,
      int64_t lenfromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexU8_carry_64(
          toindex,
          fromindex,
          carry,
          lenfromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_carry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_carry_64");
      }
    }

    template<>
    ERROR Index_carry_64<int32_t>(
      kernel::lib ptr_lib,
      int32_t *toindex,
      const int32_t *fromindex,
      const int64_t *carry,
      int64_t lenfromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Index32_carry_64(
          toindex,
          fromindex,
          carry,
          lenfromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_carry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_carry_64");
      }
    }

    template<>
    ERROR Index_carry_64<uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toindex,
      const uint32_t *fromindex,
      const int64_t *carry,
      int64_t lenfromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexU32_carry_64(
          toindex,
          fromindex,
          carry,
          lenfromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_carry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_carry_64");
      }
    }

    template<>
    ERROR Index_carry_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int64_t *fromindex,
      const int64_t *carry,
      int64_t lenfromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Index64_carry_64(
          toindex,
          fromindex,
          carry,
          lenfromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_carry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_carry_64");
      }
    }

    template<>
    ERROR Index_carry_nocheck_64<int8_t>(
      kernel::lib ptr_lib,
      int8_t *toindex,
      const int8_t *fromindex,
      const int64_t *carry,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Index8_carry_nocheck_64(
          toindex,
          fromindex,
          carry,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_carry_nocheck_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_carry_nocheck_64");
      }
    }

    template<>
    ERROR Index_carry_nocheck_64<uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toindex,
      const uint8_t *fromindex,
      const int64_t *carry,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexU8_carry_nocheck_64(
          toindex,
          fromindex,
          carry,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_carry_nocheck_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_carry_nocheck_64");
      }
    }

    template<>
    ERROR Index_carry_nocheck_64<int32_t>(
      kernel::lib ptr_lib,
      int32_t *toindex,
      const int32_t *fromindex,
      const int64_t *carry,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Index32_carry_nocheck_64(
          toindex,
          fromindex,
          carry,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_carry_nocheck_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_carry_nocheck_64");
      }
    }

    template<>
    ERROR Index_carry_nocheck_64<uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toindex,
      const uint32_t *fromindex,
      const int64_t *carry,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexU32_carry_nocheck_64(
          toindex,
          fromindex,
          carry,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_carry_nocheck_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_carry_nocheck_64");
      }
    }

    template<>
    ERROR Index_carry_nocheck_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int64_t *fromindex,
      const int64_t *carry,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Index64_carry_nocheck_64(
          toindex,
          fromindex,
          carry,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Index_carry_nocheck_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Index_carry_nocheck_64");
      }
    }

    ERROR slicearray_ravel_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      int64_t ndim,
      const int64_t *shape,
      const int64_t *strides) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_slicearray_ravel_64(
          toptr,
          fromptr,
          ndim,
          shape,
          strides);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for slicearray_ravel_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for slicearray_ravel_64");
      }
    }

    ERROR slicemissing_check_same(
      kernel::lib ptr_lib,
      bool *same,
      const int8_t *bytemask,
      const int64_t *missingindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_slicemissing_check_same(
          same,
          bytemask,
          missingindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for slicemissing_check_same");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for slicemissing_check_same");
      }
    }

    template<>
    ERROR carry_arange(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_carry_arange32(
          toptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for carry_arange");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for carry_arange");
      }
    }

    template<>
    ERROR carry_arange(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_carry_arangeU32(
          toptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for carry_arange");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for carry_arange");
      }
    }

    template<>
    ERROR carry_arange(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_carry_arange64(
          toptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for carry_arange");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for carry_arange");
      }
    }

    template<>
    ERROR Identities_getitem_carry_64(
      kernel::lib ptr_lib,
      int32_t *newidentitiesptr,
      const int32_t *identitiesptr,
      const int64_t *carryptr,
      int64_t lencarry,
      int64_t width,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_getitem_carry_64(
          newidentitiesptr,
          identitiesptr,
          carryptr,
          lencarry,
          width,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_getitem_carry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_getitem_carry_64");
      }
    }

    template<>
    ERROR Identities_getitem_carry_64(
      kernel::lib ptr_lib,
      int64_t *newidentitiesptr,
      const int64_t *identitiesptr,
      const int64_t *carryptr,
      int64_t lencarry,
      int64_t width,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_getitem_carry_64(
          newidentitiesptr,
          identitiesptr,
          carryptr,
          lencarry,
          width,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_getitem_carry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_getitem_carry_64");
      }
    }

    ERROR NumpyArray_contiguous_init_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t skip,
      int64_t stride) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_contiguous_init_64(
          toptr,
          skip,
          stride);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_contiguous_init_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_contiguous_init_64");
      }
    }


    ERROR NumpyArray_contiguous_copy_64(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      const uint8_t *fromptr,
      int64_t len,
      int64_t stride,
      const int64_t *pos) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_contiguous_copy_64(
          toptr,
          fromptr,
          len,
          stride,
          pos);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_contiguous_copy_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_contiguous_copy_64");
      }
    }

    ERROR NumpyArray_contiguous_next_64(
      kernel::lib ptr_lib,
      int64_t *topos,
      const int64_t *frompos,
      int64_t len,
      int64_t skip,
      int64_t stride) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_contiguous_next_64(
          topos,
          frompos,
          len,
          skip,
          stride);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_contiguous_next_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_contiguous_next_64");
      }
    }

    ERROR NumpyArray_getitem_next_null_64(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      const uint8_t *fromptr,
      int64_t len,
      int64_t stride,
      const int64_t *pos) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_getitem_next_null_64(
          toptr,
          fromptr,
          len,
          stride,
          pos);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_getitem_next_null_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_getitem_next_null_64");
      }
    }

    ERROR NumpyArray_getitem_next_at_64(
      kernel::lib ptr_lib,
      int64_t *nextcarryptr,
      const int64_t *carryptr,
      int64_t lencarry,
      int64_t skip,
      int64_t at) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_getitem_next_at_64(
          nextcarryptr,
          carryptr,
          lencarry,
          skip,
          at);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_getitem_next_at_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_getitem_next_at_64");
      }
    }

    ERROR NumpyArray_getitem_next_range_64(
      kernel::lib ptr_lib,
      int64_t *nextcarryptr,
      const int64_t *carryptr,
      int64_t lencarry,
      int64_t lenhead,
      int64_t skip,
      int64_t start,
      int64_t step) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_getitem_next_range_64(
          nextcarryptr,
          carryptr,
          lencarry,
          lenhead,
          skip,
          start,
          step);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_getitem_next_range_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_getitem_next_range_64");
      }
    }

    ERROR NumpyArray_getitem_next_range_advanced_64(
      kernel::lib ptr_lib,
      int64_t *nextcarryptr,
      int64_t *nextadvancedptr,
      const int64_t *carryptr,
      const int64_t *advancedptr,
      int64_t lencarry,
      int64_t lenhead,
      int64_t skip,
      int64_t start,
      int64_t step) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_getitem_next_range_advanced_64(
          nextcarryptr,
          nextadvancedptr,
          carryptr,
          advancedptr,
          lencarry,
          lenhead,
          skip,
          start,
          step);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_getitem_next_range_advanced_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_getitem_next_range_advanced_64");
      }
    }

    ERROR NumpyArray_getitem_next_array_64(
      kernel::lib ptr_lib,
      int64_t *nextcarryptr,
      int64_t *nextadvancedptr,
      const int64_t *carryptr,
      const int64_t *flatheadptr,
      int64_t lencarry,
      int64_t lenflathead,
      int64_t skip) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_getitem_next_array_64(
          nextcarryptr,
          nextadvancedptr,
          carryptr,
          flatheadptr,
          lencarry,
          lenflathead,
          skip);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_getitem_next_array_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_getitem_next_array_64");
      }
    }

    ERROR NumpyArray_getitem_next_array_advanced_64(
      kernel::lib ptr_lib,
      int64_t *nextcarryptr,
      const int64_t *carryptr,
      const int64_t *advancedptr,
      const int64_t *flatheadptr,
      int64_t lencarry,
      int64_t skip) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_getitem_next_array_advanced_64(
          nextcarryptr,
          carryptr,
          advancedptr,
          flatheadptr,
          lencarry,
          skip);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_getitem_next_array_advanced_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_getitem_next_array_advanced_64");
      }
    }

    ERROR NumpyArray_getitem_boolean_numtrue(
      kernel::lib ptr_lib,
      int64_t *numtrue,
      const int8_t *fromptr,
      int64_t length,
      int64_t stride) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_getitem_boolean_numtrue(
          numtrue,
          fromptr,
          length,
          stride);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_getitem_boolean_numtrue");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_getitem_boolean_numtrue");
      }
    }

    ERROR NumpyArray_getitem_boolean_nonzero_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int8_t *fromptr,
      int64_t length,
      int64_t stride) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_getitem_boolean_nonzero_64(
          toptr,
          fromptr,
          length,
          stride);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_getitem_boolean_nonzero_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_getitem_boolean_nonzero_64");
      }
    }

    template<>
    ERROR ListArray_getitem_next_at_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t lenstarts,
      int64_t at) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_getitem_next_at_64(
          tocarry,
          fromstarts,
          fromstops,
          lenstarts,
          at);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_at_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_at_64");
      }
    }

    template<>
    ERROR ListArray_getitem_next_at_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t lenstarts,
      int64_t at) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_getitem_next_at_64(
          tocarry,
          fromstarts,
          fromstops,
          lenstarts,
          at);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_at_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_at_64<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_at_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t lenstarts,
      int64_t at) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_ListArray64_getitem_next_at_64(
         tocarry,
         fromstarts,
         fromstops,
         lenstarts,
         at);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_at_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_at_64<int64_t>");
      }
     }

    template<>
    ERROR ListArray_getitem_next_range_carrylength<int32_t>(
      kernel::lib ptr_lib,
      int64_t *carrylength,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t lenstarts,
      int64_t start,
      int64_t stop,
      int64_t step) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_getitem_next_range_carrylength(
          carrylength,
          fromstarts,
          fromstops,
          lenstarts,
          start,
          stop,
          step);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_range_carrylength<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_range_carrylength<int32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_range_carrylength<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *carrylength,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t lenstarts,
      int64_t start,
      int64_t stop,
      int64_t step) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_ListArrayU32_getitem_next_range_carrylength(
         carrylength,
         fromstarts,
         fromstops,
         lenstarts,
         start,
         stop,
         step);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_range_carrylength<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_range_carrylength<uint32_t>");
      }
     }

    template<>
    ERROR ListArray_getitem_next_range_carrylength<int64_t>(
      kernel::lib ptr_lib,
      int64_t *carrylength,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t lenstarts,
      int64_t start,
      int64_t stop,
      int64_t step) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_getitem_next_range_carrylength(
          carrylength,
          fromstarts,
          fromstops,
          lenstarts,
          start,
          stop,
          step);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_range_carrylength<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_range_carrylength<int64_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_range_64<int32_t>(
      kernel::lib ptr_lib,
      int32_t *tooffsets,
      int64_t *tocarry,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t lenstarts,
      int64_t start,
      int64_t stop,
      int64_t step) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_ListArray32_getitem_next_range_64(
         tooffsets,
         tocarry,
         fromstarts,
         fromstops,
         lenstarts,
         start,
         stop,
         step);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_range_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_range_64<int32_t>");
      }
     }

    template<>
    ERROR ListArray_getitem_next_range_64<uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *tooffsets,
      int64_t *tocarry,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t lenstarts,
      int64_t start,
      int64_t stop,
      int64_t step) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_getitem_next_range_64(
          tooffsets,
          tocarry,
          fromstarts,
          fromstops,
          lenstarts,
          start,
          stop,
          step);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_range_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_range_64<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_range_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      int64_t *tocarry,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t lenstarts,
      int64_t start,
      int64_t stop,
      int64_t step) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_getitem_next_range_64(
          tooffsets,
          tocarry,
          fromstarts,
          fromstops,
          lenstarts,
          start,
          stop,
          step);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_range_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_range_64<int64_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_range_counts_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *total,
      const int32_t *fromoffsets,
      int64_t lenstarts) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_getitem_next_range_counts_64(
          total,
          fromoffsets,
          lenstarts);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_range_counts_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_range_counts_64<int32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_range_counts_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *total,
      const uint32_t *fromoffsets,
      int64_t lenstarts) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_getitem_next_range_counts_64(
          total,
          fromoffsets,
          lenstarts);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_range_counts_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_range_counts_64<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_range_counts_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *total,
      const int64_t *fromoffsets,
      int64_t lenstarts) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_getitem_next_range_counts_64(
          total,
          fromoffsets,
          lenstarts);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_range_counts_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_range_counts_64<int64_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_range_spreadadvanced_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *toadvanced,
      const int64_t *fromadvanced,
      const int32_t *fromoffsets,
      int64_t lenstarts) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_ListArray32_getitem_next_range_spreadadvanced_64(
         toadvanced,
         fromadvanced,
         fromoffsets,
         lenstarts);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_range_spreadadvanced_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_range_spreadadvanced_64<int32_t>");
      }
     }

    template<>
    ERROR ListArray_getitem_next_range_spreadadvanced_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *toadvanced,
      const int64_t *fromadvanced,
      const uint32_t *fromoffsets,
      int64_t lenstarts) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_getitem_next_range_spreadadvanced_64(
          toadvanced,
          fromadvanced,
          fromoffsets,
          lenstarts);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_range_spreadadvanced_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_range_spreadadvanced_64<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_range_spreadadvanced_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toadvanced,
      const int64_t *fromadvanced,
      const int64_t *fromoffsets,
      int64_t lenstarts) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_ListArray64_getitem_next_range_spreadadvanced_64(
         toadvanced,
         fromadvanced,
         fromoffsets,
         lenstarts);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_range_spreadadvanced_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_range_spreadadvanced_64<int64_t>");
      }
     }

    template<>
    ERROR ListArray_getitem_next_array_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toadvanced,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      const int64_t *fromarray,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_getitem_next_array_64(
          tocarry,
          toadvanced,
          fromstarts,
          fromstops,
          fromarray,
          lenstarts,
          lenarray,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_array_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_array_64<int32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_array_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toadvanced,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      const int64_t *fromarray,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_getitem_next_array_64(
          tocarry,
          toadvanced,
          fromstarts,
          fromstops,
          fromarray,
          lenstarts,
          lenarray,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_array_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_array_64<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_array_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toadvanced,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      const int64_t *fromarray,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_getitem_next_array_64(
          tocarry,
          toadvanced,
          fromstarts,
          fromstops,
          fromarray,
          lenstarts,
          lenarray,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_array_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_array_64<int64_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_array_advanced_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toadvanced,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      const int64_t *fromarray,
      const int64_t *fromadvanced,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_getitem_next_array_advanced_64(
          tocarry,
          toadvanced,
          fromstarts,
          fromstops,
          fromarray,
          fromadvanced,
          lenstarts,
          lenarray,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_array_advanced_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_array_advanced_64<int32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_array_advanced_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toadvanced,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      const int64_t *fromarray,
      const int64_t *fromadvanced,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_getitem_next_array_advanced_64(
          tocarry,
          toadvanced,
          fromstarts,
          fromstops,
          fromarray,
          fromadvanced,
          lenstarts,
          lenarray,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_array_advanced_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_array_advanced_64<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_next_array_advanced_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toadvanced,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      const int64_t *fromarray,
      const int64_t *fromadvanced,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_getitem_next_array_advanced_64(
          tocarry,
          toadvanced,
          fromstarts,
          fromstops,
          fromarray,
          fromadvanced,
          lenstarts,
          lenarray,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_next_array_advanced_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_next_array_advanced_64<int64_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_carry_64<int32_t>(
      kernel::lib ptr_lib,
      int32_t *tostarts,
      int32_t *tostops,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      const int64_t *fromcarry,
      int64_t lenstarts,
      int64_t lencarry) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_getitem_carry_64(
          tostarts,
          tostops,
          fromstarts,
          fromstops,
          fromcarry,
          lenstarts,
          lencarry);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_carry_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_carry_64<int32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_carry_64<uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *tostarts,
      uint32_t *tostops,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      const int64_t *fromcarry,
      int64_t lenstarts,
      int64_t lencarry) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_ListArrayU32_getitem_carry_64(
         tostarts,
         tostops,
         fromstarts,
         fromstops,
         fromcarry,
         lenstarts,
         lencarry);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_carry_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_carry_64<uint32_t>");
      }
     }

    template<>
    ERROR ListArray_getitem_carry_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tostarts,
      int64_t *tostops,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      const int64_t *fromcarry,
      int64_t lenstarts,
      int64_t lencarry) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_getitem_carry_64(
          tostarts,
          tostops,
          fromstarts,
          fromstops,
          fromcarry,
          lenstarts,
          lencarry);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_carry_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_carry_64<int64_t>");
      }
    }

    ERROR RegularArray_getitem_next_at_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t at,
      int64_t len,
      int64_t size) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_getitem_next_at_64(
          tocarry,
          at,
          len,
          size);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_getitem_next_at_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_getitem_next_at_64");
      }
    }

    ERROR RegularArray_getitem_next_range_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t regular_start,
      int64_t step,
      int64_t len,
      int64_t size,
      int64_t nextsize) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_getitem_next_range_64(
          tocarry,
          regular_start,
          step,
          len,
          size,
          nextsize);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_getitem_next_range_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_getitem_next_range_64");
      }
    }

    ERROR RegularArray_getitem_next_range_spreadadvanced_64(
      kernel::lib ptr_lib,
      int64_t *toadvanced,
      const int64_t *fromadvanced,
      int64_t len,
      int64_t nextsize) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_getitem_next_range_spreadadvanced_64(
          toadvanced,
          fromadvanced,
          len,
          nextsize);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_getitem_next_range_spreadadvanced_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_getitem_next_range_spreadadvanced_64");
      }
    }

    ERROR RegularArray_getitem_next_array_regularize_64(
      kernel::lib ptr_lib,
      int64_t *toarray,
      const int64_t *fromarray,
      int64_t lenarray,
      int64_t size) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_getitem_next_array_regularize_64(
          toarray,
          fromarray,
          lenarray,
          size);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_getitem_next_array_regularize_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_getitem_next_array_regularize_64");
      }
    }

    ERROR RegularArray_getitem_next_array_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toadvanced,
      const int64_t *fromarray,
      int64_t len,
      int64_t lenarray,
      int64_t size) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_getitem_next_array_64(
          tocarry,
          toadvanced,
          fromarray,
          len,
          lenarray,
          size);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_getitem_next_array_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_getitem_next_array_64");
      }
    }

    ERROR RegularArray_getitem_next_array_advanced_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toadvanced,
      const int64_t *fromadvanced,
      const int64_t *fromarray,
      int64_t len,
      int64_t lenarray,
      int64_t size) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_getitem_next_array_advanced_64(
          tocarry,
          toadvanced,
          fromadvanced,
          fromarray,
          len,
          lenarray,
          size);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_getitem_next_array_advanced_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_getitem_next_array_advanced_64");
      }
    }

    ERROR RegularArray_getitem_carry_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int64_t *fromcarry,
      int64_t lencarry,
      int64_t size) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_getitem_carry_64(
          tocarry,
          fromcarry,
          lencarry,
          size);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_getitem_carry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_getitem_carry_64");
      }
    }

    template<>
    ERROR IndexedArray_numnull<int32_t>(
      kernel::lib ptr_lib,
      int64_t *numnull,
      const int32_t *fromindex,
      int64_t lenindex) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray32_numnull(
          numnull,
          fromindex,
          lenindex);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_numnull<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_numnull<int32_t>");
      }
    }

    template<>
    ERROR IndexedArray_numnull<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *numnull,
      const uint32_t *fromindex,
      int64_t lenindex) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_numnull(
          numnull,
          fromindex,
          lenindex);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_numnull<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_numnull<uint32_t>");
      }
    }

    template<>
    ERROR IndexedArray_numnull<int64_t>(
      kernel::lib ptr_lib,
      int64_t *numnull,
      const int64_t *fromindex,
      int64_t lenindex) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray64_numnull(
          numnull,
          fromindex,
          lenindex);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_numnull<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_numnull<int64_t>");
      }
    }

    template<>
    ERROR IndexedArray_getitem_nextcarry_outindex_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int32_t *toindex,
      const int32_t *fromindex,
      int64_t lenindex,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_IndexedArray32_getitem_nextcarry_outindex_64(
         tocarry,
         toindex,
         fromindex,
         lenindex,
         lencontent);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_nextcarry_outindex_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_nextcarry_outindex_64<int32_t>");
      }
     }

    template<>
    ERROR IndexedArray_getitem_nextcarry_outindex_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      uint32_t *toindex,
      const uint32_t *fromindex,
      int64_t lenindex,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_getitem_nextcarry_outindex_64(
          tocarry,
          toindex,
          fromindex,
          lenindex,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_nextcarry_outindex_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_nextcarry_outindex_64<uint32_t>");
      }
    }

    template<>
    ERROR IndexedArray_getitem_nextcarry_outindex_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toindex,
      const int64_t *fromindex,
      int64_t lenindex,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_IndexedArray64_getitem_nextcarry_outindex_64(
         tocarry,
         toindex,
         fromindex,
         lenindex,
         lencontent);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_nextcarry_outindex_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_nextcarry_outindex_64<int64_t>");
      }
     }

    template<>
    ERROR IndexedArray_getitem_nextcarry_outindex_mask_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toindex,
      const int32_t *fromindex,
      int64_t lenindex,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray32_getitem_nextcarry_outindex_mask_64(
          tocarry,
          toindex,
          fromindex,
          lenindex,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_nextcarry_outindex_mask_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_nextcarry_outindex_mask_64<int32_t>");
      }
    }

    template<>
    ERROR IndexedArray_getitem_nextcarry_outindex_mask_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toindex,
      const uint32_t *fromindex,
      int64_t lenindex,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_IndexedArrayU32_getitem_nextcarry_outindex_mask_64(
         tocarry,
         toindex,
         fromindex,
         lenindex,
         lencontent);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_nextcarry_outindex_mask_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_nextcarry_outindex_mask_64<uint32_t>");
      }
     }

    template<>
    ERROR IndexedArray_getitem_nextcarry_outindex_mask_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toindex,
      const int64_t *fromindex,
      int64_t lenindex,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray64_getitem_nextcarry_outindex_mask_64(
          tocarry,
          toindex,
          fromindex,
          lenindex,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_nextcarry_outindex_mask_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_nextcarry_outindex_mask_64<int64_t>");
      }
    }

    ERROR ListOffsetArray_getitem_adjust_offsets_64(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      int64_t *tononzero,
      const int64_t *fromoffsets,
      int64_t length,
      const int64_t *nonzero,
      int64_t nonzerolength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray_getitem_adjust_offsets_64(
          tooffsets,
          tononzero,
          fromoffsets,
          length,
          nonzero,
          nonzerolength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_getitem_adjust_offsets_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_getitem_adjust_offsets_64");
      }
    }

    ERROR ListOffsetArray_getitem_adjust_offsets_index_64(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      int64_t *tononzero,
      const int64_t *fromoffsets,
      int64_t length,
      const int64_t *index,
      int64_t indexlength,
      const int64_t *nonzero,
      int64_t nonzerolength,
      const int8_t *originalmask,
      int64_t masklength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray_getitem_adjust_offsets_index_64(
          tooffsets,
          tononzero,
          fromoffsets,
          length,
          index,
          indexlength,
          nonzero,
          nonzerolength,
          originalmask,
          masklength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_getitem_adjust_offsets_index_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_getitem_adjust_offsets_index_64");
      }
    }

    ERROR IndexedArray_getitem_adjust_outindex_64(
      kernel::lib ptr_lib,
      int8_t *tomask,
      int64_t *toindex,
      int64_t *tononzero,
      const int64_t *fromindex,
      int64_t fromindexlength,
      const int64_t *nonzero,
      int64_t nonzerolength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray_getitem_adjust_outindex_64(
          tomask,
          toindex,
          tononzero,
          fromindex,
          fromindexlength,
          nonzero,
          nonzerolength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_adjust_outindex_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_adjust_outindex_64");
      }
    }

    template<>
    ERROR IndexedArray_getitem_nextcarry_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int32_t *fromindex,
      int64_t lenindex,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray32_getitem_nextcarry_64(
          tocarry,
          fromindex,
          lenindex,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_nextcarry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_nextcarry_64");
      }
    }

    template<>
    ERROR IndexedArray_getitem_nextcarry_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const uint32_t *fromindex,
      int64_t lenindex,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_getitem_nextcarry_64(
          tocarry,
          fromindex,
          lenindex,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_nextcarry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_nextcarry_64");
      }
    }

    template<>
    ERROR IndexedArray_getitem_nextcarry_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int64_t *fromindex,
      int64_t lenindex,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray64_getitem_nextcarry_64(
          tocarry,
          fromindex,
          lenindex,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_nextcarry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_nextcarry_64");
      }
    }

    template<>
    ERROR IndexedArray_getitem_carry_64<int32_t>(
      kernel::lib ptr_lib,
      int32_t *toindex,
      const int32_t *fromindex,
      const int64_t *fromcarry,
      int64_t lenindex,
      int64_t lencarry) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_IndexedArray32_getitem_carry_64(
         toindex,
         fromindex,
         fromcarry,
         lenindex,
         lencarry);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_carry_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_carry_64<int32_t>");
      }
     }

    template<>
    ERROR IndexedArray_getitem_carry_64<uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toindex,
      const uint32_t *fromindex,
      const int64_t *fromcarry,
      int64_t lenindex,
      int64_t lencarry) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_getitem_carry_64(
          toindex,
          fromindex,
          fromcarry,
          lenindex,
          lencarry);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_carry_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_carry_64<uint32_t>");
      }
    }

    template<>
    ERROR IndexedArray_getitem_carry_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int64_t *fromindex,
      const int64_t *fromcarry,
      int64_t lenindex,
      int64_t lencarry) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_IndexedArray64_getitem_carry_64(
         toindex,
         fromindex,
         fromcarry,
         lenindex,
         lencarry);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_getitem_carry_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_getitem_carry_64<int64_t>");
      }
     }

    template<>
    ERROR UnionArray_regular_index_getsize<int8_t>(
      kernel::lib ptr_lib,
      int64_t *size,
      const int8_t *fromtags,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_regular_index_getsize(
          size,
          fromtags,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_regular_index_getsize<int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_regular_index_getsize<int8_t>");
      }
    }

    template<>
    ERROR UnionArray_regular_index<int8_t, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toindex,
      int32_t *current,
      int64_t size,
      const int8_t *fromtags,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_UnionArray8_32_regular_index(
         toindex,
         current,
         size,
         fromtags,
         length);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_regular_index<int8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_regular_index<int8_t, int32_t>");
      }
     }

    template<>
    ERROR UnionArray_regular_index<int8_t, uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toindex,
      uint32_t *current,
      int64_t size,
      const int8_t *fromtags,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_U32_regular_index(
          toindex,
          current,
          size,
          fromtags,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_regular_index<int8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_regular_index<int8_t, uint32_t>");
      }
    }

    template<>
    ERROR UnionArray_regular_index<int8_t, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t *current,
      int64_t size,
      const int8_t *fromtags,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_UnionArray8_64_regular_index(
         toindex,
         current,
         size,
         fromtags,
         length);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_regular_index<int8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_regular_index<int8_t, int64_t>");
      }
     }

    template<>
    ERROR UnionArray_project_64<int8_t, int32_t>(
      kernel::lib ptr_lib,
      int64_t *lenout,
      int64_t *tocarry,
      const int8_t *fromtags,
      const int32_t *fromindex,
      int64_t length,
      int64_t which) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_32_project_64(
          lenout,
          tocarry,
          fromtags,
          fromindex,
          length,
          which);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_project_64<int8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_project_64<int8_t, int32_t>");
      }
    }

    template<>
    ERROR UnionArray_project_64<int8_t, uint32_t>(
      kernel::lib ptr_lib,
      int64_t *lenout,
      int64_t *tocarry,
      const int8_t *fromtags,
      const uint32_t *fromindex,
      int64_t length,
      int64_t which) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_UnionArray8_U32_project_64(
         lenout,
         tocarry,
         fromtags,
         fromindex,
         length,
         which);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_project_64<int8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_project_64<int8_t, uint32_t>");
      }
     }

    template<>
    ERROR UnionArray_project_64<int8_t, int64_t>(
      kernel::lib ptr_lib,
      int64_t *lenout,
      int64_t *tocarry,
      const int8_t *fromtags,
      const int64_t *fromindex,
      int64_t length,
      int64_t which) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_64_project_64(
          lenout,
          tocarry,
          fromtags,
          fromindex,
          length,
          which);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_project_64<int8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_project_64<int8_t, int64_t>");
      }
    }

    ERROR missing_repeat_64(
      kernel::lib ptr_lib,
      int64_t *outindex,
      const int64_t *index,
      int64_t indexlength,
      int64_t repetitions,
      int64_t regularsize) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_missing_repeat_64(
          outindex,
          index,
          indexlength,
          repetitions,
          regularsize);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for missing_repeat_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for missing_repeat_64");
      }
    }

    ERROR RegularArray_getitem_jagged_expand_64(
      kernel::lib ptr_lib,
      int64_t *multistarts,
      int64_t *multistops,
      const int64_t *singleoffsets,
      int64_t regularsize,
      int64_t regularlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_getitem_jagged_expand_64(
          multistarts,
          multistops,
          singleoffsets,
          regularsize,
          regularlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_getitem_jagged_expand_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_getitem_jagged_expand_64");
      }
    }

    template<>
    ERROR ListArray_getitem_jagged_expand_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *multistarts,
      int64_t *multistops,
      const int64_t *singleoffsets,
      int64_t *tocarry,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t jaggedsize,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_getitem_jagged_expand_64(
          multistarts,
          multistops,
          singleoffsets,
          tocarry,
          fromstarts,
          fromstops,
          jaggedsize,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_jagged_expand_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_jagged_expand_64<int32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_jagged_expand_64(
      kernel::lib ptr_lib,
      int64_t *multistarts,
      int64_t *multistops,
      const int64_t *singleoffsets,
      int64_t *tocarry,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t jaggedsize,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_ListArrayU32_getitem_jagged_expand_64(
         multistarts,
         multistops,
         singleoffsets,
         tocarry,
         fromstarts,
         fromstops,
         jaggedsize,
         length);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_jagged_expand_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_jagged_expand_64");
      }
     }

    template<>
    ERROR ListArray_getitem_jagged_expand_64(
      kernel::lib ptr_lib,
      int64_t *multistarts,
      int64_t *multistops,
      const int64_t *singleoffsets,
      int64_t *tocarry,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t jaggedsize,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_getitem_jagged_expand_64(
          multistarts,
          multistops,
          singleoffsets,
          tocarry,
          fromstarts,
          fromstops,
          jaggedsize,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_jagged_expand_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_jagged_expand_64");
      }
    }

    ERROR ListArray_getitem_jagged_carrylen_64(
      kernel::lib ptr_lib,
      int64_t *carrylen,
      const int64_t *slicestarts,
      const int64_t *slicestops,
      int64_t sliceouterlen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray_getitem_jagged_carrylen_64(
          carrylen,
          slicestarts,
          slicestops,
          sliceouterlen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_jagged_carrylen_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_jagged_carrylen_64");
      }
    }

    template<>
    ERROR ListArray_getitem_jagged_apply_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      int64_t *tocarry,
      const int64_t *slicestarts,
      const int64_t *slicestops,
      int64_t sliceouterlen,
      const int64_t *sliceindex,
      int64_t sliceinnerlen,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t contentlen) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_ListArray32_getitem_jagged_apply_64(
         tooffsets,
         tocarry,
         slicestarts,
         slicestops,
         sliceouterlen,
         sliceindex,
         sliceinnerlen,
         fromstarts,
         fromstops,
         contentlen);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_jagged_apply_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_jagged_apply_64<int32_t>");
      }
     }

    template<>
    ERROR ListArray_getitem_jagged_apply_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      int64_t *tocarry,
      const int64_t *slicestarts,
      const int64_t *slicestops,
      int64_t sliceouterlen,
      const int64_t *sliceindex,
      int64_t sliceinnerlen,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t contentlen) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_ListArrayU32_getitem_jagged_apply_64(
         tooffsets,
         tocarry,
         slicestarts,
         slicestops,
         sliceouterlen,
         sliceindex,
         sliceinnerlen,
         fromstarts,
         fromstops,
         contentlen);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_jagged_apply_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_jagged_apply_64<uint32_t>");
      }
     }

    template<>
    ERROR ListArray_getitem_jagged_apply_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      int64_t *tocarry,
      const int64_t *slicestarts,
      const int64_t *slicestops,
      int64_t sliceouterlen,
      const int64_t *sliceindex,
      int64_t sliceinnerlen,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t contentlen) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_ListArray64_getitem_jagged_apply_64(
         tooffsets,
         tocarry,
         slicestarts,
         slicestops,
         sliceouterlen,
         sliceindex,
         sliceinnerlen,
         fromstarts,
         fromstops,
         contentlen);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_jagged_apply_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_jagged_apply_64<int64_t>");
      }
     }

    ERROR ListArray_getitem_jagged_numvalid_64(
      kernel::lib ptr_lib,
      int64_t *numvalid,
      const int64_t *slicestarts,
      const int64_t *slicestops,
      int64_t length,
      const int64_t *missing,
      int64_t missinglength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray_getitem_jagged_numvalid_64(
          numvalid,
          slicestarts,
          slicestops,
          length,
          missing,
          missinglength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_jagged_numvalid_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_jagged_numvalid_64");
      }
    }

    ERROR ListArray_getitem_jagged_shrink_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *tosmalloffsets,
      int64_t *tolargeoffsets,
      const int64_t *slicestarts,
      const int64_t *slicestops,
      int64_t length,
      const int64_t *missing) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray_getitem_jagged_shrink_64(
          tocarry,
          tosmalloffsets,
          tolargeoffsets,
          slicestarts,
          slicestops,
          length,
          missing);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_jagged_shrink_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_jagged_shrink_64");
      }
    }

    template<>
    ERROR ListArray_getitem_jagged_descend_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const int64_t *slicestarts,
      const int64_t *slicestops,
      int64_t sliceouterlen,
      const int32_t *fromstarts,
      const int32_t *fromstops) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_getitem_jagged_descend_64(
          tooffsets,
          slicestarts,
          slicestops,
          sliceouterlen,
          fromstarts,
          fromstops);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_jagged_descend_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_jagged_descend_64<int32_t>");
      }
    }

    template<>
    ERROR ListArray_getitem_jagged_descend_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const int64_t *slicestarts,
      const int64_t *slicestops,
      int64_t sliceouterlen,
      const uint32_t *fromstarts,
      const uint32_t *fromstops) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_ListArrayU32_getitem_jagged_descend_64(
         tooffsets,
         slicestarts,
         slicestops,
         sliceouterlen,
         fromstarts,
         fromstops);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_jagged_descend_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_jagged_descend_64<uint32_t>");
      }
     }

    template<>
    ERROR ListArray_getitem_jagged_descend_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const int64_t *slicestarts,
      const int64_t *slicestops,
      int64_t sliceouterlen,
      const int64_t *fromstarts,
      const int64_t *fromstops) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_getitem_jagged_descend_64(
          tooffsets,
          slicestarts,
          slicestops,
          sliceouterlen,
          fromstarts,
          fromstops);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_getitem_jagged_descend_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_getitem_jagged_descend_64<int64_t>");
      }
    }

    template<>
    int8_t index_getitem_at_nowrap(kernel::lib ptr_lib,
                                   int8_t *ptr,
                                   int64_t at) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Index8_getitem_at_nowrap(
          ptr,
          at);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_Index8_getitem_at_nowrap, ptr_lib);
        return (*awkward_Index8_getitem_at_nowrap_fcn)(
          ptr,
          at);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in int8_t index_getitem_at_nowrap");
      }
    }

    template<>
    uint8_t index_getitem_at_nowrap(kernel::lib ptr_lib,
                                    uint8_t *ptr,
                                    int64_t at) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexU8_getitem_at_nowrap(
          ptr,
          at);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_IndexU8_getitem_at_nowrap, ptr_lib);
        return (*awkward_IndexU8_getitem_at_nowrap_fcn)(
          ptr,
          at);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in uint8_t index_getitem_at_nowrap");
      }
    }

    template<>
    int32_t index_getitem_at_nowrap(kernel::lib ptr_lib,
                                    int32_t *ptr,
                                    int64_t at) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Index32_getitem_at_nowrap(
          ptr,
          at);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_Index32_getitem_at_nowrap, ptr_lib);
        return (*awkward_Index32_getitem_at_nowrap_fcn)(
          ptr,
          at);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in int32_t index_getitem_at_nowrap");
      }
    }

    template<>
    uint32_t index_getitem_at_nowrap(kernel::lib ptr_lib,
                                     uint32_t *ptr,
                                     int64_t at) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexU32_getitem_at_nowrap(
          ptr,
          at);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_IndexU32_getitem_at_nowrap, ptr_lib);
        return (*awkward_IndexU32_getitem_at_nowrap_fcn)(
          ptr,
          at);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in uint32_t index_getitem_at_nowrap");
      }
    }

    template<>
    int64_t index_getitem_at_nowrap(kernel::lib ptr_lib,
                                    int64_t *ptr,
                                    int64_t at) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Index64_getitem_at_nowrap(
          ptr,
          at);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_Index64_getitem_at_nowrap, ptr_lib);
        return (*awkward_Index64_getitem_at_nowrap_fcn)(
          ptr,
          at);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in int64_t index_getitem_at_nowrap");
      }
    }

    template<>
    void index_setitem_at_nowrap(kernel::lib ptr_lib,
                                 int8_t *ptr,
                                 int64_t at,
                                 int8_t value) {
      if (ptr_lib == kernel::lib::cpu) {
        awkward_Index8_setitem_at_nowrap(
          ptr,
          at,
          value);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_Index8_setitem_at_nowrap, ptr_lib);
        (*awkward_Index8_setitem_at_nowrap_fcn)(
          ptr,
          at,
          value);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in void index_setitem_at_nowrap");
      }
    }

    template<>
    void index_setitem_at_nowrap(kernel::lib ptr_lib,
                                 uint8_t *ptr,
                                 int64_t at,
                                 uint8_t value) {
      if (ptr_lib == kernel::lib::cpu) {
        awkward_IndexU8_setitem_at_nowrap(
          ptr,
          at,
          value);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_IndexU8_setitem_at_nowrap, ptr_lib);
        (*awkward_IndexU8_setitem_at_nowrap_fcn)(
          ptr,
          at,
          value);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in void index_setitem_at_nowrap");
      }
    }

    template<>
    void index_setitem_at_nowrap(kernel::lib ptr_lib,
                                 int32_t *ptr,
                                 int64_t at,
                                 int32_t value) {
      if (ptr_lib == kernel::lib::cpu) {
        awkward_Index32_setitem_at_nowrap(
          ptr,
          at,
          value);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_Index32_setitem_at_nowrap, ptr_lib);
        (*awkward_Index32_setitem_at_nowrap_fcn)(
          ptr,
          at,
          value);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in void index_setitem_at_nowrap");
      }
    }

    template<>
    void index_setitem_at_nowrap(kernel::lib ptr_lib,
                                 uint32_t *ptr,
                                 int64_t at,
                                 uint32_t value) {
      if (ptr_lib == kernel::lib::cpu) {
        awkward_IndexU32_setitem_at_nowrap(
          ptr,
          at,
          value);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_IndexU32_setitem_at_nowrap, ptr_lib);
        (*awkward_IndexU32_setitem_at_nowrap_fcn)(
          ptr,
          at,
          value);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in void index_setitem_at_nowrap");
      }
    }

    template<>
    void index_setitem_at_nowrap(kernel::lib ptr_lib,
                                 int64_t *ptr,
                                 int64_t at,
                                 int64_t value) {
      if (ptr_lib == kernel::lib::cpu) {
        awkward_Index64_setitem_at_nowrap(
          ptr,
          at,
          value);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_Index64_setitem_at_nowrap, ptr_lib);
        (*awkward_Index64_setitem_at_nowrap_fcn)(
          ptr,
          at,
          value);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in void index_setitem_at_nowrap");
      }
    }

    ERROR ByteMaskedArray_getitem_carry_64(
      kernel::lib ptr_lib,
      int8_t *tomask,
      const int8_t *frommask,
      int64_t lenmask,
      const int64_t *fromcarry,
      int64_t lencarry) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ByteMaskedArray_getitem_carry_64(
          tomask,
          frommask,
          lenmask,
          fromcarry,
          lencarry);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ByteMaskedArray_getitem_carry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ByteMaskedArray_getitem_carry_64");
      }
    }

    ERROR ByteMaskedArray_numnull(
      kernel::lib ptr_lib,
      int64_t *numnull,
      const int8_t *mask,
      int64_t length,
      bool validwhen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ByteMaskedArray_numnull(
          numnull,
          mask,
          length,
          validwhen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ByteMaskedArray_numnull");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ByteMaskedArray_numnull");
      }
    }

    ERROR ByteMaskedArray_getitem_nextcarry_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int8_t *mask,
      int64_t length,
      bool validwhen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ByteMaskedArray_getitem_nextcarry_64(
          tocarry,
          mask,
          length,
          validwhen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ByteMaskedArray_getitem_nextcarry_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ByteMaskedArray_getitem_nextcarry_64");
      }
    }

    ERROR ByteMaskedArray_getitem_nextcarry_outindex_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      int64_t *toindex,
      const int8_t *mask,
      int64_t length,
      bool validwhen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ByteMaskedArray_getitem_nextcarry_outindex_64(
          tocarry,
          toindex,
          mask,
          length,
          validwhen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ByteMaskedArray_getitem_nextcarry_outindex_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ByteMaskedArray_getitem_nextcarry_outindex_64");
      }
    }

    ERROR ByteMaskedArray_toIndexedOptionArray64(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int8_t *mask,
      int64_t length,
      bool validwhen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ByteMaskedArray_toIndexedOptionArray64(
          toindex,
          mask,
          length,
          validwhen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ByteMaskedArray_toIndexedOptionArray64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ByteMaskedArray_toIndexedOptionArray64");
      }
    }

    ERROR Content_getitem_next_missing_jagged_getmaskstartstop(
      kernel::lib ptr_lib,
      int64_t *index_in,
      int64_t *offsets_in,
      int64_t *mask_out,
      int64_t *starts_out,
      int64_t *stops_out,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Content_getitem_next_missing_jagged_getmaskstartstop(
          index_in,
          offsets_in,
          mask_out,
          starts_out,
          stops_out,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Content_getitem_next_missing_jagged_getmaskstartstop");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Content_getitem_next_missing_jagged_getmaskstartstop");
      }
    }

    template <>
    ERROR MaskedArray_getitem_next_jagged_project(
      kernel::lib ptr_lib,
      int32_t *index,
      int64_t *starts_in,
      int64_t *stops_in,
      int64_t *starts_out,
      int64_t *stops_out,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_MaskedArray32_getitem_next_jagged_project(
          index,
          starts_in,
          stops_in,
          starts_out,
          stops_out,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for MaskedArray_getitem_next_jagged_project");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for MaskedArray_getitem_next_jagged_project");
      }
    }
    template <>
    ERROR MaskedArray_getitem_next_jagged_project(
      kernel::lib ptr_lib,
      uint32_t *index,
      int64_t *starts_in,
      int64_t *stops_in,
      int64_t *starts_out,
      int64_t *stops_out,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_MaskedArrayU32_getitem_next_jagged_project(
          index,
          starts_in,
          stops_in,
          starts_out,
          stops_out,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for MaskedArray_getitem_next_jagged_project");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for MaskedArray_getitem_next_jagged_project");
      }
    }
    template <>
    ERROR MaskedArray_getitem_next_jagged_project(
      kernel::lib ptr_lib,
      int64_t *index,
      int64_t *starts_in,
      int64_t *stops_in,
      int64_t *starts_out,
      int64_t *stops_out,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_MaskedArray64_getitem_next_jagged_project(
          index,
          starts_in,
          stops_in,
          starts_out,
          stops_out,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for MaskedArray_getitem_next_jagged_project");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for MaskedArray_getitem_next_jagged_project");
      }
    }

    /////////////////////////////////// awkward/kernels/identities.h

    template<>
    ERROR new_Identities(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_new_Identities32(
          toptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for new_Identities");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for new_Identities");
      }
    }

    template<>
    ERROR new_Identities(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_new_Identities64(
          toptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for new_Identities");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for new_Identities");
      }
    }

    template<>
    ERROR Identities_to_Identities64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int32_t *fromptr,
      int64_t length,
      int64_t width) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_to_Identities64(
          toptr,
          fromptr,
          length,
          width);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_to_Identities64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_to_Identities64");
      }
    }

    template<>
    ERROR Identities_from_ListOffsetArray<int32_t, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int32_t *fromptr,
      const int32_t *fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_from_ListOffsetArray32(
          toptr,
          fromptr,
          fromoffsets,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_ListOffsetArray<int32_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_ListOffsetArray<int32_t, int32_t>");
      }
    }

    template<>
    ERROR Identities_from_ListOffsetArray<int32_t, uint32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int32_t *fromptr,
      const uint32_t *fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_from_ListOffsetArrayU32(
          toptr,
          fromptr,
          fromoffsets,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_ListOffsetArray<int32_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_ListOffsetArray<int32_t, uint32_t>");
      }
    }

    template<>
    ERROR Identities_from_ListOffsetArray<int32_t, int64_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int32_t *fromptr,
      const int64_t *fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_from_ListOffsetArray64(
          toptr,
          fromptr,
          fromoffsets,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_ListOffsetArray<int32_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_ListOffsetArray<int32_t, int64_t>");
      }
    }

    template<>
    ERROR Identities_from_ListOffsetArray<int64_t, int32_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      const int32_t *fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_from_ListOffsetArray32(
          toptr,
          fromptr,
          fromoffsets,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_ListOffsetArray<int64_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_ListOffsetArray<int64_t, int32_t>");
      }
    }

    template<>
    ERROR Identities_from_ListOffsetArray<int64_t, uint32_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      const uint32_t *fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_from_ListOffsetArrayU32(
          toptr,
          fromptr,
          fromoffsets,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_ListOffsetArray<int64_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_ListOffsetArray<int64_t, uint32_t>");
      }
    }

    template<>
    ERROR Identities_from_ListOffsetArray<int64_t, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      const int64_t *fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_from_ListOffsetArray64(
          toptr,
          fromptr,
          fromoffsets,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_ListOffsetArray<int64_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_ListOffsetArray<int64_t, int64_t>");
      }
    }

    template<>
    ERROR Identities_from_ListArray<int32_t, int32_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int32_t *toptr,
      const int32_t *fromptr,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_from_ListArray32(
          uniquecontents,
          toptr,
          fromptr,
          fromstarts,
          fromstops,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_ListArray<int32_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_ListArray<int32_t, int32_t>");
      }
    }

    template<>
    ERROR Identities_from_ListArray<int32_t, uint32_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int32_t *toptr,
      const int32_t *fromptr,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_from_ListArrayU32(
          uniquecontents,
          toptr,
          fromptr,
          fromstarts,
          fromstops,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_ListArray<int32_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_ListArray<int32_t, uint32_t>");
      }
    }

    template<>
    ERROR Identities_from_ListArray<int32_t, int64_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int32_t *toptr,
      const int32_t *fromptr,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_from_ListArray64(
          uniquecontents,
          toptr,
          fromptr,
          fromstarts,
          fromstops,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_ListArray<int32_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_ListArray<int32_t, int64_t>");
      }
    }

    template<>
    ERROR Identities_from_ListArray<int64_t, int32_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int64_t *toptr,
      const int64_t *fromptr,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_from_ListArray32(
          uniquecontents,
          toptr,
          fromptr,
          fromstarts,
          fromstops,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_ListArray<int64_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_ListArray<int64_t, int32_t>");
      }
    }

    template<>
    ERROR Identities_from_ListArray<int64_t, uint32_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int64_t *toptr,
      const int64_t *fromptr,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_from_ListArrayU32(
          uniquecontents,
          toptr,
          fromptr,
          fromstarts,
          fromstops,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_ListArray<int64_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_ListArray<int64_t, uint32_t>");
      }
    }

    template<>
    ERROR Identities_from_ListArray<int64_t, int64_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int64_t *toptr,
      const int64_t *fromptr,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_from_ListArray64(
          uniquecontents,
          toptr,
          fromptr,
          fromstarts,
          fromstops,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_ListArray<int64_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_ListArray<int64_t, int64_t>");
      }
    }

    template<>
    ERROR Identities_from_RegularArray(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int32_t *fromptr,
      int64_t size,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_from_RegularArray(
          toptr,
          fromptr,
          size,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_RegularArray");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_RegularArray");
      }
    }

    template<>
    ERROR Identities_from_RegularArray(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      int64_t size,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_from_RegularArray(
          toptr,
          fromptr,
          size,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_RegularArray");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_RegularArray");
      }
    }

    template<>
    ERROR Identities_from_IndexedArray<int32_t, int32_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int32_t *toptr,
      const int32_t *fromptr,
      const int32_t *fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_from_IndexedArray32(
          uniquecontents,
          toptr,
          fromptr,
          fromindex,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_IndexedArray<int32_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_IndexedArray<int32_t, int32_t>");
      }
    }

    template<>
    ERROR Identities_from_IndexedArray<int32_t, uint32_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int32_t *toptr,
      const int32_t *fromptr,
      const uint32_t *fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_Identities32_from_IndexedArrayU32(
         uniquecontents,
         toptr,
         fromptr,
         fromindex,
         tolength,
         fromlength,
         fromwidth);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_IndexedArray<int32_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_IndexedArray<int32_t, uint32_t>");
      }
     }

    template<>
    ERROR Identities_from_IndexedArray<int32_t, int64_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int32_t *toptr,
      const int32_t *fromptr,
      const int64_t *fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_from_IndexedArray64(
          uniquecontents,
          toptr,
          fromptr,
          fromindex,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_IndexedArray<int32_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_IndexedArray<int32_t, int64_t>");
      }
    }

    template<>
    ERROR Identities_from_IndexedArray<int64_t, int32_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int64_t *toptr,
      const int64_t *fromptr,
      const int32_t *fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_Identities64_from_IndexedArray32(
         uniquecontents,
         toptr,
         fromptr,
         fromindex,
         tolength,
         fromlength,
         fromwidth);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_IndexedArray<int64_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_IndexedArray<int64_t, int32_t>");
      }
     }

    template<>
    ERROR Identities_from_IndexedArray<int64_t, uint32_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int64_t *toptr,
      const int64_t *fromptr,
      const uint32_t *fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_from_IndexedArrayU32(
          uniquecontents,
          toptr,
          fromptr,
          fromindex,
          tolength,
          fromlength,
          fromwidth);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_IndexedArray<int64_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_IndexedArray<int64_t, uint32_t>");
      }
    }

    template<>
    ERROR Identities_from_IndexedArray<int64_t, int64_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int64_t *toptr,
      const int64_t *fromptr,
      const int64_t *fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth) {
      if (ptr_lib == kernel::lib::cpu) {
       return awkward_Identities64_from_IndexedArray64(
         uniquecontents,
         toptr,
         fromptr,
         fromindex,
         tolength,
         fromlength,
         fromwidth);
     }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_IndexedArray<int64_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_IndexedArray<int64_t, int64_t>");
      }
     }

    template<>
    ERROR Identities_from_UnionArray<int32_t, int8_t, int32_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int32_t *toptr,
      const int32_t *fromptr,
      const int8_t *fromtags,
      const int32_t *fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_from_UnionArray8_32(
          uniquecontents,
          toptr,
          fromptr,
          fromtags,
          fromindex,
          tolength,
          fromlength,
          fromwidth,
          which);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_UnionArray<int32_t, int8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_UnionArray<int32_t, int8_t, int32_t>");
      }
    }

    template<>
    ERROR Identities_from_UnionArray<int32_t, int8_t, uint32_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int32_t *toptr,
      const int32_t *fromptr,
      const int8_t *fromtags,
      const uint32_t *fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_from_UnionArray8_U32(
          uniquecontents,
          toptr,
          fromptr,
          fromtags,
          fromindex,
          tolength,
          fromlength,
          fromwidth,
          which);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_UnionArray<int32_t, int8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_UnionArray<int32_t, int8_t, uint32_t>");
      }
    }

    template<>
    ERROR Identities_from_UnionArray<int32_t, int8_t, int64_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int32_t *toptr,
      const int32_t *fromptr,
      const int8_t *fromtags,
      const int64_t *fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_from_UnionArray8_64(
          uniquecontents,
          toptr,
          fromptr,
          fromtags,
          fromindex,
          tolength,
          fromlength,
          fromwidth,
          which);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_UnionArray<int32_t, int8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_UnionArray<int32_t, int8_t, int64_t>");
      }
    }

    template<>
    ERROR Identities_from_UnionArray<int64_t, int8_t, int32_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int64_t *toptr,
      const int64_t *fromptr,
      const int8_t *fromtags,
      const int32_t *fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_from_UnionArray8_32(
          uniquecontents,
          toptr,
          fromptr,
          fromtags,
          fromindex,
          tolength,
          fromlength,
          fromwidth,
          which);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_UnionArray<int64_t, int8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_UnionArray<int64_t, int8_t, int32_t>");
      }
    }

    template<>
    ERROR Identities_from_UnionArray<int64_t, int8_t, uint32_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int64_t *toptr,
      const int64_t *fromptr,
      const int8_t *fromtags,
      const uint32_t *fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_from_UnionArray8_U32(
          uniquecontents,
          toptr,
          fromptr,
          fromtags,
          fromindex,
          tolength,
          fromlength,
          fromwidth,
          which);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_UnionArray<int64_t, int8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_UnionArray<int64_t, int8_t, uint32_t>");
      }
    }

    template<>
    ERROR Identities_from_UnionArray<int64_t, int8_t, int64_t>(
      kernel::lib ptr_lib,
      bool *uniquecontents,
      int64_t *toptr,
      const int64_t *fromptr,
      const int8_t *fromtags,
      const int64_t *fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_from_UnionArray8_64(
          uniquecontents,
          toptr,
          fromptr,
          fromtags,
          fromindex,
          tolength,
          fromlength,
          fromwidth,
          which);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_from_UnionArray<int64_t, int8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_from_UnionArray<int64_t, int8_t, int64_t>");
      }
    }

    template<>
    ERROR Identities_extend(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int32_t *fromptr,
      int64_t fromlength,
      int64_t tolength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities32_extend(
          toptr,
          fromptr,
          fromlength,
          tolength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_extend");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_extend");
      }
    }

    template<>
    ERROR Identities_extend(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      int64_t fromlength,
      int64_t tolength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_Identities64_extend(
          toptr,
          fromptr,
          fromlength,
          tolength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for Identities_extend");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for Identities_extend");
      }
    }

    /////////////////////////////////// awkward/kernels/operations.h

    template<>
    ERROR ListArray_num_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tonum,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_num_64(
          tonum,
          fromstarts,
          fromstops,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_ListArray32_num_64, ptr_lib);
        return (*awkward_ListArray32_num_64_fcn)(
          tonum,
          fromstarts,
          fromstops,
          length);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in ListArray_num_64<int32_t>");
      }
    }

    template<>
    ERROR ListArray_num_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tonum,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_num_64(
          tonum,
          fromstarts,
          fromstops,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_ListArrayU32_num_64, ptr_lib);
        return (*awkward_ListArrayU32_num_64_fcn)(
          tonum,
          fromstarts,
          fromstops,
          length);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in ListArray_num_64<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_num_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tonum,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_num_64(
          tonum,
          fromstarts,
          fromstops,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_ListArray64_num_64, ptr_lib);
        return (*awkward_ListArray64_num_64_fcn)(
          tonum,
          fromstarts,
          fromstops,
          length);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in ListArray_num_64<int64_t>");
      }
    }

    ERROR RegularArray_num_64(
      kernel::lib ptr_lib,
      int64_t *tonum,
      int64_t size,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_num_64(
          tonum,
          size,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        FORM_KERNEL(awkward_RegularArray_num_64, ptr_lib);
        return (*awkward_RegularArray_num_64_fcn)(
          tonum,
          size,
          length);
      }
      else {
        throw std::runtime_error("unrecognized ptr_lib in RegularArray_num_64");
      }
    }

    template<>
    ERROR ListOffsetArray_flatten_offsets_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const int32_t *outeroffsets,
      int64_t outeroffsetslen,
      const int64_t *inneroffsets,
      int64_t inneroffsetslen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray32_flatten_offsets_64(
          tooffsets,
          outeroffsets,
          outeroffsetslen,
          inneroffsets,
          inneroffsetslen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_flatten_offsets_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_flatten_offsets_64<int32_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_flatten_offsets_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const uint32_t *outeroffsets,
      int64_t outeroffsetslen,
      const int64_t *inneroffsets,
      int64_t inneroffsetslen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArrayU32_flatten_offsets_64(
          tooffsets,
          outeroffsets,
          outeroffsetslen,
          inneroffsets,
          inneroffsetslen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_flatten_offsets_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_flatten_offsets_64<uint32_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_flatten_offsets_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const int64_t *outeroffsets,
      int64_t outeroffsetslen,
      const int64_t *inneroffsets,
      int64_t inneroffsetslen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray64_flatten_offsets_64(
          tooffsets,
          outeroffsets,
          outeroffsetslen,
          inneroffsets,
          inneroffsetslen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_flatten_offsets_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_flatten_offsets_64<int64_t>");
      }
    }

    template<>
    ERROR IndexedArray_flatten_none2empty_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *outoffsets,
      const int32_t *outindex,
      int64_t outindexlength,
      const int64_t *offsets,
      int64_t offsetslength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray32_flatten_none2empty_64(
          outoffsets,
          outindex,
          outindexlength,
          offsets,
          offsetslength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_flatten_none2empty_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_flatten_none2empty_64<int32_t>");
      }
    }

    template<>
    ERROR IndexedArray_flatten_none2empty_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *outoffsets,
      const uint32_t *outindex,
      int64_t outindexlength,
      const int64_t *offsets,
      int64_t offsetslength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_flatten_none2empty_64(
          outoffsets,
          outindex,
          outindexlength,
          offsets,
          offsetslength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_flatten_none2empty_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_flatten_none2empty_64<uint32_t>");
      }
    }

    template<>
    ERROR IndexedArray_flatten_none2empty_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *outoffsets,
      const int64_t *outindex,
      int64_t outindexlength,
      const int64_t *offsets,
      int64_t offsetslength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray64_flatten_none2empty_64(
          outoffsets,
          outindex,
          outindexlength,
          offsets,
          offsetslength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_flatten_none2empty_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_flatten_none2empty_64<int64_t>");
      }
    }

    template<>
    ERROR UnionArray_flatten_length_64<int8_t, int32_t>(
      kernel::lib ptr_lib,
      int64_t *total_length,
      const int8_t *fromtags,
      const int32_t *fromindex,
      int64_t length,
      int64_t **offsetsraws) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray32_flatten_length_64(
          total_length,
          fromtags,
          fromindex,
          length,
          offsetsraws);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_flatten_length_64<int8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_flatten_length_64<int8_t, int32_t>");
      }
    }

    template<>
    ERROR UnionArray_flatten_length_64<int8_t, uint32_t>(
      kernel::lib ptr_lib,
      int64_t *total_length,
      const int8_t *fromtags,
      const uint32_t *fromindex,
      int64_t length,
      int64_t **offsetsraws) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArrayU32_flatten_length_64(
          total_length,
          fromtags,
          fromindex,
          length,
          offsetsraws);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_flatten_length_64<int8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_flatten_length_64<int8_t, uint32_t>");
      }
    }

    template<>
    ERROR UnionArray_flatten_length_64<int8_t, int64_t>(
      kernel::lib ptr_lib,
      int64_t *total_length,
      const int8_t *fromtags,
      const int64_t *fromindex,
      int64_t length,
      int64_t **offsetsraws) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray64_flatten_length_64(
          total_length,
          fromtags,
          fromindex,
          length,
          offsetsraws);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_flatten_length_64<int8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_flatten_length_64<int8_t, int64_t>");
      }
    }

    template<>
    ERROR UnionArray_flatten_combine_64<int8_t, int32_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      int64_t *tooffsets,
      const int8_t *fromtags,
      const int32_t *fromindex,
      int64_t length,
      int64_t **offsetsraws) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray32_flatten_combine_64(
          totags,
          toindex,
          tooffsets,
          fromtags,
          fromindex,
          length,
          offsetsraws);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_flatten_combine_64<int8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_flatten_combine_64<int8_t, int32_t>");
      }
    }

    template<>
    ERROR UnionArray_flatten_combine_64<int8_t, uint32_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      int64_t *tooffsets,
      const int8_t *fromtags,
      const uint32_t *fromindex,
      int64_t length,
      int64_t **offsetsraws) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArrayU32_flatten_combine_64(
          totags,
          toindex,
          tooffsets,
          fromtags,
          fromindex,
          length,
          offsetsraws);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_flatten_combine_64<int8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_flatten_combine_64<int8_t, uint32_t>");
      }
    }

    template<>
    ERROR UnionArray_flatten_combine_64<int8_t, int64_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      int64_t *tooffsets,
      const int8_t *fromtags,
      const int64_t *fromindex,
      int64_t length,
      int64_t **offsetsraws) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray64_flatten_combine_64(
          totags,
          toindex,
          tooffsets,
          fromtags,
          fromindex,
          length,
          offsetsraws);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_flatten_combine_64<int8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_flatten_combine_64<int8_t, int64_t>");
      }
    }

    template<>
    ERROR IndexedArray_flatten_nextcarry_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int32_t *fromindex,
      int64_t lenindex,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray32_flatten_nextcarry_64(
          tocarry,
          fromindex,
          lenindex,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_flatten_nextcarry_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_flatten_nextcarry_64<int32_t>");
      }
    }

    template<>
    ERROR IndexedArray_flatten_nextcarry_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const uint32_t *fromindex,
      int64_t lenindex,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_flatten_nextcarry_64(
          tocarry,
          fromindex,
          lenindex,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_flatten_nextcarry_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_flatten_nextcarry_64<uint32_t>");
      }
    }

    template<>
    ERROR IndexedArray_flatten_nextcarry_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int64_t *fromindex,
      int64_t lenindex,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray64_flatten_nextcarry_64(
          tocarry,
          fromindex,
          lenindex,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_flatten_nextcarry_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_flatten_nextcarry_64<int64_t>");
      }
    }

    template<>
    ERROR IndexedArray_overlay_mask8_to64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int8_t *mask,
      const int32_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray32_overlay_mask8_to64(
          toindex,
          mask,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_overlay_mask8_to64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_overlay_mask8_to64<int32_t>");
      }
    }

    template<>
    ERROR IndexedArray_overlay_mask8_to64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int8_t *mask,
      const uint32_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_overlay_mask8_to64(
          toindex,
          mask,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_overlay_mask8_to64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_overlay_mask8_to64<uint32_t>");
      }
    }

    template<>
    ERROR IndexedArray_overlay_mask8_to64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int8_t *mask,
      const int64_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray64_overlay_mask8_to64(
          toindex,
          mask,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_overlay_mask8_to64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_overlay_mask8_to64<int64_t>");
      }
    }

    template<>
    ERROR IndexedArray_mask8<int32_t>(
      kernel::lib ptr_lib,
      int8_t *tomask,
      const int32_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray32_mask8(
          tomask,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_mask8<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_mask8<int32_t>");
      }
    }

    template<>
    ERROR IndexedArray_mask8<uint32_t>(
      kernel::lib ptr_lib,
      int8_t *tomask,
      const uint32_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_mask8(
          tomask,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_mask8<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_mask8<uint32_t>");
      }
    }

    template<>
    ERROR IndexedArray_mask8<int64_t>(
      kernel::lib ptr_lib,
      int8_t *tomask,
      const int64_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray64_mask8(
          tomask,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_mask8<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_mask8<int64_t>");
      }
    }

    ERROR ByteMaskedArray_mask8(
      kernel::lib ptr_lib,
      int8_t *tomask,
      const int8_t *frommask,
      int64_t length,
      bool validwhen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ByteMaskedArray_mask8(
          tomask,
          frommask,
          length,
          validwhen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ByteMaskedArray_mask8");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ByteMaskedArray_mask8");
      }
    }

    ERROR zero_mask8(
      kernel::lib ptr_lib,
      int8_t *tomask,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_zero_mask8(tomask, length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for zero_mask8");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for zero_mask8");
      }
    }

    template<>
    ERROR IndexedArray_simplify32_to64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int32_t *outerindex,
      int64_t outerlength,
      const int32_t *innerindex,
      int64_t innerlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray32_simplify32_to64(
          toindex,
          outerindex,
          outerlength,
          innerindex,
          innerlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_simplify32_to64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_simplify32_to64<int32_t>");
      }
    }

    template<>
    ERROR IndexedArray_simplify32_to64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const uint32_t *outerindex,
      int64_t outerlength,
      const int32_t *innerindex,
      int64_t innerlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_simplify32_to64(
          toindex,
          outerindex,
          outerlength,
          innerindex,
          innerlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_simplify32_to64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_simplify32_to64<uint32_t>");
      }
    }

    template<>
    ERROR IndexedArray_simplify32_to64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int64_t *outerindex,
      int64_t outerlength,
      const int32_t *innerindex,
      int64_t innerlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray64_simplify32_to64(
          toindex,
          outerindex,
          outerlength,
          innerindex,
          innerlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_simplify32_to64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_simplify32_to64<int64_t>");
      }
    }

    template<>
    ERROR IndexedArray_simplifyU32_to64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int32_t *outerindex,
      int64_t outerlength,
      const uint32_t *innerindex,
      int64_t innerlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray32_simplifyU32_to64(
          toindex,
          outerindex,
          outerlength,
          innerindex,
          innerlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_simplifyU32_to64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_simplifyU32_to64<int32_t>");
      }
    }

    template<>
    ERROR IndexedArray_simplifyU32_to64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const uint32_t *outerindex,
      int64_t outerlength,
      const uint32_t *innerindex,
      int64_t innerlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_simplifyU32_to64(
          toindex,
          outerindex,
          outerlength,
          innerindex,
          innerlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_simplifyU32_to64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_simplifyU32_to64<uint32_t>");
      }
    }

    template<>
    ERROR IndexedArray_simplifyU32_to64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int64_t *outerindex,
      int64_t outerlength,
      const uint32_t *innerindex,
      int64_t innerlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray64_simplifyU32_to64(
          toindex,
          outerindex,
          outerlength,
          innerindex,
          innerlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_simplifyU32_to64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_simplifyU32_to64<int64_t>");
      }
    }

    template<>
    ERROR IndexedArray_simplify64_to64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int32_t *outerindex,
      int64_t outerlength,
      const int64_t *innerindex,
      int64_t innerlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray32_simplify64_to64(
          toindex,
          outerindex,
          outerlength,
          innerindex,
          innerlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_simplify64_to64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_simplify64_to64<int32_t>");
      }
    }

    template<>
    ERROR IndexedArray_simplify64_to64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const uint32_t *outerindex,
      int64_t outerlength,
      const int64_t *innerindex,
      int64_t innerlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_simplify64_to64(
          toindex,
          outerindex,
          outerlength,
          innerindex,
          innerlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_simplify64_to64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_simplify64_to64<uint32_t>");
      }
    }

    template<>
    ERROR IndexedArray_simplify64_to64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int64_t *outerindex,
      int64_t outerlength,
      const int64_t *innerindex,
      int64_t innerlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray64_simplify64_to64(
          toindex,
          outerindex,
          outerlength,
          innerindex,
          innerlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_simplify64_to64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_simplify64_to64<int64_t>");
      }
    }

    template<>
    ERROR ListArray_compact_offsets_64(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_compact_offsets_64(
          tooffsets,
          fromstarts,
          fromstops,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_compact_offsets_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_compact_offsets_64");
      }
    }

    template<>
    ERROR ListArray_compact_offsets_64(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_compact_offsets_64(
          tooffsets,
          fromstarts,
          fromstops,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_compact_offsets_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_compact_offsets_64");
      }
    }

    template<>
    ERROR ListArray_compact_offsets_64(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_compact_offsets_64(
          tooffsets,
          fromstarts,
          fromstops,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_compact_offsets_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_compact_offsets_64");
      }
    }

    ERROR RegularArray_compact_offsets_64(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      int64_t length,
      int64_t size) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_compact_offsets64(
          tooffsets,
          length,
          size);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_compact_offsets_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_compact_offsets_64");
      }
    }

    template<>
    ERROR ListOffsetArray_compact_offsets_64(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const int32_t *fromoffsets,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray32_compact_offsets_64(
          tooffsets,
          fromoffsets,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_compact_offsets_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_compact_offsets_64");
      }
    }

    template<>
    ERROR ListOffsetArray_compact_offsets_64(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const uint32_t *fromoffsets,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArrayU32_compact_offsets_64(
          tooffsets,
          fromoffsets,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_compact_offsets_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_compact_offsets_64");
      }
    }

    template<>
    ERROR ListOffsetArray_compact_offsets_64(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const int64_t *fromoffsets,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray64_compact_offsets_64(
          tooffsets,
          fromoffsets,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_compact_offsets_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_compact_offsets_64");
      }
    }

    template<>
    ERROR ListArray_broadcast_tooffsets_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int64_t *fromoffsets,
      int64_t offsetslength,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_broadcast_tooffsets_64(
          tocarry,
          fromoffsets,
          offsetslength,
          fromstarts,
          fromstops,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_broadcast_tooffsets_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_broadcast_tooffsets_64<int32_t>");
      }
    }

    template<>
    ERROR ListArray_broadcast_tooffsets_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int64_t *fromoffsets,
      int64_t offsetslength,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_broadcast_tooffsets_64(
          tocarry,
          fromoffsets,
          offsetslength,
          fromstarts,
          fromstops,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_broadcast_tooffsets_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_broadcast_tooffsets_64<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_broadcast_tooffsets_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int64_t *fromoffsets,
      int64_t offsetslength,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_broadcast_tooffsets_64(
          tocarry,
          fromoffsets,
          offsetslength,
          fromstarts,
          fromstops,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_broadcast_tooffsets_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_broadcast_tooffsets_64<int64_t>");
      }
    }

    ERROR RegularArray_broadcast_tooffsets_64(
      kernel::lib ptr_lib,
      const int64_t *fromoffsets,
      int64_t offsetslength,
      int64_t size) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_broadcast_tooffsets_64(
          fromoffsets,
          offsetslength,
          size);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_broadcast_tooffsets_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_broadcast_tooffsets_64");
      }
    }

    ERROR RegularArray_broadcast_tooffsets_size1_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int64_t *fromoffsets,
      int64_t offsetslength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_broadcast_tooffsets_size1_64(
          tocarry,
          fromoffsets,
          offsetslength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_broadcast_tooffsets_size1_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_broadcast_tooffsets_size1_64");
      }
    }

    template<>
    ERROR ListOffsetArray_toRegularArray<int32_t>(
      kernel::lib ptr_lib,
      int64_t *size,
      const int32_t *fromoffsets,
      int64_t offsetslength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray32_toRegularArray(
          size,
          fromoffsets,
          offsetslength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_toRegularArray<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_toRegularArray<int32_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_toRegularArray<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *size,
      const uint32_t *fromoffsets,
      int64_t offsetslength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArrayU32_toRegularArray(
          size,
          fromoffsets,
          offsetslength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_toRegularArray<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_toRegularArray<uint32_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_toRegularArray(
      kernel::lib ptr_lib,
      int64_t *size,
      const int64_t *fromoffsets,
      int64_t offsetslength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray64_toRegularArray(
          size,
          fromoffsets,
          offsetslength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_toRegularArray");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_toRegularArray");
      }
    }

    template<>
    ERROR NumpyArray_fill_frombool<bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tobool_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill_frombool<bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill_frombool<bool>");
      }
    }

    template<>
    ERROR NumpyArray_fill_frombool<int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint8_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill_frombool<int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill_frombool<int8_t>");
      }
    }

    template<>
    ERROR NumpyArray_fill_frombool<int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint16_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill_frombool<int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill_frombool<int16_t>");
      }
    }

    template<>
    ERROR NumpyArray_fill_frombool<int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint32_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill_frombool<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill_frombool<int32_t>");
      }
    }

    template<>
    ERROR NumpyArray_fill_frombool<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint64_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill_frombool<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill_frombool<int64_t>");
      }
    }

    template<>
    ERROR NumpyArray_fill_frombool<uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint8_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill_frombool<uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill_frombool<uint8_t>");
      }
    }

    template<>
    ERROR NumpyArray_fill_frombool<uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint16_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill_frombool<uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill_frombool<uint16_t>");
      }
    }

    template<>
    ERROR NumpyArray_fill_frombool<uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint32_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill_frombool<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill_frombool<uint32_t>");
      }
    }

    template<>
    ERROR NumpyArray_fill_frombool<uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint64_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill_frombool<uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill_frombool<uint64_t>");
      }
    }

    template<>
    ERROR NumpyArray_fill_frombool<float>(
      kernel::lib ptr_lib,
      float *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat32_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill_frombool<float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill_frombool<float>");
      }
    }

    template<>
    ERROR NumpyArray_fill_frombool<double>(
      kernel::lib ptr_lib,
      double *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat64_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill_frombool<double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill_frombool<double>");
      }
    }

    template <>
    ERROR NumpyArray_fill<bool, bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tobool_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<bool, bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<bool, bool>");
      }
    }
    template <>
    ERROR NumpyArray_fill<bool, int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint8_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<bool, int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<bool, int8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<bool, int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint16_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<bool, int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<bool, int16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<bool, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint32_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<bool, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<bool, int32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<bool, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint64_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<bool, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<bool, int64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<bool, uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint8_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<bool, uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<bool, uint8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<bool, uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint16_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<bool, uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<bool, uint16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<bool, uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint32_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<bool, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<bool, uint32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<bool, uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint64_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<bool, uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<bool, uint64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<bool, float>(
      kernel::lib ptr_lib,
      float *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat32_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<bool, float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<bool, float>");
      }
    }
    template <>
    ERROR NumpyArray_fill<bool, double>(
      kernel::lib ptr_lib,
      double *toptr,
      int64_t tooffset,
      const bool *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat64_frombool(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<bool, double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<bool, double>");
      }
    }

    template <>
    ERROR NumpyArray_fill<int8_t, bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      int64_t tooffset,
      const int8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tobool_fromint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int8_t, bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int8_t, bool>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int8_t, int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      int64_t tooffset,
      const int8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint8_fromint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int8_t, int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int8_t, int8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int8_t, int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      int64_t tooffset,
      const int8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint16_fromint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int8_t, int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int8_t, int16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int8_t, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t tooffset,
      const int8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint32_fromint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int8_t, int32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int8_t, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t tooffset,
      const int8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint64_fromint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int8_t, int64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int8_t, uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      int64_t tooffset,
      const int8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint8_fromint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int8_t, uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int8_t, uint8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int8_t, uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      int64_t tooffset,
      const int8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint16_fromint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int8_t, uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int8_t, uint16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int8_t, uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t tooffset,
      const int8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint32_fromint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int8_t, uint32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int8_t, uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      int64_t tooffset,
      const int8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint64_fromint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int8_t, uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int8_t, uint64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int8_t, float>(
      kernel::lib ptr_lib,
      float *toptr,
      int64_t tooffset,
      const int8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat32_fromint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int8_t, float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int8_t, float>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int8_t, double>(
      kernel::lib ptr_lib,
      double *toptr,
      int64_t tooffset,
      const int8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat64_fromint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int8_t, double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int8_t, double>");
      }
    }

    template <>
    ERROR NumpyArray_fill<int16_t, bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      int64_t tooffset,
      const int16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tobool_fromint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int16_t, bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int16_t, bool>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int16_t, int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      int64_t tooffset,
      const int16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint8_fromint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int16_t, int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int16_t, int8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int16_t, int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      int64_t tooffset,
      const int16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint16_fromint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int16_t, int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int16_t, int16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int16_t, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t tooffset,
      const int16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint32_fromint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int16_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int16_t, int32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int16_t, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t tooffset,
      const int16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint64_fromint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int16_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int16_t, int64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int16_t, uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      int64_t tooffset,
      const int16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint8_fromint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int16_t, uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int16_t, uint8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int16_t, uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      int64_t tooffset,
      const int16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint16_fromint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int16_t, uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int16_t, uint16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int16_t, uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t tooffset,
      const int16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint32_fromint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int16_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int16_t, uint32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int16_t, uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      int64_t tooffset,
      const int16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint64_fromint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int16_t, uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int16_t, uint64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int16_t, float>(
      kernel::lib ptr_lib,
      float *toptr,
      int64_t tooffset,
      const int16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat32_fromint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int16_t, float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int16_t, float>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int16_t, double>(
      kernel::lib ptr_lib,
      double *toptr,
      int64_t tooffset,
      const int16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat64_fromint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int16_t, double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int16_t, double>");
      }
    }

    template <>
    ERROR NumpyArray_fill<int32_t, bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      int64_t tooffset,
      const int32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tobool_fromint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int32_t, bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int32_t, bool>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int32_t, int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      int64_t tooffset,
      const int32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint8_fromint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int32_t, int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int32_t, int8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int32_t, int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      int64_t tooffset,
      const int32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint16_fromint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int32_t, int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int32_t, int16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int32_t, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t tooffset,
      const int32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint32_fromint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int32_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int32_t, int32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int32_t, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t tooffset,
      const int32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint64_fromint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int32_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int32_t, int64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int32_t, uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      int64_t tooffset,
      const int32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint8_fromint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int32_t, uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int32_t, uint8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int32_t, uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      int64_t tooffset,
      const int32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint16_fromint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int32_t, uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int32_t, uint16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int32_t, uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t tooffset,
      const int32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint32_fromint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int32_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int32_t, uint32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int32_t, uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      int64_t tooffset,
      const int32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint64_fromint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int32_t, uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int32_t, uint64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int32_t, float>(
      kernel::lib ptr_lib,
      float *toptr,
      int64_t tooffset,
      const int32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat32_fromint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int32_t, float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int32_t, float>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int32_t, double>(
      kernel::lib ptr_lib,
      double *toptr,
      int64_t tooffset,
      const int32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat64_fromint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int32_t, double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int32_t, double>");
      }
    }

    template <>
    ERROR NumpyArray_fill<int64_t, bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      int64_t tooffset,
      const int64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tobool_fromint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int64_t, bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int64_t, bool>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int64_t, int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      int64_t tooffset,
      const int64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint8_fromint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int64_t, int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int64_t, int8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int64_t, int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      int64_t tooffset,
      const int64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint16_fromint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int64_t, int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int64_t, int16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int64_t, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t tooffset,
      const int64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint32_fromint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int64_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int64_t, int32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int64_t, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t tooffset,
      const int64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint64_fromint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int64_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int64_t, int64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int64_t, uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      int64_t tooffset,
      const int64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint8_fromint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int64_t, uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int64_t, uint8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int64_t, uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      int64_t tooffset,
      const int64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint16_fromint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int64_t, uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int64_t, uint16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int64_t, uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t tooffset,
      const int64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint32_fromint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int64_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int64_t, uint32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int64_t, uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      int64_t tooffset,
      const int64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint64_fromint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int64_t, uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int64_t, uint64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int64_t, float>(
      kernel::lib ptr_lib,
      float *toptr,
      int64_t tooffset,
      const int64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat32_fromint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int64_t, float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int64_t, float>");
      }
    }
    template <>
    ERROR NumpyArray_fill<int64_t, double>(
      kernel::lib ptr_lib,
      double *toptr,
      int64_t tooffset,
      const int64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat64_fromint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<int64_t, double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<int64_t, double>");
      }
    }

    template <>
    ERROR NumpyArray_fill<uint8_t, bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      int64_t tooffset,
      const uint8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tobool_fromuint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint8_t, bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint8_t, bool>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint8_t, int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      int64_t tooffset,
      const uint8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint8_fromuint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint8_t, int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint8_t, int8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint8_t, int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      int64_t tooffset,
      const uint8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint16_fromuint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint8_t, int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint8_t, int16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint8_t, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t tooffset,
      const uint8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint32_fromuint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint8_t, int32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint8_t, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t tooffset,
      const uint8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint64_fromuint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint8_t, int64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint8_t, uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      int64_t tooffset,
      const uint8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint8_fromuint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint8_t, uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint8_t, uint8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint8_t, uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      int64_t tooffset,
      const uint8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint16_fromuint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint8_t, uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint8_t, uint16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint8_t, uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t tooffset,
      const uint8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint32_fromuint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint8_t, uint32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint8_t, uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      int64_t tooffset,
      const uint8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint64_fromuint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint8_t, uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint8_t, uint64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint8_t, float>(
      kernel::lib ptr_lib,
      float *toptr,
      int64_t tooffset,
      const uint8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat32_fromuint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint8_t, float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint8_t, float>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint8_t, double>(
      kernel::lib ptr_lib,
      double *toptr,
      int64_t tooffset,
      const uint8_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat64_fromuint8(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint8_t, double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint8_t, double>");
      }
    }

    template <>
    ERROR NumpyArray_fill<uint16_t, bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      int64_t tooffset,
      const uint16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tobool_fromuint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint16_t, bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint16_t, bool>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint16_t, int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      int64_t tooffset,
      const uint16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint8_fromuint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint16_t, int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint16_t, int8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint16_t, int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      int64_t tooffset,
      const uint16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint16_fromuint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint16_t, int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint16_t, int16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint16_t, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t tooffset,
      const uint16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint32_fromuint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint16_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint16_t, int32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint16_t, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t tooffset,
      const uint16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint64_fromuint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint16_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint16_t, int64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint16_t, uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      int64_t tooffset,
      const uint16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint8_fromuint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint16_t, uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint16_t, uint8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint16_t, uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      int64_t tooffset,
      const uint16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint16_fromuint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint16_t, uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint16_t, uint16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint16_t, uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t tooffset,
      const uint16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint32_fromuint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint16_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint16_t, uint32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint16_t, uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      int64_t tooffset,
      const uint16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint64_fromuint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint16_t, uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint16_t, uint64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint16_t, float>(
      kernel::lib ptr_lib,
      float *toptr,
      int64_t tooffset,
      const uint16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat32_fromuint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint16_t, float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint16_t, float>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint16_t, double>(
      kernel::lib ptr_lib,
      double *toptr,
      int64_t tooffset,
      const uint16_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat64_fromuint16(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint16_t, double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint16_t, double>");
      }
    }

    template <>
    ERROR NumpyArray_fill<uint32_t, bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      int64_t tooffset,
      const uint32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tobool_fromuint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint32_t, bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint32_t, bool>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint32_t, int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      int64_t tooffset,
      const uint32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint8_fromuint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint32_t, int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint32_t, int8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint32_t, int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      int64_t tooffset,
      const uint32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint16_fromuint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint32_t, int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint32_t, int16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint32_t, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t tooffset,
      const uint32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint32_fromuint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint32_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint32_t, int32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint32_t, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t tooffset,
      const uint32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint64_fromuint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint32_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint32_t, int64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint32_t, uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      int64_t tooffset,
      const uint32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint8_fromuint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint32_t, uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint32_t, uint8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint32_t, uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      int64_t tooffset,
      const uint32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint16_fromuint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint32_t, uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint32_t, uint16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint32_t, uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t tooffset,
      const uint32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint32_fromuint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint32_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint32_t, uint32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint32_t, uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      int64_t tooffset,
      const uint32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint64_fromuint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint32_t, uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint32_t, uint64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint32_t, float>(
      kernel::lib ptr_lib,
      float *toptr,
      int64_t tooffset,
      const uint32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat32_fromuint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint32_t, float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint32_t, float>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint32_t, double>(
      kernel::lib ptr_lib,
      double *toptr,
      int64_t tooffset,
      const uint32_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat64_fromuint32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint32_t, double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint32_t, double>");
      }
    }

    template <>
    ERROR NumpyArray_fill<uint64_t, bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      int64_t tooffset,
      const uint64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tobool_fromuint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint64_t, bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint64_t, bool>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint64_t, int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      int64_t tooffset,
      const uint64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint8_fromuint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint64_t, int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint64_t, int8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint64_t, int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      int64_t tooffset,
      const uint64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint16_fromuint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint64_t, int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint64_t, int16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint64_t, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t tooffset,
      const uint64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint32_fromuint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint64_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint64_t, int32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint64_t, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t tooffset,
      const uint64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint64_fromuint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint64_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint64_t, int64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint64_t, uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      int64_t tooffset,
      const uint64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint8_fromuint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint64_t, uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint64_t, uint8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint64_t, uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      int64_t tooffset,
      const uint64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint16_fromuint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint64_t, uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint64_t, uint16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint64_t, uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t tooffset,
      const uint64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint32_fromuint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint64_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint64_t, uint32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint64_t, uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      int64_t tooffset,
      const uint64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint64_fromuint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint64_t, uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint64_t, uint64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint64_t, float>(
      kernel::lib ptr_lib,
      float *toptr,
      int64_t tooffset,
      const uint64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat32_fromuint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint64_t, float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint64_t, float>");
      }
    }
    template <>
    ERROR NumpyArray_fill<uint64_t, double>(
      kernel::lib ptr_lib,
      double *toptr,
      int64_t tooffset,
      const uint64_t *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat64_fromuint64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<uint64_t, double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<uint64_t, double>");
      }
    }

    template <>
    ERROR NumpyArray_fill<float, bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      int64_t tooffset,
      const float *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tobool_fromfloat32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<float, bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<float, bool>");
      }
    }
    template <>
    ERROR NumpyArray_fill<float, int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      int64_t tooffset,
      const float *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint8_fromfloat32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<float, int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<float, int8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<float, int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      int64_t tooffset,
      const float *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint16_fromfloat32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<float, int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<float, int16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<float, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t tooffset,
      const float *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint32_fromfloat32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<float, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<float, int32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<float, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t tooffset,
      const float *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint64_fromfloat32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<float, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<float, int64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<float, uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      int64_t tooffset,
      const float *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint8_fromfloat32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<float, uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<float, uint8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<float, uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      int64_t tooffset,
      const float *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint16_fromfloat32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<float, uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<float, uint16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<float, uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t tooffset,
      const float *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint32_fromfloat32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<float, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<float, uint32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<float, uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      int64_t tooffset,
      const float *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint64_fromfloat32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<float, uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<float, uint64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<float, float>(
      kernel::lib ptr_lib,
      float *toptr,
      int64_t tooffset,
      const float *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat32_fromfloat32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<float, float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<float, float>");
      }
    }
    template <>
    ERROR NumpyArray_fill<float, double>(
      kernel::lib ptr_lib,
      double *toptr,
      int64_t tooffset,
      const float *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat64_fromfloat32(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<float, double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<float, double>");
      }
    }

    template <>
    ERROR NumpyArray_fill<double, bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      int64_t tooffset,
      const double *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tobool_fromfloat64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<double, bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<double, bool>");
      }
    }
    template <>
    ERROR NumpyArray_fill<double, int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      int64_t tooffset,
      const double *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint8_fromfloat64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<double, int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<double, int8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<double, int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      int64_t tooffset,
      const double *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint16_fromfloat64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<double, int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<double, int16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<double, int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      int64_t tooffset,
      const double *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint32_fromfloat64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<double, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<double, int32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<double, int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      int64_t tooffset,
      const double *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_toint64_fromfloat64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<double, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<double, int64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<double, uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      int64_t tooffset,
      const double *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint8_fromfloat64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<double, uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<double, uint8_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<double, uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      int64_t tooffset,
      const double *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint16_fromfloat64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<double, uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<double, uint16_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<double, uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      int64_t tooffset,
      const double *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint32_fromfloat64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<double, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<double, uint32_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<double, uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      int64_t tooffset,
      const double *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_touint64_fromfloat64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<double, uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<double, uint64_t>");
      }
    }
    template <>
    ERROR NumpyArray_fill<double, float>(
      kernel::lib ptr_lib,
      float *toptr,
      int64_t tooffset,
      const double *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat32_fromfloat64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<double, float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<double, float>");
      }
    }
    template <>
    ERROR NumpyArray_fill<double, double>(
      kernel::lib ptr_lib,
      double *toptr,
      int64_t tooffset,
      const double *fromptr,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_fill_tofloat64_fromfloat64(
          toptr,
          tooffset,
          fromptr,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_fill<double, double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_fill<double, double>");
      }
    }

    template<>
    ERROR ListArray_fill(
      kernel::lib ptr_lib,
      int64_t *tostarts,
      int64_t tostartsoffset,
      int64_t *tostops,
      int64_t tostopsoffset,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray_fill_to64_from32(
          tostarts,
          tostartsoffset,
          tostops,
          tostopsoffset,
          fromstarts,
          fromstops,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_fill");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_fill");
      }
    }

    template<>
    ERROR ListArray_fill(
      kernel::lib ptr_lib,
      int64_t *tostarts,
      int64_t tostartsoffset,
      int64_t *tostops,
      int64_t tostopsoffset,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray_fill_to64_fromU32(
          tostarts,
          tostartsoffset,
          tostops,
          tostopsoffset,
          fromstarts,
          fromstops,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_fill");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_fill");
      }
    }

    template<>
    ERROR ListArray_fill(
      kernel::lib ptr_lib,
      int64_t *tostarts,
      int64_t tostartsoffset,
      int64_t *tostops,
      int64_t tostopsoffset,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray_fill_to64_from64(
          tostarts,
          tostartsoffset,
          tostops,
          tostopsoffset,
          fromstarts,
          fromstops,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_fill");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_fill");
      }
    }

    template<>
    ERROR IndexedArray_fill(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t toindexoffset,
      const int32_t *fromindex,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray_fill_to64_from32(
          toindex,
          toindexoffset,
          fromindex,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_fill");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_fill");
      }
    }

    template<>
    ERROR IndexedArray_fill(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t toindexoffset,
      const uint32_t *fromindex,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray_fill_to64_fromU32(
          toindex,
          toindexoffset,
          fromindex,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_fill");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_fill");
      }
    }

    template<>
    ERROR IndexedArray_fill(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t toindexoffset,
      const int64_t *fromindex,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray_fill_to64_from64(
          toindex,
          toindexoffset,
          fromindex,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_fill");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_fill");
      }
    }

    ERROR IndexedArray_fill_to64_count(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t toindexoffset,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray_fill_to64_count(
          toindex,
          toindexoffset,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_fill_to64_count");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_fill_to64_count");
      }
    }

    ERROR UnionArray_filltags_to8_from8(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t totagsoffset,
      const int8_t *fromtags,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray_filltags_to8_from8(
          totags,
          totagsoffset,
          fromtags,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_filltags_to8_from8");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_filltags_to8_from8");
      }
    }

    template<>
    ERROR UnionArray_fillindex(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t toindexoffset,
      const int32_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray_fillindex_to64_from32(
          toindex,
          toindexoffset,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_fillindex");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_fillindex");
      }
    }

    template<>
    ERROR UnionArray_fillindex(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t toindexoffset,
      const uint32_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray_fillindex_to64_fromU32(
          toindex,
          toindexoffset,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_fillindex");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_fillindex");
      }
    }

    template<>
    ERROR UnionArray_fillindex(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t toindexoffset,
      const int64_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray_fillindex_to64_from64(
          toindex,
          toindexoffset,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_fillindex");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_fillindex");
      }
    }

    ERROR UnionArray_filltags_to8_const(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t totagsoffset,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray_filltags_to8_const(
          totags,
          totagsoffset,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_filltags_to8_const");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_filltags_to8_const");
      }
    }

    ERROR UnionArray_fillindex_count_64(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t toindexoffset,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray_fillindex_to64_count(
          toindex,
          toindexoffset,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_fillindex_count_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_fillindex_count_64");
      }
    }

    template<>
    ERROR UnionArray_simplify8_32_to8_64<int8_t, int32_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      const int8_t *outertags,
      const int32_t *outerindex,
      const int8_t *innertags,
      const int32_t *innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_32_simplify8_32_to8_64(
          totags,
          toindex,
          outertags,
          outerindex,
          innertags,
          innerindex,
          towhich,
          innerwhich,
          outerwhich,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_simplify8_32_to8_64<int8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_simplify8_32_to8_64<int8_t, int32_t>");
      }
    }

    template<>
    ERROR UnionArray_simplify8_32_to8_64<int8_t, uint32_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      const int8_t *outertags,
      const uint32_t *outerindex,
      const int8_t *innertags,
      const int32_t *innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_U32_simplify8_32_to8_64(
          totags,
          toindex,
          outertags,
          outerindex,
          innertags,
          innerindex,
          towhich,
          innerwhich,
          outerwhich,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_simplify8_32_to8_64<int8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_simplify8_32_to8_64<int8_t, uint32_t>");
      }
    }

    template<>
    ERROR UnionArray_simplify8_32_to8_64<int8_t, int64_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      const int8_t *outertags,
      const int64_t *outerindex,
      const int8_t *innertags,
      const int32_t *innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_64_simplify8_32_to8_64(
          totags,
          toindex,
          outertags,
          outerindex,
          innertags,
          innerindex,
          towhich,
          innerwhich,
          outerwhich,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_simplify8_32_to8_64<int8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_simplify8_32_to8_64<int8_t, int64_t>");
      }
    }

    template<>
    ERROR UnionArray_simplify8_U32_to8_64<int8_t, int32_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      const int8_t *outertags,
      const int32_t *outerindex,
      const int8_t *innertags,
      const uint32_t *innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_32_simplify8_U32_to8_64(
          totags,
          toindex,
          outertags,
          outerindex,
          innertags,
          innerindex,
          towhich,
          innerwhich,
          outerwhich,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_simplify8_U32_to8_64<int8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_simplify8_U32_to8_64<int8_t, int32_t>");
      }
    }

    template<>
    ERROR UnionArray_simplify8_U32_to8_64<int8_t, uint32_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      const int8_t *outertags,
      const uint32_t *outerindex,
      const int8_t *innertags,
      const uint32_t *innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_U32_simplify8_U32_to8_64(
          totags,
          toindex,
          outertags,
          outerindex,
          innertags,
          innerindex,
          towhich,
          innerwhich,
          outerwhich,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_simplify8_U32_to8_64<int8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_simplify8_U32_to8_64<int8_t, uint32_t>");
      }
    }

    template<>
    ERROR UnionArray_simplify8_U32_to8_64<int8_t, int64_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      const int8_t *outertags,
      const int64_t *outerindex,
      const int8_t *innertags,
      const uint32_t *innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_64_simplify8_U32_to8_64(
          totags,
          toindex,
          outertags,
          outerindex,
          innertags,
          innerindex,
          towhich,
          innerwhich,
          outerwhich,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_simplify8_U32_to8_64<int8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_simplify8_U32_to8_64<int8_t, int64_t>");
      }
    }

    template<>
    ERROR UnionArray_simplify8_64_to8_64<int8_t, int32_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      const int8_t *outertags,
      const int32_t *outerindex,
      const int8_t *innertags,
      const int64_t *innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_32_simplify8_64_to8_64(
          totags,
          toindex,
          outertags,
          outerindex,
          innertags,
          innerindex,
          towhich,
          innerwhich,
          outerwhich,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_simplify8_64_to8_64<int8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_simplify8_64_to8_64<int8_t, int32_t>");
      }
    }

    template<>
    ERROR UnionArray_simplify8_64_to8_64<int8_t, uint32_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      const int8_t *outertags,
      const uint32_t *outerindex,
      const int8_t *innertags,
      const int64_t *innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_U32_simplify8_64_to8_64(
          totags,
          toindex,
          outertags,
          outerindex,
          innertags,
          innerindex,
          towhich,
          innerwhich,
          outerwhich,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_simplify8_64_to8_64<int8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_simplify8_64_to8_64<int8_t, uint32_t>");
      }
    }

    template<>
    ERROR UnionArray_simplify8_64_to8_64<int8_t, int64_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      const int8_t *outertags,
      const int64_t *outerindex,
      const int8_t *innertags,
      const int64_t *innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_64_simplify8_64_to8_64(
          totags,
          toindex,
          outertags,
          outerindex,
          innertags,
          innerindex,
          towhich,
          innerwhich,
          outerwhich,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_simplify8_64_to8_64<int8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_simplify8_64_to8_64<int8_t, int64_t>");
      }
    }

    template<>
    ERROR UnionArray_simplify_one_to8_64<int8_t, int32_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      const int8_t *fromtags,
      const int32_t *fromindex,
      int64_t towhich,
      int64_t fromwhich,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_32_simplify_one_to8_64(
          totags,
          toindex,
          fromtags,
          fromindex,
          towhich,
          fromwhich,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_simplify_one_to8_64<int8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_simplify_one_to8_64<int8_t, int32_t>");
      }
    }

    template<>
    ERROR UnionArray_simplify_one_to8_64<int8_t, uint32_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      const int8_t *fromtags,
      const uint32_t *fromindex,
      int64_t towhich,
      int64_t fromwhich,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_U32_simplify_one_to8_64(
          totags,
          toindex,
          fromtags,
          fromindex,
          towhich,
          fromwhich,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_simplify_one_to8_64<int8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_simplify_one_to8_64<int8_t, uint32_t>");
      }
    }

    template<>
    ERROR UnionArray_simplify_one_to8_64<int8_t, int64_t>(
      kernel::lib ptr_lib,
      int8_t *totags,
      int64_t *toindex,
      const int8_t *fromtags,
      const int64_t *fromindex,
      int64_t towhich,
      int64_t fromwhich,
      int64_t length,
      int64_t base) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_64_simplify_one_to8_64(
          totags,
          toindex,
          fromtags,
          fromindex,
          towhich,
          fromwhich,
          length,
          base);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_simplify_one_to8_64<int8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_simplify_one_to8_64<int8_t, int64_t>");
      }
    }

    template<>
    ERROR ListArray_validity<int32_t>(
      kernel::lib ptr_lib,
      const int32_t *starts,
      const int32_t *stops,
      int64_t length,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_validity(
          starts,
          stops,
          length,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_validity<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_validity<int32_t>");
      }
    }

    template<>
    ERROR ListArray_validity<uint32_t>(
      kernel::lib ptr_lib,
      const uint32_t *starts,
      const uint32_t *stops,
      int64_t length,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_validity(
          starts,
          stops,
          length,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_validity<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_validity<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_validity<int64_t>(
      kernel::lib ptr_lib,
      const int64_t *starts,
      const int64_t *stops,
      int64_t length,
      int64_t lencontent) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_validity(
          starts,
          stops,
          length,
          lencontent);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_validity<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_validity<int64_t>");
      }
    }

    template<>
    ERROR IndexedArray_validity<int32_t>(
      kernel::lib ptr_lib,
      const int32_t *index,
      int64_t length,
      int64_t lencontent,
      bool isoption) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray32_validity(
          index,
          length,
          lencontent,
          isoption);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_validity<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_validity<int32_t>");
      }
    }

    template<>
    ERROR IndexedArray_validity<uint32_t>(
      kernel::lib ptr_lib,
      const uint32_t *index,
      int64_t length,
      int64_t lencontent,
      bool isoption) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_validity(
          index,
          length,
          lencontent,
          isoption);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_validity<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_validity<uint32_t>");
      }
    }

    template<>
    ERROR IndexedArray_validity<int64_t>(
      kernel::lib ptr_lib,
      const int64_t *index,
      int64_t length,
      int64_t lencontent,
      bool isoption) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray64_validity(
          index,
          length,
          lencontent,
          isoption);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_validity<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_validity<int64_t>");
      }
    }

    template<>
    ERROR UnionArray_validity<int8_t, int32_t>(
      kernel::lib ptr_lib,
      const int8_t *tags,
      const int32_t *index,
      int64_t length,
      int64_t numcontents,
      const int64_t *lencontents) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_32_validity(
          tags,
          index,
          length,
          numcontents,
          lencontents);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_validity<int8_t, int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_validity<int8_t, int32_t>");
      }
    }

    template<>
    ERROR UnionArray_validity<int8_t, uint32_t>(
      kernel::lib ptr_lib,
      const int8_t *tags,
      const uint32_t *index,
      int64_t length,
      int64_t numcontents,
      const int64_t *lencontents) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_U32_validity(
          tags,
          index,
          length,
          numcontents,
          lencontents);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_validity<int8_t, uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_validity<int8_t, uint32_t>");
      }
    }

    template<>
    ERROR UnionArray_validity<int8_t, int64_t>(
      kernel::lib ptr_lib,
      const int8_t *tags,
      const int64_t *index,
      int64_t length,
      int64_t numcontents,
      const int64_t *lencontents) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray8_64_validity(
          tags,
          index,
          length,
          numcontents,
          lencontents);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_validity<int8_t, int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_validity<int8_t, int64_t>");
      }
    }

    template<>
    ERROR UnionArray_fillna_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int32_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray_fillna_from32_to64(
          toindex,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_fillna_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_fillna_64<int32_t>");
      }
    }

    template<>
    ERROR UnionArray_fillna_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const uint32_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray_fillna_fromU32_to64(
          toindex,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_fillna_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_fillna_64<uint32_t>");
      }
    }

    template<>
    ERROR UnionArray_fillna_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int64_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_UnionArray_fillna_from64_to64(
          toindex,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for UnionArray_fillna_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for UnionArray_fillna_64<int64_t>");
      }
    }

    ERROR IndexedOptionArray_rpad_and_clip_mask_axis1_64(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int8_t *frommask,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64(
          toindex,
          frommask,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedOptionArray_rpad_and_clip_mask_axis1_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedOptionArray_rpad_and_clip_mask_axis1_64");
      }
    }

    ERROR index_rpad_and_clip_axis0_64(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t target,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_index_rpad_and_clip_axis0_64(
          toindex,
          target,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for index_rpad_and_clip_axis0_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for index_rpad_and_clip_axis0_64");
      }
    }

    ERROR index_rpad_and_clip_axis1_64(
      kernel::lib ptr_lib,
      int64_t *tostarts,
      int64_t *tostops,
      int64_t target,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_index_rpad_and_clip_axis1_64(
          tostarts,
          tostops,
          target,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for index_rpad_and_clip_axis1_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for index_rpad_and_clip_axis1_64");
      }
    }

    ERROR RegularArray_rpad_and_clip_axis1_64(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t target,
      int64_t size,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_rpad_and_clip_axis1_64(
          toindex,
          target,
          size,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_rpad_and_clip_axis1_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_rpad_and_clip_axis1_64");
      }
    }

    template<>
    ERROR ListArray_min_range<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tomin,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t lenstarts) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_min_range(
          tomin,
          fromstarts,
          fromstops,
          lenstarts);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_min_range<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_min_range<int32_t>");
      }
    }

    template<>
    ERROR ListArray_min_range<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tomin,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t lenstarts) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_min_range(
          tomin,
          fromstarts,
          fromstops,
          lenstarts);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_min_range<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_min_range<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_min_range<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tomin,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t lenstarts) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_min_range(
          tomin,
          fromstarts,
          fromstops,
          lenstarts);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_min_range<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_min_range<int64_t>");
      }
    }

    template<>
    ERROR ListArray_rpad_and_clip_length_axis1<int32_t>(
      kernel::lib ptr_lib,
      int64_t *tolength,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int64_t target,
      int64_t lenstarts) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_rpad_and_clip_length_axis1(
          tolength,
          fromstarts,
          fromstops,
          target,
          lenstarts);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_rpad_and_clip_length_axis1<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_rpad_and_clip_length_axis1<int32_t>");
      }
    }

    template<>
    ERROR ListArray_rpad_and_clip_length_axis1<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *tolength,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      int64_t target,
      int64_t lenstarts) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_rpad_and_clip_length_axis1(
          tolength,
          fromstarts,
          fromstops,
          target,
          lenstarts);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_rpad_and_clip_length_axis1<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_rpad_and_clip_length_axis1<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_rpad_and_clip_length_axis1<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tolength,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t target,
      int64_t lenstarts) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_rpad_and_clip_length_axis1(
          tolength,
          fromstarts,
          fromstops,
          target,
          lenstarts);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_rpad_and_clip_length_axis1<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_rpad_and_clip_length_axis1<int64_t>");
      }
    }

    template<>
    ERROR ListArray_rpad_axis1_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int32_t *fromstarts,
      const int32_t *fromstops,
      int32_t *tostarts,
      int32_t *tostops,
      int64_t target,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_rpad_axis1_64(
          toindex,
          fromstarts,
          fromstops,
          tostarts,
          tostops,
          target,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_rpad_axis1_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_rpad_axis1_64<int32_t>");
      }
    }

    template<>
    ERROR ListArray_rpad_axis1_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const uint32_t *fromstarts,
      const uint32_t *fromstops,
      uint32_t *tostarts,
      uint32_t *tostops,
      int64_t target,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_rpad_axis1_64(
          toindex,
          fromstarts,
          fromstops,
          tostarts,
          tostops,
          target,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_rpad_axis1_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_rpad_axis1_64<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_rpad_axis1_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int64_t *fromstarts,
      const int64_t *fromstops,
      int64_t *tostarts,
      int64_t *tostops,
      int64_t target,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_rpad_axis1_64(
          toindex,
          fromstarts,
          fromstops,
          tostarts,
          tostops,
          target,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_rpad_axis1_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_rpad_axis1_64<int64_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_rpad_and_clip_axis1_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int32_t *fromoffsets,
      int64_t length,
      int64_t target) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray32_rpad_and_clip_axis1_64(
          toindex,
          fromoffsets,
          length,
          target);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_rpad_and_clip_axis1_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_rpad_and_clip_axis1_64<int32_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_rpad_and_clip_axis1_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const uint32_t *fromoffsets,
      int64_t length,
      int64_t target) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArrayU32_rpad_and_clip_axis1_64(
          toindex,
          fromoffsets,
          length,
          target);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_rpad_and_clip_axis1_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_rpad_and_clip_axis1_64<uint32_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_rpad_and_clip_axis1_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int64_t *fromoffsets,
      int64_t length,
      int64_t target) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray64_rpad_and_clip_axis1_64(
          toindex,
          fromoffsets,
          length,
          target);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_rpad_and_clip_axis1_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_rpad_and_clip_axis1_64<int64_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_rpad_length_axis1<int32_t>(
      kernel::lib ptr_lib,
      int32_t *tooffsets,
      const int32_t *fromoffsets,
      int64_t fromlength,
      int64_t length,
      int64_t *tocount) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray32_rpad_length_axis1(
          tooffsets,
          fromoffsets,
          fromlength,
          length,
          tocount);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_rpad_length_axis1<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_rpad_length_axis1<int32_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_rpad_length_axis1<uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *tooffsets,
      const uint32_t *fromoffsets,
      int64_t fromlength,
      int64_t length,
      int64_t *tocount) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArrayU32_rpad_length_axis1(
          tooffsets,
          fromoffsets,
          fromlength,
          length,
          tocount);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_rpad_length_axis1<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_rpad_length_axis1<uint32_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_rpad_length_axis1<int64_t>(
      kernel::lib ptr_lib,
      int64_t *tooffsets,
      const int64_t *fromoffsets,
      int64_t fromlength,
      int64_t length,
      int64_t *tocount) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray64_rpad_length_axis1(
          tooffsets,
          fromoffsets,
          fromlength,
          length,
          tocount);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_rpad_length_axis1<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_rpad_length_axis1<int64_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_rpad_axis1_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int32_t *fromoffsets,
      int64_t fromlength,
      int64_t target) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray32_rpad_axis1_64(
          toindex,
          fromoffsets,
          fromlength,
          target);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_rpad_axis1_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_rpad_axis1_64<int32_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_rpad_axis1_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const uint32_t *fromoffsets,
      int64_t fromlength,
      int64_t target) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArrayU32_rpad_axis1_64(
          toindex,
          fromoffsets,
          fromlength,
          target);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_rpad_axis1_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_rpad_axis1_64<uint32_t>");
      }
    }

    template<>
    ERROR ListOffsetArray_rpad_axis1_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int64_t *fromoffsets,
      int64_t fromlength,
      int64_t target) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray64_rpad_axis1_64(
          toindex,
          fromoffsets,
          fromlength,
          target);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_rpad_axis1_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_rpad_axis1_64<int64_t>");
      }
    }

    ERROR localindex_64(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_localindex_64(
          toindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for localindex_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for localindex_64");
      }
    }

    template<>
    ERROR ListArray_localindex_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int32_t *offsets,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_localindex_64(
          toindex,
          offsets,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_localindex_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_localindex_64<int32_t>");
      }
    }

    template<>
    ERROR ListArray_localindex_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const uint32_t *offsets,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_localindex_64(
          toindex,
          offsets,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_localindex_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_localindex_64<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_localindex_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const int64_t *offsets,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_localindex_64(
          toindex,
          offsets,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_localindex_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_localindex_64<int64_t>");
      }
    }

    ERROR RegularArray_localindex_64(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t size,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_localindex_64(
          toindex,
          size,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_localindex_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_localindex_64");
      }
    }

    template<>
    ERROR combinations(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t n,
      bool replacement,
      int64_t singlelen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_combinations_64(
          toindex,
          n,
          replacement,
          singlelen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for combinations");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for combinations");
      }
    }

    template<>
    ERROR ListArray_combinations_length_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *totallen,
      int64_t *tooffsets,
      int64_t n,
      bool replacement,
      const int32_t *starts,
      const int32_t *stops,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_combinations_length_64(
          totallen,
          tooffsets,
          n,
          replacement,
          starts,
          stops,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_combinations_length_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_combinations_length_64<int32_t>");
      }
    }

    template<>
    ERROR ListArray_combinations_length_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *totallen,
      int64_t *tooffsets,
      int64_t n,
      bool replacement,
      const uint32_t *starts,
      const uint32_t *stops,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_combinations_length_64(
          totallen,
          tooffsets,
          n,
          replacement,
          starts,
          stops,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_combinations_length_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_combinations_length_64<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_combinations_length_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *totallen,
      int64_t *tooffsets,
      int64_t n,
      bool replacement,
      const int64_t *starts,
      const int64_t *stops,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_combinations_length_64(
          totallen,
          tooffsets,
          n,
          replacement,
          starts,
          stops,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_combinations_length_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_combinations_length_64<int64_t>");
      }
    }

    template<>
    ERROR ListArray_combinations_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t **tocarry,
      int64_t *toindex,
      int64_t *fromindex,
      int64_t n,
      bool replacement,
      const int32_t *starts,
      const int32_t *stops,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray32_combinations_64(
          tocarry,
          toindex,
          fromindex,
          n,
          replacement,
          starts,
          stops,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_combinations_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_combinations_64<int32_t>");
      }
    }

    template<>
    ERROR ListArray_combinations_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t **tocarry,
      int64_t *toindex,
      int64_t *fromindex,
      int64_t n,
      bool replacement,
      const uint32_t *starts,
      const uint32_t *stops,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArrayU32_combinations_64(
          tocarry,
          toindex,
          fromindex,
          n,
          replacement,
          starts,
          stops,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_combinations_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_combinations_64<uint32_t>");
      }
    }

    template<>
    ERROR ListArray_combinations_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t **tocarry,
      int64_t *toindex,
      int64_t *fromindex,
      int64_t n,
      bool replacement,
      const int64_t *starts,
      const int64_t *stops,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListArray64_combinations_64(
          tocarry,
          toindex,
          fromindex,
          n,
          replacement,
          starts,
          stops,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListArray_combinations_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListArray_combinations_64<int64_t>");
      }
    }

    ERROR RegularArray_combinations_64(
      kernel::lib ptr_lib,
      int64_t **tocarry,
      int64_t *toindex,
      int64_t *fromindex,
      int64_t n,
      bool replacement,
      int64_t size,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_RegularArray_combinations_64(
          tocarry,
          toindex,
          fromindex,
          n,
          replacement,
          size,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for RegularArray_combinations_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for RegularArray_combinations_64");
      }
    }

    ERROR ByteMaskedArray_overlay_mask8(
      kernel::lib ptr_lib,
      int8_t *tomask,
      const int8_t *theirmask,
      const int8_t *mymask,
      int64_t length,
      bool validwhen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ByteMaskedArray_overlay_mask8(
          tomask,
          theirmask,
          mymask,
          length,
          validwhen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ByteMaskedArray_overlay_mask8");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ByteMaskedArray_overlay_mask8");
      }
    }

    ERROR BitMaskedArray_to_ByteMaskedArray(
      kernel::lib ptr_lib,
      int8_t *tobytemask,
      const uint8_t *frombitmask,
      int64_t bitmasklength,
      bool validwhen,
      bool lsb_order) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_BitMaskedArray_to_ByteMaskedArray(
          tobytemask,
          frombitmask,
          bitmasklength,
          validwhen,
          lsb_order);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for BitMaskedArray_to_ByteMaskedArray");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for BitMaskedArray_to_ByteMaskedArray");
      }
    }

    ERROR BitMaskedArray_to_IndexedOptionArray64(
      kernel::lib ptr_lib,
      int64_t *toindex,
      const uint8_t *frombitmask,
      int64_t bitmasklength,
      bool validwhen,
      bool lsb_order) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_BitMaskedArray_to_IndexedOptionArray64(
          toindex,
          frombitmask,
          bitmasklength,
          validwhen,
          lsb_order);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for BitMaskedArray_to_IndexedOptionArray64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for BitMaskedArray_to_IndexedOptionArray64");
      }
    }

    /////////////////////////////////// awkward/kernels/reducers.h

    ERROR reduce_count_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_count_64(
          toptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_count_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_count_64");
      }
    }

    template<>
    ERROR reduce_countnonzero_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const bool *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_countnonzero_bool_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_countnonzero_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_countnonzero_64");
      }
    }

    template<>
    ERROR reduce_countnonzero_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_countnonzero_uint8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_countnonzero_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_countnonzero_64");
      }
    }

    template<>
    ERROR reduce_countnonzero_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_countnonzero_int8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_countnonzero_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_countnonzero_64");
      }
    }

    template<>
    ERROR reduce_countnonzero_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_countnonzero_int16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_countnonzero_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_countnonzero_64");
      }
    }

    template<>
    ERROR reduce_countnonzero_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_countnonzero_uint16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_countnonzero_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_countnonzero_64");
      }
    }

    template<>
    ERROR reduce_countnonzero_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_countnonzero_int32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_countnonzero_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_countnonzero_64");
      }
    }

    template<>
    ERROR reduce_countnonzero_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_countnonzero_uint32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_countnonzero_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_countnonzero_64");
      }
    }

    template<>
    ERROR reduce_countnonzero_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_countnonzero_int64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_countnonzero_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_countnonzero_64");
      }
    }

    template<>
    ERROR reduce_countnonzero_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_countnonzero_uint64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_countnonzero_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_countnonzero_64");
      }
    }

    template<>
    ERROR reduce_countnonzero_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const float *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_countnonzero_float32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_countnonzero_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_countnonzero_64");
      }
    }

    template<>
    ERROR reduce_countnonzero_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const double *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_countnonzero_float64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_countnonzero_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_countnonzero_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const bool *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_int64_bool_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_int64_int8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      const uint8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_uint64_uint8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_int64_int16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      const uint16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_uint64_uint16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_int64_int32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      const uint32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_uint64_uint32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_int64_int64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      const uint64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_uint64_uint64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      float *toptr,
      const float *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_float32_float32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      double *toptr,
      const double *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_float64_float64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const bool *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_int32_bool_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_int32_int8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      const uint8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_uint32_uint8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_int32_int16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      const uint16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_uint32_uint16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_int32_int32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      const uint32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_uint32_uint32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_64");
      }
    }

    template<>
    ERROR reduce_sum_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const bool *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_bool_bool_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_bool_64");
      }
    }

    template<>
    ERROR reduce_sum_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const int8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_bool_int8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_bool_64");
      }
    }

    template<>
    ERROR reduce_sum_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const uint8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_bool_uint8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_bool_64");
      }
    }

    template<>
    ERROR reduce_sum_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const int16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_bool_int16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_bool_64");
      }
    }

    template<>
    ERROR reduce_sum_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const uint16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_bool_uint16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_bool_64");
      }
    }

    template<>
    ERROR reduce_sum_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const int32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_bool_int32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_bool_64");
      }
    }

    template<>
    ERROR reduce_sum_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const uint32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_bool_uint32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_bool_64");
      }
    }

    template<>
    ERROR reduce_sum_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const int64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_bool_int64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_bool_64");
      }
    }

    template<>
    ERROR reduce_sum_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const uint64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_bool_uint64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_bool_64");
      }
    }

    template<>
    ERROR reduce_sum_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const float *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_bool_float32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_bool_64");
      }
    }

    template<>
    ERROR reduce_sum_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const double *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_sum_bool_float64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_sum_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_sum_bool_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const bool *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_int64_bool_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_int64_int8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      const uint8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_uint64_uint8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_int64_int16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      const uint16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_uint64_uint16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_int64_int32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      const uint32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_uint64_uint32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_int64_int64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      const uint64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_uint64_uint64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      float *toptr,
      const float *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_float32_float32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      double *toptr,
      const double *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_float64_float64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const bool *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_int32_bool_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_int32_int8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      const uint8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_uint32_uint8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_int32_int16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      const uint16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_uint32_uint16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_int32_int32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      const uint32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_uint32_uint32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_64");
      }
    }

    template<>
    ERROR reduce_prod_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const bool *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_bool_bool_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_bool_64");
      }
    }

    template<>
    ERROR reduce_prod_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const int8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_bool_int8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_bool_64");
      }
    }

    template<>
    ERROR reduce_prod_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const uint8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_bool_uint8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_bool_64");
      }
    }

    template<>
    ERROR reduce_prod_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const int16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_bool_int16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_bool_64");
      }
    }

    template<>
    ERROR reduce_prod_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const uint16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_bool_uint16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_bool_64");
      }
    }

    template<>
    ERROR reduce_prod_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const int32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_bool_int32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_bool_64");
      }
    }

    template<>
    ERROR reduce_prod_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const uint32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_bool_uint32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_bool_64");
      }
    }

    template<>
    ERROR reduce_prod_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const int64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_bool_int64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_bool_64");
      }
    }

    template<>
    ERROR reduce_prod_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const uint64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_bool_uint64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_bool_64");
      }
    }

    template<>
    ERROR reduce_prod_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const float *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_bool_float32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_bool_64");
      }
    }

    template<>
    ERROR reduce_prod_bool_64(
      kernel::lib ptr_lib,
      bool *toptr,
      const double *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_prod_bool_float64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_prod_bool_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_prod_bool_64");
      }
    }

    template<>
    ERROR reduce_min_64(
      kernel::lib ptr_lib,
      int8_t *toptr,
      const int8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      int8_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_min_int8_int8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_min_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_min_64");
      }
    }

    template<>
    ERROR reduce_min_64(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      const uint8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      uint8_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_min_uint8_uint8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_min_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_min_64");
      }
    }

    template<>
    ERROR reduce_min_64(
      kernel::lib ptr_lib,
      int16_t *toptr,
      const int16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      int16_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_min_int16_int16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_min_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_min_64");
      }
    }

    template<>
    ERROR reduce_min_64(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      const uint16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      uint16_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_min_uint16_uint16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_min_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_min_64");
      }
    }

    template<>
    ERROR reduce_min_64(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      int32_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_min_int32_int32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_min_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_min_64");
      }
    }

    template<>
    ERROR reduce_min_64(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      const uint32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      uint32_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_min_uint32_uint32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_min_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_min_64");
      }
    }

    template<>
    ERROR reduce_min_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      int64_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_min_int64_int64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_min_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_min_64");
      }
    }

    template<>
    ERROR reduce_min_64(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      const uint64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      uint64_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_min_uint64_uint64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_min_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_min_64");
      }
    }

    template<>
    ERROR reduce_min_64(
      kernel::lib ptr_lib,
      float *toptr,
      const float *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      float identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_min_float32_float32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_min_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_min_64");
      }
    }

    template<>
    ERROR reduce_min_64(
      kernel::lib ptr_lib,
      double *toptr,
      const double *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      double identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_min_float64_float64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_min_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_min_64");
      }
    }

    template<>
    ERROR reduce_max_64(
      kernel::lib ptr_lib,
      int8_t *toptr,
      const int8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      int8_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_max_int8_int8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_max_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_max_64");
      }
    }

    template<>
    ERROR reduce_max_64(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      const uint8_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      uint8_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_max_uint8_uint8_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_max_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_max_64");
      }
    }

    template<>
    ERROR reduce_max_64(
      kernel::lib ptr_lib,
      int16_t *toptr,
      const int16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      int16_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_max_int16_int16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_max_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_max_64");
      }
    }

    template<>
    ERROR reduce_max_64(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      const uint16_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      uint16_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_max_uint16_uint16_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_max_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_max_64");
      }
    }

    template<>
    ERROR reduce_max_64(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      int32_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_max_int32_int32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_max_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_max_64");
      }
    }

    template<>
    ERROR reduce_max_64(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      const uint32_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      uint32_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_max_uint32_uint32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_max_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_max_64");
      }
    }

    template<>
    ERROR reduce_max_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      int64_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_max_int64_int64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_max_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_max_64");
      }
    }

    template<>
    ERROR reduce_max_64(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      const uint64_t *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      uint64_t identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_max_uint64_uint64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_max_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_max_64");
      }
    }

    template<>
    ERROR reduce_max_64(
      kernel::lib ptr_lib,
      float *toptr,
      const float *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      float identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_max_float32_float32_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_max_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_max_64");
      }
    }

    template<>
    ERROR reduce_max_64(
      kernel::lib ptr_lib,
      double *toptr,
      const double *fromptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength,
      double identity) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_max_float64_float64_64(
          toptr,
          fromptr,
          parents,
          lenparents,
          outlength,
          identity);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_max_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_max_64");
      }
    }

    template<>
    ERROR reduce_argmin_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const bool *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmin_bool_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmin_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmin_64");
      }
    }

    template<>
    ERROR reduce_argmin_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int8_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmin_int8_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmin_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmin_64");
      }
    }

    template<>
    ERROR reduce_argmin_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint8_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmin_uint8_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmin_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmin_64");
      }
    }

    template<>
    ERROR reduce_argmin_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int16_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmin_int16_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmin_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmin_64");
      }
    }

    template<>
    ERROR reduce_argmin_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint16_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmin_uint16_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmin_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmin_64");
      }
    }

    template<>
    ERROR reduce_argmin_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int32_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmin_int32_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmin_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmin_64");
      }
    }

    template<>
    ERROR reduce_argmin_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint32_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmin_uint32_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmin_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmin_64");
      }
    }

    template<>
    ERROR reduce_argmin_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmin_int64_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmin_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmin_64");
      }
    }

    template<>
    ERROR reduce_argmin_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint64_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmin_uint64_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmin_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmin_64");
      }
    }

    template<>
    ERROR reduce_argmin_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const float *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmin_float32_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmin_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmin_64");
      }
    }

    template<>
    ERROR reduce_argmin_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const double *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmin_float64_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmin_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmin_64");
      }
    }


    template<>
    ERROR reduce_argmax_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const bool *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmax_bool_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmax_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmax_64");
      }
    }

    template<>
    ERROR reduce_argmax_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int8_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmax_int8_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmax_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmax_64");
      }
    }

    template<>
    ERROR reduce_argmax_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint8_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmax_uint8_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmax_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmax_64");
      }
    }

    template<>
    ERROR reduce_argmax_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int16_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmax_int16_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmax_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmax_64");
      }
    }

    template<>
    ERROR reduce_argmax_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint16_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmax_uint16_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmax_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmax_64");
      }
    }

    template<>
    ERROR reduce_argmax_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int32_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmax_int32_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmax_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmax_64");
      }
    }

    template<>
    ERROR reduce_argmax_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint32_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmax_uint32_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmax_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmax_64");
      }
    }

    template<>
    ERROR reduce_argmax_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmax_int64_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmax_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmax_64");
      }
    }

    template<>
    ERROR reduce_argmax_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint64_t *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmax_uint64_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmax_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmax_64");
      }
    }

    template<>
    ERROR reduce_argmax_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const float *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmax_float32_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmax_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmax_64");
      }
    }

    template<>
    ERROR reduce_argmax_64(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const double *fromptr,
      const int64_t *starts,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_reduce_argmax_float64_64(
          toptr,
          fromptr,
          starts,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for reduce_argmax_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for reduce_argmax_64");
      }
    }

    ERROR content_reduce_zeroparents_64(
      kernel::lib ptr_lib,
      int64_t *toparents,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_content_reduce_zeroparents_64(
          toparents,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for content_reduce_zeroparents_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for content_reduce_zeroparents_64");
      }
    }

    ERROR ListOffsetArray_reduce_global_startstop_64(
      kernel::lib ptr_lib,
      int64_t *globalstart,
      int64_t *globalstop,
      const int64_t *offsets,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray_reduce_global_startstop_64(
          globalstart,
          globalstop,
          offsets,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_reduce_global_startstop_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_reduce_global_startstop_64");
      }
    }

    ERROR ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64(
      kernel::lib ptr_lib,
      int64_t *maxcount,
      int64_t *offsetscopy,
      const int64_t *offsets,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64(
          maxcount,
          offsetscopy,
          offsets,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64");
      }
    }

    ERROR ListOffsetArray_reduce_nonlocal_preparenext_64(
      kernel::lib ptr_lib,
      int64_t *nextcarry,
      int64_t *nextparents,
      int64_t nextlen,
      int64_t *maxnextparents,
      int64_t *distincts,
      int64_t distinctslen,
      int64_t *offsetscopy,
      const int64_t *offsets,
      int64_t length,
      const int64_t *parents,
      int64_t maxcount) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray_reduce_nonlocal_preparenext_64(
          nextcarry,
          nextparents,
          nextlen,
          maxnextparents,
          distincts,
          distinctslen,
          offsetscopy,
          offsets,
          length,
          parents,
          maxcount);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_reduce_nonlocal_preparenext_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_reduce_nonlocal_preparenext_64");
      }
    }

    ERROR ListOffsetArray_reduce_nonlocal_nextstarts_64(
      kernel::lib ptr_lib,
      int64_t *nextstarts,
      const int64_t *nextparents,
      int64_t nextlen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64(
          nextstarts,
          nextparents,
          nextlen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_reduce_nonlocal_nextstarts_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_reduce_nonlocal_nextstarts_64");
      }
    }

    ERROR ListOffsetArray_reduce_nonlocal_findgaps_64(
      kernel::lib ptr_lib,
      int64_t *gaps,
      const int64_t *parents,
      int64_t lenparents) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray_reduce_nonlocal_findgaps_64(
          gaps,
          parents,
          lenparents);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_reduce_nonlocal_findgaps_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_reduce_nonlocal_findgaps_64");
      }
    }

    ERROR ListOffsetArray_reduce_nonlocal_outstartsstops_64(
      kernel::lib ptr_lib,
      int64_t *outstarts,
      int64_t *outstops,
      const int64_t *distincts,
      int64_t lendistincts,
      const int64_t *gaps,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64(
          outstarts,
          outstops,
          distincts,
          lendistincts,
          gaps,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_reduce_nonlocal_outstartsstops_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_reduce_nonlocal_outstartsstops_64");
      }
    }

    ERROR ListOffsetArray_reduce_local_nextparents_64(
      kernel::lib ptr_lib,
      int64_t *nextparents,
      const int64_t *offsets,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray_reduce_local_nextparents_64(
          nextparents,
          offsets,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_reduce_local_nextparents_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_reduce_local_nextparents_64");
      }
    }

    ERROR ListOffsetArray_reduce_local_outoffsets_64(
      kernel::lib ptr_lib,
      int64_t *outoffsets,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray_reduce_local_outoffsets_64(
          outoffsets,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_reduce_local_outoffsets_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_reduce_local_outoffsets_64");
      }
    }

    template<>
    ERROR IndexedArray_reduce_next_64<int32_t>(
      kernel::lib ptr_lib,
      int64_t *nextcarry,
      int64_t *nextparents,
      int64_t *outindex,
      const int32_t *index,
      int64_t *parents,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray32_reduce_next_64(
          nextcarry,
          nextparents,
          outindex,
          index,
          parents,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_reduce_next_64<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_reduce_next_64<int32_t>");
      }
    }

    template<>
    ERROR IndexedArray_reduce_next_64<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *nextcarry,
      int64_t *nextparents,
      int64_t *outindex,
      const uint32_t *index,
      int64_t *parents,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArrayU32_reduce_next_64(
          nextcarry,
          nextparents,
          outindex,
          index,
          parents,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_reduce_next_64<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_reduce_next_64<uint32_t>");
      }
    }

    template<>
    ERROR IndexedArray_reduce_next_64<int64_t>(
      kernel::lib ptr_lib,
      int64_t *nextcarry,
      int64_t *nextparents,
      int64_t *outindex,
      const int64_t *index,
      int64_t *parents,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray64_reduce_next_64(
          nextcarry,
          nextparents,
          outindex,
          index,
          parents,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_reduce_next_64<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_reduce_next_64<int64_t>");
      }
    }

    ERROR IndexedArray_reduce_next_fix_offsets_64(
      kernel::lib ptr_lib,
      int64_t *outoffsets,
      const int64_t *starts,
      int64_t startslength,
      int64_t outindexlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray_reduce_next_fix_offsets_64(
          outoffsets,
          starts,
          startslength,
          outindexlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_reduce_next_fix_offsets_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_reduce_next_fix_offsets_64");
      }
    }

    ERROR NumpyArray_reduce_mask_ByteMaskedArray_64(
      kernel::lib ptr_lib,
      int8_t *toptr,
      const int64_t *parents,
      int64_t lenparents,
      int64_t outlength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_reduce_mask_ByteMaskedArray_64(
          toptr,
          parents,
          lenparents,
          outlength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_reduce_mask_ByteMaskedArray_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_reduce_mask_ByteMaskedArray_64");
      }
    }

    ERROR ByteMaskedArray_reduce_next_64(
      kernel::lib ptr_lib,
      int64_t *nextcarry,
      int64_t *nextparents,
      int64_t *outindex,
      const int8_t *mask,
      const int64_t *parents,
      int64_t length,
      bool validwhen) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ByteMaskedArray_reduce_next_64(
          nextcarry,
          nextparents,
          outindex,
          mask,
          parents,
          length,
          validwhen);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ByteMaskedArray_reduce_next_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ByteMaskedArray_reduce_next_64");
      }
    }

    /////////////////////////////////// awkward/kernels/sorting.h

    ERROR sorting_ranges(
      kernel::lib ptr_lib,
      int64_t *toindex,
      int64_t tolength,
      const int64_t *parents,
      int64_t parentslength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sorting_ranges(
          toindex,
          tolength,
          parents,
          parentslength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for sorting_ranges");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for sorting_ranges");
      }
    }

    ERROR sorting_ranges_length(
      kernel::lib ptr_lib,
      int64_t *tolength,
      const int64_t *parents,
      int64_t parentslength) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sorting_ranges_length(
          tolength,
          parents,
          parentslength);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for sorting_ranges_length");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for sorting_ranges_length");
      }
    }

    template<>
    ERROR NumpyArray_argsort<bool>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const bool *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_argsort_bool(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_argsort<bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_argsort<bool>");
      }
    }

    template<>
    ERROR NumpyArray_argsort<int8_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int8_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_argsort_int8(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_argsort<int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_argsort<int8_t>");
      }
    }

    template<>
    ERROR NumpyArray_argsort<uint8_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint8_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_argsort_uint8(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_argsort<uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_argsort<uint8_t>");
      }
    }

    template<>
    ERROR NumpyArray_argsort<int16_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int16_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_argsort_int16(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_argsort<int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_argsort<int16_t>");
      }
    }

    template<>
    ERROR NumpyArray_argsort<uint16_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint16_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_argsort_uint16(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_argsort<uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_argsort<uint16_t>");
      }
    }

    template<>
    ERROR NumpyArray_argsort<int32_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int32_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_argsort_int32(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_argsort<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_argsort<int32_t>");
      }
    }

    template<>
    ERROR NumpyArray_argsort<uint32_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint32_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_argsort_uint32(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_argsort<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_argsort<uint32_t>");
      }
    }

    template<>
    ERROR NumpyArray_argsort<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_argsort_int64(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_argsort<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_argsort<int64_t>");
      }
    }

    template<>
    ERROR NumpyArray_argsort<uint64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const uint64_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_argsort_uint64(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_argsort<uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_argsort<uint64_t>");
      }
    }

    template<>
    ERROR NumpyArray_argsort<float>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const float *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_argsort_float32(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_argsort<float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_argsort<float>");
      }
    }

    template<>
    ERROR NumpyArray_argsort<double>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const double *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_argsort_float64(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_argsort<double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_argsort<double>");
      }
    }

    template<>
    ERROR NumpyArray_sort<bool>(
      kernel::lib ptr_lib,
      bool *toptr,
      const bool *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sort_bool(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          parentslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_sort<bool>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_sort<bool>");
      }
    }

    template<>
    ERROR NumpyArray_sort<uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      const uint8_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sort_uint8(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          parentslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_sort<uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_sort<uint8_t>");
      }
    }

    template<>
    ERROR NumpyArray_sort<int8_t>(
      kernel::lib ptr_lib,
      int8_t *toptr,
      const int8_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sort_int8(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          parentslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_sort<int8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_sort<int8_t>");
      }
    }

    template<>
    ERROR NumpyArray_sort<uint16_t>(
      kernel::lib ptr_lib,
      uint16_t *toptr,
      const uint16_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sort_uint16(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          parentslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_sort<uint16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_sort<uint16_t>");
      }
    }

    template<>
    ERROR NumpyArray_sort<int16_t>(
      kernel::lib ptr_lib,
      int16_t *toptr,
      const int16_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sort_int16(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          parentslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_sort<int16_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_sort<int16_t>");
      }
    }

    template<>
    ERROR NumpyArray_sort<uint32_t>(
      kernel::lib ptr_lib,
      uint32_t *toptr,
      const uint32_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sort_uint32(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          parentslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_sort<uint32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_sort<uint32_t>");
      }
    }

    template<>
    ERROR NumpyArray_sort<int32_t>(
      kernel::lib ptr_lib,
      int32_t *toptr,
      const int32_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sort_int32(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          parentslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_sort<int32_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_sort<int32_t>");
      }
    }

    template<>
    ERROR NumpyArray_sort<uint64_t>(
      kernel::lib ptr_lib,
      uint64_t *toptr,
      const uint64_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sort_uint64(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          parentslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_sort<uint64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_sort<uint64_t>");
      }
    }

    template<>
    ERROR NumpyArray_sort<int64_t>(
      kernel::lib ptr_lib,
      int64_t *toptr,
      const int64_t *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sort_int64(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          parentslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_sort<int64_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_sort<int64_t>");
      }
    }

    template<>
    ERROR NumpyArray_sort<float>(
      kernel::lib ptr_lib,
      float *toptr,
      const float *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sort_float32(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          parentslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_sort<float>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_sort<float>");
      }
    }

    template<>
    ERROR NumpyArray_sort<double>(
      kernel::lib ptr_lib,
      double *toptr,
      const double *fromptr,
      int64_t length,
      const int64_t *offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_sort_float64(
          toptr,
          fromptr,
          length,
          offsets,
          offsetslength,
          parentslength,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_sort<double>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_sort<double>");
      }
    }

    template<>
    ERROR NumpyArray_sort_asstrings<uint8_t>(
      kernel::lib ptr_lib,
      uint8_t *toptr,
      const uint8_t *fromptr,
      const int64_t *offsets,
      int64_t offsetslength,
      int64_t *outoffsets,
      bool ascending,
      bool stable) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_NumpyArray_sort_asstrings_uint8(
          toptr,
          fromptr,
          offsets,
          offsetslength,
          outoffsets,
          ascending,
          stable);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for NumpyArray_sort_asstrings<uint8_t>");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for NumpyArray_sort_asstrings<uint8_t>");
      }
    }

    ERROR ListOffsetArray_local_preparenext_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int64_t *fromindex,
      int64_t length) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_ListOffsetArray_local_preparenext_64(
          tocarry,
          fromindex,
          length);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for ListOffsetArray_local_preparenext_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for ListOffsetArray_local_preparenext_64");
      }
    }

    ERROR IndexedArray_local_preparenext_64(
      kernel::lib ptr_lib,
      int64_t *tocarry,
      const int64_t *starts,
      const int64_t *parents,
      int64_t parentslength,
      const int64_t *nextparents) {
      if (ptr_lib == kernel::lib::cpu) {
        return awkward_IndexedArray_local_preparenext_64(
          tocarry,
          starts,
          parents,
          parentslength,
          nextparents);
      }
      else if (ptr_lib == kernel::lib::cuda) {
        throw std::runtime_error(
          "not implemented: ptr_lib == cuda_kernels for IndexedArray_local_preparenext_64");
      }
      else {
        throw std::runtime_error(
          "unrecognized ptr_lib for IndexedArray_local_preparenext_64");
      }
    }

  }
}
