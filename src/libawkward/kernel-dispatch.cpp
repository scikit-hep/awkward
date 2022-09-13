// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/kernel-dispatch.cpp", line)

#include <complex>

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"
#include "awkward/cuda-utils.h"

#include "awkward/kernel-dispatch.h"

#define CREATE_KERNEL(libFnName, ptr_lib)          \
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

    void* acquire_handle(kernel::lib ptr_lib) {
#ifndef _MSC_VER
      void *handle = nullptr;
      std::string path = lib_callback->awkward_library_path(ptr_lib);
      if (!path.empty()) {
        handle = dlopen(path.c_str(), RTLD_LAZY);
      }
      if (!handle) {
        if (ptr_lib == kernel::lib::cuda) {
          throw std::invalid_argument(
            std::string("array resides on a GPU, but 'awkward-cuda-kernels' is not "
                        "installed; install it with:\n\n    "
                        "pip install awkward[cuda] --upgrade")
            + FILENAME(__LINE__));
        }
        else {
          throw std::runtime_error(
            std::string("unrecognized ptr_lib in acquire_handle")
            + FILENAME(__LINE__));
        }
      }
      return handle;
#else
      throw std::invalid_argument(
          std::string("array resides on a GPU, but 'awkward-cuda-kernels' is not"
                      "supported on Windows") + FILENAME(__LINE__));
#endif
    }

    void *acquire_symbol(void* handle, const std::string& symbol_name) {
      void *symbol_ptr = nullptr;
#ifndef _MSC_VER
      symbol_ptr = dlsym(handle, symbol_name.c_str());
      if (!symbol_ptr) {
        throw std::runtime_error(
          symbol_name + std::string(" not found in kernels library")
          + FILENAME(__LINE__));
      }
#endif
      return symbol_ptr;
    }

    const int64_t
    lib_device_num(
      kernel::lib ptr_lib,
      void* ptr) {
      if(ptr_lib == kernel::lib::cuda) {
        int64_t num;
        {
          CREATE_KERNEL(awkward_cuda_ptr_device_num, ptr_lib);
          struct Error err1 = (*awkward_cuda_ptr_device_num_fcn)(&num, ptr);
          util::handle_error(err1);
        }
        return num;
      }
      else {
        throw std::runtime_error(
          std::string("unrecognized ptr_lib in kernel::lib_device_num")
          + FILENAME(__LINE__));
      }
    }

    const std::string
    lib_tostring(
      kernel::lib ptr_lib,
      void* ptr,
      const std::string& indent,
      const std::string& pre,
      const std::string& post) {
      if (ptr_lib == kernel::lib::cpu) {
        return std::string("");
      }

      else if (ptr_lib == kernel::lib::cuda) {
        const int64_t num = lib_device_num(ptr_lib, ptr);

        char name[256];
        CREATE_KERNEL(awkward_cuda_ptr_device_name, ptr_lib);
        struct Error err2 = (*awkward_cuda_ptr_device_name_fcn)(name, ptr);
        util::handle_error(err2);

        std::stringstream out;
        out << indent << pre << "<Kernels lib=\"cuda\" device=\"" << num
            << "\" device_name=\"" << name << "\"/>" << post;
        return out.str();
      }

      else {
        throw std::runtime_error(
          std::string("unrecognized ptr_lib in kernel::lib_tostring")
          + FILENAME(__LINE__));
      }
    }

    ERROR copy_to(
      kernel::lib to_lib,
      kernel::lib from_lib,
      void* to_ptr,
      void* from_ptr,
      int64_t bytelength) {
      if (from_lib == lib::cpu  &&  to_lib == lib::cuda) {
        CREATE_KERNEL(awkward_cuda_host_to_device, kernel::lib::cuda);
        return (*awkward_cuda_host_to_device_fcn)(to_ptr, from_ptr, bytelength);
      }
      else if (from_lib == lib::cuda  &&  to_lib == lib::cpu) {
        CREATE_KERNEL(awkward_cuda_device_to_host, kernel::lib::cuda);
        return (*awkward_cuda_device_to_host_fcn)(to_ptr, from_ptr, bytelength);
      }
      else {
        throw std::runtime_error(
          std::string("unrecognized combination of from_lib and to_lib")
          + FILENAME(__LINE__));
      }
    }

    const std::string
    fully_qualified_cache_key(kernel::lib ptr_lib, const std::string& cache_key) {
      switch (ptr_lib) {
        case kernel::lib::cuda:
          return cache_key + std::string(":cuda");
        default:
          return cache_key;
      }
    }

  /////////////////////////////////// awkward/kernels/getitem.h

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
          std::string("not implemented: ptr_lib == cuda_kernels for UnionArray_regular_index<int8_t, int64_t>")
          + FILENAME(__LINE__));
      }
      else {
        throw std::runtime_error(
          std::string("unrecognized ptr_lib for UnionArray_regular_index<int8_t, int64_t>")
          + FILENAME(__LINE__));
      }
     }
  }
}
