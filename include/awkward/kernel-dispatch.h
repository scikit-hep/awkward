// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_KERNEL_DISPATCH_H_
#define AWKWARD_KERNEL_DISPATCH_H_

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/kernel-utils.h"
#include "awkward/kernels.h"

#include <sstream>

#ifndef _MSC_VER
  #include "dlfcn.h"
#endif

namespace awkward {
  namespace kernel {

    enum class lib {
        cpu,
        cuda,
        size
    };

    class LibraryPathCallback {
    public:
        LibraryPathCallback() = default;
        virtual std::string library_path() = 0;
    };

    class LibraryCallback {
    public:
        LibraryCallback();

        void add_library_path_callback(
          kernel::lib ptr_lib,
          const std::shared_ptr<LibraryPathCallback> &callback);

        std::string awkward_library_path(kernel::lib ptr_lib);

    private:
        std::map<kernel::lib, std::vector<std::shared_ptr<LibraryPathCallback>>> lib_path_callbacks;
        std::mutex lib_path_callbacks_mutex;
    };

    extern std::shared_ptr<LibraryCallback> lib_callback;

    /// @brief Internal utility function to return an opaque ptr if an handle is
    /// acquired for the specified ptr_lib. If not, then it raises an appropriate
    /// exception
    void* acquire_handle(kernel::lib ptr_lib);

    /// @brief Internal utility function to return an opaque ptr if an symbol is
    /// found for the corresponding handle. If not, then it raises an appropriate
    /// exception
    void* acquire_symbol(void* handle, const std::string& symbol_name);

    /// @class array_deleter
    ///
    /// @brief Used as a `std::shared_ptr` deleter (second argument) to
    /// overload `delete ptr` with `delete[] ptr`.
    ///
    /// This is necessary for `std::shared_ptr` to contain array buffers.
    ///
    /// See also
    ///   - cuda_array_deleter, which deletes an array on a GPU.
    ///   - no_deleter, which does not free memory at all (for borrowed
    ///     references).
    ///   - pyobject_deleter, which reduces the reference count of a
    ///     Python object when there are no more C++ shared pointers
    ///     referencing it.
    template <typename T>
    class LIBAWKWARD_EXPORT_SYMBOL array_deleter {
    public:
        /// @brief Called by `std::shared_ptr` when its reference count reaches
        /// zero.
        void operator()(T const *ptr) {
          awkward_free(reinterpret_cast<void const*>(ptr));
        }
    };

    /// @class cuda_array_deleter
    ///
    /// @brief Used as a `std::shared_ptr` deleter (second argument) to
    /// overload `delete ptr` with `cudaFree`.
    ///
    /// This is necessary for `std::shared_ptr` to contain array buffers.
    ///
    /// See also
    ///   - array_deleter, which deletes array buffers in main memory.
    ///   - no_deleter, which does not free memory at all (for borrowed
    ///     references).
    ///   - pyobject_deleter, which reduces the reference count of a
    ///     Python object when there are no more C++ shared pointers
    ///     referencing it.
    template <typename T>
    class LIBAWKWARD_EXPORT_SYMBOL cuda_array_deleter {
    public:
        /// @brief Called by `std::shared_ptr` when its reference count reaches
        /// zero.
        void operator()(T const *ptr) {
          auto handle = acquire_handle(lib::cuda);
          typedef decltype(awkward_free) functor_type;
          auto* awkward_free_fcn = reinterpret_cast<functor_type*>(
                                     acquire_symbol(handle, "awkward_free"));
          (*awkward_free_fcn)(reinterpret_cast<void const*>(ptr));
        }
    };

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
    ///   - cuda_array_deleter, which frees array buffers on GPUs.
    ///   - pyobject_deleter, which reduces the reference count of a
    ///     Python object when there are no more C++ shared pointers
    ///     referencing it.
    template <typename T>
    class LIBAWKWARD_EXPORT_SYMBOL no_deleter {
    public:
        /// @brief Called by `std::shared_ptr` when its reference count reaches
        /// zero.
        void operator()(T const *ptr) { }
    };

    /// @brief Returns the number of the device associated with the pointer
    const int64_t
    lib_device_num(
      kernel::lib ptr_lib,
      void* ptr);

    /// @brief Produces a <Lib/> element for 'tostring' to indicate the kernel
    /// library.
    const std::string lib_tostring(
      kernel::lib ptr_lib,
      void* ptr,
      const std::string& indent,
      const std::string& pre,
      const std::string& post);

    /// @brief Internal Function an array buffer from library `FROM` to library
    /// `TO`, usually between main memory and a GPU.
    ///
    /// @note This function has not been implemented to handle Multi-GPU setups.
    ERROR copy_to(
      kernel::lib to_lib,
      kernel::lib from_lib,
      void* to_ptr,
      void* from_ptr,
      int64_t bytelength);

    /// @brief FIXME
    const std::string
      fully_qualified_cache_key(
        kernel::lib ptr_lib,
        const std::string& cache_key);

    /// @brief Internal Function to allocate an empty array of a given length
    /// with a given type. The `bytelength` parameter is the number of bytes,
    /// so be sure to multiply by sizeof(...) when using this function.
    ///
    /// @note This function has not been implemented to handle Multi-GPU setups.
    template <typename T>
    std::shared_ptr<T> malloc(
      kernel::lib ptr_lib,
      int64_t bytelength) {
      if (ptr_lib == lib::cpu) {
        return std::shared_ptr<T>(
          reinterpret_cast<T*>(awkward_malloc(bytelength)),
          kernel::array_deleter<T>());
      }
      else if (ptr_lib == lib::cuda) {
        auto handle = acquire_handle(lib::cuda);
        typedef decltype(awkward_malloc) functor_type;
        auto* awkward_malloc_fcn = reinterpret_cast<functor_type*>(
                                     acquire_symbol(handle, "awkward_malloc"));
        return std::shared_ptr<T>(
          reinterpret_cast<T*>((*awkward_malloc_fcn)(bytelength)),
          kernel::cuda_array_deleter<T>());
      }
      else {
        throw std::runtime_error(
          std::string("unrecognized ptr_lib in ptr_alloc<bool>"));
      }
    }

    template<class T>
    using UniquePtrDeleter = decltype(kernel::array_deleter<T>());
    template<class T>
    using UniquePtr = std::unique_ptr<T, UniquePtrDeleter<T>>;

    template <typename T>
    UniquePtr<T> unique_malloc(
      kernel::lib ptr_lib,
      int64_t bytelength) {
      if (ptr_lib == lib::cpu) {
        return UniquePtr<T>(
          reinterpret_cast<T*>(awkward_malloc(bytelength)),
          kernel::array_deleter<T>());
      }
      else {
        throw std::runtime_error(
          std::string("unrecognized ptr_lib in ptr_alloc<bool>"));
      }
    }

    /////////////////////////////////// libawkward/layoutbuilder/UnionArrayBuilder.cpp

    template <typename T, typename I>
    ERROR UnionArray_regular_index(
      kernel::lib ptr_lib,
      I* toindex,
      I* current,
      int64_t size,
      const T* fromtags,
      int64_t length);
  }
}

#endif //AWKWARD_KERNEL_DISPATCH_H_
