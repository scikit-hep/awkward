// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNEL_H_
#define AWKWARD_KERNEL_H_

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

  class LibraryPathCallback {
    public:
    LibraryPathCallback() {}

    // @brief This is not a pure virtual function, because that would make this
    // an abstract class and C++ 11 does let us iterate over abstract classes
    virtual const std::string library_path() const {
      return std::string("/");
    };
  };

  class LibraryCallback {
    public:
    LibraryCallback();

    void add_library_path_callback(
      kernel::Lib ptr_lib,
      const std::shared_ptr<LibraryPathCallback> &callback);

    std::string awkward_library_path(kernel::Lib ptr_lib);

    private:
    std::map<kernel::Lib, std::vector<std::shared_ptr<LibraryPathCallback>>> lib_path_callbacks;

    std::mutex lib_path_callbacks_mutex;
  };

  extern std::shared_ptr<LibraryCallback> lib_callback;

  /// @brief Internal utility function to return an opaque ptr if an handle is
  /// acquired for the specified ptr_lib. If not, then it raises an appropriate
  /// exception
  void* acquire_handle(kernel::Lib ptr_lib, std::string path);

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

  /// @class cuda_array_deleter
  ///
  /// @brief Used as a `std::shared_ptr` deleter (second argument) to
  /// overload `delete ptr` with `cudaFree`.
  ///
  /// This is necessary for `std::shared_ptr` to contain array buffers.
  ///
  /// See also
  ///   - array_deleter, which deletes array buffers on the main memory
  ///   - no_deleter, which does not free memory at all (for borrowed
  ///     references).
  ///   - pyobject_deleter, which reduces the reference count of a
  ///     Python object when there are no more C++ shared pointers
  ///     referencing it.
  template<typename T>
  class EXPORT_SYMBOL cuda_array_deleter {
    public:
    /// @brief Called by `std::shared_ptr` when its reference count reaches
    /// zero.
    void operator()(T const *p);
  };

  /// @brief A utility function to get the device number on which the
  /// array is located
  ///
  /// Returns the corresponding device number, else -1 if it's on main memory
  template<typename T>
  int
    get_ptr_device_num(kernel::Lib ptr_lib, T *ptr);

  /// @brief A utility function to get the device name on which the
  /// array is located
  ///
  /// Returns the corresponding device name, else empty string if it's on main memory
  template <typename T>
  std::string
    get_ptr_device_name(kernel::Lib ptr_lib, T* ptr);

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

  /// @brief Internal Function to transfer an array buffer contained
  /// on the main memory to the GPU memory
  ///
  /// @note This function has not been implemented to handle Multi-GPU setups
  template<typename T>
  Error
  H2D(kernel::Lib ptr_lib,
      T **to_ptr,
      T *from_ptr,
      int64_t length);

  /// @brief Internal Function to transfer an array buffer contained on the
  /// main memory to the GPU memory
  ///
  /// @note This function has not been implemented to handle Multi-GPU setups
  template<typename T>
  Error D2H(kernel::Lib ptr_lib,
            T **to_ptr,
            T *from_ptr,
            int64_t length);


  /// @brief Internal Function to allocate an empty array of a given length on
  /// the GPU
  ///
  /// @note This function has not been implemented to handle Multi-GPU setups
  template<typename T>
  std::shared_ptr<T> ptr_alloc(kernel::Lib ptr_lib,
                               int64_t length);

  /// @brief Internal utility kernel to avoid raw pointer access
  ///
  /// @note This does not have a corresponding cpu_kernel
  template <typename T>
  T numpyarray_getitem_at(kernel::Lib ptr_lib,
                          T* ptr,
                          int64_t at);
  ////////////////////////////////// awkward/getitem

  template <typename T>
  T index_getitem_at_nowrap(kernel::Lib ptr_lib,
                            T* ptr,
                            int64_t offset,
                            int64_t at);

  template <typename T>
  void index_setitem_at_nowrap(kernel::Lib ptr_lib,
                               T* ptr,
                               int64_t offset,
                               int64_t at,
                               T value);

  /////////////////////////////////// awkward/operations
  template <typename T>
  ERROR ListArray_num_64(
    kernel::Lib ptr_lib,
    int64_t* tonum,
    const T* fromstarts,
    int64_t startsoffset,
    const T* fromstops,
    int64_t stopsoffset,
    int64_t length);

}

#endif //AWKWARD_KERNEL_H_
