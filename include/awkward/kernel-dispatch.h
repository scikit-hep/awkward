// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNEL_DISPATCH_H_
#define AWKWARD_KERNEL_DISPATCH_H_

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/kernels/allocators.h"
#include "awkward/kernels/getitem.h"
#include "awkward/kernels/sorting.h"

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

    /////////////////////////////////// awkward/kernels/getitem.h

    /// @brief Internal utility kernel to avoid raw pointer access.
    template <typename T>
    T NumpyArray_getitem_at0(
      kernel::lib ptr_lib,
      T* ptr);

    // FIXME: move regularize_rangeslice to common.h; it's not a kernel.
    void regularize_rangeslice(
      int64_t* start,
      int64_t* stop,
      bool posstep,
      bool hasstart,
      bool hasstop,
      int64_t length);

    ERROR regularize_arrayslice_64(
      kernel::lib ptr_lib,
      int64_t* flatheadptr,
      int64_t lenflathead,
      int64_t length);

    template <typename T>
    ERROR Index_to_Index64(
      kernel::lib ptr_lib,
      int64_t* toptr,
      const T* fromptr,
      int64_t length);

    template <typename T>
    ERROR Index_carry_64(
      kernel::lib ptr_lib,
      T* toindex,
      const T* fromindex,
      const int64_t* carry,
      int64_t lenfromindex,
      int64_t length);

    template <typename T>
    ERROR Index_carry_nocheck_64(
      kernel::lib ptr_lib,
      T* toindex,
      const T* fromindex,
      const int64_t* carry,
      int64_t length);

    ERROR slicearray_ravel_64(
      kernel::lib ptr_lib,
      int64_t* toptr,
      const int64_t* fromptr,
      int64_t ndim,
      const int64_t* shape,
      const int64_t* strides);

    ERROR slicemissing_check_same(
      kernel::lib ptr_lib,
      bool* same,
      const int8_t* bytemask,
      const int64_t* missingindex,
      int64_t length);

    template <typename T>
    ERROR carry_arange(
      kernel::lib ptr_lib,
      T* toptr,
      int64_t length);

    template <typename ID>
    ERROR Identities_getitem_carry_64(
      kernel::lib ptr_lib,
      ID* newidentitiesptr,
      const ID* identitiesptr,
      const int64_t* carryptr,
      int64_t lencarry,
      int64_t width,
      int64_t length);

    ERROR NumpyArray_contiguous_init_64(
      kernel::lib ptr_lib,
      int64_t* toptr,
      int64_t skip,
      int64_t stride);

    ERROR NumpyArray_contiguous_copy_64(
      kernel::lib ptr_lib,
      uint8_t* toptr,
      const uint8_t* fromptr,
      int64_t len,
      int64_t stride,
      const int64_t* pos);

    ERROR NumpyArray_contiguous_next_64(
      kernel::lib ptr_lib,
      int64_t* topos,
      const int64_t* frompos,
      int64_t len,
      int64_t skip,
      int64_t stride);

    ERROR NumpyArray_getitem_next_null_64(
      kernel::lib ptr_lib,
      uint8_t* toptr,
      const uint8_t* fromptr,
      int64_t len,
      int64_t stride,
      const int64_t* pos);

    ERROR NumpyArray_getitem_next_at_64(
      kernel::lib ptr_lib,
      int64_t* nextcarryptr,
      const int64_t* carryptr,
      int64_t lencarry,
      int64_t skip,
      int64_t at);

    ERROR NumpyArray_getitem_next_range_64(
      kernel::lib ptr_lib,
      int64_t* nextcarryptr,
      const int64_t* carryptr,
      int64_t lencarry,
      int64_t lenhead,
      int64_t skip,
      int64_t start,
      int64_t step);

    ERROR NumpyArray_getitem_next_range_advanced_64(
      kernel::lib ptr_lib,
      int64_t* nextcarryptr,
      int64_t* nextadvancedptr,
      const int64_t* carryptr,
      const int64_t* advancedptr,
      int64_t lencarry,
      int64_t lenhead,
      int64_t skip,
      int64_t start,
      int64_t step);

    ERROR NumpyArray_getitem_next_array_64(
      kernel::lib ptr_lib,
      int64_t* nextcarryptr,
      int64_t* nextadvancedptr,
      const int64_t* carryptr,
      const int64_t* flatheadptr,
      int64_t lencarry,
      int64_t lenflathead,
      int64_t skip);

    ERROR NumpyArray_getitem_next_array_advanced_64(
      kernel::lib ptr_lib,
      int64_t* nextcarryptr,
      const int64_t* carryptr,
      const int64_t* advancedptr,
      const int64_t* flatheadptr,
      int64_t lencarry,
      int64_t skip);

    ERROR NumpyArray_getitem_boolean_numtrue(
      kernel::lib ptr_lib,
      int64_t* numtrue,
      const int8_t* fromptr,
      int64_t length,
      int64_t stride);

    ERROR NumpyArray_getitem_boolean_nonzero_64(
      kernel::lib ptr_lib,
      int64_t* toptr,
      const int8_t* fromptr,
      int64_t length,
      int64_t stride);

    template <typename T>
    ERROR ListArray_getitem_next_at_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      const T* fromstarts,
      const T* fromstops,
      int64_t lenstarts,
      int64_t at);


    template <typename T>
    ERROR ListArray_getitem_next_range_carrylength(
      kernel::lib ptr_lib,
      int64_t* carrylength,
      const T* fromstarts,
      const T* fromstops,
      int64_t lenstarts,
      int64_t start,
      int64_t stop,
      int64_t step);

    template <typename T>
    ERROR ListArray_getitem_next_range_64(
      kernel::lib ptr_lib,
      T* tooffsets,
      int64_t* tocarry,
      const T* fromstarts,
      const T* fromstops,
      int64_t lenstarts,
      int64_t start,
      int64_t stop,
      int64_t step);


    template <typename T>
    ERROR ListArray_getitem_next_range_counts_64(
      kernel::lib ptr_lib,
      int64_t* total,
      const T* fromoffsets,
      int64_t lenstarts);


    template <typename T>
    ERROR ListArray_getitem_next_range_spreadadvanced_64(
      kernel::lib ptr_lib,
      int64_t* toadvanced,
      const int64_t* fromadvanced,
      const T* fromoffsets,
      int64_t lenstarts);


    template <typename T>
    ERROR ListArray_getitem_next_array_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      int64_t* toadvanced,
      const T* fromstarts,
      const T* fromstops,
      const int64_t* fromarray,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent);


    template <typename T>
    ERROR ListArray_getitem_next_array_advanced_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      int64_t* toadvanced,
      const T* fromstarts,
      const T* fromstops,
      const int64_t* fromarray,
      const int64_t* fromadvanced,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent);


    template <typename T>
    ERROR ListArray_getitem_carry_64(
      kernel::lib ptr_lib,
      T* tostarts,
      T* tostops,
      const T* fromstarts,
      const T* fromstops,
      const int64_t* fromcarry,
      int64_t lenstarts,
      int64_t lencarry);

    ERROR RegularArray_getitem_next_at_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      int64_t at,
      int64_t len,
      int64_t size);

    ERROR RegularArray_getitem_next_range_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      int64_t regular_start,
      int64_t step,
      int64_t len,
      int64_t size,
      int64_t nextsize);

    ERROR RegularArray_getitem_next_range_spreadadvanced_64(
      kernel::lib ptr_lib,
      int64_t* toadvanced,
      const int64_t* fromadvanced,
      int64_t len,
      int64_t nextsize);

    ERROR RegularArray_getitem_next_array_regularize_64(
      kernel::lib ptr_lib,
      int64_t* toarray,
      const int64_t* fromarray,
      int64_t lenarray,
      int64_t size);

    ERROR RegularArray_getitem_next_array_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      int64_t* toadvanced,
      const int64_t* fromarray,
      int64_t len,
      int64_t lenarray,
      int64_t size);

    ERROR RegularArray_getitem_next_array_advanced_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      int64_t* toadvanced,
      const int64_t* fromadvanced,
      const int64_t* fromarray,
      int64_t len,
      int64_t lenarray,
      int64_t size);

    ERROR RegularArray_getitem_carry_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      const int64_t* fromcarry,
      int64_t lencarry,
      int64_t size);

    template <typename T>
    ERROR IndexedArray_numnull(
      kernel::lib ptr_lib,
      int64_t* numnull,
      const T* fromindex,
      int64_t lenindex);

    template <typename T>
    ERROR IndexedArray_getitem_nextcarry_outindex_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      T* toindex,
      const T* fromindex,
      int64_t lenindex,
      int64_t lencontent);

    template <typename T>
    ERROR IndexedArray_getitem_nextcarry_outindex_mask_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      int64_t* toindex,
      const T* fromindex,
      int64_t lenindex,
      int64_t lencontent);

    ERROR ListOffsetArray_getitem_adjust_offsets_64(
      kernel::lib ptr_lib,
      int64_t* tooffsets,
      int64_t* tononzero,
      const int64_t* fromoffsets,
      int64_t length,
      const int64_t* nonzero,
      int64_t nonzerolength);

    ERROR ListOffsetArray_getitem_adjust_offsets_index_64(
      kernel::lib ptr_lib,
      int64_t* tooffsets,
      int64_t* tononzero,
      const int64_t* fromoffsets,
      int64_t length,
      const int64_t* index,
      int64_t indexlength,
      const int64_t* nonzero,
      int64_t nonzerolength,
      const int8_t* originalmask,
      int64_t masklength);

    ERROR IndexedArray_getitem_adjust_outindex_64(
      kernel::lib ptr_lib,
      int8_t* tomask,
      int64_t* toindex,
      int64_t* tononzero,
      const int64_t* fromindex,
      int64_t fromindexlength,
      const int64_t* nonzero,
      int64_t nonzerolength);

    template <typename T>
    ERROR IndexedArray_getitem_nextcarry_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      const T* fromindex,
      int64_t lenindex,
      int64_t lencontent);


    template <typename T>
    ERROR IndexedArray_getitem_carry_64(
      kernel::lib ptr_lib,
      T* toindex,
      const T* fromindex,
      const int64_t* fromcarry,
      int64_t lenindex,
      int64_t lencarry);

    template <typename T>
    ERROR UnionArray_regular_index_getsize(
      kernel::lib ptr_lib,
      int64_t* size,
      const T* fromtags,
      int64_t length);

    template <typename T, typename I>
    ERROR UnionArray_regular_index(
      kernel::lib ptr_lib,
      I* toindex,
      I* current,
      int64_t size,
      const T* fromtags,
      int64_t length);

    template <typename T, typename I>
    ERROR UnionArray_project_64(
      kernel::lib ptr_lib,
      int64_t* lenout,
      int64_t* tocarry,
      const T* fromtags,
      const I* fromindex,
      int64_t length,
      int64_t which);

    ERROR missing_repeat_64(
      kernel::lib ptr_lib,
      int64_t* outindex,
      const int64_t* index,
      int64_t indexlength,
      int64_t repetitions,
      int64_t regularsize);

    ERROR RegularArray_getitem_jagged_expand_64(
      kernel::lib ptr_lib,
      int64_t* multistarts,
      int64_t * multistops,
      const int64_t* singleoffsets,
      int64_t regularsize,
      int64_t regularlength);

    template <typename T>
    ERROR ListArray_getitem_jagged_expand_64(
      kernel::lib ptr_lib,
      int64_t* multistarts,
      int64_t* multistops,
      const int64_t* singleoffsets,
      int64_t* tocarry,
      const T* fromstarts,
      const T* fromstops,
      int64_t jaggedsize,
      int64_t length);

    ERROR ListArray_getitem_jagged_carrylen_64(
      kernel::lib ptr_lib,
      int64_t* carrylen,
      const int64_t* slicestarts,
      const int64_t* slicestops,
      int64_t sliceouterlen);

    template <typename T>
    ERROR ListArray_getitem_jagged_apply_64(
      kernel::lib ptr_lib,
      int64_t* tooffsets,
      int64_t* tocarry,
      const int64_t* slicestarts,
      const int64_t* slicestops,
      int64_t sliceouterlen,
      const int64_t* sliceindex,
      int64_t sliceinnerlen,
      const T* fromstarts,
      const T* fromstops,
      int64_t contentlen);

    ERROR ListArray_getitem_jagged_numvalid_64(
      kernel::lib ptr_lib,
      int64_t* numvalid,
      const int64_t* slicestarts,
      const int64_t* slicestops,
      int64_t length,
      const int64_t* missing,
      int64_t missinglength);

    ERROR ListArray_getitem_jagged_shrink_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      int64_t* tosmalloffsets,
      int64_t* tolargeoffsets,
      const int64_t* slicestarts,
      const int64_t* slicestops,
      int64_t length,
      const int64_t* missing);

    template <typename T>
    ERROR ListArray_getitem_jagged_descend_64(
      kernel::lib ptr_lib,
      int64_t* tooffsets,
      const int64_t* slicestarts,
      const int64_t* slicestops,
      int64_t sliceouterlen,
      const T* fromstarts,
      const T* fromstops);

      template <typename T>
      T index_getitem_at_nowrap(
        kernel::lib ptr_lib,
        T* ptr,
        int64_t at);

      template <typename T>
      void index_setitem_at_nowrap(
        kernel::lib ptr_lib,
        T* ptr,
        int64_t at,
        T value);

    ERROR ByteMaskedArray_getitem_carry_64(
      kernel::lib ptr_lib,
      int8_t* tomask,
      const int8_t* frommask,
      int64_t lenmask,
      const int64_t* fromcarry,
      int64_t lencarry);

    ERROR ByteMaskedArray_numnull(
      kernel::lib ptr_lib,
      int64_t* numnull,
      const int8_t* mask,
      int64_t length,
      bool validwhen);

    ERROR ByteMaskedArray_getitem_nextcarry_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      const int8_t* mask,
      int64_t length,
      bool validwhen);

    ERROR ByteMaskedArray_getitem_nextcarry_outindex_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      int64_t* toindex,
      const int8_t* mask,
      int64_t length,
      bool validwhen);

    ERROR ByteMaskedArray_toIndexedOptionArray64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      const int8_t* mask,
      int64_t length,
      bool validwhen);

    ERROR Content_getitem_next_missing_jagged_getmaskstartstop(
      kernel::lib ptr_lib,
      int64_t* index_in,
      int64_t* offsets_in,
      int64_t* mask_out,
      int64_t* starts_out,
      int64_t* stops_out,
      int64_t length);

    template <typename T>
    ERROR MaskedArray_getitem_next_jagged_project(
      kernel::lib ptr_lib,
      T* index,
      int64_t* starts_in,
      int64_t* stops_in,
      int64_t* starts_out,
      int64_t* stops_out,
      int64_t length);

    /////////////////////////////////// awkward/kernels/identities.h

    template <typename T>
    ERROR new_Identities(
      kernel::lib ptr_lib,
      T* toptr,
      int64_t length);

    template <typename T>
    ERROR Identities_to_Identities64(
      kernel::lib ptr_lib,
      int64_t* toptr,
      const T* fromptr,
      int64_t length,
      int64_t width);

    template <typename C, typename T>
    ERROR Identities_from_ListOffsetArray(
      kernel::lib ptr_lib,
      C* toptr,
      const C* fromptr,
      const T* fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);

    template <typename C, typename T>
    ERROR Identities_from_ListArray(
      kernel::lib ptr_lib,
      bool* uniquecontents,
      C* toptr,
      const C* fromptr,
      const T* fromstarts,
      const T* fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);

    template <typename ID>
    ERROR Identities_from_RegularArray(
      kernel::lib ptr_lib,
      ID* toptr,
      const ID* fromptr,
      int64_t size,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);


    template <typename C, typename T>
    ERROR Identities_from_IndexedArray(
      kernel::lib ptr_lib,
      bool* uniquecontents,
      C* toptr,
      const C* fromptr,
      const T* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);


    template <typename C, typename T, typename I>
    ERROR Identities_from_UnionArray(
      kernel::lib ptr_lib,
      bool* uniquecontents,
      C* toptr,
      const C* fromptr,
      const T* fromtags,
      const I* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which);

    template <typename ID>
    ERROR Identities_extend(
      kernel::lib ptr_lib,
      ID* toptr,
      const ID* fromptr,
      int64_t fromlength,
      int64_t tolength);

    /////////////////////////////////// awkward/kernels/operations.h

    template <typename T>
    ERROR ListArray_num_64(
      kernel::lib ptr_lib,
      int64_t* tonum,
      const T* fromstarts,
      const T* fromstops,
      int64_t length);

    ERROR RegularArray_num_64(
      kernel::lib ptr_lib,
      int64_t* tonum,
      int64_t size,
      int64_t length);

    template <typename T>
    ERROR ListOffsetArray_flatten_offsets_64(
      kernel::lib ptr_lib,
      int64_t* tooffsets,
      const T* outeroffsets,
      int64_t outeroffsetslen,
      const int64_t* inneroffsets,
      int64_t inneroffsetslen);


    template <typename T>
    ERROR IndexedArray_flatten_none2empty_64(
      kernel::lib ptr_lib,
      int64_t* outoffsets,
      const T* outindex,
      int64_t outindexlength,
      const int64_t* offsets,
      int64_t offsetslength);


    template <typename T, typename I>
    ERROR UnionArray_flatten_length_64(
      kernel::lib ptr_lib,
      int64_t* total_length,
      const T* fromtags,
      const I* fromindex,
      int64_t length,
      int64_t** offsetsraws);


    template <typename T, typename I>
    ERROR UnionArray_flatten_combine_64(
      kernel::lib ptr_lib,
      int8_t* totags,
      int64_t* toindex,
      int64_t* tooffsets,
      const T* fromtags,
      const I* fromindex,
      int64_t length,
      int64_t** offsetsraws);


    template <typename T>
    ERROR IndexedArray_flatten_nextcarry_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      const T* fromindex,
      int64_t lenindex,
      int64_t lencontent);

    template <typename T>
    ERROR IndexedArray_overlay_mask8_to64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      const int8_t* mask,
      const T* fromindex,
      int64_t length);


    template <typename T>
    ERROR IndexedArray_mask8(
      kernel::lib ptr_lib,
      int8_t* tomask,
      const T* fromindex,
      int64_t length);

    ERROR ByteMaskedArray_mask8(
      kernel::lib ptr_lib,
      int8_t* tomask,
      const int8_t* frommask,
      int64_t length,
      bool validwhen);

    ERROR zero_mask8(
      kernel::lib ptr_lib,
      int8_t* tomask,
      int64_t length);

    template <typename T>
    ERROR IndexedArray_simplify32_to64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      const T* outerindex,
      int64_t outerlength,
      const int32_t* innerindex,
      int64_t innerlength);


    template <typename T>
    ERROR IndexedArray_simplifyU32_to64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      const T* outerindex,
      int64_t outerlength,
      const uint32_t* innerindex,
      int64_t innerlength);


    template <typename T>
    ERROR IndexedArray_simplify64_to64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      const T* outerindex,
      int64_t outerlength,
      const int64_t* innerindex,
      int64_t innerlength);

    template <typename T>
    ERROR ListArray_compact_offsets_64(
      kernel::lib ptr_lib,
      int64_t* tooffsets,
      const T* fromstarts,
      const T* fromstops,
      int64_t length);

    ERROR RegularArray_compact_offsets_64(
      kernel::lib ptr_lib,
      int64_t* tooffsets,
      int64_t length,
      int64_t size);

    template <typename T>
    ERROR ListOffsetArray_compact_offsets_64(
      kernel::lib ptr_lib,
      int64_t* tooffsets,
      const T* fromoffsets,
      int64_t length);


    template <typename T>
    ERROR ListArray_broadcast_tooffsets_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      const int64_t* fromoffsets,
      int64_t offsetslength,
      const T* fromstarts,
      const T* fromstops,
      int64_t lencontent);

    ERROR RegularArray_broadcast_tooffsets_64(
      kernel::lib ptr_lib,
      const int64_t* fromoffsets,
      int64_t offsetslength,
      int64_t size);

    ERROR RegularArray_broadcast_tooffsets_size1_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      const int64_t* fromoffsets,
      int64_t offsetslength);

    template <typename T>
    ERROR ListOffsetArray_toRegularArray(
      kernel::lib ptr_lib,
      int64_t* size,
      const T* fromoffsets,
      int64_t offsetslength);

    template <typename FROM, typename TO>
    ERROR NumpyArray_fill(
      kernel::lib ptr_lib,
      TO* toptr,
      int64_t tooffset,
      const FROM* fromptr,
      int64_t length);

    template <typename TO>
    ERROR NumpyArray_fill_frombool(
      kernel::lib ptr_lib,
      TO* toptr,
      int64_t tooffset,
      const bool* fromptr,
      int64_t length);

    template <typename FROM, typename TO>
    ERROR ListArray_fill(
      kernel::lib ptr_lib,
      TO* tostarts,
      int64_t tostartsoffset,
      TO* tostops,
      int64_t tostopsoffset,
      const FROM* fromstarts,
      const FROM* fromstops,
      int64_t length,
      int64_t base);

    template <typename FROM, typename TO>
    ERROR IndexedArray_fill(
      kernel::lib ptr_lib,
      TO* toindex,
      int64_t toindexoffset,
      const FROM* fromindex,
      int64_t length,
      int64_t base);

    ERROR IndexedArray_fill_to64_count(
      kernel::lib ptr_lib,
      int64_t* toindex,
      int64_t toindexoffset,
      int64_t length,
      int64_t base);

    ERROR UnionArray_filltags_to8_from8(
      kernel::lib ptr_lib,
      int8_t* totags,
      int64_t totagsoffset,
      const int8_t* fromtags,
      int64_t length,
      int64_t base);

    template <typename FROM, typename TO>
    ERROR UnionArray_fillindex(
      kernel::lib ptr_lib,
      TO* toindex,
      int64_t toindexoffset,
      const FROM* fromindex,
      int64_t length);

    ERROR UnionArray_filltags_to8_const(
      kernel::lib ptr_lib,
      int8_t* totags,
      int64_t totagsoffset,
      int64_t length,
      int64_t base);

    ERROR UnionArray_fillindex_count_64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      int64_t toindexoffset,
      int64_t length);

    template <typename T, typename I>
    ERROR UnionArray_simplify8_32_to8_64(
      kernel::lib ptr_lib,
      int8_t* totags,
      int64_t* toindex,
      const T* outertags,
      const I* outerindex,
      const int8_t* innertags,
      const int32_t* innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base);


    template <typename T, typename I>
    ERROR UnionArray_simplify8_U32_to8_64(
      kernel::lib ptr_lib,
      int8_t* totags,
      int64_t* toindex,
      const T* outertags,
      const I* outerindex,
      const int8_t* innertags,
      const uint32_t* innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base);


    template <typename T, typename I>
    ERROR UnionArray_simplify8_64_to8_64(
      kernel::lib ptr_lib,
      int8_t* totags,
      int64_t* toindex,
      const T* outertags,
      const I* outerindex,
      const int8_t* innertags,
      const int64_t* innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base);


    template <typename T, typename I>
    ERROR UnionArray_simplify_one_to8_64(
      kernel::lib ptr_lib,
      int8_t* totags,
      int64_t* toindex,
      const T* fromtags,
      const I* fromindex,
      int64_t towhich,
      int64_t fromwhich,
      int64_t length,
      int64_t base);

    template <typename T>
    ERROR ListArray_validity(
      kernel::lib ptr_lib,
      const T* starts,
      const T* stops,
      int64_t length,
      int64_t lencontent);


    template <typename T>
    ERROR IndexedArray_validity(
      kernel::lib ptr_lib,
      const T* index,
      int64_t length,
      int64_t lencontent,
      bool isoption);


    template <typename T, typename I>
    ERROR UnionArray_validity(
      kernel::lib ptr_lib,
      const T* tags,
      const I* index,
      int64_t length,
      int64_t numcontents,
      const int64_t* lencontents);

    template <typename T>
    ERROR UnionArray_fillna_64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      const T* fromindex,
      int64_t length);

    ERROR IndexedOptionArray_rpad_and_clip_mask_axis1_64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      const int8_t* frommask,
      int64_t length);

    ERROR index_rpad_and_clip_axis0_64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      int64_t target,
      int64_t length);

    ERROR index_rpad_and_clip_axis1_64(
      kernel::lib ptr_lib,
      int64_t* tostarts,
      int64_t* tostops,
      int64_t target,
      int64_t length);

    ERROR RegularArray_rpad_and_clip_axis1_64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      int64_t target,
      int64_t size,
      int64_t length);

    template <typename T>
    ERROR ListArray_min_range(
      kernel::lib ptr_lib,
      int64_t* tomin,
      const T* fromstarts,
      const T* fromstops,
      int64_t lenstarts);

    template <typename T>
    ERROR ListArray_rpad_and_clip_length_axis1(
      kernel::lib ptr_lib,
      int64_t* tolength,
      const T* fromstarts,
      const T* fromstops,
      int64_t target,
      int64_t lenstarts);

    template <typename T>
    ERROR ListArray_rpad_axis1_64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      const T* fromstarts,
      const T* fromstops,
      T* tostarts,
      T* tostops,
      int64_t target,
      int64_t length);

    template <typename T>
    ERROR ListOffsetArray_rpad_and_clip_axis1_64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      const T* fromoffsets,
      int64_t length,
      int64_t target);

    template <typename T>
    ERROR ListOffsetArray_rpad_length_axis1(
      kernel::lib ptr_lib,
      T* tooffsets,
      const T* fromoffsets,
      int64_t fromlength,
      int64_t target,
      int64_t* tolength);

    template <typename T>
    ERROR ListOffsetArray_rpad_axis1_64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      const T* fromoffsets,
      int64_t fromlength,
      int64_t target);

    ERROR localindex_64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      int64_t length);

    template <typename T>
    ERROR ListArray_localindex_64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      const T* offsets,
      int64_t length);

    ERROR RegularArray_localindex_64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      int64_t size,
      int64_t length);

    template <typename T>
    ERROR combinations(
      kernel::lib ptr_lib,
      T* toindex,
      int64_t n,
      bool replacement,
      int64_t singlelen);

    template <typename T>
    ERROR ListArray_combinations_length_64(
      kernel::lib ptr_lib,
      int64_t* totallen,
      int64_t* tooffsets,
      int64_t n,
      bool replacement,
      const T* starts,
      const T* stops,
      int64_t length);

    template <typename T>
    ERROR ListArray_combinations_64(
      kernel::lib ptr_lib,
      int64_t** tocarry,
      int64_t* toindex,
      int64_t* fromindex,
      int64_t n,
      bool replacement,
      const T* starts,
      const T* stops,
      int64_t length);

    ERROR RegularArray_combinations_64(
      kernel::lib ptr_lib,
      int64_t** tocarry,
      int64_t* toindex,
      int64_t* fromindex,
      int64_t n,
      bool replacement,
      int64_t size,
      int64_t length);

    ERROR ByteMaskedArray_overlay_mask8(
      kernel::lib ptr_lib,
      int8_t* tomask,
      const int8_t* theirmask,
      const int8_t* mymask,
      int64_t length,
      bool validwhen);

    ERROR BitMaskedArray_to_ByteMaskedArray(
      kernel::lib ptr_lib,
      int8_t* tobytemask,
      const uint8_t* frombitmask,
      int64_t bitmasklength,
      bool validwhen,
      bool lsb_order);

    ERROR BitMaskedArray_to_IndexedOptionArray64(
      kernel::lib ptr_lib,
      int64_t* toindex,
      const uint8_t* frombitmask,
      int64_t bitmasklength,
      bool validwhen,
      bool lsb_order);

    /////////////////////////////////// awkward/kernels/reducers.h

    ERROR reduce_count_64(
      kernel::lib ptr_lib,
      int64_t* toptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

    template <typename IN>
    ERROR reduce_countnonzero_64(
      kernel::lib ptr_lib,
      int64_t* toptr,
      const IN* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

    template <typename OUT, typename IN>
    ERROR reduce_sum_64(
      kernel::lib ptr_lib,
      OUT* toptr,
      const IN* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

    template <typename IN>
    ERROR reduce_sum_bool_64(
      kernel::lib ptr_lib,
      bool* toptr,
      const IN* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

    template <typename OUT, typename IN>
    ERROR reduce_prod_64(
      kernel::lib ptr_lib,
      OUT* toptr,
      const IN* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

    template <typename IN>
    ERROR reduce_prod_bool_64(
      kernel::lib ptr_lib,
      bool* toptr,
      const IN* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

    template <typename OUT, typename IN>
    ERROR reduce_min_64(
      kernel::lib ptr_lib,
      OUT* toptr,
      const IN* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      OUT identity);

    template <typename OUT, typename IN>
    ERROR reduce_max_64(
      kernel::lib ptr_lib,
      OUT* toptr,
      const IN* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      OUT identity);

    template <typename OUT, typename IN>
    ERROR reduce_argmin_64(
      kernel::lib ptr_lib,
      OUT* toptr,
      const IN* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

    template <typename OUT, typename IN>
    ERROR reduce_argmax_64(
      kernel::lib ptr_lib,
      OUT* toptr,
      const IN* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

    ERROR content_reduce_zeroparents_64(
      kernel::lib ptr_lib,
      int64_t* toparents,
      int64_t length);

    ERROR ListOffsetArray_reduce_global_startstop_64(
      kernel::lib ptr_lib,
      int64_t* globalstart,
      int64_t* globalstop,
      const int64_t* offsets,
      int64_t length);

    ERROR ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64(
      kernel::lib ptr_lib,
      int64_t* maxcount,
      int64_t* offsetscopy,
      const int64_t* offsets,
      int64_t length);

    ERROR ListOffsetArray_reduce_nonlocal_preparenext_64(
      kernel::lib ptr_lib,
      int64_t* nextcarry,
      int64_t* nextparents,
      int64_t nextlen,
      int64_t* maxnextparents,
      int64_t* distincts,
      int64_t distinctslen,
      int64_t* offsetscopy,
      const int64_t* offsets,
      int64_t length,
      const int64_t* parents,
      int64_t maxcount);

    ERROR ListOffsetArray_reduce_nonlocal_nextstarts_64(
      kernel::lib ptr_lib,
      int64_t* nextstarts,
      const int64_t* nextparents,
      int64_t nextlen);

    ERROR ListOffsetArray_reduce_nonlocal_findgaps_64(
      kernel::lib ptr_lib,
      int64_t* gaps,
      const int64_t* parents,
      int64_t lenparents);

    ERROR ListOffsetArray_reduce_nonlocal_outstartsstops_64(
      kernel::lib ptr_lib,
      int64_t* outstarts,
      int64_t* outstops,
      const int64_t* distincts,
      int64_t lendistincts,
      const int64_t* gaps,
      int64_t outlength);

    ERROR ListOffsetArray_reduce_nonlocal_nextshifts_64(
      kernel::lib ptr_lib,
      int64_t* nummissing,
      int64_t* missing,
      int64_t* nextshifts,
      const int64_t* offsets,
      int64_t length,
      const int64_t* starts,
      const int64_t* parents,
      int64_t maxcount,
      int64_t nextlen,
      const int64_t* nextcarry);

    ERROR ListOffsetArray_reduce_local_nextparents_64(
      kernel::lib ptr_lib,
      int64_t* nextparents,
      const int64_t* offsets,
      int64_t length);

    ERROR ListOffsetArray_reduce_local_outoffsets_64(
      kernel::lib ptr_lib,
      int64_t* outoffsets,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

    template <typename T>
    ERROR IndexedArray_reduce_next_64(
      kernel::lib ptr_lib,
      int64_t* nextcarry,
      int64_t* nextparents,
      int64_t* outindex,
      const T* index,
      int64_t* parents,
      int64_t length);

    template <typename T>
    ERROR IndexedArray_reduce_next_nonlocal_nextshifts_64(
      kernel::lib ptr_lib,
      int64_t* nextshifts,
      const T* index,
      int64_t length);

    template <typename T>
    ERROR IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64(
      kernel::lib ptr_lib,
      int64_t* nextshifts,
      const T* index,
      int64_t length,
      const int64_t* shifts);

    ERROR IndexedArray_reduce_next_fix_offsets_64(
      kernel::lib ptr_lib,
      int64_t* outoffsets,
      const int64_t* starts,
      int64_t startslength,
      int64_t outindexlength);

    ERROR NumpyArray_reduce_adjust_starts_64(
      kernel::lib ptr_lib,
      int64_t* toptr,
      int64_t outlength,
      const int64_t* parents,
      const int64_t* starts);

    ERROR NumpyArray_reduce_adjust_starts_shifts_64(
      kernel::lib ptr_lib,
      int64_t* toptr,
      int64_t outlength,
      const int64_t* parents,
      const int64_t* starts,
      const int64_t* shifts);

    ERROR NumpyArray_reduce_mask_ByteMaskedArray_64(
      kernel::lib ptr_lib,
      int8_t* toptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

    ERROR NumpyArray_reduce_mask_ByteMaskedArray_64(
      kernel::lib ptr_lib,
      int8_t* toptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

    ERROR ByteMaskedArray_reduce_next_64(
      kernel::lib ptr_lib,
      int64_t* nextcarry,
      int64_t* nextparents,
      int64_t* outindex,
      const int8_t* mask,
      const int64_t* parents,
      int64_t length,
      bool validwhen);

    ERROR ByteMaskedArray_reduce_next_nonlocal_nextshifts_64(
      kernel::lib ptr_lib,
      int64_t* nextshifts,
      const int8_t* mask,
      int64_t length,
      bool valid_when);

    ERROR ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64(
      kernel::lib ptr_lib,
      int64_t* nextshifts,
      const int8_t* mask,
      int64_t length,
      bool valid_when,
      const int64_t* shifts);

    /////////////////////////////////// awkward/kernels/sorting.h

    ERROR sorting_ranges(
      kernel::lib ptr_lib,
      int64_t* toindex,
      int64_t tolength,
      const int64_t* parents,
      int64_t parentslength);

    ERROR sorting_ranges_length(
      kernel::lib ptr_lib,
      int64_t* tolength,
      const int64_t* parents,
      int64_t parentslength);

    template <typename T>
    ERROR NumpyArray_argsort(
      kernel::lib ptr_lib,
      int64_t* toptr,
      const T* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      bool ascending,
      bool stable);

    template <typename T>
    ERROR NumpyArray_sort(
      kernel::lib ptr_lib,
      T* toptr,
      const T* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable);

    template <typename T>
    ERROR NumpyArray_sort_asstrings(
      kernel::lib ptr_lib,
      T* toptr,
      const T* fromptr,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t* outoffsets,
      bool ascending,
      bool stable);

    ERROR ListOffsetArray_local_preparenext_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      const int64_t* fromindex,
      int64_t length);

    ERROR IndexedArray_local_preparenext_64(
      kernel::lib ptr_lib,
      int64_t* tocarry,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentslength,
      const int64_t* nextparents);

  }
}

#endif //AWKWARD_KERNEL_DISPATCH_H_
