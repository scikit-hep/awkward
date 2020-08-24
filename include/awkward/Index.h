// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEX_H_
#define AWKWARD_INDEX_H_

#include <string>
#include <map>
#include <memory>

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/kernel-dispatch.h"

namespace awkward {
  template <typename T>
  class IndexOf;

  /// @class Index
  ///
  /// @brief A contiguous, one-dimensional array of integers used to
  /// represent data structures, rather than numerical data in the arrays
  /// themselves.
  ///
  /// The Index superclass abstracts over templated specializations:
  ///
  ///    - {@link IndexOf Index8}, which is `IndexOf<int8_t>`
  ///    - {@link IndexOf IndexU8}, which is `IndexOf<uint8_t>`
  ///    - {@link IndexOf Index32}, which is `IndexOf<int32_t>`
  ///    - {@link IndexOf IndexU32}, which is `IndexOf<uint32_t>`
  ///    - {@link IndexOf Index64}, which is `IndexOf<int64_t>`
  class LIBAWKWARD_EXPORT_SYMBOL Index {
  public:
    /// @brief Integer type of an Index, used by ListForm, IndexedForm, etc.
    enum class Form {i8, u8, i32, u32, i64, kNumIndexForm};

    /// @brief Converts a string into a Form enumeration.
    static Form
      str2form(const std::string& str);

    /// @brief Converts a Form enumeration into a string.
    static const std::string
      form2str(Form form);

    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~Index();

    /// @brief Copies this Index node without copying its buffer.
    ///
    /// See also #deep_copy.
    virtual const std::shared_ptr<Index>
      shallow_copy() const = 0;

    /// @brief Converts this Index to an {@link IndexOf Index64}.
    virtual IndexOf<int64_t>
      to64() const = 0;
  };

  /// @class IndexOf
  ///
  /// @brief A contiguous, one-dimensional array of integers used to
  /// represent data structures, rather than numerical data in the arrays
  /// themselves.
  ///
  /// The Index superclass abstracts over templated specializations:
  ///
  ///    - {@link IndexOf Index8}, which is `IndexOf<int8_t>`
  ///    - {@link IndexOf IndexU8}, which is `IndexOf<uint8_t>`
  ///    - {@link IndexOf Index32}, which is `IndexOf<int32_t>`
  ///    - {@link IndexOf IndexU32}, which is `IndexOf<uint32_t>`
  ///    - {@link IndexOf Index64}, which is `IndexOf<int64_t>`
  template <typename T>
  class
#ifdef AWKWARD_INDEX_NO_EXTERN_TEMPLATE
  LIBAWKWARD_EXPORT_SYMBOL
#endif
  IndexOf: public Index {
  public:
    /// @brief Creates an IndexOf from a full set of parameters.
    ///
    /// @param ptr Reference-counted pointer to the integer array buffer.
    /// @param offset Location of item zero in the buffer, relative to
    /// `ptr`, measured in the number of elements. We keep this information in
    /// two parameters (`ptr` and `offset`) rather than moving `ptr` so that
    /// `ptr` can be reference counted among all arrays that use the same
    /// buffer.
    /// @param length Number of elements in the array.
    /// @param Choose the Kernel Library for this array, default:= kernel::lib::cpu
    IndexOf<T>(const std::shared_ptr<T>& ptr,
               int64_t offset,
               int64_t length,
               kernel::lib ptr_lib);

    /// @brief Allocates a new integer array buffer with a given #length.
    IndexOf<T>(int64_t length, kernel::lib ptr_lib = kernel::lib::cpu);

    /// @brief Reference-counted pointer to the integer array buffer.
    const std::shared_ptr<T>
      ptr() const;

    /// @brief The Kernel Library that ptr uses.
    kernel::lib
      ptr_lib() const;

    /// @brief Raw pointer to the beginning of data (i.e. offset accounted for).
    T*
      data() const;

    /// @brief Location of item zero in the buffer, relative to
    /// #ptr, measured in the number of elements.
    ///
    /// We keep this information in two parameters
    /// (#ptr and #offset) rather than moving #ptr so that #ptr can be
    /// reference counted among all arrays that use the same buffer.
    int64_t
      offset() const;

    /// @brief Number of elements in the array.
    int64_t
      length() const;

    /// @brief User-friendly name of this class: `"Index8"`, `"IndexU8"`,
    /// `"Index32"`, `"IndexU32"`, or `"Index64"`.
    const std::string
      classname() const;

    /// @brief Returns a string representation of this array (single-line XML).
    const std::string
      tostring() const;

    /// @brief Internal function to build an output string for #tostring.
    ///
    /// @param indent Indentation depth as a string of spaces.
    /// @param pre Prefix string, usually an opening XML tag.
    /// @param post Postfix string, usually a closing XML tag and carriage
    /// return.
    const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const;

    /// @brief Returns the enum describing this Index's integer specialization.
    Form
      form() const;

    /// @brief Returns the element at a given position in the array, handling
    /// negative indexing and bounds-checking like Python.
    ///
    /// The first item in the array is at `0`, the second at `1`, the last at
    /// `-1`, the penultimate at `-2`, etc.
    T
      getitem_at(int64_t at) const;

    /// @brief Returns the element at a given position in the array, without
    /// handling negative indexing or bounds-checking.
    T
      getitem_at_nowrap(int64_t at) const;

    /// @brief Assigns an integer value (type `T`) in-place.
    ///
    /// This modifies the array itself.
    void
      setitem_at_nowrap(int64_t at, T value) const;

    /// @brief Subinterval of this array, handling negative indexing
    /// and bounds-checking like Python.
    ///
    /// The first item in the array is at `0`, the second at `1`, the last at
    /// `-1`, the penultimate at `-2`, etc.
    ///
    /// Ranges beyond the array are not an error; they are trimmed to
    /// `start = 0` on the left and `stop = length() - 1` on the right.
    ///
    /// This operation only affects the node metadata; its calculation time
    /// does not scale with the size of the array.
    IndexOf<T>
      getitem_range(int64_t start, int64_t stop) const;

    /// @brief Subinterval of this array, without handling negative
    /// indexing or bounds-checking.
    ///
    /// If the array has Identities, the identity bounds are checked.
    ///
    /// This operation only affects the node metadata; its calculation time
    /// does not scale with the size of the array.
    IndexOf<T>
      getitem_range_nowrap(int64_t start, int64_t stop) const;

    /// @brief Internal function used to calculate Content#nbytes.
    ///
    /// @param largest The largest range of bytes used in each
    /// reference-counted pointer (`size_t`).
    ///
    /// @note This method of accounting for overlapping buffers is
    /// insufficient: two nodes could use parts of the same buffer in which
    /// one doesn't completely overlap the other. It's not likely, but
    /// currently the calculation of Content#nbytes is an underestimate.
    void
      nbytes_part(std::map<size_t, int64_t>& largest) const;

    const std::shared_ptr<Index>
      shallow_copy() const override;

    IndexOf<int64_t>
      to64() const override;

    /// @brief Copies this Index node and all the data in its buffer.
    ///
    /// See also #shallow_copy.
    const IndexOf<T>
      deep_copy() const;

    const IndexOf<T>
      copy_to(kernel::lib ptr_lib) const;

  private:
    /// @brief See #ptr.
    const std::shared_ptr<T> ptr_;
    /// @brief See #ptr_lib
    const kernel::lib ptr_lib_;
    /// @brief See #offset.
    const int64_t offset_;
    /// @brief See #length.
    const int64_t length_;
  };

#ifndef AWKWARD_INDEX_NO_EXTERN_TEMPLATE
  extern template class IndexOf<int8_t>;
  extern template class IndexOf<uint8_t>;
  extern template class IndexOf<int32_t>;
  extern template class IndexOf<uint32_t>;
  extern template class IndexOf<int64_t>;
#endif

  using Index8   = IndexOf<int8_t>;
  using IndexU8  = IndexOf<uint8_t>;
  using Index32  = IndexOf<int32_t>;
  using IndexU32 = IndexOf<uint32_t>;
  using Index64  = IndexOf<int64_t>;
}

#endif // AWKWARD_INDEX_H_
