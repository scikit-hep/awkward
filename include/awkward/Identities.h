// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IDENTITIES_H_
#define AWKWARD_IDENTITIES_H_

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "awkward/common.h"
#include "awkward/Index.h"

namespace awkward {
  class Identities;
  using IdentitiesPtr = std::shared_ptr<Identities>;

  /// @class Identities
  ///
  /// @brief A contiguous, two-dimensional array of integers and a list
  /// of strings used to represent a path from the root of an array structure
  /// to each item in an array.
  ///
  /// A single value's identity is equivalent to a tuple of integers and
  /// strings:
  ///
  /// @code{.py}
  /// (999, "muons", 1, "pt")
  /// @endcode
  ///
  /// which corresponds to the "getitem" path from the root of the array
  /// to that value:
  ///
  /// @code{.py}
  /// array[999, "muons", 1, "pt"]
  /// @endcode
  ///
  /// The #fieldloc is a set of integer-string pairs, such as
  ///
  /// @code{.py}
  /// [(1, "muons"), (3, "pt")]
  /// @endcode
  ///
  /// in the above example. The identities array is two-dimensional so that
  /// a single value can be an array, such as
  ///
  /// @code{.py}
  /// [999, 1]
  /// @endcode
  ///
  /// in the above example.
  ///
  /// The Identities superclass abstracts over templated specializations:
  ///
  ///    - {@link IdentitiesOf Identities32}, which is `IdentitiesOf<int32_t>`
  ///    - {@link IdentitiesOf Identities64}, which is `IdentitiesOf<int64_t>`
  class LIBAWKWARD_EXPORT_SYMBOL Identities {
  public:
    /// @brief Identities reference type (64-bit integer).
    using Ref = int64_t;

    /// @brief Identities field location type (integer-string pairs).
    using FieldLoc = std::vector<std::pair<int64_t, std::string>>;

    /// @brief Returns a new Identities reference that is globally unique
    /// in the current process.
    ///
    /// This is thread-safe: the global counter is an atomic integer.
    static Ref
      newref();

    /// @brief A constant, empty Identities pointer (`nullptr`).
    static IdentitiesPtr
      none();

    /// @brief Called by all subclass constructors; assigns #ref,
    /// #fieldloc, #offset, #width, and #length upon construction.
    ///
    /// @param ref A globally unique reference to this set of identities.
    /// @param fieldloc A list of integer-string pairs indicating the positions
    /// of all tuple/record field indicators within the identity tuple.
    /// @param offset Location of item zero in the buffer, relative to
    /// `ptr`, measured in the number of elements. We keep this information in
    /// two parameters (`ptr` and `offset`) rather than moving `ptr` so that
    /// `ptr` can be reference counted among all arrays that use the same
    /// buffer.
    /// @param width The number of integers in each identity tuple.
    /// @param length The number of identities in the array.
    Identities(const Ref ref,
               const FieldLoc& fieldloc,
               int64_t offset,
               int64_t width,
               int64_t length);

    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~Identities();

    /// @brief A globally unique reference to this set of identities.
    const Ref
      ref() const;

    /// @brief A list of integer-string pairs indicating the positions
    /// of all tuple/record field indicators within the identity tuple.
    const FieldLoc
      fieldloc() const;

    /// @brief Location of item zero in the buffer, relative to
    /// `ptr`, measured in the number of elements. We keep this information in
    /// two parameters (`ptr` and `offset`) rather than moving `ptr` so that
    /// `ptr` can be reference counted among all arrays that use the same
    /// buffer.
    const int64_t
      offset() const;

    /// @brief The number of integers in each identity tuple.
    const int64_t
      width() const;

    /// @brief The number of identities in the array.
    const int64_t
      length() const;

    /// @brief User-friendly name of this class: `"Identities32"` or
    /// `"Identities64"`.
    virtual const std::string
      classname() const = 0;

    /// @brief Return a string representing an identity tuple at `where`.
    virtual const std::string
      identity_at(int64_t where) const = 0;

    /// @brief Converts this Identities to an {@link Identities Identities64}.
    virtual const IdentitiesPtr
      to64() const = 0;

    /// @brief Internal function to build an output string for #tostring.
    ///
    /// @param indent Indentation depth as a string of spaces.
    /// @param pre Prefix string, usually an opening XML tag.
    /// @param post Postfix string, usually a closing XML tag and carriage
    /// return.
    virtual const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const = 0;

    /// @brief Subinterval of this array, without handling negative
    /// indexing or bounds-checking.
    ///
    /// This operation only affects the node metadata; its calculation time
    /// does not scale with the size of the array.
    virtual const IdentitiesPtr
      getitem_range_nowrap(int64_t start, int64_t stop) const = 0;

    /// @brief Internal function used to calculate Content#nbytes.
    ///
    /// @param largest The largest range of bytes used in each
    /// reference-counted pointer (`size_t`).
    ///
    /// @note This method of accounting for overlapping buffers is
    /// insufficient: two nodes could use parts of the same buffer in which
    /// one doesn't completely overlap the other. It's not likely, but
    /// currently the calculation of Content#nbytes is an underestimate.
    virtual void
      nbytes_part(std::map<size_t, int64_t>& largest) const = 0;

    /// @brief Copies this Identities node without copying its buffer.
    ///
    /// See also #deep_copy.
    virtual const IdentitiesPtr
      shallow_copy() const = 0;

    /// @brief Copies this Identities node and all the data in its buffer.
    ///
    /// See also #shallow_copy.
    virtual const IdentitiesPtr
      deep_copy() const = 0;

    /// @brief Returns an Identities array with elements filtered,
    /// rearranged, and possibly duplicated by the `carry` array of integers.
    ///
    /// The output has the same length as the `carry` index, not the `array`
    /// that is being manipulated. For each item `i` in `carry`, the output
    /// is `array[index[i]]`.
    ///
    /// This operation is called
    /// [take](https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html)
    /// in NumPy and Arrow, although this carry is a low-level function that
    /// does not handle negative indexes and is not exposed to the Python
    /// layer. It is used by many operations to pass
    /// filters/rearrangements/duplications from one typed array node to
    /// another without knowing the latter's type.
    ///
    /// Taking {@link IdentitiesOf#getitem_at_nowrap getitem_at_nowrap} as a
    /// function from integers to the array's item type, `A: [0, len(a)) → T`,
    /// and the `carry` array's
    /// {@link IndexOf#getitem_at_nowrap Index64::getitem_at_nowrap} as a
    /// function `C: [0, len(c)) → [0, len(a))`, this method represents
    /// function composition, `A ∘ C: [0, len(c)) → T`.
    virtual const IdentitiesPtr
      getitem_carry_64(const Index64& carry) const = 0;

    /// @brief Returns this Identities node with a different #fieldloc.
    virtual const IdentitiesPtr
      withfieldloc(const FieldLoc& fieldloc) const = 0;

    /// @brief Returns the integer value of the two-dimensional array at
    /// `row` and `col`.
    virtual int64_t
      value(int64_t row, int64_t col) const = 0;

    /// @brief Moves the identity ptr buffer of the array between devices
    ///
    /// Returns a std::shared_ptr<IdentitiesOf> which is, by default, allocated
    /// on the first device(device [0])
    ///
    /// @note This function has not been implemented to handle Multi-GPU setups
    virtual const IdentitiesPtr
      copy_to(kernel::lib ptr_lib) const = 0;

    /// @brief Returns a string representation of this array (multi-line XML).
    const std::string
      tostring() const;

  protected:
    /// @brief See #ref.
    const Ref ref_;
    /// @brief See #fieldloc.
    const FieldLoc fieldloc_;
    /// @brief See #offset.
    int64_t offset_;
    /// @brief See #width.
    int64_t width_;
    /// @brief See #length.
    int64_t length_;
  };

  /// @class IdentitiesOf
  ///
  /// @brief A contiguous, two-dimensional array of integers and a list
  /// of strings used to represent a path from the root of an array structure
  /// to each item in an array.
  ///
  /// A single value's identity is equivalent to a tuple of integers and
  /// strings:
  ///
  /// @code{.py}
  /// (999, "muons", 1, "pt")
  /// @endcode
  ///
  /// which corresponds to the "getitem" path from the root of the array
  /// to that value:
  ///
  /// @code{.py}
  /// array[999, "muons", 1, "pt"]
  /// @endcode
  ///
  /// The #fieldloc is a set of integer-string pairs, such as
  ///
  /// @code{.py}
  /// [(1, "muons"), (3, "pt")]
  /// @endcode
  ///
  /// in the above example. The identities array is two-dimensional so that
  /// a single value can be an array, such as
  ///
  /// @code{.py}
  /// [999, 1]
  /// @endcode
  ///
  /// in the above example.
  ///
  /// The Identities superclass abstracts over templated specializations:
  ///
  ///    - {@link IdentitiesOf Identities32}, which is `IdentitiesOf<int32_t>`
  ///    - {@link IdentitiesOf Identities64}, which is `IdentitiesOf<int64_t>`
  template <typename T>
  class
#ifdef AWKWARD_IDENTITIES_NO_EXTERN_TEMPLATE
  LIBAWKWARD_EXPORT_SYMBOL
#endif
  IdentitiesOf: public Identities {
  public:
    /// @brief Creates an IdentitiesOf from a full set of parameters.
    ///
    /// @param ref A globally unique reference to this set of identities.
    /// @param fieldloc A list of integer-string pairs indicating the positions
    /// of all tuple/record field indicators within the identity tuple.
    /// @param offset Location of item zero in the buffer, relative to
    /// `ptr`, measured in the number of elements. We keep this information in
    /// two parameters (`ptr` and `offset`) rather than moving `ptr` so that
    /// `ptr` can be reference counted among all arrays that use the same
    /// buffer.
    /// @param width The number of integers in each identity tuple.
    /// @param length The number of identities in the array.
    /// @param ptr Reference-counted pointer to the array buffer.
    /// @param Choose the Kernel Library for this array, default:= cpu_kernels.
    IdentitiesOf<T>(const Ref ref,
                    const FieldLoc& fieldloc,
                    int64_t offset,
                    int64_t width,
                    int64_t length,
                    const std::shared_ptr<T> ptr,
                    kernel::lib ptr_lib = kernel::lib::cpu);

    /// @brief Allocates a new array buffer with a given #ref, #fieldloc,
    /// #length and #width.
    IdentitiesOf<T>(const Ref ref,
                    const FieldLoc& fieldloc,
                    int64_t width,
                    int64_t length,
                    kernel::lib ptr_lib = kernel::lib::cpu);

    /// @brief Reference-counted pointer to the array buffer.
    const std::shared_ptr<T>
      ptr() const;

    /// @brief The Kernel Library that ptr uses.
    kernel::lib
      ptr_lib() const;

    /// @brief Raw pointer to the beginning of data (i.e. offset accounted for).
    T*
      data() const;

    const std::string
      classname() const override;

    const std::string
      identity_at(int64_t at) const override;

    const IdentitiesPtr
      to64() const override;

    const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const override;

    const IdentitiesPtr
      getitem_range_nowrap(int64_t start, int64_t stop) const override;

    void
      nbytes_part(std::map<size_t, int64_t>& largest) const override;

    const IdentitiesPtr
      shallow_copy() const override;

    const IdentitiesPtr
      deep_copy() const override;

    const IdentitiesPtr
      getitem_carry_64(const Index64& carry) const override;

    const IdentitiesPtr
      withfieldloc(const FieldLoc& fieldloc) const override;

    int64_t
      value(int64_t row, int64_t col) const override;

    const IdentitiesPtr
      copy_to(kernel::lib ptr_lib) const override;

    /// @brief Returns the element at a given position in the array, handling
    /// negative indexing and bounds-checking like Python.
    ///
    /// The first item in the array is at `0`, the second at `1`, the last at
    /// `-1`, the penultimate at `-2`, etc.
    const std::vector<T>
      getitem_at(int64_t at) const;

    /// @brief Returns the element at a given position in the array, without
    /// handling negative indexing or bounds-checking.
    const std::vector<T>
      getitem_at_nowrap(int64_t at) const;

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
    const IdentitiesPtr
      getitem_range(int64_t start, int64_t stop) const;

  private:
    /// @brief See #ptr.
    const std::shared_ptr<T> ptr_;
    /// @brief See #ptr_lib.
    const kernel::lib ptr_lib_;
  };

#ifndef AWKWARD_IDENTITIES_NO_EXTERN_TEMPLATE
  extern template class IdentitiesOf<int32_t>;
  extern template class IdentitiesOf<int64_t>;
#endif

  using Identities32 = IdentitiesOf<int32_t>;
  using Identities64 = IdentitiesOf<int64_t>;
}

#endif // AWKWARD_IDENTITIES_H_
