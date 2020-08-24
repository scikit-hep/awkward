// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_NUMPYARRAY_H_
#define AWKWARD_NUMPYARRAY_H_

#include <string>
#include <unordered_map>
#include <memory>
#include <typeindex>
#include <vector>

#include "awkward/common.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"
#include "awkward/kernel-dispatch.h"

namespace awkward {
  /// @class NumpyForm
  ///
  /// @brief Form describing NumpyArray.
  class LIBAWKWARD_EXPORT_SYMBOL NumpyForm: public Form {
  public:
    /// @brief Creates a NumpyForm. See NumpyArray for documentation.
    NumpyForm(bool has_identities,
              const util::Parameters& parameters,
              const FormKey& form_key,
              const std::vector<int64_t>& inner_shape,
              int64_t itemsize,
              const std::string& format,
              util::dtype dtype);

    const std::vector<int64_t>
      inner_shape() const;

    int64_t
      itemsize() const;

    const std::string
      format() const;

    util::dtype
      dtype() const;

    const std::string
      primitive() const;

    const TypePtr
      type(const util::TypeStrs& typestrs) const override;

    const std::string
      tostring() const override;

    const std::string
      tojson(bool pretty, bool verbose) const override;

    void
      tojson_part(ToJson& builder, bool verbose) const override;

    void
      tojson_part(ToJson& builder, bool verbose, bool toplevel) const;

    const FormPtr
      shallow_copy() const override;

    const std::string
      purelist_parameter(const std::string& key) const override;

    bool
      purelist_isregular() const override;

    int64_t
      purelist_depth() const override;

    const std::pair<int64_t, int64_t>
      minmax_depth() const override;

    const std::pair<bool, int64_t>
      branch_depth() const override;

    int64_t
      numfields() const override;

    int64_t
      fieldindex(const std::string& key) const override;

    const std::string
      key(int64_t fieldindex) const override;

    bool
      haskey(const std::string& key) const override;

    const std::vector<std::string>
      keys() const override;

    bool
      equal(const FormPtr& other,
            bool check_identities,
            bool check_parameters,
            bool check_form_key,
            bool compatibility_check) const override;

    const FormPtr
      getitem_field(const std::string& key) const override;

  private:
    const std::vector<int64_t> inner_shape_;
    int64_t itemsize_;
    const std::string format_;
    const util::dtype dtype_;
  };

  /// @class NumpyArray
  ///
  /// @brief Represents a rectilinear numerical array that can be converted to
  /// and from NumPy without loss of information or copying the underlying
  /// buffer.
  ///
  /// See #NumpyArray for the meaning of each parameter.
  ///
  /// Any NumPy array can be passed through Awkward Array operations, even
  /// slicing, but operations that have to interpret the array's values (such
  /// as reducers like "sum" and "max") only work on numeric types:
  ///
  ///  - 32-bit and 64-bit floating point numbers
  ///  - 8-bit, 16-bit, 32-bit, and 64-bit signed and unsigned integers
  ///  - 8-bit booleans
  ///
  ///(native endian only).
  class LIBAWKWARD_EXPORT_SYMBOL NumpyArray: public Content {
  public:

    /// @brief Creates a NumpyArray from a full set of parameters.
    ///
    /// @param identities Optional Identities for each element of the array
    /// (may be `nullptr`).
    /// @param parameters String-to-JSON map that augments the meaning of this
    /// array.
    /// @param ptr Reference-counted pointer to the array buffer.
    /// @param shape Number of elements in each dimension. A one-dimensional
    /// array has a shape of length one. A zero-length shape represents a
    /// numerical scalar, which is only valid as an output (immediately
    /// converted into a number in Python). Each element in shape must be
    /// non-negative. If not scalar, the total number of elements in the array
    /// is the product of the shape (which can be zero).
    /// @param strides Length of each dimension in number of bytes. The length
    /// of strides must match the length of `shape`. Strides may be zero or
    /// negative, but they must be multiples of itemsize. An array is only
    /// "contiguous" if `strides[i] == itemsize * prod(shape[i + 1:])` for all
    /// valid `i`.
    /// @param byteoffset Location of item zero in the buffer, relative to
    /// #ptr, measured in bytes, rather than number of elements (must be a
    /// multiple of `itemsize`). We keep this information in two parameters
    /// (#ptr and #byteoffset) rather than moving #ptr so that #ptr can be
    /// reference counted among all arrays that use the same buffer.
    /// @param itemsize Number of bytes per item; should agree with the format.
    /// @param format String representing the NumPy dtype (as defined by
    /// pybind11). Note that 32-bit systems and Windows use "`q/Q`" for
    /// signed/unsigned 64-bit and "`l/L`" for 32-bit, while all other systems
    /// use "`l/L`" for 64-bit and "`i/I`" for 32-bit.
    /// @param dtype Enumeration for the type without the platform-dependence
    /// that `format` has.
    /// @param ptr_lib Indicates the kernel libraries to use for this `ptr`.
    ///
    /// Note that the integer type for all of these parameters is `ssize_t`,
    /// for consistency with pybind11. Most integers in Awkward are `int64_t`.
    NumpyArray(const IdentitiesPtr& identities,
               const util::Parameters& parameters,
               const std::shared_ptr<void>& ptr,
               const std::vector<ssize_t>& shape,
               const std::vector<ssize_t>& strides,
               ssize_t byteoffset,
               ssize_t itemsize,
               const std::string format,
               util::dtype dtype,
               const kernel::lib ptr_lib);

    /// @brief Creates a NumpyArray from an {@link IndexOf Index8}.
    NumpyArray(const Index8 index);
    /// @brief Creates a NumpyArray from an {@link IndexOf IndexU8}.
    NumpyArray(const IndexU8 index);
    /// @brief Creates a NumpyArray from an {@link IndexOf Index32}.
    NumpyArray(const Index32 index);
    /// @brief Creates a NumpyArray from an {@link IndexOf IndexU32}.
    NumpyArray(const IndexU32 index);
    /// @brief Creates a NumpyArray from an {@link IndexOf Index64}.
    NumpyArray(const Index64 index);

    /// @brief Reference-counted pointer to the array buffer.
    const std::shared_ptr<void>
      ptr() const;

    /// @param ptr_lib Indicates the kernel libraries to use for this `ptr`.
    kernel::lib
      ptr_lib() const;

    /// @brief Raw pointer to the beginning of data (i.e. offset accounted for).
    void*
      data() const;

    /// @brief Number of elements in each dimension. A one-dimensional
    /// array has a shape of length one.
    ///
    /// A zero-length shape represents a
    /// numerical scalar, which is only valid as an output (immediately
    /// converted into a number in Python). Each element in shape must be
    /// non-negative. If not scalar, the total number of elements in the array
    /// is the product of the shape (which can be zero).
    ///
    /// Note that the integer type is `ssize_t` for consistency with pybind11.
    /// Most integers in Awkward are `int64_t`.
    const std::vector<ssize_t>
      shape() const;

    /// @brief Length of each dimension in number of bytes.
    ///
    /// The length
    /// of strides must match the length of `shape`. Strides may be zero or
    /// negative, but they must be multiples of itemsize. An array is only
    /// "contiguous" if `strides[i] == itemsize * prod(shape[i + 1:])` for all
    /// valid `i`.
    ///
    /// Note that the integer type is `ssize_t` for consistency with pybind11.
    /// Most integers in Awkward are `int64_t`.
    const std::vector<ssize_t>
      strides() const;

    /// @brief Location of item zero in the buffer, relative to
    /// `ptr`, measured in bytes, rather than number of elements (must be a
    /// multiple of `itemsize`).
    ///
    /// We keep this information in two parameters
    /// (`ptr` and `byteoffset`) rather than moving `ptr` so that `ptr` can be
    /// reference counted among all arrays that use the same buffer.
    ///
    /// Note that the integer type is `ssize_t` for consistency with pybind11.
    /// Most integers in Awkward are `int64_t`.
    ssize_t
      byteoffset() const;

    /// @brief Number of bytes per item; should agree with the format.
    ///
    /// Note that the integer type is `ssize_t` for consistency with pybind11.
    /// Most integers in Awkward are `int64_t`.
    ssize_t
      itemsize() const;

    /// @brief String representing the NumPy dtype (as defined by
    /// pybind11).
    ///
    /// Note that 32-bit systems and Windows use "`q/Q`" for
    /// signed/unsigned 64-bit and "`l/L`" for 32-bit, while all other systems
    /// use "`l/L`" for 64-bit and "`i/I`" for 32-bit.
    const std::string
      format() const;

    /// @param dtype Enumeration for the type without the platform-dependence
    /// that `format` has.
    util::dtype
      dtype() const;

    /// @brief The number of dimensions, which is `shape.size()`.
    ///
    /// Note that the integer type is `ssize_t` for consistency with pybind11.
    /// Most integers in Awkward are `int64_t`.
    ssize_t
      ndim() const;

    /// @brief Returns `true` if any element of the #shape is zero; `false`
    /// otherwise.
    ///
    /// Note that a NumpyArray can be empty with non-zero #length.
    bool
      isempty() const;

    /// @brief The length of the array (or scalar, if `shape.empty()`) in
    /// bytes.
    ///
    /// If scalar, the return value is equal to #itemsize; otherwise, it is
    /// equal to `shape[0] * strides[0]`.
    ssize_t
      bytelength() const;

    /// @brief Dereferences a selected item as a `uint8_t`.
    ///
    /// Note that the integer type is `ssize_t` for consistency with pybind11.
    /// Most integers in Awkward are `int64_t`.
    uint8_t
      getbyte(ssize_t at) const;

    /// @brief A contiguous version of this array with multidimensional
    /// #shape replaced by nested RegularArray nodes.
    ///
    /// If the #shape has zero dimensions (it's a scalar), it is passed through
    /// unaffected.
    const ContentPtr
      toRegularArray() const;

    /// @brief Returns `true` if the #shape is zero-dimensional; `false` otherwise.
    bool
      isscalar() const override;

    /// @brief User-friendly name of this class: `"NumpyArray"`.
    const std::string
      classname() const override;

    void
      setidentities() override;

    void
      setidentities(const IdentitiesPtr& identities) override;

    const TypePtr
      type(const util::TypeStrs& typestrs) const override;

    const FormPtr
      form(bool materialize) const override;

    bool
      has_virtual_form() const override;

    bool
      has_virtual_length() const override;

    const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const override;

    void
      tojson_part(ToJson& builder, bool include_beginendlist) const override;

    void
      nbytes_part(std::map<size_t, int64_t>& largest) const override;

    int64_t
      length() const override;

    const ContentPtr
      shallow_copy() const override;

    const ContentPtr
      deep_copy(bool copyarrays,
                bool copyindexes,
                bool copyidentities) const override;

    void
      check_for_iteration() const override;

    const ContentPtr
      getitem_nothing() const override;

    const ContentPtr
      getitem_at(int64_t at) const override;

    const ContentPtr
      getitem_at_nowrap(int64_t at) const override;

    const ContentPtr
      getitem_range(int64_t start, int64_t stop) const override;

    const ContentPtr
      getitem_range_nowrap(int64_t start, int64_t stop) const override;

    const ContentPtr
      getitem_field(const std::string& key) const override;

    const ContentPtr
      getitem_fields(const std::vector<std::string>& keys) const override;

    const ContentPtr
      getitem(const Slice& where) const override;

    const ContentPtr
      getitem_next(const SliceItemPtr& head,
                   const Slice& tail,
                   const Index64& advanced) const override;

    const ContentPtr
      carry(const Index64& carry, bool allow_lazy) const override;

    int64_t
      numfields() const override;

    int64_t
      fieldindex(const std::string& key) const override;

    const std::string
      key(int64_t fieldindex) const override;

    bool
      haskey(const std::string& key) const override;

    const std::vector<std::string>
      keys() const override;

    // operations
    const std::string
      validityerror(const std::string& path) const override;

    /// @copydoc Content::shallow_simplify()
    ///
    /// For NumpyArray, this method returns #shallow_copy (pass-through).
    const ContentPtr
      shallow_simplify() const override;

    const ContentPtr
      num(int64_t axis, int64_t depth) const override;

    const std::pair<Index64, ContentPtr>
      offsets_and_flattened(int64_t axis, int64_t depth) const override;

    bool
      mergeable(const ContentPtr& other, bool mergebool) const override;

    const ContentPtr
      merge(const ContentPtr& other) const override;

    const SliceItemPtr
      asslice() const override;

    const ContentPtr
      fillna(const ContentPtr& value) const override;

    const ContentPtr
      rpad(int64_t target, int64_t axis, int64_t depth) const override;

    const ContentPtr
      rpad_and_clip(int64_t target,
                    int64_t axis,
                    int64_t depth) const override;

    const ContentPtr
      reduce_next(const Reducer& reducer,
                  int64_t negaxis,
                  const Index64& starts,
                  const Index64& shifts,
                  const Index64& parents,
                  int64_t outlength,
                  bool mask,
                  bool keepdims) const override;

    const ContentPtr
      sort_next(int64_t negaxis,
                const Index64& starts,
                const Index64& parents,
                int64_t outlength,
                bool ascending,
                bool stable,
                bool keepdims) const override;

    const ContentPtr
      sort_asstrings(const Index64& offsets,
                     bool ascending,
                     bool stable) const;

    const ContentPtr
      argsort_next(int64_t negaxis,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength,
                   bool ascending,
                   bool stable,
                   bool keepdims) const override;

    const ContentPtr
      localindex(int64_t axis, int64_t depth) const override;

    const ContentPtr
      combinations(int64_t n,
                   bool replacement,
                   const util::RecordLookupPtr& recordlookup,
                   const util::Parameters& parameters,
                   int64_t axis,
                   int64_t depth) const override;

    /// @brief Returns `true` if this array is contiguous; `false` otherwise.
    ///
    /// An array is "contiguous" if
    /// `strides[i] = itemsize * prod(shape[i + 1:])`
    /// for all `i` in the #shape and #strides.
    ///
    /// This means that there are no unreachable gaps between values and
    /// each user-visible value is represented by one value in memory.
    ///
    /// It does not mean that #byteoffset is `0` or the #length covers the
    /// entire allocated buffer. There may be unreachable elements before
    /// or after the user-visible data.
    bool
      iscontiguous() const;

    /// @brief A contiguous version of this array (or this array if
    /// `iscontiguous()`).
    ///
    /// An array is "contiguous" if
    /// `strides[i] = itemsize * prod(shape[i + 1:])`
    /// for all `i` in the #shape and #strides.
    ///
    /// This means that there are no unreachable gaps between values and
    /// each user-visible value is represented by one value in memory.
    ///
    /// It does not mean that #byteoffset is `0` or the #length covers the
    /// entire allocated buffer. There may be unreachable elements before
    /// or after the user-visible data.
    const NumpyArray
      contiguous() const;

    /// @brief Inhibited general function (see 7 argument `getitem_next`
    /// specific to NumpyArray).
    const ContentPtr
      getitem_next(const SliceAt& at,
                   const Slice& tail,
                   const Index64& advanced) const override;

    /// @brief Inhibited general function (see 7 argument `getitem_next`
    /// specific to NumpyArray).
    const ContentPtr
      getitem_next(const SliceRange& range,
                   const Slice& tail,
                   const Index64& advanced) const override;

    /// @brief Inhibited general function (see 7 argument `getitem_next`
    /// specific to NumpyArray).
    const ContentPtr
      getitem_next(const SliceArray64& array,
                   const Slice& tail,
                   const Index64& advanced) const override;

    /// @brief Inhibited general function (see 7 argument `getitem_next`
    /// specific to NumpyArray).
    const ContentPtr
      getitem_next(const SliceField& field,
                   const Slice& tail,
                   const Index64& advanced) const override;

    /// @brief Inhibited general function (see 7 argument `getitem_next`
    /// specific to NumpyArray).
    const ContentPtr
      getitem_next(const SliceFields& fields,
                   const Slice& tail,
                   const Index64& advanced) const override;

    /// @brief Inhibited general function (see 7 argument `getitem_next`
    /// specific to NumpyArray).
    const ContentPtr
      getitem_next(const SliceJagged64& jagged,
                   const Slice& tail,
                   const Index64& advanced) const override;

    /// @brief An utility function to create a new instance of NumpyArray on the
    /// GPU identical to this one.
    const ContentPtr
      copy_to(kernel::lib ptr_lib) const override;

    const ContentPtr
      numbers_to_type(const std::string& name) const override;

  protected:
    /// @brief Internal function to merge two byte arrays without promoting
    /// the types to int64.
    const ContentPtr
      merge_bytes(const std::shared_ptr<NumpyArray>& other) const;

    /// @brief Internal function that propagates the derivation of a contiguous
    /// version of this array from one axis to the next.
    ///
    /// This may be thought of as a specialized application of #carry indexed
    /// by byte positions, rather than item positions. (If it had been written
    /// after #getitem_next and #carry, it probably would have used $carry.)
    const NumpyArray
      contiguous_next(const Index64& bytepos) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis by propagating strides (non-advanced indexing only).
    ///
    /// @param head First element of the Slice tuple.
    /// @param tail The rest of the elements of the Slice tuple.
    /// @param length The length of the output array (after propagating
    /// through all axis levels).
    const NumpyArray
      getitem_bystrides(const SliceItemPtr& head,
                        const Slice& tail,
                        int64_t length) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis by propagating strides (non-advanced indexing only).
    ///
    /// See generic #getitem_bystrides for details.
    const NumpyArray
      getitem_bystrides(const SliceAt& at,
                        const Slice& tail,
                        int64_t length) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis by propagating strides (non-advanced indexing only).
    ///
    /// See generic #getitem_bystrides for details.
    const NumpyArray
      getitem_bystrides(const SliceRange& range,
                        const Slice& tail,
                        int64_t length) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis by propagating strides (non-advanced indexing only).
    ///
    /// See generic #getitem_bystrides for details.
    const NumpyArray
      getitem_bystrides(const SliceEllipsis& ellipsis,
                        const Slice& tail,
                        int64_t length) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis by propagating strides (non-advanced indexing only).
    ///
    /// See generic #getitem_bystrides for details.
    const NumpyArray
      getitem_bystrides(const SliceNewAxis& newaxis,
                        const Slice& tail,
                        int64_t length) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis by building up a `carry` {@link IndexOf Index}.
    ///
    /// @param head First element of the Slice tuple.
    /// @param tail The rest of the elements of the Slice tuple.
    /// @param carry The filter/rearrangement/duplication {@link IndexOf Index}
    /// that will be applied to the last axis.
    /// @param advanced If empty, no array slices (integer or boolean) have
    /// been encountered yet; otherwise, positions in any subsequent array
    /// slices to select.
    /// @param length The length of the output array (after propagating
    /// through all axis levels).
    /// @param stride The size of the data to carry at each item of the
    /// carry {@link IndexOf Index}.
    /// @param first If `true`, this axis is the first axis to be considered.
    ///
    /// In the [NumPy documentation](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer-array-indexing),
    /// advanced indexes are described as iterating "as one," which requires
    /// an {@link IndexOf Index} to be propagated when implemented recursively.
    ///
    /// Identities are only sliced at the `first` axis.
    const NumpyArray
      getitem_next(const SliceItemPtr& head,
                   const Slice& tail,
                   const Index64& carry,
                   const Index64& advanced,
                   int64_t length,
                   int64_t stride,
                   bool first) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis by building up a `carry` {@link IndexOf Index}.
    ///
    /// See generic #getitem_bystrides for details.
    const NumpyArray
      getitem_next(const SliceAt& at,
                   const Slice& tail,
                   const Index64& carry,
                   const Index64& advanced,
                   int64_t length,
                   int64_t stride,
                   bool first) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis by building up a `carry` {@link IndexOf Index}.
    ///
    /// See generic #getitem_bystrides for details.
    const NumpyArray
      getitem_next(const SliceRange& range,
                   const Slice& tail,
                   const Index64& carry,
                   const Index64& advanced,
                   int64_t length,
                   int64_t stride,
                   bool first) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis by building up a `carry` {@link IndexOf Index}.
    ///
    /// See generic #getitem_bystrides for details.
    const NumpyArray
      getitem_next(const SliceEllipsis& ellipsis,
                   const Slice& tail,
                   const Index64& carry,
                   const Index64& advanced,
                   int64_t length,
                   int64_t stride,
                   bool first) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis by building up a `carry` {@link IndexOf Index}.
    ///
    /// See generic #getitem_bystrides for details.
    const NumpyArray
      getitem_next(const SliceNewAxis& newaxis,
                   const Slice& tail,
                   const Index64& carry,
                   const Index64& advanced,
                   int64_t length,
                   int64_t stride,
                   bool first) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis by building up a `carry` {@link IndexOf Index}.
    ///
    /// See generic #getitem_bystrides for details.
    const NumpyArray
      getitem_next(const SliceArray64& array,
                   const Slice& tail,
                   const Index64& carry,
                   const Index64& advanced,
                   int64_t length,
                   int64_t stride,
                   bool first) const;

    const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceArray64& slicecontent,
                          const Slice& tail) const override;

    const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceMissing64& slicecontent,
                          const Slice& tail) const override;

    const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceJagged64& slicecontent,
                          const Slice& tail) const override;

  /// @brief Internal function to fill JSON with boolean values.
  void
    tojson_boolean(ToJson& builder, bool include_beginendlist) const;

  /// @brief Internal function to fill JSON with integer values.
  template <typename T>
  void
    tojson_integer(ToJson& builder, bool include_beginendlist) const;

  /// @brief Internal function to fill JSON with floating-point values.
  template <typename T>
  void
    tojson_real(ToJson& builder, bool include_beginendlist) const;

  /// @brief Internal function to fill JSON with string values.
  void
    tojson_string(ToJson& builder, bool include_beginendlist) const;

  private:

  /// @brief std::sort uses intro-sort
  ///        std::stable_sort uses mergesort
    template<typename T>
    const std::shared_ptr<void> index_sort(const T* data,
                                           int64_t length,
                                           const Index64& starts,
                                           const Index64& parents,
                                           int64_t outlength,
                                           bool ascending,
                                           bool stable) const;

    template<typename T>
    const std::shared_ptr<void> array_sort(const T* data,
                                           int64_t length,
                                           const Index64& starts,
                                           const Index64& parents,
                                           int64_t outlength,
                                           bool ascending,
                                           bool stable) const;

   template<typename T>
   const std::shared_ptr<void> string_sort(const T* data,
                                           int64_t length,
                                           const Index64& offsets,
                                           Index64& outoffsets,
                                           bool ascending,
                                           bool stable) const;

  template<typename T>
  const std::shared_ptr<void> as_type(const T* data,
                                      int64_t length,
                                      const util::dtype dtype) const;

  template<typename TO, typename FROM>
  const std::shared_ptr<void> cast_to_type(const FROM* data,
                                           int64_t length) const;

  /// @brief See #ptr.
  std::shared_ptr<void> ptr_;
  /// @brief See #ptr_lib
  const kernel::lib ptr_lib_;
  /// @brief See #shape.
  std::vector<ssize_t> shape_;
  /// @brief See #strides.
  std::vector<ssize_t> strides_;
  /// @brief See #byteoffset.
  ssize_t byteoffset_;
  /// @brief See #itemsize.
  const ssize_t itemsize_;
  /// @brief See #format.
  const std::string format_;
  /// @brief See #dtype.
  const util::dtype dtype_;

  };
}

#endif // AWKWARD_NUMPYARRAY_H_
