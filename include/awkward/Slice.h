// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_SLICE_H_
#define AWKWARD_SLICE_H_

#include <string>
#include <vector>
#include <memory>

#include "awkward/kernels/getitem.h"
#include "awkward/type/Type.h"
#include "awkward/Index.h"

namespace awkward {
  class SliceItem;
  using SliceItemPtr = std::shared_ptr<SliceItem>;

  /// @class SliceItem
  ///
  /// @brief Abstract class for slice items, which are elements of a tuple
  /// passed to an array's `__getitem__` in Python.
  class LIBAWKWARD_EXPORT_SYMBOL SliceItem {
  public:
    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~SliceItem();

    /// @brief Copies this node without copying any associated arrays.
    virtual const SliceItemPtr
      shallow_copy() const = 0;

    /// @brief Returns a string representation of this slice item (single-line
    /// custom format).
    virtual const std::string
      tostring() const = 0;

    /// @brief Returns `true` if this slice would preserve an array's slice
    /// and therefore should pass on
    /// {@link Content#parameters Content::parameters}.
    ///
    /// @param advanced The index that is passed through
    /// {@link Content#getitem_next Content::getitem_next}.
    virtual bool
      preserves_type(const Index64& advanced) const = 0;
  };

  /// @class SliceAt
  ///
  /// @brief Represents an integer in a tuple of slices passed to
  /// `__getitem__` in Python.
  class LIBAWKWARD_EXPORT_SYMBOL SliceAt: public SliceItem {
  public:
    /// @brief Creates a SliceAt from a full set of parameters.
    ///
    /// @param at The integer that this slice item represents.
    SliceAt(int64_t at);

    /// @brief The integer that this slice item represents.
    int64_t
      at() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    /// @copydoc SliceItem::preserves_type()
    ///
    /// Always `false` for SliceAt.
    bool
      preserves_type(const Index64& advanced) const override;

  private:
    /// @brief See #at.
    const int64_t at_;
  };

  /// @class SliceRange
  ///
  /// @brief Represents a Python `slice` object (usual syntax:
  /// `array[start:stop:step]`).
  class LIBAWKWARD_EXPORT_SYMBOL SliceRange: public SliceItem {
  public:
    /// @brief Creates a SliceRange from a full set of parameters.
    ///
    /// @param start The inclusive starting position.
    /// @param stop The exclusive stopping position.
    /// @param step The step size, which may be negative but must not be zero.
    ///
    /// Any #start, #stop, or #step may be
    /// {@link Slice#none Slice::none}. Appropriate values are
    /// derived from an array's {@link Content#length length} the same way
    /// they are in Python.
    SliceRange(int64_t start, int64_t stop, int64_t step);

    /// @brief The inclusive starting position.
    ///
    /// This value may be {@link Slice#none Slice::none}; if so,
    /// the value used would be derived from an array's
    /// {@link Content#length length} the same way they are in Python.
    int64_t
      start() const;

    /// @brief The exclusive stopping position.
    ///
    /// This value may be {@link Slice#none Slice::none}; if so,
    /// the value used would be derived from an array's
    /// {@link Content#length length} the same way they are in Python.
    int64_t
      stop() const;

    /// @brief The step size, which may be negative but must not be zero.
    ///
    /// This value may be {@link Slice#none Slice::none}; if so,
    /// the value used would be derived from an array's
    /// {@link Content#length length} the same way they are in Python.
    int64_t
      step() const;

    /// @brief Returns `true` if #start is not
    /// {@link Slice#none Slice::none}; `false` otherwise.
    bool
      hasstart() const;

    /// @brief Returns `true` if #stop is not
    /// {@link Slice#none Slice::none}; `false` otherwise.
    bool
      hasstop() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    /// @copydoc SliceItem::preserves_type()
    ///
    /// Always `true` for SliceRange.
    bool
      preserves_type(const Index64& advanced) const override;

  private:
    /// @brief See #start.
    const int64_t start_;
    /// @brief See #stop.
    const int64_t stop_;
    /// @brief See #step.
    const int64_t step_;
  };

  /// @class SliceEllipsis
  ///
  /// @brief Represents a Python `Ellipsis` object (usual syntax:
  /// `array[...]`).
  class LIBAWKWARD_EXPORT_SYMBOL SliceEllipsis: public SliceItem {
  public:
    /// @brief Creates a SliceEllipsis.
    SliceEllipsis();

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    /// @copydoc SliceItem::preserves_type()
    ///
    /// Always `true` for SliceEllipsis.
    bool
      preserves_type(const Index64& advanced) const override;
  };

  /// @class SliceNewAxis
  ///
  /// @brief Represents NumPy's
  /// [newaxis](https://docs.scipy.org/doc/numpy/reference/constants.html#numpy.newaxis)
  /// marker (a.k.a. `None`), which prompts `__getitem__` to insert a
  /// length-1 regular dimension (RegularArray) at some point in the slice
  /// tuple.
  class LIBAWKWARD_EXPORT_SYMBOL SliceNewAxis: public SliceItem {
  public:
    /// @brief Creates a SliceNewAxis.
    SliceNewAxis();

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    /// @copydoc SliceItem::preserves_type()
    ///
    /// Always `false` for SliceNewAxis.
    bool
      preserves_type(const Index64& advanced) const override;
  };

  /// @class SliceArrayOf
  ///
  /// @brief Represents an array of integers in a slice (possibly converted
  /// from an array of booleans).
  ///
  /// Currently, the only type specialization is `T = int64_t`.
  template <typename T>
  class
#ifdef AWKWARD_SLICE_NO_EXTERN_TEMPLATE
  LIBAWKWARD_EXPORT_SYMBOL
#endif
  SliceArrayOf: public SliceItem {
  public:
    /// @brief Creates a SliceArrayOf from a full set of parameters.
    ///
    /// @param index A flattened version of the array used for slicing.
    /// If the original array was multidimensional, the shape/strides are
    /// stored separately.
    /// @param shape Number of elements in each dimension, like NumPy's
    /// array shape.
    /// Note that unlike {@link NumpyArray#shape NumpyArray::shape}, the
    /// integer type is `int64_t`, rather than `ssize_t`, and it must be
    /// at least one-dimensional.
    /// @param strides Length of each dimension in number of items. The length
    /// of strides must match the length of `shape`.
    /// Note that unlike {@link NumpyArray#shape NumpyArray::strides}, the
    /// integer type is `int64_t`, rather than `ssize_t`, and it quantifies
    /// the number of items, not the number of bytes.
    /// @param frombool If `true`, this integer array of positions was
    /// derived from a boolean array mask (via `numpy.nonzero` or
    /// equivalent); `false` otherwise.
    SliceArrayOf<T>(const IndexOf<T>& index,
                    const std::vector<int64_t>& shape,
                    const std::vector<int64_t>& strides, bool frombool);

    /// @brief A flattened version of the array used for slicing.
    ///
    /// If the original array was multidimensional, the shape/strides are
    /// stored separately.
    const IndexOf<T>
      index() const;

    /// @brief Number of elements in each dimension, like NumPy's
    /// array shape.
    ///
    /// Note that unlike {@link NumpyArray#shape NumpyArray::shape}, the
    /// integer type is `int64_t`, rather than `ssize_t`, and it must be
    /// at least one-dimensional.
    const std::vector<int64_t>
      shape() const;

    /// @brief Length of each dimension in number of items. The length
    /// of strides must match the length of `shape`.
    ///
    /// Note that unlike {@link NumpyArray#shape NumpyArray::strides}, the
    /// integer type is `int64_t`, rather than `ssize_t`, and it quantifies
    /// the number of items, not the number of bytes.
    const std::vector<int64_t>
      strides() const;

    /// @brief If `true`, this integer array of positions was
    /// derived from a boolean array mask (from NumPy's
    /// [nonzero](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html)
    /// or equivalent); `false` otherwise.
    bool
      frombool() const;

    /// @brief The length of the logical array: `shape[0]`.
    ///
    /// If the array is one-dimensional, this is equal to `array.length()`.
    const int64_t
      length() const;

    /// @brief The number of dimensions: `shape.size()`.
    int64_t
      ndim() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    const std::string
      tostring_part() const;

    /// @copydoc SliceItem::preserves_type()
    ///
    /// This is `true` for SliceArrayOf if `advanced.length() == 0`, `false`
    /// otherwise.
    bool
      preserves_type(const Index64& advanced) const override;

    /// @brief Returns a one-dimensional contiguous version of the array,
    /// like NumPy's [ravel](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html).
    const IndexOf<T>
      ravel() const;

  private:
    /// @brief See #index.
    const IndexOf<T> index_;
    /// @brief See #shape.
    const std::vector<int64_t> shape_;
    /// @brief See #strides.
    const std::vector<int64_t> strides_;
    /// @brief See #frombool.
    bool frombool_;
  };

#ifndef AWKWARD_SLICE_NO_EXTERN_TEMPLATE
  extern template class SliceArrayOf<int64_t>;
#endif

  using SliceArray64 = SliceArrayOf<int64_t>;

  /// @class SliceField
  ///
  /// @brief Represents a single string in a slice tuple, indicating that a
  /// RecordArray should be replaced by one of its fields.
  class LIBAWKWARD_EXPORT_SYMBOL SliceField: public SliceItem {
  public:
    /// @brief Creates a SliceField from a full set of parameters.
    ///
    /// @param key The name of the field to select.
    /// This may be an element of a
    /// {@link RecordArray#recordlookup RecordArray::recordlookup} or a
    /// {@link RecordArray#fieldindex RecordArray::fieldindex} integer
    /// as a string.
    SliceField(const std::string& key);

    /// @brief The name of the field to select.
    ///
    /// This may be an element of a
    /// {@link RecordArray#recordlookup RecordArray::recordlookup} or a
    /// {@link RecordArray#fieldindex RecordArray::fieldindex} integer
    /// as a string.
    const std::string
      key() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    /// @copydoc SliceItem::preserves_type()
    ///
    /// Always `false` for SliceField.
    bool
      preserves_type(const Index64& advanced) const override;

  private:
    /// @brief See #key.
    const std::string key_;
  };

  /// @class SliceField
  ///
  /// @brief Represents a list of strings in a slice tuple, indicating that a
  /// RecordArray should be replaced by a subset of its fields.
  class LIBAWKWARD_EXPORT_SYMBOL SliceFields: public SliceItem {
  public:
    /// @brief Creates a SliceFields from a full set of parameters.
    ///
    /// @param keys The names of the fields to select.
    /// This may be elements of a
    /// {@link RecordArray#recordlookup RecordArray::recordlookup} or
    /// {@link RecordArray#fieldindex RecordArray::fieldindex} integers
    /// as strings.
    SliceFields(const std::vector<std::string>& keys);

    /// @brief The names of the fields to select.
    ///
    /// This may be elements of a
    /// {@link RecordArray#recordlookup RecordArray::recordlookup} or
    /// {@link RecordArray#fieldindex RecordArray::fieldindex} integers
    /// as strings.
    const std::vector<std::string>
      keys() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    /// @copydoc SliceItem::preserves_type()
    ///
    /// Always `false` for SliceFields.
    bool
      preserves_type(const Index64& advanced) const override;

  private:
    /// @brief See #keys.
    const std::vector<std::string> keys_;
  };

  /// @class SliceMissingOf
  ///
  /// @brief Represents a SliceArrayOf, SliceMissingOf, or SliceJaggedOf
  /// with missing values: `None` (no equivalent in NumPy).
  ///
  /// Currently, the only type specialization is `T = int64_t`.
  template <typename T>
  class
#ifdef AWKWARD_SLICE_NO_EXTERN_TEMPLATE
  LIBAWKWARD_EXPORT_SYMBOL
#endif
  SliceMissingOf: public SliceItem {
  public:
    /// @brief Creates a SliceMissingOf with a full set of parameters.
    ///
    /// @param index Positions in the #content or negative values representing
    /// `None` in the same sense as {@link IndexedArrayOf IndexedOptionArray}.
    /// @param originalmask The array of booleans from which the #index was
    /// derived.
    /// @param content The non-`None` values of the array, much like an
    /// IndexedOptionArray's {@link IndexedArrayOf#content content}.
    SliceMissingOf(const IndexOf<T>& index,
                   const Index8& originalmask,
                   const SliceItemPtr& content);

    /// @brief Positions in the #content or negative values representing
    /// `None` in the same sense as {@link IndexedArrayOf IndexedOptionArray}.
    const IndexOf<T>
      index() const;

    /// @brief The array of booleans from which the #index was
    /// derived.
    const Index8
      originalmask() const;

    /// @brief The non-`None` values of the array, much like an
    /// IndexedOptionArray's {@link IndexedArrayOf#content content}.
    const SliceItemPtr
      content() const;

    /// @brief The length of the array: `len(index)`.
    int64_t
      length() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    const std::string
      tostring_part() const;

    /// @copydoc SliceItem::preserves_type()
    ///
    /// Always `true` for SliceMissingOf.
    ///
    /// @note This might not be the right choice; it should be revisited.
    bool
      preserves_type(const Index64& advanced) const override;

  private:
    /// @brief See #index.
    const IndexOf<T> index_;
    /// @brief See #originalmask.
    const Index8 originalmask_;
    /// @brief See #content.
    const SliceItemPtr content_;
  };

#ifndef AWKWARD_SLICE_NO_EXTERN_TEMPLATE
  extern template class SliceMissingOf<int64_t>;
#endif

  using SliceMissing64 = SliceMissingOf<int64_t>;

  /// @class SliceJaggedOf
  ///
  /// @brief Represents an array of nested lists, where the #content
  /// may be SliceArrayOf, SliceMissingOf, or SliceJaggedOf
  /// (no equivalent in NumPy).
  ///
  /// Currently, the only type specialization is `T = int64_t`.
  template <typename T>
  class
#ifdef AWKWARD_SLICE_NO_EXTERN_TEMPLATE
  LIBAWKWARD_EXPORT_SYMBOL
#endif
  SliceJaggedOf: public SliceItem {
  public:
    /// @brief Creates a SliceJaggedOf with a full set of parameters.
    ///
    /// @param offsets Positions where one nested list stops and the next
    /// starts in the #content in the same sense as
    /// {@link ListOffsetArrayOf ListOffsetArray}.
    /// The `offsets` must be monotonically increasing and its length is one
    /// greater than the length of the array it represents. As such, it must
    /// always have at least one element.
    /// @param content The contiguous content of the nested lists, like
    /// ListOffsetArray's {@link ListOffsetArrayOf#content content}.
    SliceJaggedOf(const IndexOf<T>& offsets, const SliceItemPtr& content);

    /// @brief Positions where one nested list stops and the next
    /// starts in the #content in the same sense as
    /// {@link ListOffsetArrayOf ListOffsetArray}.
    ///
    /// The `offsets` must be monotonically increasing and its length is one
    /// greater than the length of the array it represents. As such, it must
    /// always have at least one element.
    const IndexOf<T>
      offsets() const;

    /// @brief The contiguous content of the nested lists, like
    /// ListOffsetArray's {@link ListOffsetArrayOf#content content}.
    const SliceItemPtr
      content() const;

    /// @brief The length of the array: `len(offsets) - 1`.
    int64_t
      length() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    const std::string
      tostring_part() const;

    /// @copydoc SliceItem::preserves_type()
    ///
    /// Always `true` for SliceJaggedOf.
    ///
    /// @note This might not be the right choice; it should be revisited.
    bool
      preserves_type(const Index64& advanced) const override;

  private:
    /// @brief See #offsets.
    const IndexOf<T> offsets_;
    /// @brief See #content.
    const SliceItemPtr content_;
  };

#ifndef AWKWARD_SLICE_NO_EXTERN_TEMPLATE
  extern template class SliceJaggedOf<int64_t>;
#endif

  using SliceJagged64 = SliceJaggedOf<int64_t>;

  /// @class Slice
  ///
  /// @brief A sequence of SliceItem objects representing a tuple passed
  /// to Python's `__getitem__`.
  class LIBAWKWARD_EXPORT_SYMBOL Slice {
  public:
    /// @brief Represents a missing {@link SliceRange#start start},
    /// {@link SliceRange#stop stop}, or {@link SliceRange#step step}
    /// in a SliceRange.
    static int64_t none();

    /// @brief Creates a Slice with a full set of parameters.
    ///
    /// @param items The SliceItem objects in this Slice.
    /// @param sealed If `true`, the Slice is immutable and #append will fail.
    /// Otherwise, the #items may be appended to.
    Slice(const std::vector<SliceItemPtr>& items, bool sealed);

    /// @brief Creates an "unsealed" Slice, to which we can still add
    /// SliceItem objects (with #append).
    /// @param items The SliceItem objects in this Slice.
    Slice(const std::vector<SliceItemPtr>& items);

    /// @brief Creates an empty Slice.
    Slice();

    /// @brief The SliceItem objects in this Slice.
    const std::vector<SliceItemPtr>
      items() const;

    /// @brief If `true`, the Slice is immutable and #append will fail.
    /// Otherwise, the #items may be appended to.
    bool
      sealed() const;

    /// @brief The number of SliceItem objects in #items.
    int64_t
      length() const;

    /// @brief The number of SliceAt, SliceRange, and SliceArrayOf objects
    /// in the #items.
    int64_t
      dimlength() const;

    /// @brief Returns a pointer to the first SliceItem.
    const SliceItemPtr
      head() const;

    /// @brief Returns a Slice representing all but the first SliceItem.
    const Slice
      tail() const;

    /// @brief Returns a string representation of this slice item (single-line
    /// custom format).
    const std::string
      tostring() const;

    /// @brief Returns a new Slice with `item` prepended.
    const Slice
      prepended(const SliceItemPtr& item) const;

    /// @brief Inserts a SliceItem in-place at the end of the #items.
    void
      append(const SliceItemPtr& item);

    /// @brief Inserts a SliceAt in-place at the end of the #items.
    void
      append(const SliceAt& item);

    /// @brief Inserts a SliceRange in-place at the end of the #items.
    void
      append(const SliceRange& item);

    /// @brief Inserts a SliceEllipsis in-place at the end of the #items.
    void
      append(const SliceEllipsis& item);

    /// @brief Inserts a SliceNewAxis in-place at the end of the #items.
    void
      append(const SliceNewAxis& item);

    /// @brief Inserts a {@link SliceArrayOf SliceArray64} in-place at the end
    /// of the #items.
    void
      append(const SliceArray64& item);

    /// @brief Inserts a SliceField in-place at the end of the #items.
    void
      append(const SliceField& item);

    /// @brief Inserts a SliceFields in-place at the end of the #items.
    void
      append(const SliceFields& item);

    /// @brief Inserts a {@link SliceMissingOf SliceMissing64} in-place at the
    /// end of the #items.
    void
      append(const SliceMissing64& item);

    /// @brief Inserts a {@link SliceJaggedOf SliceJagged64} in-place at the
    /// end of the #items.
    void
      append(const SliceJagged64& item);

    /// @brief Seal this Slice so that it is no longer open to #append.
    void
      become_sealed();

    /// @brief Returns `true` if the Slice contains SliceArrayOf; `false`
    /// otherwise.
    ///
    /// This function can only be called when the Slice is sealed (see
    /// #Slice and #become_sealed).
    bool
      isadvanced() const;

  private:
    /// @brief See #items.
    std::vector<SliceItemPtr> items_;
    /// @brief See #sealed.
    bool sealed_;
  };

}

#endif // AWKWARD_SLICE_H_
