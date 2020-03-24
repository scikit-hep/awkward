// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_SLICE_H_
#define AWKWARD_SLICE_H_

#include <string>
#include <vector>
#include <memory>

#include "awkward/cpu-kernels/getitem.h"
#include "awkward/type/Type.h"
#include "awkward/Index.h"

namespace awkward {
  class SliceItem;
  using SliceItemPtr = std::shared_ptr<SliceItem>;

  /// @class SliceItem
  ///
  /// @brief Abstract class for slice items, which are elements of a tuple
  /// passed to an array's `__getitem__` in Python.
  class EXPORT_SYMBOL SliceItem {
  public:
    /// @brief Represents a missing {@link SliceRange#start start},
    /// {@link SliceRange#stop stop}, or {@link SliceRange#step step}
    /// in a SliceRange.
    static int64_t none();

    /// @brief Empty destructor; required for some C++ reason.
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

  class EXPORT_SYMBOL SliceAt: public SliceItem {
  public:
    SliceAt(int64_t at);

    int64_t
      at() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    bool
      preserves_type(const Index64& advanced) const override;

  private:
    const int64_t at_;
  };

  class EXPORT_SYMBOL SliceRange: public SliceItem {
  public:
    SliceRange(int64_t start, int64_t stop, int64_t step);

    int64_t
      start() const;

    int64_t
      stop() const;

    int64_t
      step() const;

    bool
      hasstart() const;

    bool
      hasstop() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    bool
      preserves_type(const Index64& advanced) const override;

  private:
    const int64_t start_;
    const int64_t stop_;
    const int64_t step_;
  };

  class EXPORT_SYMBOL SliceEllipsis: public SliceItem {
  public:
    SliceEllipsis();

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    bool
      preserves_type(const Index64& advanced) const override;
  };

  class EXPORT_SYMBOL SliceNewAxis: public SliceItem {
  public:
    SliceNewAxis();

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    bool
      preserves_type(const Index64& advanced) const override;
  };

  template <typename T>
  class EXPORT_SYMBOL SliceArrayOf: public SliceItem {
  public:
    SliceArrayOf<T>(const IndexOf<T>& index,
                    const std::vector<int64_t>& shape,
                    const std::vector<int64_t>& strides, bool frombool);

    const IndexOf<T>
      index() const;

    const int64_t
      length() const;

    const std::vector<int64_t>
      shape() const;

    const std::vector<int64_t>
      strides() const;

    bool
      frombool() const;

    int64_t
      ndim() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    const std::string
      tostring_part() const;

    bool
      preserves_type(const Index64& advanced) const override;

    const IndexOf<T>
      ravel() const;

  private:
    const IndexOf<T> index_;
    const std::vector<int64_t> shape_;
    const std::vector<int64_t> strides_;
    bool frombool_;
  };

  using SliceArray64 = SliceArrayOf<int64_t>;

  class EXPORT_SYMBOL SliceField: public SliceItem {
  public:
    SliceField(const std::string& key);

    const std::string
      key() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    bool
      preserves_type(const Index64& advanced) const override;

  private:
    const std::string key_;
  };

  class EXPORT_SYMBOL SliceFields: public SliceItem {
  public:
    SliceFields(const std::vector<std::string>& keys);

    const std::vector<std::string>
      keys() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    bool
      preserves_type(const Index64& advanced) const override;

  private:
    const std::vector<std::string> keys_;
  };

  template <typename T>
  class EXPORT_SYMBOL SliceMissingOf: public SliceItem {
  public:
    SliceMissingOf(const IndexOf<T>& index,
                   const Index8& originalmask,
                   const SliceItemPtr& content);

    int64_t
      length() const;

    const IndexOf<T>
      index() const;

    const Index8
      originalmask() const;

    const SliceItemPtr
      content() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    const std::string
      tostring_part() const;

    bool
      preserves_type(const Index64& advanced) const override;

  private:
    const IndexOf<T> index_;
    const Index8 originalmask_;
    const SliceItemPtr content_;
  };

  using SliceMissing64 = SliceMissingOf<int64_t>;

  template <typename T>
  class EXPORT_SYMBOL SliceJaggedOf: public SliceItem {
  public:
    SliceJaggedOf(const IndexOf<T>& offsets, const SliceItemPtr& content);

    int64_t
      length() const;

    const IndexOf<T>
      offsets() const;

    const SliceItemPtr
      content() const;

    const SliceItemPtr
      shallow_copy() const override;

    const std::string
      tostring() const override;

    const std::string
      tostring_part() const;

    bool
      preserves_type(const Index64& advanced) const override;

  private:
    const IndexOf<T> offsets_;
    const SliceItemPtr content_;
  };

  using SliceJagged64 = SliceJaggedOf<int64_t>;

  class EXPORT_SYMBOL Slice {
  public:
    static int64_t none();

    Slice();

    Slice(const std::vector<SliceItemPtr>& items);

    Slice(const std::vector<SliceItemPtr>& items, bool sealed);

    const std::vector<SliceItemPtr>
      items() const;

    bool
      sealed() const;

    int64_t
      length() const;

    int64_t
      dimlength() const;

    const SliceItemPtr
      head() const;

    const Slice
      tail() const;

    const std::string
      tostring() const;

    void
      append(const SliceItemPtr& item);

    void
      append(const SliceAt& item);

    void
      append(const SliceRange& item);

    void
      append(const SliceEllipsis& item);

    void
      append(const SliceNewAxis& item);

    template <typename T>
    void
      append(const SliceArrayOf<T>& item);

    void
      become_sealed();

    bool
      isadvanced() const;

  private:
    std::vector<SliceItemPtr> items_;
    bool sealed_;
  };
}

#endif // AWKWARD_SLICE_H_
