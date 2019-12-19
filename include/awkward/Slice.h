// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_SLICE_H_
#define AWKWARD_SLICE_H_

#include <cassert>
#include <string>
#include <vector>
#include <memory>

#include "awkward/cpu-kernels/getitem.h"
#include "awkward/type/Type.h"
#include "awkward/Index.h"

namespace awkward {
  class SliceItem {
  public:
    static int64_t none();
    virtual ~SliceItem();
    virtual const std::shared_ptr<SliceItem> shallow_copy() const = 0;
    virtual const std::string tostring() const = 0;
    virtual bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const = 0;
  };

  class SliceAt: public SliceItem {
  public:
    SliceAt(int64_t at);
    int64_t at() const;
    const std::shared_ptr<SliceItem> shallow_copy() const override;
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override;
  private:
    const int64_t at_;
  };

  class SliceRange: public SliceItem {
  public:
    SliceRange(int64_t start, int64_t stop, int64_t step);
    int64_t start() const;
    int64_t stop() const;
    int64_t step() const;
    bool hasstart() const;
    bool hasstop() const;
    const std::shared_ptr<SliceItem> shallow_copy() const override;
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override;
  private:
    const int64_t start_;
    const int64_t stop_;
    const int64_t step_;
  };

  class SliceEllipsis: public SliceItem {
  public:
    SliceEllipsis();
    const std::shared_ptr<SliceItem> shallow_copy() const override;
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override;
  };

  class SliceNewAxis: public SliceItem {
  public:
    SliceNewAxis();
    const std::shared_ptr<SliceItem> shallow_copy() const override;
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override;
  };

  template <typename T>
  class SliceArrayOf: public SliceItem {
  public:
    SliceArrayOf<T>(const IndexOf<T>& index, const std::vector<int64_t>& shape, const std::vector<int64_t>& strides);
    const IndexOf<T> index() const;
    const int64_t length() const;
    const std::vector<int64_t> shape() const;
    const std::vector<int64_t> strides() const;
    int64_t ndim() const;
    const std::shared_ptr<SliceItem> shallow_copy() const override;
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override;
    const std::string tostring_part() const;
    const IndexOf<T> ravel() const;
  private:
    const IndexOf<T> index_;
    const std::vector<int64_t> shape_;
    const std::vector<int64_t> strides_;
  };

  typedef SliceArrayOf<int64_t> SliceArray64;

  class SliceField: public SliceItem {
  public:
    SliceField(const std::string& key);
    const std::string key() const;
    const std::shared_ptr<SliceItem> shallow_copy() const override;
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override;
  private:
    const std::string key_;
  };

  class SliceFields: public SliceItem {
  public:
    SliceFields(const std::vector<std::string>& keys);
    const std::vector<std::string> keys() const;
    const std::shared_ptr<SliceItem> shallow_copy() const override;
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override;
  private:
    const std::vector<std::string> keys_;
  };

  class Slice {
  public:
    static int64_t none();

    Slice();
    Slice(const std::vector<std::shared_ptr<SliceItem>>& items);
    Slice(const std::vector<std::shared_ptr<SliceItem>>& items, bool sealed);
    const std::vector<std::shared_ptr<SliceItem>> items() const;
    bool sealed() const;
    int64_t length() const;
    int64_t dimlength() const;
    const std::shared_ptr<SliceItem> head() const;
    const Slice tail() const;
    const std::string tostring() const;
    void append(const std::shared_ptr<SliceItem>& item);
    void append(const SliceAt& item);
    void append(const SliceRange& item);
    void append(const SliceEllipsis& item);
    void append(const SliceNewAxis& item);
    template <typename T>
    void append(const SliceArrayOf<T>& item);
    void become_sealed();
    bool isadvanced() const;

  private:
    std::vector<std::shared_ptr<SliceItem>> items_;
    bool sealed_;
  };
}

#endif // AWKWARD_SLICE_H_
