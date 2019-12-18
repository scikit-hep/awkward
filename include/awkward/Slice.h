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
    static int64_t none() { return kSliceNone; }

    virtual ~SliceItem() { }

    virtual const std::shared_ptr<SliceItem> shallow_copy() const = 0;
    virtual const std::string tostring() const = 0;
    virtual bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const = 0;
  };

  class SliceAt: public SliceItem {
  public:
    SliceAt(int64_t at): at_(at) { }
    int64_t at() const { return at_; }
    const std::shared_ptr<SliceItem> shallow_copy() const override {
      return std::shared_ptr<SliceItem>(new SliceAt(at_));
    }
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override {
      return false;
    }
  private:
    const int64_t at_;
  };

  class SliceRange: public SliceItem {
  public:
    SliceRange(int64_t start, int64_t stop, int64_t step): start_(start), stop_(stop), step_(step == none() ? 1 : step) {
      assert(step_ != 0);
    }
    int64_t start() const { return start_; }
    int64_t stop() const { return stop_; }
    int64_t step() const { return step_; }
    bool hasstart() const { return start_ != none(); }
    bool hasstop() const { return stop_ != none(); }
    const std::shared_ptr<SliceItem> shallow_copy() const override {
      return std::shared_ptr<SliceItem>(new SliceRange(start_, stop_, step_));
    }
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override {
      return true;
    }
  private:
    const int64_t start_;
    const int64_t stop_;
    const int64_t step_;
  };

  class SliceEllipsis: public SliceItem {
  public:
    SliceEllipsis() { }
    const std::shared_ptr<SliceItem> shallow_copy() const override {
      return std::shared_ptr<SliceItem>(new SliceEllipsis());
    }
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override {
      return true;
    }
  };

  class SliceNewAxis: public SliceItem {
  public:
    SliceNewAxis() { }
    const std::shared_ptr<SliceItem> shallow_copy() const override {
      return std::shared_ptr<SliceItem>(new SliceNewAxis());
    }
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override {
      return false;
    }
  };

  template <typename T>
  class SliceArrayOf: public SliceItem {
  public:
    SliceArrayOf<T>(const IndexOf<T>& index, const std::vector<int64_t>& shape, const std::vector<int64_t>& strides): index_(index), shape_(shape), strides_(strides) {
      assert(shape_.size() != 0);
      assert(shape_.size() == strides_.size());
    }
    const IndexOf<T> index() const { return index_; }
    const int64_t length() const { return shape_[0]; }
    const std::vector<int64_t> shape() const { return shape_; }
    const std::vector<int64_t> strides() const { return strides_; }
    int64_t ndim() const { return (int64_t)shape_.size(); }
    const std::shared_ptr<SliceItem> shallow_copy() const override {
      return std::shared_ptr<SliceItem>(new SliceArrayOf<T>(index_, shape_, strides_));
    }
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override {
      return advanced.length() == 0;
    }
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
    SliceField(const std::string& key): key_(key) { }
    const std::string key() const { return key_; }
    const std::shared_ptr<SliceItem> shallow_copy() const override {
      return std::shared_ptr<SliceItem>(new SliceField(key_));
    }
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override {
      return false;
    }
  private:
    const std::string key_;
  };

  class SliceFields: public SliceItem {
  public:
    SliceFields(const std::vector<std::string>& keys): keys_(keys) { }
    const std::vector<std::string> keys() const { return keys_; }
    const std::shared_ptr<SliceItem> shallow_copy() const override {
      return std::shared_ptr<SliceItem>(new SliceFields(keys_));
    }
    const std::string tostring() const override;
    bool preserves_type(const std::shared_ptr<Type>& type, const Index64& advanced) const override {
      return type.get() != nullptr  &&  type.get()->numfields() != -1  &&  util::subset(keys_, type.get()->keys());
    }
  private:
    const std::vector<std::string> keys_;
  };

  class Slice {
  public:
    static int64_t none() { return SliceItem::none(); }

    Slice(): items_(std::vector<std::shared_ptr<SliceItem>>()), sealed_(false) { }
    Slice(const std::vector<std::shared_ptr<SliceItem>> items): items_(items), sealed_(false) { }
    Slice(const std::vector<std::shared_ptr<SliceItem>> items, bool sealed): items_(items), sealed_(sealed) { }
    const std::vector<std::shared_ptr<SliceItem>> items() const { return items_; }
    bool sealed() const { return sealed_; }

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
