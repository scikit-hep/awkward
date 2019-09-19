// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_SLICE_H_
#define AWKWARD_SLICE_H_

#include <cassert>
#include <string>
#include <vector>
#include <memory>

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"
#include "awkward/Index.h"

namespace awkward {
  class SliceItem {
  public:
    static int64_t none() { return kMaxInt64 + 1; }

    virtual const std::string tostring() const = 0;
  };

  class SliceAt: public SliceItem {
  public:
    SliceAt(int64_t at): at_(at) { }
    int64_t at() const { return at_; }
    virtual const std::string tostring() const;
  private:
    const int64_t at_;
  };

  class SliceRange: public SliceItem {
  public:
    SliceRange(int64_t start, int64_t stop, int64_t step): start_(start), stop_(stop), step_(step) { assert(step_ != 0); }
    int64_t start() const { return start_; }
    int64_t stop() const { return stop_; }
    int64_t step() const { return step_; }
    bool hasstart() const { return start_ != none(); }
    bool hasstop() const { return stop_ != none(); }
    bool hasstep() const { return step_ != none(); }
    virtual const std::string tostring() const;
  private:
    const int64_t start_;
    const int64_t stop_;
    const int64_t step_;
  };

  template <typename T>
  class SliceArrayOf: public SliceItem {
  public:
    SliceArrayOf<T>(const IndexOf<T>& index, const std::vector<int64_t>& shape, const std::vector<int64_t>& strides): index_(index), shape_(shape), strides_(strides) {
      assert(shape_.size() != 0);
      assert(shape_.size() == strides_.size());
    }
    const IndexOf<T> index() const { return index_; }
    virtual const std::string tostring() const;
  private:
    const IndexOf<T> index_;
    const std::vector<int64_t> shape_;
    const std::vector<int64_t> strides_;
  };

  typedef SliceArrayOf<int64_t> SliceArray64;

  class SliceEllipsis: public SliceItem {
  public:
    SliceEllipsis() { }
    virtual const std::string tostring() const;
  };

  class SliceNewAxis: public SliceItem {
  public:
    SliceNewAxis() { }
    virtual const std::string tostring() const;
  };

  class Slice {
  public:
    static int64_t none() { return SliceItem::none(); }

    Slice(): items_(std::vector<std::shared_ptr<SliceItem>>()) { }
    Slice(const std::vector<std::shared_ptr<SliceItem>> items): items_(items) { }
    const SliceItem* borrow(int64_t which) const { return items_[(size_t)which].get(); }
    const int64_t length() const { return (int64_t)items_.size(); }
    const std::string tostring() const;

  private:
    std::vector<std::shared_ptr<SliceItem>> items_;
  };
}

#endif // AWKWARD_SLICE_H_
