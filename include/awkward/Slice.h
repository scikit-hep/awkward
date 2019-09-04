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

    virtual const std::string tostring_part() const = 0;
  };

  class SliceAt: public SliceItem {
  public:
    SliceAt(int64_t at): at_(at) { }
    int64_t at() const { return at_; }
    virtual const std::string tostring_part() const { return std::to_string(at_); }
  private:
    const int64_t at_;
  };

  class SliceStartStop: public SliceItem {
  public:
    SliceStartStop(int64_t start, int64_t stop): start_(start), stop_(stop) { }
    int64_t start() const { return start_; }
    int64_t stop() const { return stop_; }
    bool hasstart() const { return start_ != none(); }
    bool hasstop() const { return stop_ != none(); }
    virtual const std::string tostring_part() const {
      return (hasstart() ? std::to_string(start_) : std::string("")) + std::string(":") + (hasstop() ? std::to_string(stop_) : std::string(""));
    }
  private:
    const int64_t start_;
    const int64_t stop_;
  };

  class SliceStartStopStep: public SliceItem {
  public:
    SliceStartStopStep(int64_t start, int64_t stop, int64_t step): start_(start), stop_(stop), step_(step) { }
    int64_t start() const { return start_; }
    int64_t stop() const { return stop_; }
    int64_t step() const { return step_; }
    bool hasstart() const { return start_ != none(); }
    bool hasstop() const { return stop_ != none(); }
    bool hasstep() const { return step_ != none(); }
    virtual const std::string tostring_part() const {
      return (hasstart() ? std::to_string(start_) : std::string("")) + std::string(":") + (hasstop() ? std::to_string(stop_) : std::string("")) + std::string(":") + (hasstep() ? std::to_string(step_) : std::string(""));
    }
  private:
    const int64_t start_;
    const int64_t stop_;
    const int64_t step_;
  };

  class SliceByteMask: public SliceItem {
  public:
    SliceByteMask(Index8 mask): mask_(mask) { }
    Index8 mask() const { return mask_; }
    int64_t length() const { return mask_.length(); }
    virtual const std::string tostring_part() const { return std::string("<bytemask[") + std::to_string(length()) + std::string("]>"); }
  private:
    const Index8 mask_;
  };

  class SliceIndex32: public SliceItem {
  public:
    SliceIndex32(Index32 index): index_(index) { }
    Index32 index() const { return index_; }
    int64_t length() const { return index_.length(); }
    virtual const std::string tostring_part() const { return std::string("<index32[") + std::to_string(length()) + std::string("]>"); }
  private:
    const Index32 index_;
  };

  class SliceIndex64: public SliceItem {
  public:
    SliceIndex64(Index64 index): index_(index) { }
    Index64 index() const { return index_; }
    int64_t length() const { return index_.length(); }
    virtual const std::string tostring_part() const { return std::string("<index64[") + std::to_string(length()) + std::string("]>"); }
  private:
    const Index64 index_;
  };

  class SliceEllipsis: public SliceItem {
  public:
    SliceEllipsis() { }
    virtual const std::string tostring_part() const { return std::string("<ellipsis>"); }
  };

  class SliceNewAxis: public SliceItem {
  public:
    SliceNewAxis() { }
    virtual const std::string tostring_part() const { return std::string("<newaxis>"); }
  };

  class Slice {
  public:
    static int64_t none() { return SliceItem::none(); }

    Slice(): items_(std::vector<std::shared_ptr<SliceItem>>()) { }
    Slice(const std::vector<std::shared_ptr<SliceItem>> items): items_(items) { }
    const SliceItem* borrow(int64_t which) const { return items_[(size_t)which].get(); }
    const int64_t length() const { return (int64_t)items_.size(); }
    const std::shared_ptr<SliceItem> head() const {
      assert(items_.size() != 0);
      return items_[0];
    }
    const Slice tail() const {
      assert(items_.size() != 0);
      return Slice(std::vector<std::shared_ptr<SliceItem>>(items_.begin() + 1, items_.end()));
    }
    void append(std::shared_ptr<SliceItem> x) {
      items_.push_back(x);
    }
    Slice with(SliceAt x) {
      Slice out(std::vector<std::shared_ptr<SliceItem>>(items_.begin(), items_.end()));
      out.items_.push_back(std::shared_ptr<SliceItem>(new SliceAt(x)));
      return out;
    }
    Slice with(SliceStartStop x) {
      Slice out(std::vector<std::shared_ptr<SliceItem>>(items_.begin(), items_.end()));
      out.items_.push_back(std::shared_ptr<SliceItem>(new SliceStartStop(x)));
      return out;
    }
    Slice with(SliceStartStopStep x) {
      Slice out(std::vector<std::shared_ptr<SliceItem>>(items_.begin(), items_.end()));
      out.items_.push_back(std::shared_ptr<SliceItem>(new SliceStartStopStep(x)));
      return out;
    }
    Slice with(SliceByteMask x) {
      Slice out(std::vector<std::shared_ptr<SliceItem>>(items_.begin(), items_.end()));
      out.items_.push_back(std::shared_ptr<SliceItem>(new SliceByteMask(x)));
      return out;
    }
    Slice with(SliceIndex32 x) {
      Slice out(std::vector<std::shared_ptr<SliceItem>>(items_.begin(), items_.end()));
      out.items_.push_back(std::shared_ptr<SliceItem>(new SliceIndex32(x)));
      return out;
    }
    Slice with(SliceIndex64 x) {
      Slice out(std::vector<std::shared_ptr<SliceItem>>(items_.begin(), items_.end()));
      out.items_.push_back(std::shared_ptr<SliceItem>(new SliceIndex64(x)));
      return out;
    }
    Slice with(SliceEllipsis x) {
      Slice out(std::vector<std::shared_ptr<SliceItem>>(items_.begin(), items_.end()));
      out.items_.push_back(std::shared_ptr<SliceItem>(new SliceEllipsis(x)));
      return out;
    }
    Slice with(SliceNewAxis x) {
      Slice out(std::vector<std::shared_ptr<SliceItem>>(items_.begin(), items_.end()));
      out.items_.push_back(std::shared_ptr<SliceItem>(new SliceNewAxis(x)));
      return out;
    }
    virtual const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const {
      std::string out;
      out += indent + pre + std::string("<Slice expr=\"");
      if (length() != 1) {
        out += std::string("(");
      }
      for (size_t i = 0;  i < items_.size();  i++) {
        if (i != 0) {
          out += ", ";
        }
        out += items_[i].get()->tostring_part();
      }
      if (length() != 1) {
        out += std::string(")");
      }
      return out + std::string("\"/>") + post;
    }
    const std::string tostring() const {
      return tostring_part("", "", "");
    }
  private:
    std::vector<std::shared_ptr<SliceItem>> items_;
  };

}

#endif // AWKWARD_SLICE_H_
