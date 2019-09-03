// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_SLICE_H_
#define AWKWARD_SLICE_H_

#include <cassert>
#include <string>
#include <vector>
#include <memory>

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"

namespace awkward {
  class Slice {
  public:
    virtual const std::string tostring() const = 0;
  };

  class Slice1: public Slice {
  public:
    Slice1(int64_t at): at_(at) { }
    const int64_t at() const { return at_; }
    virtual const std::string tostring() const {
      return std::to_string(at_);
    }
  private:
    const int64_t at_;
  };

  class Slice2: public Slice {
  public:
    Slice2(int64_t start, int64_t stop): start_(start), stop_(stop) { }
    const int64_t start() const { return start_; }
    const int64_t stop() const { return stop_; }
    virtual const std::string tostring() const {
      return std::to_string(start_) + std::string(":") + std::to_string(stop_);
    }
  private:
    const int64_t start_;
    const int64_t stop_;
  };

  class Slice3: public Slice {
  public:
    Slice3(int64_t start, int64_t stop, int64_t step): start_(start), stop_(stop), step_(step) {
      assert(step != 0);
    }
    const int64_t start() const { return start_; }
    const int64_t stop() const { return stop_; }
    const int64_t step() const { return step_; }
    virtual const std::string tostring() const {
      return std::to_string(start_) + std::string(":") + std::to_string(stop_) + std::string(":") + std::to_string(step_);
    }
  private:
    const int64_t start_;
    const int64_t stop_;
    const int64_t step_;
  };

  // class SliceByteMask: public Slice {
  // public:
  //   SliceByteMask(const std::vector<uint8_t> mask, bool maskedwhen)
  //       : mask_(std::move(mask)), maskedwhen_(maskedwhen) { }
  //   SliceByteMask(SliceByteMask&& other) { *this = std::move(other); }
  //   SliceByteMask(SliceByteMask& other) { *this = other; }
  //   SliceByteMask& operator=(const SliceByteMask& other) {
  //     if (&other == this) {
  //       return *this;
  //     }
  //     mask_ = other.mask_;
  //     maskedwhen_ = other.maskedwhen_;
  //     return *this;
  //   }
  //   const uint8_t* maskdata() const { return mask_.data(); }
  //   const int64_t length() const { return (int64_t)mask_.size(); }
  //   virtual const std::string tostring() const {
  //     return std::string("<bytemask>");
  //   }
  // private:
  //   std::vector<uint8_t> mask_;
  //   bool maskedwhen_;  // true for Numpy
  // };
  //
  // class SliceBitMask: public Slice {
  // public:
  //   SliceBitMask(const std::vector<uint8_t> mask, bool maskedwhen, bool lsborder, int64_t bitlength)
  //       : mask_(std::move(mask)), maskedwhen_(maskedwhen), lsborder_(lsborder), bitlength_(bitlength) { }
  //   SliceBitMask(SliceBitMask&& other) { *this = std::move(other); }
  //   SliceBitMask(SliceBitMask& other) { *this = other; }
  //   SliceBitMask& operator=(const SliceBitMask& other) {
  //     if (&other == this) {
  //       return *this;
  //     }
  //     mask_ = other.mask_;
  //     maskedwhen_ = other.maskedwhen_;
  //     lsborder_ = other.lsborder_;
  //     bitlength_ = other.bitlength_;
  //     return *this;
  //   }
  //   const uint8_t* maskdata() const { return mask_.data(); }
  //   const int64_t bytelength() const { return (int64_t)mask_.size(); }
  //   const int64_t bitlength() const { return bitlength_; }
  //   virtual const std::string tostring() const {
  //     return std::string("<bitmask>");
  //   }
  // private:
  //   std::vector<uint8_t> mask_;
  //   bool maskedwhen_;  // false for Arrow
  //   bool lsborder_;    // true for Arrow
  //   int64_t bitlength_;
  // };
  //
  // class SliceIndex32: public Slice {
  // public:
  //   SliceIndex32(const std::vector<int32_t> index): index_(std::move(index)) { }
  //   SliceIndex32(SliceIndex32&& other) { *this = std::move(other); }
  //   SliceIndex32(SliceIndex32& other) { *this = other; }
  //   SliceIndex32& operator=(const SliceIndex32& other) {
  //     if (&other == this) {
  //       return *this;
  //     }
  //     index_ = other.index_;
  //     return *this;
  //   }
  //   const int32_t* indexdata() const { return index_.data(); }
  //   const int64_t length() const { return (int64_t)index_.size(); }
  //   virtual const std::string tostring() const {
  //     return std::string("<index32>");
  //   }
  // private:
  //   std::vector<int32_t> index_;
  // };
  //
  // class SliceIndex64: public Slice {
  // public:
  //   SliceIndex64(const std::vector<int64_t> index): index_(std::move(index)) { }
  //   SliceIndex64(SliceIndex64&& other) { *this = std::move(other); }
  //   SliceIndex64(SliceIndex64& other) { *this = other; }
  //   SliceIndex64& operator=(const SliceIndex64& other) {
  //     if (&other == this) {
  //       return *this;
  //     }
  //     index_ = other.index_;
  //     return *this;
  //   }
  //   const int64_t* indexdata() const { return index_.data(); }
  //   const int64_t length() const { return (int64_t)index_.size(); }
  //   virtual const std::string tostring() const {
  //     return std::string("<index64>");
  //   }
  // private:
  //   std::vector<int64_t> index_;
  // };
  //
  // class SliceEllipsis: public Slice {
  // public:
  //   SliceEllipsis() { }
  //   virtual const std::string tostring() const {
  //     return std::string("...");
  //   }
  // };
  //
  // class SliceNewAxis: public Slice {
  // public:
  //   SliceNewAxis() { }
  //   virtual const std::string tostring() const {
  //     return std::string("<newaxis>");
  //   }
  // };

  class Slices {
  public:
    Slices(): items_() { }
    Slices(const std::vector<std::shared_ptr<Slice>> items): items_(std::move(items)) { }
    Slices(Slices&& other) { *this = std::move(other); }
    Slices(Slices& other) { *this = other; }
    Slices& operator=(const Slices& other) {
      if (&other == this) {
        return *this;
      }
      items_ = other.items_;
      return *this;
    }
    const Slice* item(int64_t where) const { return items_[where].get(); }
    const int64_t length() const { return (int64_t)items_.size(); }
    void append(std::shared_ptr<Slice> slice) {
      items_.push_back(slice);
    }
    void append(Slice1& slice) {
      items_.push_back(std::shared_ptr<Slice>(new Slice1(slice.at())));
    }
    void append(Slice2& slice) {
      items_.push_back(std::shared_ptr<Slice>(new Slice2(slice.start(), slice.stop())));
    }
    void append(Slice3& slice) {
      items_.push_back(std::shared_ptr<Slice>(new Slice3(slice.start(), slice.stop(), slice.step())));
    }
    virtual const std::string tostring() const {
      std::string out("(");
      for (std::vector<std::shared_ptr<Slice>>::size_type i = 0;  i < items_.size();  i++) {
        if (i != 0) {
          out += ", ";
        }
        out += items_[i].get()->tostring();
      }
      return out + std::string(")");
    }
  private:
    std::vector<std::shared_ptr<Slice>> items_;
  };

}

#endif // AWKWARD_SLICE_H_
