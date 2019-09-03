// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_SLICE_H_
#define AWKWARD_SLICE_H_

#include <cassert>
#include <vector>
#include <memory>

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"

namespace awkward {
  class Slice { };

  class Slice1: public Slice {
  public:
    Slice1(int64_t where): where_(where) { }
    const int64_t where() const { return where_; }
  private:
    const int64_t where_;
  };

  class Slice2: public Slice {
  public:
    Slice2(int64_t start, int64_t stop): start_(start), stop_(stop) { }
    const int64_t start() const { return start_; }
    const int64_t stop() const { return stop_; }
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
  private:
    const int64_t start_;
    const int64_t stop_;
    const int64_t step_;
  };

  class SliceByteMask: public Slice {
  public:
    SliceByteMask(const std::vector<uint8_t> mask, bool maskedwhen)
        : mask_(std::move(mask)), maskedwhen_(maskedwhen) { }
    SliceByteMask(SliceByteMask&& other) { *this = std::move(other); }
    SliceByteMask(SliceByteMask& other) { *this = other; }
    SliceByteMask& operator=(const SliceByteMask& other) {
      if (&other == this) {
        return *this;
      }
      mask_ = other.mask_;
      return *this;
    }
    const uint8_t* maskdata() const { return mask_.data(); }
    const int64_t length() const { return (int64_t)mask_.size(); }
  private:
    std::vector<uint8_t> mask_;
    bool maskedwhen_;  // true for Numpy
  };

  class SliceBitMask: public Slice {
  public:
    SliceBitMask(const std::vector<uint8_t> mask, bool maskedwhen, bool lsborder)
        : mask_(std::move(mask)), maskedwhen_(maskedwhen), lsborder_(lsborder) { }
    SliceBitMask(SliceBitMask&& other) { *this = std::move(other); }
    SliceBitMask(SliceBitMask& other) { *this = other; }
    SliceBitMask& operator=(const SliceBitMask& other) {
      if (&other == this) {
        return *this;
      }
      mask_ = other.mask_;
      return *this;
    }
    const uint8_t* maskdata() const { return mask_.data(); }
    const int64_t length() const { return (int64_t)mask_.size(); }
  private:
    std::vector<uint8_t> mask_;
    bool maskedwhen_;  // false for Arrow
    bool lsborder_;    // true for Arrow
  };

  class SliceIndex32: public Slice {
  public:
    SliceIndex32(const std::vector<int32_t> index): index_(std::move(index)) { }
    SliceIndex32(SliceIndex32&& other) { *this = std::move(other); }
    SliceIndex32(SliceIndex32& other) { *this = other; }
    SliceIndex32& operator=(const SliceIndex32& other) {
      if (&other == this) {
        return *this;
      }
      index_ = other.index_;
      return *this;
    }
    const int32_t* indexdata() const { return index_.data(); }
    const int64_t length() const { return (int64_t)index_.size(); }
  private:
    std::vector<int32_t> index_;
  };

  class SliceIndex64: public Slice {
  public:
    SliceIndex64(const std::vector<int64_t> index): index_(std::move(index)) { }
    SliceIndex64(SliceIndex64&& other) { *this = std::move(other); }
    SliceIndex64(SliceIndex64& other) { *this = other; }
    SliceIndex64& operator=(const SliceIndex64& other) {
      if (&other == this) {
        return *this;
      }
      index_ = other.index_;
      return *this;
    }
    const int64_t* indexdata() const { return index_.data(); }
    const int64_t length() const { return (int64_t)index_.size(); }
  private:
    std::vector<int64_t> index_;
  };

  class SliceTuple {
  public:
    SliceTuple(const std::vector<std::unique_ptr> items): items_(std::move(items)) { }
    SliceTuple(SliceTuple&& other) { *this = std::move(other); }
    SliceTuple(SliceTuple& other) { *this = other; }
    SliceTuple& operator=(const SliceTuple& other) {
      if (&other == this) {
        return *this;
      }
      items_ = other.items_;
      return *this;
    }
    const Slice* item(int64_t where) const { return items_[where].get(); }
    const int64_t length() const { return (int64_t)items_.size(); }
  private:
    std::vector<std::unique_ptr> items_;
  };

}

#endif // AWKWARD_SLICE_H_
