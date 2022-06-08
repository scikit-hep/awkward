#ifndef GROWABLEBUFFER_H_
#define GROWABLEBUFFER_H_

#include <iostream>
#include <stdint.h>
#include <vector>
#include <memory>
#include <numeric>
#include <cmath>

template <typename PRIMITIVE>
class GrowableBuffer {
public:
  GrowableBuffer(size_t initial)
      : initial_(initial) {
    initial_ = initial;
    ptr_.push_back(std::unique_ptr<PRIMITIVE>(new PRIMITIVE[initial]));
    length_.push_back(0);
    reserved_.push_back(initial);
    }

  const std::unique_ptr<PRIMITIVE>&
  ptr() const {
    return ptr_[0];
  }

  size_t
  length() const {
    return std::accumulate(length_.begin(), length_.end(), (size_t)0);
  }

  size_t
  reserved() const {
    return std::accumulate(reserved_.begin(), reserved_.end(), (size_t)0);
  }

  void
  clear() {
    length_.clear();
    length_.push_back(0);
    reserved_.clear();
    reserved_.push_back(initial_);
    ptr_.clear();
    ptr_.push_back(std::unique_ptr<PRIMITIVE>(new PRIMITIVE[initial_]));
  }

  void
  fill_panel(PRIMITIVE datum) {
    if (length_[ptr_.size()-1] < reserved_[ptr_.size()-1]) {
      ptr_[ptr_.size()-1].get()[length_[ptr_.size()-1]] = datum;
      length_[ptr_.size()-1]++;
    }
  }

  void
  add_panel(size_t reserved) {
    ptr_.push_back(std::unique_ptr<PRIMITIVE>(new PRIMITIVE[reserved]));
    length_.push_back(0);
    reserved_.push_back(reserved);
  }

  void
  append(PRIMITIVE datum) {
    if (length_[ptr_.size()-1] == reserved_[ptr_.size()-1]) {
      add_panel(reserved_[ptr_.size()-1]);
    }
    fill_panel(datum);
  }

  PRIMITIVE
  getitem_at_nowrap(int64_t at) const {
    return ptr_[0].get()[at];
  }

  void
  concatenate() {
    if (!is_contiguous()) {
      auto ptr = std::unique_ptr<PRIMITIVE>(new PRIMITIVE[length()]);
      size_t new_length = length();
      int64_t next_panel = 0;
      for (int64_t i = 0;  i < ptr_.size();  i++) {
        memcpy(ptr.get() + next_panel, reinterpret_cast<void*>(ptr_[i].get()), length_[i]*sizeof(PRIMITIVE));
        next_panel += length_[i];
      }
      length_.clear();
      reserved_.clear();
      ptr_.clear();
      length_.push_back(new_length);
      reserved_.push_back(new_length);
      ptr_.push_back(std::move(ptr));
    }
  }

  int64_t is_contiguous() {
    return (ptr_.size() == 1);
  }

private:
  size_t initial_;
  std::vector<std::unique_ptr<PRIMITIVE>> ptr_;
  std::vector<size_t> length_;
  std::vector<size_t> reserved_;
};
  
#endif // GROWABLEBUFFER_H_