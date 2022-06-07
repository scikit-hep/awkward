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

  const std::unique_ptr<PRIMITIVE>
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

  // when length of each panel is different
  /*PRIMITIVE
  getitem_at_nowrap(int64_t at, int64_t index) const {
    if (at<length_[index])
      return ptr_[index].get()[at%length_[index]];
    return getitem_at_nowrap(at - length_[index], index+1);
  }*/

  PRIMITIVE
  getitem_at_nowrap(int64_t at) const {
    return ptr_[0].get()[at];
    return ptr_[floor(at/initial_)].get()[at%initial_];
  }

  void
  concatenate() {
    auto ptr = std::unique_ptr<PRIMITIVE>(new PRIMITIVE[length()]);
    size_t new_length = length();
    int64_t next_panel = 0;
    for (int64_t i = 0;  i < ptr_.size();  i++) {
      std::cout << ptr_[i].get()[0] << ", " << length_[i] << std::endl;
      memcpy(ptr.get() + next_panel, reinterpret_cast<void*>(ptr_[i].get()), length_[i]*sizeof(PRIMITIVE));
      next_panel += length_[i];
    }
    clear();
    ptr_[0] = std::move(ptr);
    length_[0] = new_length;
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

int main(int argc, const char * argv[]) {
    int data_size = 13;
    double data[13] = { 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
        2.1, 2.2, 2.3, 2.4};

    size_t initial = 4;
    GrowableBuffer<float> buffer(initial);
    for (int i = 0; i < data_size; i++) {
        buffer.append(data[i]);
    }
    buffer.concatenate();
    for (int at = 0; at < buffer.length(); at++) {
      std::cout << buffer.getitem_at_nowrap(at) << ", ";
    }

    return 0;
}
