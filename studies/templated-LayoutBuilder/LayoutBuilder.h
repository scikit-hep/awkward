// only needed for 'dump', which is a temporary debugging tool
#include <iostream>
#include <string>

// really needed
#include <stdint.h>
#include <vector>

template <typename PRIMITIVE>
class GrowableBuffer {
public:
  void append(PRIMITIVE datum) {
    data_.push_back(datum);
  }

  int64_t length() const {
    return data_.size();
  }

  void dump() const {
    for (auto x : data_) {
      std::cout << x << " ";
    }
  }

private:
  std::vector<PRIMITIVE> data_;
};


template <typename PRIMITIVE>
class NumpyLayoutBuilder {
public:
  void append(PRIMITIVE x) {
    data_.append(x);
  }

  int64_t length() const {
    return data_.length();
  }

  void dump(std::string indent) const {
    std::cout << indent << "NumpyLayoutBuilder" << std::endl;
    std::cout << indent << "    data ";
    data_.dump();
    std::cout << std::endl;
  }

private:
  GrowableBuffer<PRIMITIVE> data_;
};


template <typename INDEX, typename BUILDER>
class ListOffsetLayoutBuilder {
public:
  ListOffsetLayoutBuilder() {
    offsets_.append(0);
  }

  BUILDER* begin_list() {
    return &content_;
  }

  void end_list() {
    offsets_.append(content_.length());
  }

  void dump(std::string indent) const {
    std::cout << indent << "ListOffsetLayoutBuilder" << std::endl;
    std::cout << indent << "    offsets ";
    offsets_.dump();
    std::cout << std::endl;
    content_.dump("    ");
  }

private:
  GrowableBuffer<INDEX> offsets_;
  BUILDER content_;
};
