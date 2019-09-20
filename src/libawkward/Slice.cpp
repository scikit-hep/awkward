// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Slice.h"

using namespace awkward;

const std::string SliceAt::tostring() const {
  return std::to_string(at_);
}

const std::string SliceRange::tostring() const {
  return (hasstart() ? std::to_string(start_) : std::string("")) + std::string(":") +
         (hasstop() ? std::to_string(stop_) : std::string("")) + std::string(":") +
         (hasstep() ? std::to_string(step_) : std::string(""));
}

const std::string SliceEllipsis::tostring() const {
  return std::string("...");
}

const std::string SliceNewAxis::tostring() const {
  return std::string("newaxis");
}

template <typename T>
const std::string SliceArrayOf<T>::tostring() const {
  return std::string("array(") + tostring_part() + std::string(")");
}

template <typename T>
const std::string SliceArrayOf<T>::tostring_part() const {
  std::stringstream out;
  out << "[";
  if (shape_.size() == 1) {
    for (int64_t i = 0;  i < shape_[0];  i++) {
      if (i != 0) {
        out << ", ";
      }
      out << (T)index_.get(i*strides_[0]);
    }
  }
  else {
    std::vector<int64_t> shape(shape_.begin() + 1, shape_.end());
    std::vector<int64_t> strides(strides_.begin() + 1, strides_.end());
    for (int64_t i = 0;  i < shape_[0];  i++) {
      if (i != 0) {
        out << ", ";
      }
      IndexOf<T> index(index_.ptr(), index_.offset() + i*strides_[0], shape_[1]);
      SliceArrayOf<T> subarray(index, shape, strides);
      out << subarray.tostring_part();
    }
  }
  out << "]";
  return out.str();
}

namespace awkward {
  template class SliceArrayOf<int64_t>;
}

const std::string Slice::tostring() const {
  std::stringstream out;
  out << "[";
  for (size_t i = 0;  i < items_.size();  i++) {
    if (i != 0) {
      out << ", ";
    }
    out << items_[i].get()->tostring();
  }
  out << "]";
  return out.str();
}

void Slice::append(const std::shared_ptr<SliceItem>& item) {
  items_.push_back(item);
}
