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

template <typename T>
const std::string SliceArrayOf<T>::tostring() const {
  std::stringstream out;
  out << "array([";
  out << "])";
  return out.str();
}

const std::string SliceEllipsis::tostring() const {
  return std::string("...");
}

const std::string SliceNewAxis::tostring() const {
  return std::string("newaxis");
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
