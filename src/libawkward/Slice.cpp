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
    if (shape_[0] < 6) {
      for (int64_t i = 0;  i < shape_[0];  i++) {
        if (i != 0) {
          out << ", ";
        }
        out << (T)index_.get(i*strides_[0]);
      }
    }
    else {
      for (int64_t i = 0;  i < 3;  i++) {
        if (i != 0) {
          out << ", ";
        }
        out << (T)index_.get(i*strides_[0]);
      }
      out << ", ..., ";
      for (int64_t i = shape_[0] - 3;  i < shape_[0];  i++) {
        if (i != shape_[0] - 3) {
          out << ", ";
        }
        out << (T)index_.get(i*strides_[0]);
      }
    }
  }
  else {
    std::vector<int64_t> shape(shape_.begin() + 1, shape_.end());
    std::vector<int64_t> strides(strides_.begin() + 1, strides_.end());
    if (shape_[0] < 6) {
      for (int64_t i = 0;  i < shape_[0];  i++) {
        if (i != 0) {
          out << ", ";
        }
        IndexOf<T> index(index_.ptr(), index_.offset() + i*strides_[0], shape_[1]);
        SliceArrayOf<T> subarray(index, shape, strides);
        out << subarray.tostring_part();
      }
    }
    else {
      for (int64_t i = 0;  i < 3;  i++) {
        if (i != 0) {
          out << ", ";
        }
        IndexOf<T> index(index_.ptr(), index_.offset() + i*strides_[0], shape_[1]);
        SliceArrayOf<T> subarray(index, shape, strides);
        out << subarray.tostring_part();
      }
      out << ", ..., ";
      for (int64_t i = shape_[0] - 3;  i < shape_[0];  i++) {
        if (i != shape_[0] - 3) {
          out << ", ";
        }
        IndexOf<T> index(index_.ptr(), index_.offset() + i*strides_[0], shape_[1]);
        SliceArrayOf<T> subarray(index, shape, strides);
        out << subarray.tostring_part();
      }
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

#include <iostream>

void Slice::broadcast() {
  std::cout << "broadcast" << std::endl;

  std::vector<int64_t> shape;
  for (int64_t i = 0;  i < items_.size();  i++) {
    if (SliceArray64* array = dynamic_cast<SliceArray64*>(items_[i].get())) {
      if (shape.size() == 0) {
        shape = array->shape();
      }
      else if (shape.size() != array->ndim()) {
        throw std::invalid_argument("cannot broadcast arrays in slice");
      }
      else {
        std::vector<int64_t> arrayshape = array->shape();
        for (int64_t j = 0;  j < shape.size();  j++) {
          if (arrayshape[j] > shape[j]) {
            shape[j] = arrayshape[j];
          }
        }
      }
    }
  }

  if (shape.size() != 0) {
    std::cout << "shape ";
    for (auto x : shape) {
      std::cout << x << " ";
    }
    std::cout << std::endl;

    for (int64_t i = 0;  i < items_.size();  i++) {
      if (SliceAt* at = dynamic_cast<SliceAt*>(items_[i].get())) {
        Index64 index(1);
        index.ptr().get()[0] = at->at();
        std::vector<int64_t> strides;
        for (int64_t j = 0;  j < shape.size();  j++) {
          strides.push_back(0);
        }
        items_[i] = std::shared_ptr<SliceItem>(new SliceArray64(index, shape, strides));
      }
      else if (SliceArray64* array = dynamic_cast<SliceArray64*>(items_[i].get())) {
        std::vector<int64_t> arrayshape = array->shape();
        std::vector<int64_t> arraystrides = array->strides();
        std::vector<int64_t> strides;
        for (int64_t j = 0;  j < shape.size();  j++) {
          if (arrayshape[j] == shape[j]) {
            strides.push_back(arraystrides[j]);
          }
          else if (arrayshape[j] == 1) {
            strides.push_back(0);
          }
          else {
            throw std::invalid_argument("cannot broadcast arrays in slice");
          }
        }
        items_[i] = std::shared_ptr<SliceItem>(new SliceArray64(array->index(), shape, strides));
      }
    }

    // more checks...

  }
}
