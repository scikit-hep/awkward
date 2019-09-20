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

int64_t Slice::length() const {
  return (int64_t)items_.size();
}

const std::shared_ptr<SliceItem> Slice::head() const {
  if (items_.size() != 0) {
    return items_[0];
  }
  else {
    return std::shared_ptr<SliceItem>(nullptr);
  }
}

const Slice Slice::tail() const {
  std::vector<std::shared_ptr<SliceItem>> items;
  if (items_.size() != 0) {
    items.insert(items.end(), items_.begin() + 1, items_.end());
  }
  return Slice(items, true);
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
  assert(!sealed_);
  items_.push_back(item);
}

void Slice::seal() {
  assert(!sealed_);

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

    std::string types;
    for (int64_t i = 0;  i < items_.size();  i++) {
      if (dynamic_cast<SliceAt*>(items_[i].get()) != nullptr) {
        types.push_back('@');
      }
      else if (dynamic_cast<SliceRange*>(items_[i].get()) != nullptr) {
        types.push_back(':');
      }
      else if (dynamic_cast<SliceEllipsis*>(items_[i].get()) != nullptr) {
        types.push_back('.');
      }
      else if (dynamic_cast<SliceNewAxis*>(items_[i].get()) != nullptr) {
        types.push_back('1');
      }
      else if (dynamic_cast<SliceArray64*>(items_[i].get()) != nullptr) {
        types.push_back('A');
      }
    }

    if (std::count(types.begin(), types.end(), '.') > 1) {
      throw std::invalid_argument("a slice can have no more than one ellipsis ('...')");
    }

    int64_t numadvanced = std::count(types.begin(), types.end(), 'A');
    if (numadvanced != 0) {
      types = types.substr(0, types.find_last_of("A") + 1).substr(types.find_first_of("A"));
      if (numadvanced != types.size()) {
        throw std::invalid_argument("advanced indexes separated by basic indexes is not permitted (simple integers are advanced when any arrays are present)");
      }
    }
  }

  sealed_ = true;
}

bool Slice::isadvanced() const {
  assert(sealed_);
  for (int64_t i = 0;  i < items_.size();  i++) {
    if (dynamic_cast<SliceArray64*>(items_[i].get()) != nullptr) {
      return true;
    }
  }
  return false;
}
