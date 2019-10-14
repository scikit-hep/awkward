// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/Identity.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"

#include "awkward/fillable/BoolFillable.h"

namespace awkward {
  int64_t BoolFillable::length() const {
    return (int64_t)data_.size();
  }

  void BoolFillable::clear() {
    data_.clear();
  }

  const std::shared_ptr<Type> BoolFillable::type() const {
    return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::boolean));
  }

  const std::shared_ptr<Content> BoolFillable::toarray() const {
    bool* rawptr = new bool[data_.size()];
    std::shared_ptr<void> ptr(rawptr, awkward::util::array_deleter<bool>());
    std::copy(std::begin(data_), std::end(data_), rawptr);
    std::vector<ssize_t> shape = { (ssize_t)data_.size() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(bool) };
    std::shared_ptr<Content> out(new NumpyArray(Identity::none(), ptr, shape, strides, 0, sizeof(bool), "?"));
  }

  Fillable* BoolFillable::boolean(bool x) {
    data_.push_back(x);
    return this;
  }
}
