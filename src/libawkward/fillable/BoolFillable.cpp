// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identity.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/fillable/MostlyValidFillable.h"

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

  const std::shared_ptr<Content> BoolFillable::tolayout() {
    // std::shared_ptr<void> ptr = vector_to_sharedptr<uint8_t>(data_);

    uint8_t* rawptr = data_.data();
    std::shared_ptr<void> ptr(reinterpret_cast<void*>(rawptr), vector_deleter<uint8_t>(&data_));

    // bool* rawptr = new bool[data_.size()];
    // std::shared_ptr<void> ptr(rawptr, awkward::util::array_deleter<bool>());
    // std::copy(data_.begin(), data_.end(), rawptr);

    std::vector<ssize_t> shape = { (ssize_t)data_.size() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(bool) };
    return std::shared_ptr<Content>(new NumpyArray(Identity::none(), ptr, shape, strides, 0, sizeof(bool), "?"));
  }

  Fillable* BoolFillable::null() {
    Fillable* out = new MostlyValidFillable(this);
    out->null();
    return out;
  }

  Fillable* BoolFillable::boolean(bool x) {
    data_.push_back(x);
    return this;
  }
}
