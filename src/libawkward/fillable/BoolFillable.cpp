// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identity.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/fillable/MostlyValidFillable.h"

#include "awkward/fillable/BoolFillable.h"

namespace awkward {
  int64_t BoolFillable::length() const {
    return buffer_.length();
  }

  void BoolFillable::clear() {
    buffer_.clear();
  }

  const std::shared_ptr<Type> BoolFillable::type() const {
    return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::boolean));
  }

  const std::shared_ptr<Content> BoolFillable::snapshot() {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(bool) };
    return std::shared_ptr<Content>(new NumpyArray(Identity::none(), buffer_.ptr(), shape, strides, 0, sizeof(bool), "?"));
  }

  Fillable* BoolFillable::null() {
    Fillable* out = new MostlyValidFillable(this);
    out->null();
    return out;
  }

  Fillable* BoolFillable::boolean(bool x) {
    buffer_.append(x);
    return this;
  }
}
