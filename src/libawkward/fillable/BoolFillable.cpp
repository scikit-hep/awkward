// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identity.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/fillable/FillableArray.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/UnionFillable.h"

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

  const std::shared_ptr<Content> BoolFillable::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(bool) };
    return std::shared_ptr<Content>(new NumpyArray(Identity::none(), buffer_.ptr(), shape, strides, 0, sizeof(bool), "?"));
  }

  Fillable* BoolFillable::null() {
    Fillable* out = OptionFillable::fromvalids(fillablearray_, options_, this);
    out->null();
    return out;
  }

  Fillable* BoolFillable::boolean(bool x) {
    buffer_.append(x);
    return this;
  }

  Fillable* BoolFillable::integer(int64_t x) {
    Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
    out->integer(x);
    return out;
  }

  Fillable* BoolFillable::real(double x) {
    Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
    out->real(x);
    return out;
  }

  Fillable* BoolFillable::beginlist() {
    Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
    out->beginlist();
    return out;
  }

  Fillable* BoolFillable::endlist() {
    return nullptr;
  }

  Fillable* BoolFillable::begintuple(int64_t numfields) {
    Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
    out->begintuple(numfields);
    return out;
  }

  Fillable* BoolFillable::index(int64_t index) {
    throw std::invalid_argument("'index' should only be called in a tuple (did you forget to call 'begintuple'?)");
  }

  Fillable* BoolFillable::endtuple() {
    throw std::invalid_argument("'endtuple' should only be called in a tuple (did you forget to call 'begintuple'?)");
  }

}
