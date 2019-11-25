// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identity.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/UnionFillable.h"

#include "awkward/fillable/Float64Fillable.h"

namespace awkward {
  int64_t Float64Fillable::length() const {
    return buffer_.length();
  }

  void Float64Fillable::clear() {
    buffer_.clear();
  }

  const std::shared_ptr<Type> Float64Fillable::type() const {
    return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::float64));
  }

  const std::shared_ptr<Content> Float64Fillable::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(double) };
    return std::shared_ptr<Content>(new NumpyArray(Identity::none(), buffer_.ptr(), shape, strides, 0, sizeof(double), "d"));
  }

  Fillable* Float64Fillable::null() {
    Fillable* out = OptionFillable::fromvalids(options_, this);
    out->null();
    return out;
  }

  Fillable* Float64Fillable::boolean(bool x) {
    Fillable* out = UnionFillable::fromsingle(options_, this);
    out->boolean(x);
    return out;
  }

  Fillable* Float64Fillable::integer(int64_t x) {
    buffer_.append((double)x);
    return this;
  }

  Fillable* Float64Fillable::real(double x) {
    buffer_.append(x);
    return this;
  }

  Fillable* Float64Fillable::beginlist() {
    Fillable* out = UnionFillable::fromsingle(options_, this);
    out->beginlist();
    return out;
  }

  Fillable* Float64Fillable::endlist() {
    return nullptr;
  }

  Fillable* Float64Fillable::beginrec(const std::vector<std::string>& keys) {
    throw std::runtime_error("FIXME: Float64Fillable::beginrec");
  }

  Fillable* Float64Fillable::reckey(int64_t index) {
    throw std::runtime_error("FIXME: Float64Fillable::reckey(int)");
  }

  Fillable* Float64Fillable::endrec() {
    throw std::runtime_error("FIXME: Float64Fillable::endrec");
  }

}
