// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identity.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/UnionFillable.h"
#include "awkward/fillable/Float64Fillable.h"

#include "awkward/fillable/Int64Fillable.h"

namespace awkward {
  int64_t Int64Fillable::length() const {
    return buffer_.length();
  }

  void Int64Fillable::clear() {
    buffer_.clear();
  }

  const std::shared_ptr<Type> Int64Fillable::type() const {
    return std::shared_ptr<Type>(new PrimitiveType(PrimitiveType::int64));
  }

  const std::shared_ptr<Content> Int64Fillable::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(int64_t) };
#ifdef _MSC_VER
    return std::shared_ptr<Content>(new NumpyArray(Identity::none(), buffer_.ptr(), shape, strides, 0, sizeof(int64_t), "q"));
#else
    return std::shared_ptr<Content>(new NumpyArray(Identity::none(), buffer_.ptr(), shape, strides, 0, sizeof(int64_t), "l"));
#endif
  }

  Fillable* Int64Fillable::null() {
    Fillable* out = OptionFillable::fromvalids(options_, this);
    out->null();
    return out;
  }

  Fillable* Int64Fillable::boolean(bool x) {
    Fillable* out = UnionFillable::fromsingle(options_, this);
    out->boolean(x);
    return out;
  }

  Fillable* Int64Fillable::integer(int64_t x) {
    buffer_.append(x);
    return this;
  }

  Fillable* Int64Fillable::real(double x) {
    Float64Fillable* out = Float64Fillable::fromint64(options_, buffer_);
    out->real(x);
    return out;
  }

  Fillable* Int64Fillable::beginlist() {
    Fillable* out = UnionFillable::fromsingle(options_, this);
    out->beginlist();
    return out;
  }

  Fillable* Int64Fillable::endlist() {
    return nullptr;
  }

  Fillable* Int64Fillable::beginrec() {
    throw std::runtime_error("FIXME: Int64Fillable::beginrec");
  }

  Fillable* Int64Fillable::reckey(const char* key) {
    throw std::runtime_error("FIXME: Int64Fillable::reckey(string)");
  }

  Fillable* Int64Fillable::reckey(int64_t index) {
    throw std::runtime_error("FIXME: Int64Fillable::reckey(int)");
  }

  Fillable* Int64Fillable::endrec() {
    throw std::runtime_error("FIXME: Int64Fillable::endrec");
  }

}
