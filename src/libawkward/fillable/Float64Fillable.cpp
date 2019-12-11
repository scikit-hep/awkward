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
    return std::shared_ptr<Type>(new PrimitiveType(Type::Parameters(), PrimitiveType::float64));
  }

  const std::shared_ptr<Content> Float64Fillable::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(double) };
    return std::shared_ptr<Content>(new NumpyArray(Identity::none(), Type::none(), buffer_.ptr(), shape, strides, 0, sizeof(double), "d"));   // FIXME: Type::none()
  }

  bool Float64Fillable::active() const {
    return false;
  }

  Fillable* Float64Fillable::null() {
    Fillable* out = OptionFillable::fromvalids(options_, this);
    try {
      out->null();
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* Float64Fillable::boolean(bool x) {
    Fillable* out = UnionFillable::fromsingle(options_, this);
    try {
      out->boolean(x);
    }
    catch (...) {
      delete out;
      throw;
    }
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
    try {
      out->beginlist();
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* Float64Fillable::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  Fillable* Float64Fillable::begintuple(int64_t numfields) {
    Fillable* out = UnionFillable::fromsingle(options_, this);
    try {
      out->begintuple(numfields);
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* Float64Fillable::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  Fillable* Float64Fillable::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  Fillable* Float64Fillable::beginrecord(int64_t disambiguator) {
    Fillable* out = UnionFillable::fromsingle(options_, this);
    try {
      out->beginrecord(disambiguator);
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* Float64Fillable::field_fast(const char* key) {
    throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
  }

  Fillable* Float64Fillable::field_check(const char* key) {
    throw std::invalid_argument("called 'field_check' without 'beginrecord' at the same level before it");
  }

  Fillable* Float64Fillable::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

}
