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
    return std::shared_ptr<Content>(new NumpyArray(Identity::none(), Type::none(), buffer_.ptr(), shape, strides, 0, sizeof(int64_t), "q"));
#else
    return std::shared_ptr<Content>(new NumpyArray(Identity::none(), Type::none(), buffer_.ptr(), shape, strides, 0, sizeof(int64_t), "l"));
#endif
  }

  bool Int64Fillable::active() const {
    return false;
  }

  Fillable* Int64Fillable::null() {
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

  Fillable* Int64Fillable::boolean(bool x) {
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

  Fillable* Int64Fillable::integer(int64_t x) {
    buffer_.append(x);
    return this;
  }

  Fillable* Int64Fillable::real(double x) {
    Float64Fillable* out = Float64Fillable::fromint64(options_, buffer_);
    try {
      out->real(x);
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* Int64Fillable::beginlist() {
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

  Fillable* Int64Fillable::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  Fillable* Int64Fillable::begintuple(int64_t numfields) {
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

  Fillable* Int64Fillable::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  Fillable* Int64Fillable::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  Fillable* Int64Fillable::beginrecord(int64_t disambiguator) {
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

  Fillable* Int64Fillable::field_fast(const char* key) {
    throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
  }

  Fillable* Int64Fillable::field_check(const char* key) {
    throw std::invalid_argument("called 'field_check' without 'beginrecord' at the same level before it");
  }

  Fillable* Int64Fillable::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

}
