// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identity.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
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
    return std::shared_ptr<Content>(new NumpyArray(Identity::none(), Type::none(), buffer_.ptr(), shape, strides, 0, sizeof(bool), "?"));   // FIXME: Type::none()
  }

  bool BoolFillable::active() const {
    return false;
  }

  Fillable* BoolFillable::null() {
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

  Fillable* BoolFillable::boolean(bool x) {
    buffer_.append(x);
    return this;
  }

  Fillable* BoolFillable::integer(int64_t x) {
    Fillable* out = UnionFillable::fromsingle(options_, this);
    try {
      out->integer(x);
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* BoolFillable::real(double x) {
    Fillable* out = UnionFillable::fromsingle(options_, this);
    try {
      out->real(x);
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* BoolFillable::beginlist() {
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

  Fillable* BoolFillable::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  Fillable* BoolFillable::begintuple(int64_t numfields) {
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

  Fillable* BoolFillable::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  Fillable* BoolFillable::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  Fillable* BoolFillable::beginrecord(int64_t disambiguator) {
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

  Fillable* BoolFillable::field_fast(const char* key) {
    throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
  }

  Fillable* BoolFillable::field_check(const char* key) {
    throw std::invalid_argument("called 'field_check' without 'beginrecord' at the same level before it");
  }

  Fillable* BoolFillable::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

}
