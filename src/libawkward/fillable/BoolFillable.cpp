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

  bool BoolFillable::active() const {
    throw std::runtime_error("FIXME: BoolFillable::active");
  }

  Fillable* BoolFillable::null() {
    throw std::runtime_error("FIXME: BoolFillable::null");
  }

  Fillable* BoolFillable::boolean(bool x) {
    throw std::runtime_error("FIXME: BoolFillable::boolean");
  }

  Fillable* BoolFillable::integer(int64_t x) {
    throw std::runtime_error("FIXME: BoolFillable::integer");
  }

  Fillable* BoolFillable::real(double x) {
    throw std::runtime_error("FIXME: BoolFillable::real");
  }

  Fillable* BoolFillable::beginlist() {
    throw std::runtime_error("FIXME: BoolFillable::beginlist");
  }

  Fillable* BoolFillable::endlist() {
    throw std::runtime_error("FIXME: BoolFillable::endlist");
  }

  Fillable* BoolFillable::begintuple(int64_t numfields) {
    throw std::runtime_error("FIXME: BoolFillable::begintuple");
  }

  Fillable* BoolFillable::index(int64_t index) {
    throw std::runtime_error("FIXME: BoolFillable::index");
  }

  Fillable* BoolFillable::endtuple() {
    throw std::runtime_error("FIXME: BoolFillable::endtuple");
  }

  Fillable* BoolFillable::beginrecord(int64_t disambiguator) {
    throw std::runtime_error("FIXME: BoolFillable::beginrecord");
  }

  Fillable* BoolFillable::field_fast(const char* key) {
    throw std::runtime_error("FIXME: BoolFillable::field_fast");
  }

  Fillable* BoolFillable::field_check(const char* key) {
    throw std::runtime_error("FIXME: BoolFillable::field_check");
  }

  Fillable* BoolFillable::endrecord() {
    throw std::runtime_error("FIXME: BoolFillable::endrecord");
  }

}
