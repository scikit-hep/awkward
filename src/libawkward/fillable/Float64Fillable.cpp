// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identity.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/fillable/FillableArray.h"
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

  bool Float64Fillable::active() const {
    throw std::runtime_error("FIXME: Float64Fillable::active");
  }

  Fillable* Float64Fillable::null() {
    throw std::runtime_error("FIXME: Float64Fillable::null");
  }

  Fillable* Float64Fillable::boolean(bool x) {
    throw std::runtime_error("FIXME: Float64Fillable::boolean");
  }

  Fillable* Float64Fillable::integer(int64_t x) {
    throw std::runtime_error("FIXME: Float64Fillable::integer");
  }

  Fillable* Float64Fillable::real(double x) {
    throw std::runtime_error("FIXME: Float64Fillable::real");
  }

  Fillable* Float64Fillable::beginlist() {
    throw std::runtime_error("FIXME: Float64Fillable::beginlist");
  }

  Fillable* Float64Fillable::endlist() {
    throw std::runtime_error("FIXME: Float64Fillable::endlist");
  }

  Fillable* Float64Fillable::begintuple(int64_t numfields) {
    throw std::runtime_error("FIXME: Float64Fillable::begintuple");
  }

  Fillable* Float64Fillable::index(int64_t index) {
    throw std::runtime_error("FIXME: Float64Fillable::index");
  }

  Fillable* Float64Fillable::endtuple() {
    throw std::runtime_error("FIXME: Float64Fillable::endtuple");
  }

  Fillable* Float64Fillable::beginrecord(int64_t disambiguator) {
    throw std::runtime_error("FIXME: Float64Fillable::beginrecord");
  }

  Fillable* Float64Fillable::field_fast(const char* key) {
    throw std::runtime_error("FIXME: Float64Fillable::field_fast");
  }

  Fillable* Float64Fillable::field_check(const char* key) {
    throw std::runtime_error("FIXME: Float64Fillable::field_check");
  }

  Fillable* Float64Fillable::endrecord() {
    throw std::runtime_error("FIXME: Float64Fillable::endrecord");
  }

}
