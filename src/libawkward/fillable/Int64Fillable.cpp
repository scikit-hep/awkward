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

  bool Int64Fillable::active() const {
    throw std::runtime_error("FIXME: Int64Fillable::active");
  }

  Fillable* Int64Fillable::null() {
    throw std::runtime_error("FIXME: Int64Fillable::null");
  }

  Fillable* Int64Fillable::boolean(bool x) {
    throw std::runtime_error("FIXME: Int64Fillable::boolean");
  }

  Fillable* Int64Fillable::integer(int64_t x) {
    throw std::runtime_error("FIXME: Int64Fillable::integer");
  }

  Fillable* Int64Fillable::real(double x) {
    throw std::runtime_error("FIXME: Int64Fillable::real");
  }

  Fillable* Int64Fillable::beginlist() {
    throw std::runtime_error("FIXME: Int64Fillable::beginlist");
  }

  Fillable* Int64Fillable::endlist() {
    throw std::runtime_error("FIXME: Int64Fillable::endlist");
  }

  Fillable* Int64Fillable::begintuple(int64_t numfields) {
    throw std::runtime_error("FIXME: Int64Fillable::begintuple");
  }

  Fillable* Int64Fillable::index(int64_t index) {
    throw std::runtime_error("FIXME: Int64Fillable::index");
  }

  Fillable* Int64Fillable::endtuple() {
    throw std::runtime_error("FIXME: Int64Fillable::endtuple");
  }

  Fillable* Int64Fillable::beginrecord(int64_t disambiguator) {
    throw std::runtime_error("FIXME: Int64Fillable::beginrecord");
  }

  Fillable* Int64Fillable::field_fast(const char* key) {
    throw std::runtime_error("FIXME: Int64Fillable::field_fast");
  }

  Fillable* Int64Fillable::field_check(const char* key) {
    throw std::runtime_error("FIXME: Int64Fillable::field_check");
  }

  Fillable* Int64Fillable::endrecord() {
    throw std::runtime_error("FIXME: Int64Fillable::endrecord");
  }

}
