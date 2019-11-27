// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/type/UnknownType.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/BoolFillable.h"
#include "awkward/fillable/Int64Fillable.h"
#include "awkward/fillable/Float64Fillable.h"
#include "awkward/fillable/ListFillable.h"
#include "awkward/fillable/TupleFillable.h"
#include "awkward/fillable/RecordFillable.h"

#include "awkward/fillable/UnknownFillable.h"

namespace awkward {
  int64_t UnknownFillable::length() const {
    return nullcount_;
  }

  void UnknownFillable::clear() {
    nullcount_ = 0;
  }

  const std::shared_ptr<Type> UnknownFillable::type() const {
    return std::shared_ptr<Type>(new UnknownType);
  }

  const std::shared_ptr<Content> UnknownFillable::snapshot() const {
    if (nullcount_ == 0) {
      return std::shared_ptr<Content>(new EmptyArray(Identity::none()));
    }
    else {
      throw std::runtime_error("UnknownFillable::snapshot() needs MaskedArray");
    }
  }

  bool UnknownFillable::active() const {
    throw std::runtime_error("FIXME: UnknownFillable::active");
  }

  Fillable* UnknownFillable::null() {
    throw std::runtime_error("FIXME: UnknownFillable::null");
  }

  Fillable* UnknownFillable::boolean(bool x) {
    throw std::runtime_error("FIXME: UnknownFillable::boolean");
  }

  Fillable* UnknownFillable::integer(int64_t x) {
    throw std::runtime_error("FIXME: UnknownFillable::integer");
  }

  Fillable* UnknownFillable::real(double x) {
    throw std::runtime_error("FIXME: UnknownFillable::real");
  }

  Fillable* UnknownFillable::beginlist() {
    throw std::runtime_error("FIXME: UnknownFillable::beginlist");
  }

  Fillable* UnknownFillable::endlist() {
    throw std::runtime_error("FIXME: UnknownFillable::endlist");
  }

  Fillable* UnknownFillable::begintuple(int64_t numfields) {
    throw std::runtime_error("FIXME: UnknownFillable::begintuple");
  }

  Fillable* UnknownFillable::index(int64_t index) {
    throw std::runtime_error("FIXME: UnknownFillable::index");
  }

  Fillable* UnknownFillable::endtuple() {
    throw std::runtime_error("FIXME: UnknownFillable::endtuple");
  }

  Fillable* UnknownFillable::beginrecord(int64_t disambiguator) {
    throw std::runtime_error("FIXME: UnknownFillable::beginrecord");
  }

  Fillable* UnknownFillable::field_fast(const char* key) {
    throw std::runtime_error("FIXME: UnknownFillable::field_fast");
  }

  Fillable* UnknownFillable::field_check(const char* key) {
    throw std::runtime_error("FIXME: UnknownFillable::field_check");
  }

  Fillable* UnknownFillable::endrecord() {
    throw std::runtime_error("FIXME: UnknownFillable::endrecord");
  }

  template <typename T>
  Fillable* UnknownFillable::prepare() const {
    Fillable* out = new T(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    return out;
  }

}
