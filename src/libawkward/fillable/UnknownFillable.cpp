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

  Fillable* UnknownFillable::null() {
    nullcount_++;
    return this;
  }

  Fillable* UnknownFillable::boolean(bool x) {
    Fillable* out = prepare<BoolFillable>();
    out->boolean(x);
    return out;
  }

  Fillable* UnknownFillable::integer(int64_t x) {
    Fillable* out = prepare<Int64Fillable>();
    out->integer(x);
    return out;
  }

  Fillable* UnknownFillable::real(double x) {
    Fillable* out = prepare<Float64Fillable>();
    out->real(x);
    return out;
  }

  Fillable* UnknownFillable::beginlist() {
    Fillable* out = prepare<ListFillable>();
    out->beginlist();
    return out;
  }

  Fillable* UnknownFillable::endlist() {
    return nullptr;
  }

  Fillable* UnknownFillable::beginrec() {
    throw std::runtime_error("FIXME: UnknownFillable::beginrec");
  }

  Fillable* UnknownFillable::reckey(const char* key) {
    throw std::runtime_error("FIXME: UnknownFillable::reckey(string)");
  }

  Fillable* UnknownFillable::reckey(int64_t index) {
    throw std::runtime_error("FIXME: UnknownFillable::reckey(int)");
  }

  Fillable* UnknownFillable::endrec() {
    throw std::runtime_error("FIXME: UnknownFillable::endrec");
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
