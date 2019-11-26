// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/type/UnknownType.h"
#include "awkward/fillable/FillableArray.h"
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

  Fillable* UnknownFillable::begintuple(int64_t numfields) {
    Fillable* out = prepare<TupleFillable>();
    out->begintuple(numfields);
    return out;
  }

  Fillable* UnknownFillable::index(int64_t index) {
    throw std::invalid_argument("'index' should only be called in a tuple (did you forget to call 'begintuple'?)");
  }

  Fillable* UnknownFillable::endtuple() {
    throw std::invalid_argument("'endtuple' should only be called in a tuple (did you forget to call 'begintuple'?)");
  }

  template <typename T>
  Fillable* UnknownFillable::prepare() const {
    Fillable* out = new T(fillablearray_, options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(fillablearray_, options_, nullcount_, out);
    }
    return out;
  }

}
