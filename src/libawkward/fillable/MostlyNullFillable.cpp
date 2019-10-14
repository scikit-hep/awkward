// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/type/OptionType.h"

#include "awkward/fillable/MostlyNullFillable.h"

namespace awkward {
  int64_t MostlyNullFillable::length() const {
    return length_;
  }

  void MostlyNullFillable::clear() {
    validindex_.clear();
    content_.get()->clear();
    length_ = 0;
  }

  const std::shared_ptr<Type> MostlyNullFillable::type() const {
    return std::shared_ptr<Type>(new OptionType(content_.get()->type()));
  }

  const std::shared_ptr<Content> MostlyNullFillable::snapshot() {
    throw std::runtime_error("MostlyNullFillable::snapshot() needs MaskedArray");
    // FIXME: clear();
  }

  Fillable* MostlyNullFillable::null() {
    length_++;
  }

  Fillable* MostlyNullFillable::boolean(bool x) {
    validindex_.push_back(length_);
    content_.get()->boolean(x);
    length_++;
  }
}
