// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/type/OptionType.h"

#include "awkward/fillable/MostlyValidFillable.h"

namespace awkward {
  int64_t MostlyValidFillable::length() const {
    return length_;
  }

  void MostlyValidFillable::clear() {
    nullindex_.clear();
    content_.get()->clear();
    length_ = 0;
  }

  const std::shared_ptr<Type> MostlyValidFillable::type() const {
    return std::shared_ptr<Type>(new OptionType(content_.get()->type()));
  }

  const std::shared_ptr<Content> MostlyValidFillable::snapshot() {
    throw std::runtime_error("MostlyValidFillable::snapshot() needs MaskedArray");
    // FIXME: clear();
  }

  Fillable* MostlyValidFillable::null() {
    nullindex_.push_back(length_);
    length_++;
  }

  Fillable* MostlyValidFillable::boolean(bool x) {
    content_.get()->boolean(x);
    length_++;
  }
}
