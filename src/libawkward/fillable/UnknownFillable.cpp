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
      throw std::runtime_error("UnknownFillable::snapshot() needs OptionArray");
    }
  }

  bool UnknownFillable::active() const {
    return false;
  }

  Fillable* UnknownFillable::null() {
    nullcount_++;
    return this;
  }

  Fillable* UnknownFillable::boolean(bool x) {
    Fillable* out = BoolFillable::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    try {
      out->boolean(x);
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* UnknownFillable::integer(int64_t x) {
    Fillable* out = Int64Fillable::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    try {
      out->integer(x);
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* UnknownFillable::real(double x) {
    Fillable* out = Float64Fillable::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    try {
      out->real(x);
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* UnknownFillable::beginlist() {
    Fillable* out = ListFillable::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    try {
      out->beginlist();
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* UnknownFillable::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  Fillable* UnknownFillable::begintuple(int64_t numfields) {
    Fillable* out = TupleFillable::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    try {
      out->begintuple(numfields);
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* UnknownFillable::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  Fillable* UnknownFillable::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  Fillable* UnknownFillable::beginrecord(int64_t disambiguator) {
    Fillable* out = RecordFillable::fromempty(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    try {
      out->beginrecord(disambiguator);
    }
    catch (...) {
      delete out;
      throw;
    }
    return out;
  }

  Fillable* UnknownFillable::field_fast(const char* key) {
    throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
  }

  Fillable* UnknownFillable::field_check(const char* key) {
    throw std::invalid_argument("called 'field_check' without 'beginrecord' at the same level before it");
  }

  Fillable* UnknownFillable::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

}
