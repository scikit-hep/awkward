// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/type/OptionType.h"

#include "awkward/fillable/OptionFillable.h"

namespace awkward {
  int64_t OptionFillable::length() const {
    return index_.length();
  }

  void OptionFillable::clear() {
    index_.clear();
    content_.get()->clear();
  }

  const std::shared_ptr<Type> OptionFillable::type() const {
    return std::shared_ptr<Type>(new OptionType(content_.get()->type()));
  }

  const std::shared_ptr<Content> OptionFillable::snapshot() const {
    throw std::runtime_error("OptionFillable::snapshot() needs MaskedArray");
  }

  bool OptionFillable::active() const {
    throw std::runtime_error("FIXME: OptionFillable::active");
  }

  Fillable* OptionFillable::null() {
    throw std::runtime_error("FIXME: OptionFillable::null");
  }

  Fillable* OptionFillable::boolean(bool x) {
    throw std::runtime_error("FIXME: OptionFillable::boolean");
  }

  Fillable* OptionFillable::integer(int64_t x) {
    throw std::runtime_error("FIXME: OptionFillable::integer");
  }

  Fillable* OptionFillable::real(double x) {
    throw std::runtime_error("FIXME: OptionFillable::real");
  }

  Fillable* OptionFillable::beginlist() {
    throw std::runtime_error("FIXME: OptionFillable::beginlist");
  }

  Fillable* OptionFillable::endlist() {
    throw std::runtime_error("FIXME: OptionFillable::endlist");
  }

  Fillable* OptionFillable::begintuple(int64_t numfields) {
    throw std::runtime_error("FIXME: OptionFillable::begintuple");
  }

  Fillable* OptionFillable::index(int64_t index) {
    throw std::runtime_error("FIXME: OptionFillable::index");
  }

  Fillable* OptionFillable::endtuple() {
    throw std::runtime_error("FIXME: OptionFillable::endtuple");
  }

  Fillable* OptionFillable::beginrecord(int64_t disambiguator) {
    throw std::runtime_error("FIXME: OptionFillable::beginrecord");
  }

  Fillable* OptionFillable::field_fast(const char* key) {
    throw std::runtime_error("FIXME: OptionFillable::field_fast");
  }

  Fillable* OptionFillable::field_check(const char* key) {
    throw std::runtime_error("FIXME: OptionFillable::field_check");
  }

  Fillable* OptionFillable::endrecord() {
    throw std::runtime_error("FIXME: OptionFillable::endrecord");
  }

  void OptionFillable::maybeupdate(Fillable* tmp) {
    if (tmp != content_.get()  &&  tmp != nullptr) {
      content_ = std::shared_ptr<Fillable>(tmp);
    }
  }
}
