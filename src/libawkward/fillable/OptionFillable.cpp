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

  Fillable* OptionFillable::null() {
    index_.append(-1);
    return this;
  }

  Fillable* OptionFillable::boolean(bool x) {
    int64_t length = content_.get()->length();
    maybeupdate(content_.get()->boolean(x));
    index_.append(length);
    return this;
  }

  Fillable* OptionFillable::integer(int64_t x) {
    int64_t length = content_.get()->length();
    maybeupdate(content_.get()->integer(x));
    index_.append(length);
    return this;
  }

  Fillable* OptionFillable::real(double x) {
    int64_t length = content_.get()->length();
    maybeupdate(content_.get()->real(x));
    index_.append(length);
    return this;
  }

  Fillable* OptionFillable::beginlist() {
    int64_t length = content_.get()->length();
    maybeupdate(content_.get()->beginlist());
    index_.append(length);
    return this;
  }

  Fillable* OptionFillable::endlist() {
    return nullptr;
  }

  Fillable* OptionFillable::beginrec(const Slots* slots) {
    throw std::runtime_error("FIXME: OptionFillable::beginrec");
  }

  Fillable* OptionFillable::reckey(int64_t index) {
    throw std::runtime_error("FIXME: OptionFillable::reckey(int)");
  }

  Fillable* OptionFillable::endrec() {
    throw std::runtime_error("FIXME: OptionFillable::endrec");
  }

  void OptionFillable::maybeupdate(Fillable* tmp) {
    if (tmp != content_.get()  &&  tmp != nullptr) {
      content_ = std::shared_ptr<Fillable>(tmp);
    }
  }
}
