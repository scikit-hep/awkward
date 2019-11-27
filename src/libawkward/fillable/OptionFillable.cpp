// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/Index.h"
#include "awkward/type/OptionType.h"

#include "awkward/fillable/OptionFillable.h"

namespace awkward {
  int64_t OptionFillable::length() const {
    return offsets_.length();
  }

  void OptionFillable::clear() {
    offsets_.clear();
    content_.get()->clear();
  }

  const std::shared_ptr<Type> OptionFillable::type() const {
    Index64 offsets(offsets_.ptr(), 0, offsets_.length());
    return std::shared_ptr<Type>(new OptionType(content_.get()->type()));
  }

  const std::shared_ptr<Content> OptionFillable::snapshot() const {
    throw std::runtime_error("OptionFillable::snapshot() needs OptionArray");
  }

  bool OptionFillable::active() const {
    return content_.get()->active();
  }

  Fillable* OptionFillable::null() {
    if (!content_.get()->active()) {
      offsets_.append(-1);
    }
    else {
      content_.get()->null();
    }
    return this;
  }

  Fillable* OptionFillable::boolean(bool x) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->boolean(x));
      offsets_.append(length);
    }
    else {
      content_.get()->boolean(x);
    }
    return this;
  }

  Fillable* OptionFillable::integer(int64_t x) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->integer(x));
      offsets_.append(length);
    }
    else {
      content_.get()->integer(x);
    }
    return this;
  }

  Fillable* OptionFillable::real(double x) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->real(x));
      offsets_.append(length);
    }
    else {
      content_.get()->real(x);
    }
    return this;
  }

  Fillable* OptionFillable::beginlist() {
    if (!content_.get()->active()) {
      maybeupdate(content_.get()->beginlist());
    }
    else {
      content_.get()->beginlist();
    }
    return this;
  }

  Fillable* OptionFillable::endlist() {
    if (!content_.get()->active()) {
      throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
    }
    else {
      int64_t length = content_.get()->length();
      content_.get()->endlist();
      if (length != content_.get()->length()) {
        offsets_.append(length);
      }
    }
    return this;
  }

  Fillable* OptionFillable::begintuple(int64_t numfields) {
    if (!content_.get()->active()) {
      maybeupdate(content_.get()->begintuple(numfields));
    }
    else {
      content_.get()->begintuple(numfields);
    }
    return this;
  }

  Fillable* OptionFillable::index(int64_t index) {
    if (!content_.get()->active()) {
      throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
    }
    else {
      content_.get()->index(index);
    }
    return this;
  }

  Fillable* OptionFillable::endtuple() {
    if (!content_.get()->active()) {
      throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
    }
    else {
      int64_t length = content_.get()->length();
      content_.get()->endtuple();
      if (length != content_.get()->length()) {
        offsets_.append(length);
      }
    }
    return this;
  }

  Fillable* OptionFillable::beginrecord(int64_t disambiguator) {
    if (!content_.get()->active()) {
      maybeupdate(content_.get()->beginrecord(disambiguator));
    }
    else {
      content_.get()->beginrecord(disambiguator);
    }
    return this;
  }

  Fillable* OptionFillable::field_fast(const char* key) {
    if (!content_.get()->active()) {
      throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
    }
    else {
      content_.get()->field_fast(key);
    }
    return this;
  }

  Fillable* OptionFillable::field_check(const char* key) {
    if (!content_.get()->active()) {
      throw std::invalid_argument("called 'field_check' without 'beginrecord' at the same level before it");
    }
    else {
      content_.get()->field_check(key);
    }
    return this;
  }

  Fillable* OptionFillable::endrecord() {
    if (!content_.get()->active()) {
      throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
    }
    else {
      int64_t length = content_.get()->length();
      content_.get()->endrecord();
      if (length != content_.get()->length()) {
        offsets_.append(length);
      }
    }
    return this;
  }

  void OptionFillable::maybeupdate(Fillable* tmp) {
    if (tmp != content_.get()) {
      content_ = std::shared_ptr<Fillable>(tmp);
    }
  }
}
