// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/Index.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/type/ListType.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/UnionFillable.h"

#include "awkward/fillable/ListFillable.h"

namespace awkward {
  int64_t ListFillable::length() const {
    return offsets_.length() - 1;
  }

  void ListFillable::clear() {
    offsets_.clear();
    content_.get()->clear();
  }

  const std::shared_ptr<Type> ListFillable::type() const {
    return std::shared_ptr<Type>(new ListType(content_.get()->type()));
  }

  const std::shared_ptr<Content> ListFillable::snapshot() const {
    Index64 offsets(offsets_.ptr(), 0, offsets_.length());
    return std::shared_ptr<Content>(new ListOffsetArray64(Identity::none(), Type::none(), offsets, content_.get()->snapshot()));
  }

  bool ListFillable::active() const {
    return begun_;
  }

  Fillable* ListFillable::null() {
    if (!begun_) {
      Fillable* out = OptionFillable::fromvalids(options_, this);
      try {
        out->null();
      }
      catch (...) {
        delete out;
        throw;
      }
      return out;
    }
    else {
      maybeupdate(content_.get()->null());
      return this;
    }
  }

  Fillable* ListFillable::boolean(bool x) {
    if (!begun_) {
      Fillable* out = UnionFillable::fromsingle(options_, this);
      try {
        out->boolean(x);
      }
      catch (...) {
        delete out;
        throw;
      }
      return out;
    }
    else {
      maybeupdate(content_.get()->boolean(x));
      return this;
    }
  }

  Fillable* ListFillable::integer(int64_t x) {
    if (!begun_) {
      Fillable* out = UnionFillable::fromsingle(options_, this);
      try {
        out->integer(x);
      }
      catch (...) {
        delete out;
        throw;
      }
      return out;
    }
    else {
      maybeupdate(content_.get()->integer(x));
      return this;
    }
  }

  Fillable* ListFillable::real(double x) {
    if (!begun_) {
      Fillable* out = UnionFillable::fromsingle(options_, this);
      try {
        out->real(x);
      }
      catch (...) {
        delete out;
        throw;
      }
      return out;
    }
    else {
      maybeupdate(content_.get()->real(x));
      return this;
    }
  }

  Fillable* ListFillable::beginlist() {
    if (!begun_) {
      begun_ = true;
    }
    else {
      maybeupdate(content_.get()->beginlist());
    }
    return this;
  }

  Fillable* ListFillable::endlist() {
    if (!begun_) {
      throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
    }
    else if (!content_.get()->active()) {
      offsets_.append(content_.get()->length());
      begun_ = false;
    }
    else {
      maybeupdate(content_.get()->endlist());
    }
    return this;
  }

  Fillable* ListFillable::begintuple(int64_t numfields) {
    if (!begun_) {
      Fillable* out = UnionFillable::fromsingle(options_, this);
      try {
        out->begintuple(numfields);
      }
      catch (...) {
        delete out;
        throw;
      }
      return out;
    }
    else {
      maybeupdate(content_.get()->begintuple(numfields));
      return this;
    }
  }

  Fillable* ListFillable::index(int64_t index) {
    if (!begun_) {
      throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
    }
    else {
      content_.get()->index(index);
      return this;
    }
  }

  Fillable* ListFillable::endtuple() {
    if (!begun_) {
      throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
    }
    else {
      content_.get()->endtuple();
      return this;
    }
  }

  Fillable* ListFillable::beginrecord(int64_t disambiguator) {
    if (!begun_) {
      Fillable* out = UnionFillable::fromsingle(options_, this);
      try {
        out->beginrecord(disambiguator);
      }
      catch (...) {
        delete out;
        throw;
      }
      return out;
    }
    else {
      maybeupdate(content_.get()->beginrecord(disambiguator));
      return this;
    }
  }

  Fillable* ListFillable::field_fast(const char* key) {
    if (!begun_) {
      throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
    }
    else {
      content_.get()->field_fast(key);
      return this;
    }
  }

  Fillable* ListFillable::field_check(const char* key) {
    if (!begun_) {
      throw std::invalid_argument("called 'field_check' without 'beginrecord' at the same level before it");
    }
    else {
      content_.get()->field_check(key);
      return this;
    }
  }

  Fillable* ListFillable::endrecord() {
    if (!begun_) {
      throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
    }
    else {
      content_.get()->endrecord();
      return this;
    }
  }

  Fillable* ListFillable::maybeupdate(Fillable* tmp) {
    if (tmp != content_.get()) {
      content_ = std::shared_ptr<Fillable>(tmp);
    }
    return this;
  }
}
