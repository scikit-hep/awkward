// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/Index.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/type/ListType.h"
#include "awkward/fillable/FillableArray.h"
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
    return std::shared_ptr<Content>(new ListOffsetArray64(Identity::none(), offsets, content_.get()->snapshot()));
  }

  Fillable* ListFillable::null() {
    if (begun_) {
      return maybeupdate(content_.get()->null());
    }
    else {
      Fillable* out = OptionFillable::fromvalids(fillablearray_, options_, this);
      out->null();
      return out;
    }
  }

  Fillable* ListFillable::boolean(bool x) {
    if (begun_) {
      return maybeupdate(content_.get()->boolean(x));
    }
    else {
      Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
      out->boolean(x);
      return out;
    }
  }

  Fillable* ListFillable::integer(int64_t x) {
    if (begun_) {
      return maybeupdate(content_.get()->integer(x));
    }
    else {
      Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
      out->integer(x);
      return out;
    }
  }

  Fillable* ListFillable::real(double x) {
    if (begun_) {
      return maybeupdate(content_.get()->real(x));
    }
    else {
      Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
      out->real(x);
      return out;
    }
  }

  Fillable* ListFillable::beginlist() {
    if (begun_) {
      return maybeupdate(content_.get()->beginlist());
    }
    else {
      begun_ = true;
      return this;
    }
  }

  Fillable* ListFillable::endlist() {
    if (begun_) {
      Fillable* tmp = content_.get()->endlist();
      if (tmp == nullptr) {
        offsets_.append(content_.get()->length());
        begun_ = false;
        return this;
      }
      else {
        return maybeupdate(tmp);
      }
    }
    else {
      return nullptr;
    }
  }

  Fillable* ListFillable::begintuple(int64_t numfields) {
    Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
    out->begintuple(numfields);
    return out;
  }

  Fillable* ListFillable::index(int64_t index) {
    throw std::invalid_argument("'index' should only be called in a tuple (did you forget to call 'begintuple'?)");
  }

  Fillable* ListFillable::endtuple() {
    throw std::invalid_argument("'endtuple' should only be called in a tuple (did you forget to call 'begintuple'?)");
  }

  Fillable* ListFillable::beginrecord(int64_t disambiguator) {
    Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
    out->beginrecord(disambiguator);
    return out;
  }

  Fillable* ListFillable::field_fast(const char* key) {
    throw std::invalid_argument("'field_fast' should only be called in a record (did you forget to call 'beginrecord'?)");
  }

  Fillable* ListFillable::field_check(const char* key) {
    throw std::invalid_argument("'field_check' should only be called in a record (did you forget to call 'beginrecord'?)");
  }

  Fillable* ListFillable::endrecord() {
    throw std::invalid_argument("'endrecord' should only be called in a record (did you forget to call 'beginrecord'?)");
  }

  Fillable* ListFillable::maybeupdate(Fillable* tmp) {
    if (tmp != content_.get()  &&  tmp != nullptr) {
      content_ = std::shared_ptr<Fillable>(tmp);
    }
    return this;
  }
}
