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
    return std::shared_ptr<Content>(new ListOffsetArray64(Identity::none(), offsets, content_.get()->snapshot()));
  }

  Fillable* ListFillable::null() {
    if (begun_) {
      return maybeupdate(content_.get()->null());
    }
    else {
      Fillable* out = OptionFillable::fromvalids(options_, this);
      out->null();
      return out;
    }
  }

  Fillable* ListFillable::boolean(bool x) {
    if (begun_) {
      return maybeupdate(content_.get()->boolean(x));
    }
    else {
      Fillable* out = UnionFillable::fromsingle(options_, this);
      out->boolean(x);
      return out;
    }
  }

  Fillable* ListFillable::integer(int64_t x) {
    if (begun_) {
      return maybeupdate(content_.get()->integer(x));
    }
    else {
      Fillable* out = UnionFillable::fromsingle(options_, this);
      out->integer(x);
      return out;
    }
  }

  Fillable* ListFillable::real(double x) {
    if (begun_) {
      return maybeupdate(content_.get()->real(x));
    }
    else {
      Fillable* out = UnionFillable::fromsingle(options_, this);
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

  Fillable* ListFillable::beginrec(const Slots* slots) {
    throw std::runtime_error("FIXME: ListFillable::beginrec");
  }

  Fillable* ListFillable::reckey(int64_t index) {
    throw std::runtime_error("FIXME: ListFillable::reckey(int)");
  }

  Fillable* ListFillable::endrec() {
    throw std::runtime_error("FIXME: ListFillable::endrec");
  }

  Fillable* ListFillable::maybeupdate(Fillable* tmp) {
    if (tmp != content_.get()  &&  tmp != nullptr) {
      content_ = std::shared_ptr<Fillable>(tmp);
    }
    return this;
  }
}
