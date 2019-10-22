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
    Fillable* out = OptionFillable::fromvalids(options_, this);
    out->null();
    return out;
  }

  Fillable* ListFillable::boolean(bool x) {
    Fillable* out = UnionFillable::fromsingle(options_, this);
    out->boolean(x);
    return out;
  }

  Fillable* ListFillable::integer(int64_t x) {
    Fillable* out = UnionFillable::fromsingle(options_, this);
    out->integer(x);
    return out;
  }

  Fillable* ListFillable::real(double x) {
    Fillable* out = UnionFillable::fromsingle(options_, this);
    out->real(x);
    return out;
  }

  void ListFillable::maybeupdate(Fillable* tmp) {
    if (tmp != content_.get()) {
      content_ = std::shared_ptr<Fillable>(tmp);
    }
  }
}
