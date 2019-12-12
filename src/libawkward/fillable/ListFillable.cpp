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
  const std::shared_ptr<Fillable> ListFillable::fromempty(const FillableOptions& options) {
    GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::empty(options);
    offsets.append(0);
    std::shared_ptr<Fillable> out(new ListFillable(options, offsets, UnknownFillable::fromempty(options), false));
    out.get()->setthat(out);
    return out;
  }

  int64_t ListFillable::length() const {
    return offsets_.length() - 1;
  }

  void ListFillable::clear() {
    offsets_.clear();
    content_.get()->clear();
  }

  const std::shared_ptr<Type> ListFillable::type() const {
    return std::shared_ptr<Type>(new ListType(Type::Parameters(), content_.get()->type()));
  }

  const std::shared_ptr<Content> ListFillable::snapshot() const {
    Index64 offsets(offsets_.ptr(), 0, offsets_.length());
    return std::shared_ptr<Content>(new ListOffsetArray64(Identity::none(), Type::none(), offsets, content_.get()->snapshot()));   // FIXME: Type::none()
  }

  bool ListFillable::active() const {
    return begun_;
  }

  const std::shared_ptr<Fillable> ListFillable::null() {
    if (!begun_) {
      std::shared_ptr<Fillable> out = OptionFillable::fromvalids(options_, that_);
      out.get()->null();
      return out;
    }
    else {
      maybeupdate(content_.get()->null());
      return that_;
    }
  }

  const std::shared_ptr<Fillable> ListFillable::boolean(bool x) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->boolean(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->boolean(x));
      return that_;
    }
  }

  const std::shared_ptr<Fillable> ListFillable::integer(int64_t x) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->integer(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->integer(x));
      return that_;
    }
  }

  const std::shared_ptr<Fillable> ListFillable::real(double x) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->real(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->real(x));
      return that_;
    }
  }

  const std::shared_ptr<Fillable> ListFillable::beginlist() {
    if (!begun_) {
      begun_ = true;
    }
    else {
      maybeupdate(content_.get()->beginlist());
    }
    return that_;
  }

  const std::shared_ptr<Fillable> ListFillable::endlist() {
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
    return that_;
  }

  const std::shared_ptr<Fillable> ListFillable::begintuple(int64_t numfields) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->begintuple(numfields);
      return out;
    }
    else {
      maybeupdate(content_.get()->begintuple(numfields));
      return that_;
    }
  }

  const std::shared_ptr<Fillable> ListFillable::index(int64_t index) {
    if (!begun_) {
      throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
    }
    else {
      content_.get()->index(index);
      return that_;
    }
  }

  const std::shared_ptr<Fillable> ListFillable::endtuple() {
    if (!begun_) {
      throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
    }
    else {
      content_.get()->endtuple();
      return that_;
    }
  }

  const std::shared_ptr<Fillable> ListFillable::beginrecord(int64_t disambiguator) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->beginrecord(disambiguator);
      return out;
    }
    else {
      maybeupdate(content_.get()->beginrecord(disambiguator));
      return that_;
    }
  }

  const std::shared_ptr<Fillable> ListFillable::field_fast(const char* key) {
    if (!begun_) {
      throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
    }
    else {
      content_.get()->field_fast(key);
      return that_;
    }
  }

  const std::shared_ptr<Fillable> ListFillable::field_check(const char* key) {
    if (!begun_) {
      throw std::invalid_argument("called 'field_check' without 'beginrecord' at the same level before it");
    }
    else {
      content_.get()->field_check(key);
      return that_;
    }
  }

  const std::shared_ptr<Fillable> ListFillable::endrecord() {
    if (!begun_) {
      throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
    }
    else {
      content_.get()->endrecord();
      return that_;
    }
  }

  void ListFillable::maybeupdate(const std::shared_ptr<Fillable>& tmp) {
    if (tmp.get() != content_.get()) {
      content_ = tmp;
    }
  }
}
