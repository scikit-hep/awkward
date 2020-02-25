// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identities.h"
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
    std::shared_ptr<Fillable> out = std::make_shared<ListFillable>(options, offsets, UnknownFillable::fromempty(options), false);
    out.get()->setthat(out);
    return out;
  }

  ListFillable::ListFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& offsets, const std::shared_ptr<Fillable>& content, bool begun)
      : options_(options)
      , offsets_(offsets)
      , content_(content)
      , begun_(begun) { }

  const std::string ListFillable::classname() const {
    return "ListFillable";
  };

  int64_t ListFillable::length() const {
    return offsets_.length() - 1;
  }

  void ListFillable::clear() {
    offsets_.clear();
    offsets_.append(0);
    content_.get()->clear();
  }

  const std::shared_ptr<Content> ListFillable::snapshot() const {
    Index64 offsets(offsets_.ptr(), 0, offsets_.length());
    return std::make_shared<ListOffsetArray64>(Identities::none(), util::Parameters(), offsets, content_.get()->snapshot());
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

  const std::shared_ptr<Fillable> ListFillable::string(const char* x, int64_t length, const char* encoding) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->string(x, length, encoding);
      return out;
    }
    else {
      maybeupdate(content_.get()->string(x, length, encoding));
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

  const std::shared_ptr<Fillable> ListFillable::beginrecord(const char* name, bool check) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->beginrecord(name, check);
      return out;
    }
    else {
      maybeupdate(content_.get()->beginrecord(name, check));
      return that_;
    }
  }

  const std::shared_ptr<Fillable> ListFillable::field(const char* key, bool check) {
    if (!begun_) {
      throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
    }
    else {
      content_.get()->field(key, check);
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

  const std::shared_ptr<Fillable> ListFillable::append(const std::shared_ptr<Content>& array, int64_t at) {
    throw std::runtime_error("FIXME: ListFillable::append");
  }

  void ListFillable::maybeupdate(const std::shared_ptr<Fillable>& tmp) {
    if (tmp.get() != content_.get()) {
      content_ = tmp;
    }
  }
}
