// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/Index.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/type/RecordType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/UnionFillable.h"

#include "awkward/fillable/TupleFillable.h"

namespace awkward {
  const std::shared_ptr<Fillable> TupleFillable::fromempty(const FillableOptions& options) {
    std::shared_ptr<Fillable> out(new TupleFillable(options, std::vector<std::shared_ptr<Fillable>>(), -1, false, -1));
    out.get()->setthat(out);
    return out;
  }

  int64_t TupleFillable::length() const {
    return length_;
  }

  void TupleFillable::clear() {
    for (auto x : contents_) {
      x.get()->clear();
    }
    length_ = -1;
    begun_ = false;
    nextindex_ = -1;
  }

  const std::shared_ptr<Type> TupleFillable::type() const {
    if (length_ == -1) {
      return std::shared_ptr<Type>(new UnknownType(Type::Parameters()));
    }
    else {
      std::vector<std::shared_ptr<Type>> types;
      for (auto content : contents_) {
        types.push_back(content.get()->type());
      }
      return std::shared_ptr<Type>(new RecordType(Type::Parameters(), types));
    }
  }

  const std::shared_ptr<Content> TupleFillable::snapshot() const {
    if (length_ == -1) {
      return std::shared_ptr<Content>(new EmptyArray(Identity::none(), Type::none()));   // FIXME: Type::none()
    }
    else if (contents_.size() == 0) {
      return std::shared_ptr<Content>(new RecordArray(Identity::none(), Type::none(), length_, true));   // FIXME: Type::none()
    }
    else {
      std::vector<std::shared_ptr<Content>> contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->snapshot());
      }
      return std::shared_ptr<Content>(new RecordArray(Identity::none(), Type::none(), contents));   // FIXME: Type::none()
    }
  }

  bool TupleFillable::active() const {
    return begun_;
  }

  const std::shared_ptr<Fillable> TupleFillable::null() {
    if (!begun_) {
      std::shared_ptr<Fillable> out = OptionFillable::fromvalids(options_, that_);
      out.get()->null();
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'null' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->null());
    }
    else {
      contents_[(size_t)nextindex_].get()->null();
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::boolean(bool x) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->boolean(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'boolean' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->boolean(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->boolean(x);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::integer(int64_t x) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->integer(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'integer' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->integer(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->integer(x);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::real(double x) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->real(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'real' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->real(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->real(x);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::beginlist() {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->beginlist();
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'beginlist' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->beginlist());
    }
    else {
      contents_[(size_t)nextindex_].get()->beginlist();
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::endlist() {
    if (!begun_) {
      throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'endlist' immediately after 'begintuple'; needs 'index' or 'endtuple' and then 'beginlist'");
    }
    else {
      contents_[(size_t)nextindex_].get()->endlist();
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::begintuple(int64_t numfields) {
    if (length_ == -1) {
      for (int64_t i = 0;  i < numfields;  i++) {
        contents_.push_back(std::shared_ptr<Fillable>(UnknownFillable::fromempty(options_)));
      }
      length_ = 0;
    }

    if (!begun_  &&  numfields == (int64_t)contents_.size()) {
      begun_ = true;
      nextindex_ = -1;
    }
    else if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->begintuple(numfields);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'begintuple' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->begintuple(numfields));
    }
    else {
      contents_[(size_t)nextindex_].get()->begintuple(numfields);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::index(int64_t index) {
    if (!begun_) {
      throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
    }
    else if (nextindex_ == -1  ||  !contents_[(size_t)nextindex_].get()->active()) {
      nextindex_ = index;
    }
    else {
      contents_[(size_t)nextindex_].get()->index(index);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::endtuple() {
    if (!begun_) {
      throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
    }
    else if (nextindex_ == -1  ||  !contents_[(size_t)nextindex_].get()->active()) {
      int64_t i = 0;
      for (auto content : contents_) {
        if (content.get()->length() == length_) {
          maybeupdate(i, content.get()->null());
        }
        if (content.get()->length() != length_ + 1) {
          throw std::invalid_argument(std::string("tuple index ") + std::to_string(i) + std::string(" filled more than once"));
        }
        i++;
      }
      length_++;
      begun_ = false;
    }
    else {
      contents_[(size_t)nextindex_].get()->endtuple();
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::beginrecord(int64_t disambiguator) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->beginrecord(disambiguator);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'beginrecord' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->beginrecord(disambiguator));
    }
    else {
      contents_[(size_t)nextindex_].get()->beginrecord(disambiguator);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::field_fast(const char* key) {
    if (!begun_) {
      throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'field_fast' immediately after 'begintuple'; needs 'index' or 'endtuple' and then 'beginrecord'");
    }
    else {
      contents_[(size_t)nextindex_].get()->field_fast(key);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::field_check(const char* key) {
    if (!begun_) {
      throw std::invalid_argument("called 'field_check' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'field_check' immediately after 'begintuple'; needs 'index' or 'endtuple' and then 'beginrecord'");
    }
    else {
      contents_[(size_t)nextindex_].get()->field_check(key);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::endrecord() {
    if (!begun_) {
      throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'endrecord' immediately after 'begintuple'; needs 'index' or 'endtuple' and then 'beginrecord'");
    }
    else {
      contents_[(size_t)nextindex_].get()->endrecord();
    }
    return that_;
  }

  void TupleFillable::maybeupdate(int64_t i, const std::shared_ptr<Fillable>& tmp) {
    if (tmp.get() != contents_[(size_t)i].get()) {
      contents_[(size_t)i] = tmp;
    }
  }
}
