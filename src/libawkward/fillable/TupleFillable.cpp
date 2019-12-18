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
    std::shared_ptr<Fillable> out = std::make_shared<TupleFillable>(options, std::vector<std::shared_ptr<Fillable>>(), -1, false, -1);
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
      return std::make_shared<UnknownType>(Type::Parameters());
    }
    else {
      std::vector<std::shared_ptr<Type>> types;
      for (size_t i = 0;  i < contents_.size();  i++) {
        types.push_back(contents_[i].get()->type());
      }
      return std::make_shared<RecordType>(Type::Parameters(), types);
    }
  }

  const std::shared_ptr<Content> TupleFillable::snapshot(const std::shared_ptr<Type>& type) const {
    if (length_ == -1) {
      return std::make_shared<EmptyArray>(Identity::none(), type);
    }

    RecordType* raw = dynamic_cast<RecordType*>(type.get());
    std::vector<std::shared_ptr<Content>> contents;
    for (size_t i = 0;  i < contents_.size();  i++) {
      if (raw == nullptr) {
        contents.push_back(contents_[i].get()->snapshot(Type::none()));
      }
      else {
        contents.push_back(contents_[i].get()->snapshot(raw->field((int64_t)i)));
      }
    }

    if (contents.empty()) {
      return std::make_shared<RecordArray>(Identity::none(), type, length_, true);
    }
    else {
      return std::make_shared<RecordArray>(Identity::none(), type, contents);
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

  const std::shared_ptr<Fillable> TupleFillable::string(const char* x, int64_t length, const char* encoding) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->string(x, length, encoding);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'string' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->string(x, length, encoding));
    }
    else {
      contents_[(size_t)nextindex_].get()->string(x, length, encoding);
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
      for (size_t i = 0;  i < contents_.size();  i++) {
        if (contents_[i].get()->length() == length_) {
          maybeupdate(i, contents_[i].get()->null());
        }
        if (contents_[i].get()->length() != length_ + 1) {
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

  const std::shared_ptr<Fillable> TupleFillable::beginrecord(const char* name, bool check) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->beginrecord(name, check);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'beginrecord' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->beginrecord(name, check));
    }
    else {
      contents_[(size_t)nextindex_].get()->beginrecord(name, check);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> TupleFillable::field(const char* key, bool check) {
    if (!begun_) {
      throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'field_fast' immediately after 'begintuple'; needs 'index' or 'endtuple' and then 'beginrecord'");
    }
    else {
      contents_[(size_t)nextindex_].get()->field(key, check);
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
