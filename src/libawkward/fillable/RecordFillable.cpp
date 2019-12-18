// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/Index.h"

#include "awkward/Identity.h"
#include "awkward/Index.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/type/RecordType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/UnionFillable.h"

#include "awkward/fillable/RecordFillable.h"

namespace awkward {
  const std::shared_ptr<Fillable> RecordFillable::fromempty(const FillableOptions& options) {
    std::shared_ptr<Fillable> out(new RecordFillable(options, std::vector<std::shared_ptr<Fillable>>(), std::vector<std::string>(), std::vector<const char*>(), "", nullptr, -1, false, -1, -1));
    out.get()->setthat(out);
    return out;
  }

  int64_t RecordFillable::length() const {
    return length_;
  }

  void RecordFillable::clear() {
    for (auto x : contents_) {
      x.get()->clear();
    }
    keys_.clear();
    pointers_.clear();
    name_ = "";
    nameptr_ = nullptr;
    length_ = -1;
    begun_ = false;
    nextindex_ = -1;
    nexttotry_ = 0;
  }

  const std::shared_ptr<Type> RecordFillable::type() const {
    if (length_ == -1) {
      return std::shared_ptr<Type>(new UnknownType(Type::Parameters()));
    }
    else {
      std::vector<std::shared_ptr<Type>> types;
      std::shared_ptr<RecordType::Lookup> lookup(new RecordType::Lookup);
      std::shared_ptr<RecordType::ReverseLookup> reverselookup(new RecordType::ReverseLookup);
      for (size_t i = 0;  i < contents_.size();  i++) {
        types.push_back(contents_[i].get()->type());
        (*lookup.get())[keys_[i]] = i;
        reverselookup.get()->push_back(keys_[i]);
      }
      Type::Parameters parameters;
      if (nameptr_ != nullptr) {
        parameters["__class__"] = util::quote(name_, true);
      }
      return std::shared_ptr<Type>(new RecordType(parameters, types, lookup, reverselookup));
      return std::shared_ptr<Type>(new RecordType(parameters, types));
    }
  }

  const std::shared_ptr<Content> RecordFillable::snapshot() const {
    if (length_ == -1) {
      return std::shared_ptr<Content>(new EmptyArray(Identity::none(), Type::none()));
    }
    else if (contents_.empty()) {
      std::shared_ptr<Content> out(new RecordArray(Identity::none(), Type::none(), length_, false));
      if (nameptr_ != nullptr) {
        std::shared_ptr<Type> type = out.get()->type();
        type.get()->nolength().get()->setparameter("__class__", util::quote(name_, true));
        out.get()->settype(type);
      }
      return out;
    }
    else {
      std::vector<std::shared_ptr<Content>> contents;
      std::shared_ptr<RecordArray::Lookup> lookup(new RecordArray::Lookup);
      std::shared_ptr<RecordArray::ReverseLookup> reverselookup(new RecordArray::ReverseLookup);
      for (size_t i = 0;  i < contents_.size();  i++) {
        contents.push_back(contents_[i].get()->snapshot());
        (*lookup.get())[keys_[i]] = i;
        reverselookup.get()->push_back(keys_[i]);
      }
      std::shared_ptr<Content> out(new RecordArray(Identity::none(), Type::none(), contents, lookup, reverselookup));
      if (nameptr_ != nullptr) {
        std::shared_ptr<Type> type = out.get()->type();
        type.get()->nolength().get()->setparameter("__class__", util::quote(name_, true));
        out.get()->settype(type);
      }
      return out;
    }
  }

  bool RecordFillable::active() const {
    return begun_;
  }

  const std::shared_ptr<Fillable> RecordFillable::null() {
    if (!begun_) {
      std::shared_ptr<Fillable> out = OptionFillable::fromvalids(options_, that_);
      out.get()->null();
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'null' immediately after 'beginrecord'; needs 'index' or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->null());
    }
    else {
      contents_[(size_t)nextindex_].get()->null();
    }
    return that_;
  }

  const std::shared_ptr<Fillable> RecordFillable::boolean(bool x) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->boolean(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'boolean' immediately after 'beginrecord'; needs 'index' or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->boolean(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->boolean(x);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> RecordFillable::integer(int64_t x) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->integer(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'integer' immediately after 'beginrecord'; needs 'index' or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->integer(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->integer(x);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> RecordFillable::real(double x) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->real(x);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'real' immediately after 'beginrecord'; needs 'index' or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->real(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->real(x);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> RecordFillable::string(const char* x, int64_t length, const char* encoding) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->string(x, length, encoding);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'string' immediately after 'beginrecord'; needs 'index' or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->string(x, length, encoding));
    }
    else {
      contents_[(size_t)nextindex_].get()->string(x, length, encoding);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> RecordFillable::beginlist() {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->beginlist();
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'beginlist' immediately after 'beginrecord'; needs 'index' or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->beginlist());
    }
    else {
      contents_[(size_t)nextindex_].get()->beginlist();
    }
    return that_;
  }

  const std::shared_ptr<Fillable> RecordFillable::endlist() {
    if (!begun_) {
      throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'endlist' immediately after 'beginrecord'; needs 'index' or 'endrecord' and then 'beginlist'");
    }
    else {
      contents_[(size_t)nextindex_].get()->endlist();
    }
    return that_;
  }

  const std::shared_ptr<Fillable> RecordFillable::begintuple(int64_t numfields) {
    if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->begintuple(numfields);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'begintuple' immediately after 'beginrecord'; needs 'field_fast', 'field_check', or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->begintuple(numfields));
    }
    else {
      contents_[(size_t)nextindex_].get()->begintuple(numfields);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> RecordFillable::index(int64_t index) {
    if (!begun_) {
      throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'index' immediately after 'beginrecord'; needs 'field_fast', 'field_check' or 'endrecord' and then 'begintuple'");
    }
    else {
      contents_[(size_t)nextindex_].get()->index(index);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> RecordFillable::endtuple() {
    if (!begun_) {
      throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'endtuple' immediately after 'beginrecord'; needs 'field_fast', 'field_check', or 'endrecord' and then 'begintuple'");
    }
    else {
      contents_[(size_t)nextindex_].get()->endtuple();
    }
    return that_;
  }

  const std::shared_ptr<Fillable> RecordFillable::beginrecord(const char* name, bool check) {
    if (length_ == -1) {
      if (name == nullptr) {
        name_ = std::string("");
      }
      else {
        name_ = std::string(name);
      }
      nameptr_ = name;
      length_ = 0;
    }

    if (!begun_  &&  ((check  &&  name_ == name)  ||  (!check  &&  nameptr_ == name))) {
      begun_ = true;
      nextindex_ = -1;
      nexttotry_ = 0;
    }
    else if (!begun_) {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->beginrecord(name, check);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'beginrecord' immediately after 'beginrecord'; needs 'field_fast', 'field_check', or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->beginrecord(name, check));
    }
    else {
      contents_[(size_t)nextindex_].get()->beginrecord(name, check);
    }
    return that_;
  }

  const std::shared_ptr<Fillable> RecordFillable::field(const char* key, bool check) {
    if (check) {
      return field_check(key);
    }
    else {
      return field_fast(key);
    }
  }

  const std::shared_ptr<Fillable> RecordFillable::field_fast(const char* key) {
    if (!begun_) {
      throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1  ||  !contents_[(size_t)nextindex_].get()->active()) {
      int64_t wrap_around = (int64_t)pointers_.size();
      int64_t i = nexttotry_;
      do {
        if (i >= wrap_around) {
          i = 0;
          if (i == nexttotry_) {
            break;
          }
        }
        if (pointers_[(size_t)i] == key) {
          nextindex_ = i;
          nexttotry_ = i + 1;
          return that_;
        }
        i++;
      } while (i != nexttotry_);
      nextindex_ = wrap_around;
      nexttotry_ = 0;
      if (length_ == 0) {
        contents_.push_back(UnknownFillable::fromempty(options_));
      }
      else {
        contents_.push_back(OptionFillable::fromnulls(options_, length_, UnknownFillable::fromempty(options_)));
      }
      keys_.push_back(std::string(key));
      pointers_.push_back(key);
      return that_;
    }
    else {
      contents_[(size_t)nextindex_].get()->field(key, false);
      return that_;
    }
  }

  const std::shared_ptr<Fillable> RecordFillable::field_check(const char* key) {
    if (!begun_) {
      throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1  ||  !contents_[(size_t)nextindex_].get()->active()) {
      int64_t wrap_around = (int64_t)keys_.size();
      int64_t i = nexttotry_;
      do {
        if (i >= wrap_around) {
          i = 0;
          if (i == nexttotry_) {
            break;
          }
        }
        if (keys_[(size_t)i].compare(key) == 0) {
          nextindex_ = i;
          nexttotry_ = i + 1;
          return that_;
        }
        i++;
      } while (i != nexttotry_);
      nextindex_ = wrap_around;
      nexttotry_ = 0;
      if (length_ == 0) {
        contents_.push_back(UnknownFillable::fromempty(options_));
      }
      else {
        contents_.push_back(OptionFillable::fromnulls(options_, length_, UnknownFillable::fromempty(options_)));
      }
      keys_.push_back(std::string(key));
      pointers_.push_back(nullptr);
      return that_;
    }
    else {
      contents_[(size_t)nextindex_].get()->field(key, true);
      return that_;
    }
  }

  const std::shared_ptr<Fillable> RecordFillable::endrecord() {
    if (!begun_) {
      throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1  ||  !contents_[(size_t)nextindex_].get()->active()) {
      int64_t i = 0;
      for (auto content : contents_) {
        if (content.get()->length() == length_) {
          maybeupdate(i, content.get()->null());
        }
        if (content.get()->length() != length_ + 1) {
          throw std::invalid_argument(std::string("record field ") + util::quote(keys_[(size_t)i], true) + std::string(" filled more than once"));
        }
        i++;
      }
      length_++;
      begun_ = false;
    }
    else {
      contents_[(size_t)nextindex_].get()->endrecord();
    }
    return that_;
  }

  void RecordFillable::maybeupdate(int64_t i, const std::shared_ptr<Fillable>& tmp) {
    if (tmp.get() != contents_[(size_t)i].get()) {
      contents_[(size_t)i] = tmp;
    }
  }
}
