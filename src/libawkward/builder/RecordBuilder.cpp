// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identities.h"
#include "awkward/Index.h"

#include "awkward/Identities.h"
#include "awkward/Index.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/type/RecordType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/RecordBuilder.h"

namespace awkward {
  const std::shared_ptr<Builder> RecordBuilder::fromempty(const ArrayBuilderOptions& options) {
    std::shared_ptr<Builder> out = std::make_shared<RecordBuilder>(options, std::vector<std::shared_ptr<Builder>>(), std::vector<std::string>(), std::vector<const char*>(), "", nullptr, -1, false, -1, -1);
    out.get()->setthat(out);
    return out;
  }

  RecordBuilder::RecordBuilder(const ArrayBuilderOptions& options, const std::vector<std::shared_ptr<Builder>>& contents, const std::vector<std::string>& keys, const std::vector<const char*>& pointers, const std::string& name, const char* nameptr, int64_t length, bool begun, int64_t nextindex, int64_t nexttotry)
      : options_(options)
      , contents_(contents)
      , keys_(keys)
      , pointers_(pointers)
      , name_(name)
      , nameptr_(nameptr)
      , length_(length)
      , begun_(begun)
      , nextindex_(nextindex)
      , nexttotry_(nexttotry) { }

  const std::string RecordBuilder::name() const {
    return name_;
  }

  const char* RecordBuilder::nameptr() const {
    return nameptr_;
  }

  const std::string RecordBuilder::classname() const {
    return "RecordBuilder";
  };

  int64_t RecordBuilder::length() const {
    return length_;
  }

  void RecordBuilder::clear() {
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

  const ContentPtr RecordBuilder::snapshot() const {
    if (length_ == -1) {
      return std::make_shared<EmptyArray>(Identities::none(), util::Parameters());
    }
    util::Parameters parameters;
    if (nameptr_ != nullptr) {
      parameters["__record__"] = util::quote(name_, true);
    }
    std::vector<ContentPtr> contents;
    std::shared_ptr<util::RecordLookup> recordlookup = std::make_shared<util::RecordLookup>();
    for (size_t i = 0;  i < contents_.size();  i++) {
      contents.push_back(contents_[i].get()->snapshot());
      recordlookup.get()->push_back(keys_[i]);
    }
    return std::make_shared<RecordArray>(Identities::none(), parameters, contents, recordlookup, length_);
  }

  bool RecordBuilder::active() const {
    return begun_;
  }

  const std::shared_ptr<Builder> RecordBuilder::null() {
    if (!begun_) {
      std::shared_ptr<Builder> out = OptionBuilder::fromvalids(options_, that_);
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

  const std::shared_ptr<Builder> RecordBuilder::boolean(bool x) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
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

  const std::shared_ptr<Builder> RecordBuilder::integer(int64_t x) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
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

  const std::shared_ptr<Builder> RecordBuilder::real(double x) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
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

  const std::shared_ptr<Builder> RecordBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
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

  const std::shared_ptr<Builder> RecordBuilder::beginlist() {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
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

  const std::shared_ptr<Builder> RecordBuilder::endlist() {
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

  const std::shared_ptr<Builder> RecordBuilder::begintuple(int64_t numfields) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
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

  const std::shared_ptr<Builder> RecordBuilder::index(int64_t index) {
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

  const std::shared_ptr<Builder> RecordBuilder::endtuple() {
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

  const std::shared_ptr<Builder> RecordBuilder::beginrecord(const char* name, bool check) {
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
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
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

  const std::shared_ptr<Builder> RecordBuilder::field(const char* key, bool check) {
    if (check) {
      return field_check(key);
    }
    else {
      return field_fast(key);
    }
  }

  const std::shared_ptr<Builder> RecordBuilder::field_fast(const char* key) {
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
        contents_.push_back(UnknownBuilder::fromempty(options_));
      }
      else {
        contents_.push_back(OptionBuilder::fromnulls(options_, length_, UnknownBuilder::fromempty(options_)));
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

  const std::shared_ptr<Builder> RecordBuilder::field_check(const char* key) {
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
        contents_.push_back(UnknownBuilder::fromempty(options_));
      }
      else {
        contents_.push_back(OptionBuilder::fromnulls(options_, length_, UnknownBuilder::fromempty(options_)));
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

  const std::shared_ptr<Builder> RecordBuilder::endrecord() {
    if (!begun_) {
      throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1  ||  !contents_[(size_t)nextindex_].get()->active()) {
      for (size_t i = 0;  i < contents_.size();  i++) {
        if (contents_[i].get()->length() == length_) {
          maybeupdate((int64_t)i, contents_[i].get()->null());
        }
        if (contents_[i].get()->length() != length_ + 1) {
          throw std::invalid_argument(std::string("record field ") + util::quote(keys_[i], true) + std::string(" filled more than once"));
        }
      }
      length_++;
      begun_ = false;
    }
    else {
      contents_[(size_t)nextindex_].get()->endrecord();
    }
    return that_;
  }

  const std::shared_ptr<Builder> RecordBuilder::append(const ContentPtr& array, int64_t at) {
    if (!begun_) {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'append' immediately after 'beginrecord'; needs 'index' or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->append(array, at));
    }
    else {
      contents_[(size_t)nextindex_].get()->append(array, at);
    }
    return that_;
  }

  void RecordBuilder::maybeupdate(int64_t i, const std::shared_ptr<Builder>& tmp) {
    if (tmp.get() != contents_[(size_t)i].get()) {
      contents_[(size_t)i] = tmp;
    }
  }
}
