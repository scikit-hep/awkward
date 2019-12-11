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
  int64_t RecordFillable::length() const {
    return length_;
  }

  void RecordFillable::clear() {
    for (auto x : contents_) {
      x.get()->clear();
    }
    keys_.clear();
    pointers_.clear();
    disambiguator_ = 0;
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
      return std::shared_ptr<Type>(new RecordType(Type::Parameters(), types, lookup, reverselookup));
      return std::shared_ptr<Type>(new RecordType(Type::Parameters(), types));
    }
  }

  const std::shared_ptr<Content> RecordFillable::snapshot() const {
    if (length_ == -1) {
      return std::shared_ptr<Content>(new EmptyArray(Identity::none(), Type::none()));   // FIXME: Type::none()
    }
    else if (contents_.size() == 0) {
      return std::shared_ptr<Content>(new RecordArray(Identity::none(), Type::none(), length_, false));   // FIXME: Type::none()
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
      return std::shared_ptr<Content>(new RecordArray(Identity::none(), Type::none(), contents, lookup, reverselookup));   // FIXME: Type::none()
    }
  }

  bool RecordFillable::active() const {
    return begun_;
  }

  Fillable* RecordFillable::null() {
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
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'null' immediately after 'beginrecord'; needs 'index' or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->null());
    }
    else {
      contents_[(size_t)nextindex_].get()->null();
    }
    return this;
  }

  Fillable* RecordFillable::boolean(bool x) {
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
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'boolean' immediately after 'beginrecord'; needs 'index' or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->boolean(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->boolean(x);
    }
    return this;
  }

  Fillable* RecordFillable::integer(int64_t x) {
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
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'integer' immediately after 'beginrecord'; needs 'index' or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->integer(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->integer(x);
    }
    return this;
  }

  Fillable* RecordFillable::real(double x) {
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
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'real' immediately after 'beginrecord'; needs 'index' or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->real(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->real(x);
    }
    return this;
  }

  Fillable* RecordFillable::beginlist() {
    if (!begun_) {
      Fillable* out = UnionFillable::fromsingle(options_, this);
      try {
        out->beginlist();
      }
      catch (...) {
        delete out;
        throw;
      }
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
    return this;
  }

  Fillable* RecordFillable::endlist() {
    if (!begun_) {
      throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'endlist' immediately after 'beginrecord'; needs 'index' or 'endrecord' and then 'beginlist'");
    }
    else {
      contents_[(size_t)nextindex_].get()->endlist();
    }
    return this;
  }

  Fillable* RecordFillable::begintuple(int64_t numfields) {
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
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'begintuple' immediately after 'beginrecord'; needs 'field_fast', 'field_check', or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->begintuple(numfields));
    }
    else {
      contents_[(size_t)nextindex_].get()->begintuple(numfields);
    }
    return this;
  }

  Fillable* RecordFillable::index(int64_t index) {
    if (!begun_) {
      throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'index' immediately after 'beginrecord'; needs 'field_fast', 'field_check' or 'endrecord' and then 'begintuple'");
    }
    else {
      contents_[(size_t)nextindex_].get()->index(index);
    }
    return this;
  }

  Fillable* RecordFillable::endtuple() {
    if (!begun_) {
      throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'endtuple' immediately after 'beginrecord'; needs 'field_fast', 'field_check', or 'endrecord' and then 'begintuple'");
    }
    else {
      contents_[(size_t)nextindex_].get()->endtuple();
    }
    return this;
  }

  Fillable* RecordFillable::beginrecord(int64_t disambiguator) {
    if (length_ == -1) {
      disambiguator_ = disambiguator;
      length_ = 0;
    }

    if (!begun_  &&  disambiguator == disambiguator_) {
      begun_ = true;
      nextindex_ = -1;
      nexttotry_ = 0;
    }
    else if (!begun_) {
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
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'beginrecord' immediately after 'beginrecord'; needs 'field_fast', 'field_check', or 'endrecord'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->beginrecord(disambiguator));
    }
    else {
      contents_[(size_t)nextindex_].get()->beginrecord(disambiguator);
    }
    return this;
  }

  Fillable* RecordFillable::field_fast(const char* key) {
    if (!begun_) {
      throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
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
          return this;
        }
        i++;
      } while (i != nexttotry_);
      nextindex_ = wrap_around;
      nexttotry_ = 0;
      if (length_ == 0) {
        contents_.push_back(std::shared_ptr<Fillable>(UnknownFillable::fromempty(options_)));
      }
      else {
        contents_.push_back(std::shared_ptr<Fillable>(OptionFillable::fromnulls(options_, length_, UnknownFillable::fromempty(options_))));
      }
      keys_.push_back(std::string(key));
      pointers_.push_back(key);
      return this;
    }
    else {
      contents_[(size_t)nextindex_].get()->field_fast(key);
      return this;
    }
  }

  Fillable* RecordFillable::field_check(const char* key) {
    if (!begun_) {
      throw std::invalid_argument("called 'field_check' without 'beginrecord' at the same level before it");
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
          return this;
        }
        i++;
      } while (i != nexttotry_);
      nextindex_ = wrap_around;
      nexttotry_ = 0;
      if (length_ == 0) {
        contents_.push_back(std::shared_ptr<Fillable>(UnknownFillable::fromempty(options_)));
      }
      else {
        contents_.push_back(std::shared_ptr<Fillable>(OptionFillable::fromnulls(options_, length_, UnknownFillable::fromempty(options_))));
      }
      keys_.push_back(std::string(key));
      pointers_.push_back(nullptr);
      return this;
    }
    else {
      contents_[(size_t)nextindex_].get()->field_check(key);
      return this;
    }
  }

  Fillable* RecordFillable::endrecord() {
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
    return this;
  }

  void RecordFillable::maybeupdate(int64_t i, Fillable* tmp) {
    if (tmp != contents_[(size_t)i].get()) {
      contents_[(size_t)i] = std::shared_ptr<Fillable>(tmp);
    }
  }

}
