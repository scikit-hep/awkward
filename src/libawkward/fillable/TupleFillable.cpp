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
  int64_t TupleFillable::length() const {
    return length_;
  }

  void TupleFillable::clear() {
    for (auto x : contents_) {
      x.get()->clear();
    }
    length_ = 0;
    nextindex_ = -1;
  }

  const std::shared_ptr<Type> TupleFillable::type() const {
    if (nextindex_ > (int64_t)contents_.size()) {
      return std::shared_ptr<Type>(new UnknownType);
    }
    else {
      std::vector<std::shared_ptr<Type>> types;
      for (auto content : contents_) {
        types.push_back(content.get()->type());
      }
      return std::shared_ptr<Type>(new RecordType(types));
    }
  }

  const std::shared_ptr<Content> TupleFillable::snapshot() const {
    if (nextindex_ > (int64_t)contents_.size()) {
      return std::shared_ptr<Content>(new EmptyArray(Identity::none()));
    }
    else {
      std::vector<std::shared_ptr<Content>> contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->snapshot());
      }
      return std::shared_ptr<Content>(new RecordArray(Identity::none(), contents));
    }
  }

  bool TupleFillable::active() const {
    return begun_;
  }

  Fillable* TupleFillable::null() {
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
      throw std::invalid_argument("called 'null' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->null());
    }
    else {
      contents_[(size_t)nextindex_].get()->null();
    }
    return this;
  }

  Fillable* TupleFillable::boolean(bool x) {
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
      throw std::invalid_argument("called 'boolean' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->boolean(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->boolean(x);
    }
    return this;
  }

  Fillable* TupleFillable::integer(int64_t x) {
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
      throw std::invalid_argument("called 'integer' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->integer(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->integer(x);
    }
    return this;
  }

  Fillable* TupleFillable::real(double x) {
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
      throw std::invalid_argument("called 'real' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->real(x));
    }
    else {
      contents_[(size_t)nextindex_].get()->real(x);
    }
    return this;
  }

  Fillable* TupleFillable::beginlist() {
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
      throw std::invalid_argument("called 'beginlist' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->beginlist());
    }
    else {
      contents_[(size_t)nextindex_].get()->beginlist();
    }
    return this;
  }

  Fillable* TupleFillable::endlist() {
    if (!begun_) {
      throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'endlist' immediately after 'begintuple'; needs 'index' or 'endtuple' and then 'beginlist'");
    }
    else {
      contents_[(size_t)nextindex_].get()->endlist();
    }
    return this;
  }

  Fillable* TupleFillable::begintuple(int64_t numfields) {
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
      throw std::invalid_argument("called 'begintuple' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->begintuple(numfields));
    }
    else {
      contents_[(size_t)nextindex_].get()->begintuple(numfields);
    }
    return this;
  }

  Fillable* TupleFillable::index(int64_t index) {
    if (!begun_) {
      throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
    }
    else if (nextindex_ == -1  ||  !contents_[(size_t)nextindex_].get()->active()) {
      nextindex_ = index;
    }
    else {
      contents_[(size_t)nextindex_].get()->index(index);
    }
    return this;
  }

  Fillable* TupleFillable::endtuple() {
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
    return this;
  }

  Fillable* TupleFillable::beginrecord(int64_t disambiguator) {
    if (!begun_) {
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
      throw std::invalid_argument("called 'beginrecord' immediately after 'begintuple'; needs 'index' or 'endtuple'");
    }
    else if (!contents_[(size_t)nextindex_].get()->active()) {
      maybeupdate(nextindex_, contents_[(size_t)nextindex_].get()->beginrecord(disambiguator));
    }
    else {
      contents_[(size_t)nextindex_].get()->beginrecord(disambiguator);
    }
    return this;
  }

  Fillable* TupleFillable::field_fast(const char* key) {
    if (!begun_) {
      throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'field_fast' immediately after 'begintuple'; needs 'index' or 'endtuple' and then 'beginrecord'");
    }
    else {
      contents_[(size_t)nextindex_].get()->field_fast(key);
    }
    return this;
  }

  Fillable* TupleFillable::field_check(const char* key) {
    if (!begun_) {
      throw std::invalid_argument("called 'field_check' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'field_check' immediately after 'begintuple'; needs 'index' or 'endtuple' and then 'beginrecord'");
    }
    else {
      contents_[(size_t)nextindex_].get()->field_check(key);
    }
    return this;
  }

  Fillable* TupleFillable::endrecord() {
    if (!begun_) {
      throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
    }
    else if (nextindex_ == -1) {
      throw std::invalid_argument("called 'endrecord' immediately after 'begintuple'; needs 'index' or 'endtuple' and then 'beginrecord'");
    }
    else {
      contents_[(size_t)nextindex_].get()->endrecord();
    }
    return this;
  }

  void TupleFillable::maybeupdate(int64_t i, Fillable* tmp) {
    if (tmp != contents_[(size_t)i].get()  &&  tmp != nullptr) {
      contents_[(size_t)i] = std::shared_ptr<Fillable>(tmp);
    }
  }
}
