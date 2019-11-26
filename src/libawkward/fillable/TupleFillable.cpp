// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/Index.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/type/RecordType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/fillable/FillableArray.h"
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
    index_ = -1;
  }

  const std::shared_ptr<Type> TupleFillable::type() const {
    if (index_ > (int64_t)contents_.size()) {
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
    if (index_ > (int64_t)contents_.size()) {
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

  Fillable* TupleFillable::null() {
    if (index_ != -1) {
      checklength();
      maybeupdate(index_, contents_[(size_t)index_].get()->null());
      return this;
    }
    else {
      Fillable* out = OptionFillable::fromvalids(fillablearray_, options_, this);
      out->null();
      return out;
    }
  }

  Fillable* TupleFillable::boolean(bool x) {
    if (index_ != -1) {
      checklength();
      maybeupdate(index_, contents_[(size_t)index_].get()->boolean(x));
      return this;
    }
    else {
      Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
      out->boolean(x);
      return out;
    }
  }

  Fillable* TupleFillable::integer(int64_t x) {
    if (index_ != -1) {
      checklength();
      maybeupdate(index_, contents_[(size_t)index_].get()->integer(x));
      return this;
    }
    else {
      Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
      out->integer(x);
      return out;
    }
  }

  Fillable* TupleFillable::real(double x) {
    if (index_ != -1) {
      checklength();
      maybeupdate(index_, contents_[(size_t)index_].get()->real(x));
      return this;
    }
    else {
      Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
      out->real(x);
      return out;
    }
  }

  Fillable* TupleFillable::beginlist() {
    if (index_ != -1) {
      checklength();
      maybeupdate(index_, contents_[(size_t)index_].get()->beginlist());
      return this;
    }
    else {
      Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
      out->beginlist();
      return out;
    }
  }

  Fillable* TupleFillable::endlist() {
    if (index_ != -1) {
      checklength();
      maybeupdate(index_, contents_[(size_t)index_].get()->endlist());
      return this;
    }
    else {
      return nullptr;
    }
  }

  Fillable* TupleFillable::begintuple(int64_t numfields) {
    if (index_ > (int64_t)contents_.size()) {
      // first-time initialization
      for (int64_t i = 0;  i < numfields;  i++) {
        contents_.push_back(std::shared_ptr<Fillable>(new UnknownFillable(fillablearray_, options_)));
      }
    }

    if (contents_.size() == (size_t)numfields) {
      index_ = -1;
    }
    else {
      Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
      out->begintuple(numfields);
      return out;
    }

    return this;
  }

  Fillable* TupleFillable::index(int64_t index) {
    if (!(0 <= index  &&  index < contents_.size())) {
      throw std::invalid_argument(std::string("index ") + std::to_string(index) + std::string(" for tuple of length ") + std::to_string(contents_.size()));
    }
    index_ = index;
    return this;
  }

  Fillable* TupleFillable::endtuple() {
    int64_t length = length_ + 1;
    for (size_t i = 0;  i < contents_.size();  i++) {
      while (contents_[i].get()->length() < length) {
        maybeupdate(i, contents_[i].get()->null());
      }
    }
    length_ = length;
    index_ = -1;
    return this;
  }

  Fillable* TupleFillable::beginrecord(int64_t disambiguator) {
    throw std::runtime_error("FIXME: TupleFillable::beginrecord");
    // if (index_ != -1) {
    //   checklength();
    //   maybeupdate(index_, contents_[(size_t)index_].get()->beginrecord(disambiguator));
    //   return this;
    // }
    // else {
    //   Fillable* out = UnionFillable::fromsingle(fillablearray_, options_, this);
    //   out->beginrecord(disambiguator);
    //   return out;
    // }
  }

  Fillable* TupleFillable::field_fast(const char* key) {
    throw std::runtime_error("FIXME: TupleFillable::field_fast");
  }

  Fillable* TupleFillable::field_check(const char* key) {
    throw std::runtime_error("FIXME: TupleFillable::field_check");
  }

  Fillable* TupleFillable::endrecord() {
    throw std::runtime_error("FIXME: TupleFillable::endrecord");
  }

  void TupleFillable::checklength() {
    if (contents_[(size_t)index_].get()->length() > length_) {
      throw std::invalid_argument(std::string("tuple index ") + std::to_string(index_) + std::string(" filled more than once (missing call to 'index'?)"));
    }
  }

  void TupleFillable::maybeupdate(int64_t i, Fillable* tmp) {
    if (tmp != contents_[(size_t)i].get()  &&  tmp != nullptr) {
      contents_[(size_t)i] = std::shared_ptr<Fillable>(tmp);
    }
  }
}
