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
    if (index_ > contents_.size()) {
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
    if (index_ > contents_.size()) {
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
    // FIXME: null() outside of an index means the whole tuple is null
    check();
    maybeupdate(index_, contents_[index_].get()->null());
    return this;
  }

  Fillable* TupleFillable::boolean(bool x) {
    check();
    maybeupdate(index_, contents_[index_].get()->boolean(x));
    return this;
  }

  Fillable* TupleFillable::integer(int64_t x) {
    check();
    maybeupdate(index_, contents_[index_].get()->integer(x));
    return this;
  }

  Fillable* TupleFillable::real(double x) {
    check();
    maybeupdate(index_, contents_[index_].get()->real(x));
    return this;
  }

  Fillable* TupleFillable::beginlist() {
    check();
    maybeupdate(index_, contents_[index_].get()->beginlist());
    return this;
  }

  Fillable* TupleFillable::endlist() {
    check();
    maybeupdate(index_, contents_[index_].get()->endlist());
    return this;
  }

  Fillable* TupleFillable::begintuple(int64_t numfields) {
    if (index_ > contents_.size()) {
      for (int64_t i = 0;  i < numfields;  i++) {
        contents_.push_back(std::shared_ptr<Fillable>(new UnknownFillable(fillablearray_, options_)));
      }
    }

    if (contents_.size() == (size_t)numfields) {
      index_ = -1;
    }
    else {
      throw std::runtime_error("FIXME: turn this into a union");
    }

    return this;
  }

  Fillable* TupleFillable::index(int64_t index) {
    if (!(0 <= index  &&  index < contents_.size())) {
      throw std::invalid_argument(std::string("index ") + std::to_string(index) + std::string(" for tuple of length ") + std::to_string(contents_.size()));
    }
    index_ = (size_t)index;
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
    return this;
  }

  void TupleFillable::check() {
    if (index_ == -1) {
      throw std::invalid_argument("call 'index' before setting each tuple element");
    }
    if (contents_[index_].get()->length() > length_) {
      throw std::invalid_argument(std::string("tuple index ") + std::to_string(index_) + std::string(" filled more than once (missing call to 'index'?)"));
    }
  }

  void TupleFillable::maybeupdate(size_t i, Fillable* tmp) {
    if (tmp != contents_[i].get()  &&  tmp != nullptr) {
      contents_[i] = std::shared_ptr<Fillable>(tmp);
    }
  }
}
