// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/Index.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/type/ListType.h"
#include "awkward/fillable/FillableArray.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/UnionFillable.h"

#include "awkward/fillable/TupleFillable.h"

namespace awkward {
  int64_t TupleFillable::length() const {
    int64_t out = -1;
    for (auto x : contents_) {
      int64_t len = x.get()->length();
      if (out < 0  ||  out > len) {
        out = len;
      }
    }
    return out;
  }

  void TupleFillable::clear() {
    for (auto x : contents_) {
      x.get()->clear();
    }
  }

  const std::shared_ptr<Type> TupleFillable::type() const {
    throw std::runtime_error("FIXME: TupleFillable::type");
  }

  const std::shared_ptr<Content> TupleFillable::snapshot() const {
    throw std::runtime_error("FIXME: TupleFillable::snapshot");
  }

  Fillable* TupleFillable::null() {
    if (index_ == -1) {
      throw std::invalid_argument("call 'index' before setting each tuple element");
    }
    maybeupdate(contents_[index_].get()->null());
    return this;
  }

  Fillable* TupleFillable::boolean(bool x) {
    if (index_ == -1) {
      throw std::invalid_argument("call 'index' before setting each tuple element");
    }
    maybeupdate(contents_[index_].get()->boolean(x));
    return this;
  }

  Fillable* TupleFillable::integer(int64_t x) {
    if (index_ == -1) {
      throw std::invalid_argument("call 'index' before setting each tuple element");
    }
    maybeupdate(contents_[index_].get()->integer(x));
    return this;
  }

  Fillable* TupleFillable::real(double x) {
    if (index_ == -1) {
      throw std::invalid_argument("call 'index' before setting each tuple element");
    }
    maybeupdate(contents_[index_].get()->real(x));
    return this;
  }

  Fillable* TupleFillable::beginlist() {
    if (index_ == -1) {
      throw std::invalid_argument("call 'index' before setting each tuple element");
    }
    maybeupdate(contents_[index_].get()->beginlist());
    return this;
  }

  Fillable* TupleFillable::endlist() {
    if (index_ == -1) {
      throw std::invalid_argument("call 'index' before setting each tuple element");
    }
    maybeupdate(contents_[index_].get()->endlist());
    return this;
  }

  Fillable* TupleFillable::begintuple(int64_t numfields) {
    throw std::runtime_error("FIXME: TupleFillable::begintuple");
  }

  Fillable* TupleFillable::index(int64_t index) {
    if (!(0 <= index < contents_.size())) {
      throw std::invalid_argument(std::string("index ") + std::to_string(index) + std::string(" for tuple of length ") + std::to_string(contents_.size()));
    }
    index_ = (size_t)index;
  }

  Fillable* TupleFillable::endtuple() {
    throw std::runtime_error("FIXME: TupleFillable::endtuple");
  }

  void TupleFillable::maybeupdate(Fillable* tmp) {
    if (tmp != contents_[index_].get()  &&  contents_[index_] != nullptr) {
      contents_[index_] = std::shared_ptr<Fillable>(tmp);
    }
  }
}
