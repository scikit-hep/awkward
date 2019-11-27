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

  bool TupleFillable::active() const {
    throw std::runtime_error("FIXME: TupleFillable::active");
  }

  Fillable* TupleFillable::null() {
    throw std::runtime_error("FIXME: TupleFillable::null");
  }

  Fillable* TupleFillable::boolean(bool x) {
    throw std::runtime_error("FIXME: TupleFillable::boolean");
  }

  Fillable* TupleFillable::integer(int64_t x) {
    throw std::runtime_error("FIXME: TupleFillable::integer");
  }

  Fillable* TupleFillable::real(double x) {
    throw std::runtime_error("FIXME: TupleFillable::real");
  }

  Fillable* TupleFillable::beginlist() {
    throw std::runtime_error("FIXME: TupleFillable::beginlist");
  }

  Fillable* TupleFillable::endlist() {
    throw std::runtime_error("FIXME: TupleFillable::endlist");
  }

  Fillable* TupleFillable::begintuple(int64_t numfields) {
    throw std::runtime_error("FIXME: TupleFillable::begintuple");
  }

  Fillable* TupleFillable::index(int64_t index) {
    throw std::runtime_error("FIXME: TupleFillable::index");
  }

  Fillable* TupleFillable::endtuple() {
    throw std::runtime_error("FIXME: TupleFillable::endtuple");
  }

  Fillable* TupleFillable::beginrecord(int64_t disambiguator) {
    throw std::runtime_error("FIXME: TupleFillable::beginrecord");
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
