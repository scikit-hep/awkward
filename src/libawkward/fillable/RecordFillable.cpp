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
  }

  const std::shared_ptr<Type> RecordFillable::type() const {
    if (length_ == -1) {
      return std::shared_ptr<Type>(new UnknownType);
    }
    else {
      std::vector<std::shared_ptr<Type>> types;
      std::shared_ptr<RecordType::Lookup> lookup;
      std::shared_ptr<RecordType::ReverseLookup> reverselookup;
      for (size_t i = 0;  i < contents_.size();  i++) {
        types.push_back(contents_[i].get()->type());
        (*lookup.get())[keys_[i]] = i;
        reverselookup.get()->push_back(keys_[i]);
      }
      return std::shared_ptr<Type>(new RecordType(types, lookup, reverselookup));
    }
  }

  const std::shared_ptr<Content> RecordFillable::snapshot() const {
    if (length_ == -1) {
      return std::shared_ptr<Content>(new EmptyArray(Identity::none()));
    }
    else {
      std::vector<std::shared_ptr<Content>> contents;
      std::shared_ptr<RecordArray::Lookup> lookup;
      std::shared_ptr<RecordArray::ReverseLookup> reverselookup;
      for (size_t i = 0;  i < contents_.size();  i++) {
        contents.push_back(contents_[i].get()->snapshot());
        (*lookup.get())[keys_[i]] = i;
        reverselookup.get()->push_back(keys_[i]);
      }
      return std::shared_ptr<Content>(new RecordArray(Identity::none(), contents, lookup, reverselookup));
    }
  }

  bool RecordFillable::active() const {
    throw std::runtime_error("FIXME: RecordFillable::active");
  }

  Fillable* RecordFillable::null() {
    throw std::runtime_error("FIXME: RecordFillable::null");
  }

  Fillable* RecordFillable::boolean(bool x) {
    throw std::runtime_error("FIXME: RecordFillable::boolean");
  }

  Fillable* RecordFillable::integer(int64_t x) {
    throw std::runtime_error("FIXME: RecordFillable::integer");
  }

  Fillable* RecordFillable::real(double x) {
    throw std::runtime_error("FIXME: RecordFillable::real");
  }

  Fillable* RecordFillable::beginlist() {
    throw std::runtime_error("FIXME: RecordFillable::beginlist");
  }

  Fillable* RecordFillable::endlist() {
    throw std::runtime_error("FIXME: RecordFillable::endlist");
  }

  Fillable* RecordFillable::begintuple(int64_t numfields) {
    throw std::runtime_error("FIXME: RecordFillable::begintuple");
  }

  Fillable* RecordFillable::index(int64_t index) {
    throw std::runtime_error("FIXME: RecordFillable::index");
  }

  Fillable* RecordFillable::endtuple() {
    throw std::runtime_error("FIXME: RecordFillable::endtuple");
  }

  Fillable* RecordFillable::beginrecord(int64_t disambiguator) {
    throw std::runtime_error("FIXME: RecordFillable::beginrecord");
  }

  Fillable* RecordFillable::field_fast(const char* key) {
    throw std::runtime_error("FIXME: RecordFillable::field_fast");
  }

  Fillable* RecordFillable::field_check(const char* key) {
    throw std::runtime_error("FIXME: RecordFillable::field_check");
  }

  Fillable* RecordFillable::endrecord() {
    throw std::runtime_error("FIXME: RecordFillable::endrecord");
  }

}
