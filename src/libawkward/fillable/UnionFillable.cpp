// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/type/UnionType.h"
#include "awkward/fillable/FillableArray.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/BoolFillable.h"
#include "awkward/fillable/Int64Fillable.h"
#include "awkward/fillable/Float64Fillable.h"
#include "awkward/fillable/ListFillable.h"
#include "awkward/fillable/TupleFillable.h"
#include "awkward/fillable/RecordFillable.h"

#include "awkward/fillable/UnionFillable.h"

namespace awkward {
  int64_t UnionFillable::length() const {
    return types_.length();
  }

  void UnionFillable::clear() {
    types_.clear();
    offsets_.clear();
    for (auto x : contents_) {
      x.get()->clear();
    }
  }

  const std::shared_ptr<Type> UnionFillable::type() const {
    std::vector<std::shared_ptr<Type>> types;
    for (auto x : contents_) {
      types.push_back(x.get()->type());
    }
    return std::shared_ptr<Type>(new UnionType(types));
  }

  const std::shared_ptr<Content> UnionFillable::snapshot() const {
    throw std::runtime_error("UnionFillable::snapshot() needs UnionArray");
  }

  Fillable* UnionFillable::null() {
    Fillable* out = OptionFillable::fromvalids(fillablearray_, options_, this);
    out->null();
    return out;
  }

  bool UnionFillable::active() const {
    throw std::runtime_error("FIXME: UnionFillable::active");
  }

  Fillable* UnionFillable::boolean(bool x) {
    throw std::runtime_error("FIXME: UnionFillable::boolean");
  }

  Fillable* UnionFillable::integer(int64_t x) {
    throw std::runtime_error("FIXME: UnionFillable::integer");
  }

  Fillable* UnionFillable::real(double x) {
    throw std::runtime_error("FIXME: UnionFillable::real");
  }

  Fillable* UnionFillable::beginlist() {
    throw std::runtime_error("FIXME: UnionFillable::beginlist");
  }

  Fillable* UnionFillable::endlist() {
    throw std::runtime_error("FIXME: UnionFillable::endlist");
  }

  Fillable* UnionFillable::begintuple(int64_t numfields) {
    throw std::runtime_error("FIXME: UnionFillable::begintuple");
  }

  Fillable* UnionFillable::index(int64_t index) {
    throw std::runtime_error("FIXME: UnionFillable::index");
  }

  Fillable* UnionFillable::endtuple() {
    throw std::runtime_error("FIXME: UnionFillable::endtuple");
  }

  Fillable* UnionFillable::beginrecord(int64_t disambiguator) {
    throw std::runtime_error("FIXME: UnionFillable::beginrecord");
  }

  Fillable* UnionFillable::field_fast(const char* key) {
    throw std::runtime_error("FIXME: UnionFillable::field_fast");
  }

  Fillable* UnionFillable::field_check(const char* key) {
    throw std::runtime_error("FIXME: UnionFillable::field_check");
  }

  Fillable* UnionFillable::endrecord() {
    throw std::runtime_error("FIXME: UnionFillable::endrecord");
  }

  template <typename T>
  T* UnionFillable::findfillable(int8_t& type) {
    type = 0;
    for (auto x : contents_) {
      if (T* raw = dynamic_cast<T*>(x.get())) {
        return raw;
      }
      type++;
    }
    return nullptr;
  }

  TupleFillable* UnionFillable::findtuple(int8_t& type, int64_t numfields) {
    type = 0;
    for (auto x : contents_) {
      if (TupleFillable* raw = dynamic_cast<TupleFillable*>(x.get())) {
        if (raw->numfields() == numfields) {
          return raw;
        }
        type++;
      }
    }
    return nullptr;
  }

  RecordFillable* UnionFillable::findrecord(int8_t& type, int64_t disambiguator) {
    type = 0;
    for (auto x : contents_) {
      if (RecordFillable* raw = dynamic_cast<RecordFillable*>(x.get())) {
        if (raw->disambiguator() == disambiguator) {
          return raw;
        }
        type++;
      }
    }
    return nullptr;
  }

  template <typename T>
  T* UnionFillable::maybenew(T* fillable, int64_t& length) {
    if (fillable == nullptr) {
      fillable = new T(fillablearray_, options_);
      contents_.push_back(std::shared_ptr<Fillable>(fillable));
    }
    length = fillable->length();
    return fillable;
  }

  template <typename T1>
  Fillable* UnionFillable::get1(int8_t& type, int64_t& length) {
    return maybenew<T1>(findfillable<T1>(type), length);
  }

  template <typename T1, typename T2>
  Fillable* UnionFillable::get2(int8_t& type, int64_t& length) {
    Fillable* fillable = findfillable<T1>(type);
    if (fillable == nullptr) {
      return maybenew<T2>(findfillable<T2>(type), length);
    }
    else {
      length = fillable->length();
      return fillable;
    }
  }
}
