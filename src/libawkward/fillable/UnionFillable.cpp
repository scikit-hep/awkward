// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/type/UnionType.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/BoolFillable.h"
#include "awkward/fillable/Int64Fillable.h"
#include "awkward/fillable/Float64Fillable.h"

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
    Fillable* out = OptionFillable::fromvalids(options_, this);
    out->null();
    return out;
  }

  Fillable* UnionFillable::boolean(bool x) {
    int8_t type;
    int64_t length;
    get1<BoolFillable>(type, length)->boolean(x);
    offsets_.append(length);
    types_.append(type);
    return this;
  }

  Fillable* UnionFillable::integer(int64_t x) {
    int8_t type;
    int64_t length;
    get2<Int64Fillable, Float64Fillable>(type, length)->integer(x);
    offsets_.append(length);
    types_.append(type);
    return this;
  }

  Fillable* UnionFillable::real(double x) {
    int8_t type;
    int64_t length;
    get2<Int64Fillable, Float64Fillable>(type, length)->real(x);
    offsets_.append(length);
    types_.append(type);
    return this;
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

  template <typename T>
  T* UnionFillable::maybenew(T* fillable, int64_t& length) {
    if (fillable == nullptr) {
      fillable = new T(options_);
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
