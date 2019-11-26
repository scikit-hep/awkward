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

  Fillable* UnionFillable::beginlist() {
    int8_t type;
    int64_t length;
    get1<ListFillable>(type, length)->beginlist();
    offsets_.append(length);
    types_.append(type);
    return this;
  }

  Fillable* UnionFillable::endlist() {
    return nullptr;
  }

  Fillable* UnionFillable::begintuple(int64_t numfields) {
    int8_t type;
    int64_t length;
    activerec_ = maybenew<TupleFillable>(findtuple(type, numfields), length);
    return this;
  }

  Fillable* UnionFillable::index(int64_t index) {
    if (dynamic_cast<TupleFillable*>(activerec_) == nullptr) {
      throw std::invalid_argument("'index' should only be called in a tuple (did you forget to call 'begintuple'?)");
    }
    activerec_->index(index);
    return this;
  }

  Fillable* UnionFillable::endtuple() {
    if (dynamic_cast<TupleFillable*>(activerec_) == nullptr) {
      throw std::invalid_argument("'endtuple' should only be called in a tuple (did you forget to call 'begintuple'?)");
    }
    int8_t type = 0;
    int64_t length = activerec_->length();
    activerec_->endtuple();
    for (auto x : contents_) {
      if (x.get() == activerec_) {
        break;
      }
      type++;
    }
    offsets_.append(length);
    types_.append(type);
    activerec_ = nullptr;
    return this;
  }

  Fillable* UnionFillable::beginrecord(int64_t disambiguator) {
    int8_t type;
    int64_t length;
    activerec_ = maybenew<RecordFillable>(findrecord(type, disambiguator), length);
    return this;
  }

  Fillable* UnionFillable::field_fast(const char* key) {
    if (dynamic_cast<RecordFillable*>(activerec_) == nullptr) {
      throw std::invalid_argument("'field_fast' should only be called in a record (did you forget to call 'beginrecord'?)");
    }
    activerec_->field_fast(key);
    return this;
  }

  Fillable* UnionFillable::field_check(const char* key) {
    if (dynamic_cast<RecordFillable*>(activerec_) == nullptr) {
      throw std::invalid_argument("'field_check' should only be called in a record (did you forget to call 'beginrecord'?)");
    }
    activerec_->field_check(key);
    return this;
  }

  Fillable* UnionFillable::endrecord() {
    if (dynamic_cast<RecordFillable*>(activerec_) == nullptr) {
      throw std::invalid_argument("'endrecord' should only be called in a record (did you forget to call 'beginrecord'?)");
    }
    int8_t type = 0;
    int64_t length = activerec_->length();
    activerec_->endrecord();
    for (auto x : contents_) {
      if (x.get() == activerec_) {
        break;
      }
      type++;
    }
    offsets_.append(length);
    types_.append(type);
    activerec_ = nullptr;
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
