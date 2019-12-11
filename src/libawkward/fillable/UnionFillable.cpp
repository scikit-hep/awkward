// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/Index.h"
#include "awkward/type/UnionType.h"
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
    return std::shared_ptr<Type>(new UnionType(Type::Parameters(), types));
  }

  const std::shared_ptr<Content> UnionFillable::snapshot() const {
    Index8 types(types_.ptr(), 0, types_.length());
    Index64 offsets(offsets_.ptr(), 0, offsets_.length());
    throw std::runtime_error("UnionFillable::snapshot() needs UnionArray");
  }

  bool UnionFillable::active() const {
    throw std::runtime_error("FIXME: UnionFillable::active");
  }

  Fillable* UnionFillable::null() {
    if (current_ == -1) {
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
    else {
      contents_[(size_t)current_].get()->null();
      return this;
    }
  }

  Fillable* UnionFillable::boolean(bool x) {
    if (current_ == -1) {
      Fillable* tofill = nullptr;
      int8_t i = 0;
      for (auto content : contents_) {
        if (dynamic_cast<BoolFillable*>(content.get()) != nullptr) {
          tofill = content.get();
          break;
        }
        i++;
      }
      if (tofill == nullptr) {
        tofill = BoolFillable::fromempty(options_);
        contents_.push_back(std::shared_ptr<Fillable>(tofill));
      }
      int64_t length = tofill->length();
      tofill->boolean(x);
      types_.append(i);
      offsets_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->boolean(x);
    }
    return this;
  }

  Fillable* UnionFillable::integer(int64_t x) {
    if (current_ == -1) {
      Fillable* tofill = nullptr;
      int8_t i = 0;
      for (auto content : contents_) {
        if (dynamic_cast<Int64Fillable*>(content.get()) != nullptr) {
          tofill = content.get();
          break;
        }
        i++;
      }
      if (tofill == nullptr) {
        tofill = Int64Fillable::fromempty(options_);
        contents_.push_back(std::shared_ptr<Fillable>(tofill));
      }
      int64_t length = tofill->length();
      tofill->integer(x);
      types_.append(i);
      offsets_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->integer(x);
    }
    return this;
  }

  Fillable* UnionFillable::real(double x) {
    if (current_ == -1) {
      Fillable* tofill = nullptr;
      int8_t i = 0;
      for (auto content : contents_) {
        if (dynamic_cast<Float64Fillable*>(content.get()) != nullptr) {
          tofill = content.get();
          break;
        }
        i++;
      }
      if (tofill == nullptr) {
        i = 0;
        for (auto content : contents_) {
          if (dynamic_cast<Int64Fillable*>(content.get()) != nullptr) {
            tofill = content.get();
            break;
          }
          i++;
        }
        if (tofill != nullptr) {
          tofill = Float64Fillable::fromint64(options_, dynamic_cast<Int64Fillable*>(tofill)->buffer());
          contents_[(size_t)i] = std::shared_ptr<Fillable>(tofill);
        }
        else {
          tofill = Float64Fillable::fromempty(options_);
          contents_.push_back(std::shared_ptr<Fillable>(tofill));
        }
      }
      int64_t length = tofill->length();
      tofill->real(x);
      types_.append(i);
      offsets_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->real(x);
    }
    return this;
  }

  Fillable* UnionFillable::beginlist() {
    if (current_ == -1) {
      Fillable* tofill = nullptr;
      int8_t i = 0;
      for (auto content : contents_) {
        if (dynamic_cast<ListFillable*>(content.get()) != nullptr) {
          tofill = content.get();
          break;
        }
        i++;
      }
      if (tofill == nullptr) {
        tofill = ListFillable::fromempty(options_);
        contents_.push_back(std::shared_ptr<Fillable>(tofill));
      }
      tofill->beginlist();
    }
    else {
      contents_[(size_t)current_].get()->beginlist();
    }
    return this;
  }

  Fillable* UnionFillable::endlist() {
    if (current_ == -1) {
      throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
    }
    else {
      int64_t length = contents_[(size_t)current_].get()->length();
      contents_[(size_t)current_].get()->endlist();
      if (length != contents_[(size_t)current_].get()->length()) {
        types_.append(current_);
        offsets_.append(length);
        current_ = -1;
      }
    }
    return this;
  }

  Fillable* UnionFillable::begintuple(int64_t numfields) {
    if (current_ == -1) {
      Fillable* tofill = nullptr;
      int8_t i = 0;
      for (auto content : contents_) {
        if (TupleFillable* raw = dynamic_cast<TupleFillable*>(content.get())) {
          if (raw->length() == -1  ||  raw->numfields() == numfields) {
            tofill = content.get();
            break;
          }
        }
        i++;
      }
      if (tofill == nullptr) {
        tofill = TupleFillable::fromempty(options_);
        contents_.push_back(std::shared_ptr<Fillable>(tofill));
      }
      tofill->begintuple(numfields);
      current_ = i;
    }
    else {
      contents_[(size_t)current_].get()->begintuple(numfields);
    }
    return this;
  }

  Fillable* UnionFillable::index(int64_t index) {
    if (current_ == -1) {
      throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
    }
    else {
      contents_[(size_t)current_].get()->index(index);
    }
    return this;
  }

  Fillable* UnionFillable::endtuple() {
    if (current_ == -1) {
      throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
    }
    else {
      int64_t length = contents_[(size_t)current_].get()->length();
      contents_[(size_t)current_].get()->endtuple();
      if (length != contents_[(size_t)current_].get()->length()) {
        types_.append(current_);
        offsets_.append(length);
        current_ = -1;
      }
    }
    return this;
  }

  Fillable* UnionFillable::beginrecord(int64_t disambiguator) {
    if (current_ == -1) {
      Fillable* tofill = nullptr;
      int8_t i = 0;
      for (auto content : contents_) {
        if (RecordFillable* raw = dynamic_cast<RecordFillable*>(content.get())) {
          if (raw->length() == -1  ||  raw->disambiguator() == disambiguator) {
            tofill = content.get();
            break;
          }
        }
        i++;
      }
      if (tofill == nullptr) {
        tofill = RecordFillable::fromempty(options_);
        contents_.push_back(std::shared_ptr<Fillable>(tofill));
      }
      tofill->beginrecord(disambiguator);
      current_ = i;
    }
    else {
      contents_[(size_t)current_].get()->beginrecord(disambiguator);
    }
    return this;
  }

  Fillable* UnionFillable::field_fast(const char* key) {
    if (current_ == -1) {
      throw std::invalid_argument("called 'field_fast' without 'beginrecord' at the same level before it");
    }
    else {
      contents_[(size_t)current_].get()->field_fast(key);
    }
    return this;
  }

  Fillable* UnionFillable::field_check(const char* key) {
    if (current_ == -1) {
      throw std::invalid_argument("called 'field_check' without 'beginrecord' at the same level before it");
    }
    else {
      contents_[(size_t)current_].get()->field_check(key);
    }
    return this;
  }

  Fillable* UnionFillable::endrecord() {
    if (current_ == -1) {
      throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
    }
    else {
      int64_t length = contents_[(size_t)current_].get()->length();
      contents_[(size_t)current_].get()->endrecord();
      if (length != contents_[(size_t)current_].get()->length()) {
        types_.append(current_);
        offsets_.append(length);
        current_ = -1;
      }
    }
    return this;
  }

}
