// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/type/UnionType.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/BoolFillable.h"

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
    int64_t type, length;
    BoolFillable* fillable = getfillable<BoolFillable>(type, length);
    fillable->boolean(x);
    offsets_.append(length);
    types_.append(type);
    return this;
  }

  template <typename F>
  F* UnionFillable::getfillable(int64_t& type, int64_t& length) {
    type = 0;
    F* content = nullptr;
    for (auto x : contents_) {
      if (F* y = dynamic_cast<F*>(x.get())) {
        content = y;
        break;
      }
      type++;
    }
    if (content == nullptr) {
      content = new F(options_);
      contents_.push_back(std::shared_ptr<Fillable>(content));
    }
    length = content->length();
    return content;
  }
}
