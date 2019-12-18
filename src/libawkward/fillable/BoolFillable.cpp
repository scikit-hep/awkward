// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identity.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/UnionFillable.h"

#include "awkward/fillable/BoolFillable.h"

namespace awkward {
  const std::shared_ptr<Fillable> BoolFillable::fromempty(const FillableOptions& options) {
    std::shared_ptr<Fillable> out = std::make_shared<BoolFillable>(options, GrowableBuffer<uint8_t>::empty(options));
    out.get()->setthat(out);
    return out;
  }

  int64_t BoolFillable::length() const {
    return buffer_.length();
  }

  void BoolFillable::clear() {
    buffer_.clear();
  }

  const std::shared_ptr<Type> BoolFillable::type() const {
    return std::make_shared<PrimitiveType>(Type::Parameters(), PrimitiveType::boolean);
  }

  const std::shared_ptr<Content> BoolFillable::snapshot(const std::shared_ptr<Type> type) const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(bool) };
    return std::make_shared<NumpyArray>(Identity::none(), type, buffer_.ptr(), shape, strides, 0, sizeof(bool), "?");
  }

  bool BoolFillable::active() const {
    return false;
  }

  const std::shared_ptr<Fillable> BoolFillable::null() {
    std::shared_ptr<Fillable> out = OptionFillable::fromvalids(options_, that_);
    out.get()->null();
    return out;
  }

  const std::shared_ptr<Fillable> BoolFillable::boolean(bool x) {
    buffer_.append(x);
    return that_;
  }

  const std::shared_ptr<Fillable> BoolFillable::integer(int64_t x) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->integer(x);
    return out;
  }

  const std::shared_ptr<Fillable> BoolFillable::real(double x) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->real(x);
    return out;
  }

  const std::shared_ptr<Fillable> BoolFillable::string(const char* x, int64_t length, const char* encoding) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->string(x, length, encoding);
    return out;
  }

  const std::shared_ptr<Fillable> BoolFillable::beginlist() {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->beginlist();
    return out;
  }

  const std::shared_ptr<Fillable> BoolFillable::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  const std::shared_ptr<Fillable> BoolFillable::begintuple(int64_t numfields) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  const std::shared_ptr<Fillable> BoolFillable::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Fillable> BoolFillable::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Fillable> BoolFillable::beginrecord(const char* name, bool check) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  const std::shared_ptr<Fillable> BoolFillable::field(const char* key, bool check) {
    throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Fillable> BoolFillable::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }
}
