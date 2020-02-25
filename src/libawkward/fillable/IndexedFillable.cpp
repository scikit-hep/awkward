// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identities.h"
#include "awkward/Index.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/type/OptionType.h"
#include "awkward/fillable/UnionFillable.h"

#include "awkward/fillable/IndexedFillable.h"

namespace awkward {
  const std::shared_ptr<Fillable> IndexedFillable::fromnulls(const FillableOptions& options, int64_t nullcount, const std::shared_ptr<Content>& array) {
    GrowableBuffer<int64_t> index = GrowableBuffer<int64_t>::full(options, -1, nullcount);
    std::shared_ptr<Fillable> out = std::make_shared<IndexedFillable>(options, index, array, nullcount != 0);
    out.get()->setthat(out);
    return out;
  }

  IndexedFillable::IndexedFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<Content>& array, bool hasnull)
      : options_(options)
      , index_(index)
      , array_(array)
      , arraylength_(array.get()->length())
      , hasnull_(hasnull) { }

  const Content* IndexedFillable::arrayptr() const {
    return array_.get();
  }

  const std::string IndexedFillable::classname() const {
    return "IndexedFillable";
  };

  int64_t IndexedFillable::length() const {
    return index_.length();
  }

  void IndexedFillable::clear() {
    index_.clear();
  }

  const std::shared_ptr<Content> IndexedFillable::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    if (hasnull_) {
      return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, array_);
    }
    else {
      return std::make_shared<IndexedArray64>(Identities::none(), util::Parameters(), index, array_);
    }
  }

  bool IndexedFillable::active() const {
    return false;
  }

  const std::shared_ptr<Fillable> IndexedFillable::null() {
    index_.append(-1);
    hasnull_ = true;
    return that_;
  }

  const std::shared_ptr<Fillable> IndexedFillable::boolean(bool x) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->boolean(x);
    return out;
  }

  const std::shared_ptr<Fillable> IndexedFillable::integer(int64_t x) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->integer(x);
    return out;
  }

  const std::shared_ptr<Fillable> IndexedFillable::real(double x) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->real(x);
    return out;
  }

  const std::shared_ptr<Fillable> IndexedFillable::string(const char* x, int64_t length, const char* encoding) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->string(x, length, encoding);
    return out;
  }

  const std::shared_ptr<Fillable> IndexedFillable::beginlist() {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->beginlist();
    return out;
  }

  const std::shared_ptr<Fillable> IndexedFillable::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  const std::shared_ptr<Fillable> IndexedFillable::begintuple(int64_t numfields) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  const std::shared_ptr<Fillable> IndexedFillable::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Fillable> IndexedFillable::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Fillable> IndexedFillable::beginrecord(const char* name, bool check) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  const std::shared_ptr<Fillable> IndexedFillable::field(const char* key, bool check) {
    throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Fillable> IndexedFillable::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Fillable> IndexedFillable::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      int64_t regular_at = at;
      if (regular_at < 0) {
        regular_at += arraylength_;
      }
      if (!(0 <= regular_at  &&  regular_at < arraylength_)) {
        throw std::invalid_argument(std::string("'append' index (") + std::to_string(at) + std::string(") out of bounds (") + std::to_string(arraylength_) + std::string(")"));
      }
      index_.append(regular_at);
    }
    else {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

}
