// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identities.h"
#include "awkward/Index.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/type/OptionType.h"

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
    throw std::runtime_error("FIXME: IndexedFillable::active");
  }

  const std::shared_ptr<Fillable> IndexedFillable::null() {
    index_.append(-1);
    hasnull_ = true;
    return that_;
  }

  const std::shared_ptr<Fillable> IndexedFillable::boolean(bool x) {
    throw std::runtime_error("FIXME: IndexedFillable::boolean");
  }

  const std::shared_ptr<Fillable> IndexedFillable::integer(int64_t x) {
    throw std::runtime_error("FIXME: IndexedFillable::integer");
  }

  const std::shared_ptr<Fillable> IndexedFillable::real(double x) {
    throw std::runtime_error("FIXME: IndexedFillable::real");
  }

  const std::shared_ptr<Fillable> IndexedFillable::string(const char* x, int64_t length, const char* encoding) {
    throw std::runtime_error("FIXME: IndexedFillable::string");
  }

  const std::shared_ptr<Fillable> IndexedFillable::beginlist() {
    throw std::runtime_error("FIXME: IndexedFillable::beginlist");
  }

  const std::shared_ptr<Fillable> IndexedFillable::endlist() {
    throw std::runtime_error("FIXME: IndexedFillable::endlist");
  }

  const std::shared_ptr<Fillable> IndexedFillable::begintuple(int64_t numfields) {
    throw std::runtime_error("FIXME: IndexedFillable::begintuple");
  }

  const std::shared_ptr<Fillable> IndexedFillable::index(int64_t index) {
    throw std::runtime_error("FIXME: IndexedFillable::index");
  }

  const std::shared_ptr<Fillable> IndexedFillable::endtuple() {
    throw std::runtime_error("FIXME: IndexedFillable::endtuple");
  }

  const std::shared_ptr<Fillable> IndexedFillable::beginrecord(const char* name, bool check) {
    throw std::runtime_error("FIXME: IndexedFillable::beginrecord");
  }

  const std::shared_ptr<Fillable> IndexedFillable::field(const char* key, bool check) {
    throw std::runtime_error("FIXME: IndexedFillable::field");
  }

  const std::shared_ptr<Fillable> IndexedFillable::endrecord() {
    throw std::runtime_error("FIXME: IndexedFillable::endrecord");
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
      throw std::runtime_error("FIXME: IndexedFillable::append(array != array_)");
    }
    return that_;
  }

}
