// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identities.h"
#include "awkward/Index.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/type/OptionType.h"
#include "awkward/fillable/UnionFillable.h"

#include "awkward/fillable/IndexedFillable.h"

namespace awkward {
  template <typename T>
  IndexedFillable<T>::IndexedFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<T>& array, bool hasnull)
      : options_(options)
      , index_(index)
      , array_(array)
      , hasnull_(hasnull) { }

  template <typename T>
  const Content* IndexedFillable<T>::arrayptr() const {
    return array_.get();
  }

  template <typename T>
  int64_t IndexedFillable<T>::length() const {
    return index_.length();
  }

  template <typename T>
  void IndexedFillable<T>::clear() {
    index_.clear();
  }

  template <typename T>
  bool IndexedFillable<T>::active() const {
    return false;
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::null() {
    index_.append(-1);
    hasnull_ = true;
    return that_;
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::boolean(bool x) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->boolean(x);
    return out;
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::integer(int64_t x) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->integer(x);
    return out;
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::real(double x) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->real(x);
    return out;
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::string(const char* x, int64_t length, const char* encoding) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->string(x, length, encoding);
    return out;
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::beginlist() {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->beginlist();
    return out;
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::begintuple(int64_t numfields) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::beginrecord(const char* name, bool check) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::field(const char* key, bool check) {
    throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
  }

  template <typename T>
  const std::shared_ptr<Fillable> IndexedFillable<T>::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  template class IndexedFillable<Content>;

  const std::shared_ptr<Fillable> IndexedGenericFillable::fromnulls(const FillableOptions& options, int64_t nullcount, const std::shared_ptr<Content>& array) {
    GrowableBuffer<int64_t> index = GrowableBuffer<int64_t>::full(options, -1, nullcount);
    std::shared_ptr<Fillable> out;
    if (std::shared_ptr<IndexedArray32> ptr = std::dynamic_pointer_cast<IndexedArray32>(array)) {
      out = std::make_shared<IndexedI32Fillable>(options, index, ptr, nullcount != 0);
    }
    else if (std::shared_ptr<IndexedArrayU32> ptr = std::dynamic_pointer_cast<IndexedArrayU32>(array)) {
      out = std::make_shared<IndexedIU32Fillable>(options, index, ptr, nullcount != 0);
    }
    else if (std::shared_ptr<IndexedArray64> ptr = std::dynamic_pointer_cast<IndexedArray64>(array)) {
      out = std::make_shared<IndexedI64Fillable>(options, index, ptr, nullcount != 0);
    }
    else if (std::shared_ptr<IndexedOptionArray32> ptr = std::dynamic_pointer_cast<IndexedOptionArray32>(array)) {
      out = std::make_shared<IndexedIO32Fillable>(options, index, ptr, nullcount != 0);
    }
    else if (std::shared_ptr<IndexedOptionArray64> ptr = std::dynamic_pointer_cast<IndexedOptionArray64>(array)) {
      out = std::make_shared<IndexedIO64Fillable>(options, index, ptr, nullcount != 0);
    }
    else {
      out = std::make_shared<IndexedGenericFillable>(options, index, array, nullcount != 0);
    }
    out.get()->setthat(out);
    return out;
  }

  IndexedGenericFillable::IndexedGenericFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<Content>& array, bool hasnull)
      : IndexedFillable<Content>(options, index, array, hasnull) { }

  const std::string IndexedGenericFillable::classname() const {
    return "IndexedGenericFillable";
  };

  const std::shared_ptr<Content> IndexedGenericFillable::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    if (hasnull_) {
      return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, array_);
    }
    else {
      return std::make_shared<IndexedArray64>(Identities::none(), util::Parameters(), index, array_);
    }
  }

  const std::shared_ptr<Fillable> IndexedGenericFillable::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append(at);
    }
    else {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  template class IndexedFillable<IndexedArray32>;

  IndexedI32Fillable::IndexedI32Fillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedArray32>& array, bool hasnull)
      : IndexedFillable<IndexedArray32>(options, index, array, hasnull) { }

  const std::string IndexedI32Fillable::classname() const {
    return "IndexedI32Fillable";
  };

  const std::shared_ptr<Content> IndexedI32Fillable::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    if (hasnull_) {
      return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, array_.get()->content());
    }
    else {
      return std::make_shared<IndexedArray64>(Identities::none(), util::Parameters(), index, array_.get()->content());
    }
  }

  const std::shared_ptr<Fillable> IndexedI32Fillable::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append((int64_t)array_.get()->index_at_nowrap(at));
    }
    else {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  template class IndexedFillable<IndexedArrayU32>;

  IndexedIU32Fillable::IndexedIU32Fillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedArrayU32>& array, bool hasnull)
      : IndexedFillable<IndexedArrayU32>(options, index, array, hasnull) { }

  const std::string IndexedIU32Fillable::classname() const {
    return "IndexedIU32Fillable";
  };

  const std::shared_ptr<Content> IndexedIU32Fillable::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    if (hasnull_) {
      return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, array_.get()->content());
    }
    else {
      return std::make_shared<IndexedArray64>(Identities::none(), util::Parameters(), index, array_.get()->content());
    }
  }

  const std::shared_ptr<Fillable> IndexedIU32Fillable::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append((int64_t)array_.get()->index_at_nowrap(at));
    }
    else {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  template class IndexedFillable<IndexedArray64>;

  IndexedI64Fillable::IndexedI64Fillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedArray64>& array, bool hasnull)
      : IndexedFillable<IndexedArray64>(options, index, array, hasnull) { }

  const std::string IndexedI64Fillable::classname() const {
    return "IndexedI64Fillable";
  };

  const std::shared_ptr<Content> IndexedI64Fillable::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    if (hasnull_) {
      return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, array_.get()->content());
    }
    else {
      return std::make_shared<IndexedArray64>(Identities::none(), util::Parameters(), index, array_.get()->content());
    }
  }

  const std::shared_ptr<Fillable> IndexedI64Fillable::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append(array_.get()->index_at_nowrap(at));
    }
    else {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  template class IndexedFillable<IndexedOptionArray32>;

  IndexedIO32Fillable::IndexedIO32Fillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedOptionArray32>& array, bool hasnull)
      : IndexedFillable<IndexedOptionArray32>(options, index, array, hasnull) { }

  const std::string IndexedIO32Fillable::classname() const {
    return "IndexedIO32Fillable";
  };

  const std::shared_ptr<Content> IndexedIO32Fillable::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, array_.get()->content());
  }

  const std::shared_ptr<Fillable> IndexedIO32Fillable::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append((int64_t)array_.get()->index_at_nowrap(at));
    }
    else {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  template class IndexedFillable<IndexedOptionArray64>;

  IndexedIO64Fillable::IndexedIO64Fillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedOptionArray64>& array, bool hasnull)
      : IndexedFillable<IndexedOptionArray64>(options, index, array, hasnull) { }

  const std::string IndexedIO64Fillable::classname() const {
    return "IndexedIO64Fillable";
  };

  const std::shared_ptr<Content> IndexedIO64Fillable::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, array_.get()->content());
  }

  const std::shared_ptr<Fillable> IndexedIO64Fillable::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append(array_.get()->index_at_nowrap(at));
    }
    else {
      std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

}
