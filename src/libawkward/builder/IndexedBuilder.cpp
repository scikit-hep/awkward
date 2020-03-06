// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identities.h"
#include "awkward/Index.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/type/OptionType.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/IndexedBuilder.h"

namespace awkward {
  template <typename T>
  IndexedBuilder<T>::IndexedBuilder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<T>& array, bool hasnull)
      : options_(options)
      , index_(index)
      , array_(array)
      , hasnull_(hasnull) { }

  template <typename T>
  const Content* IndexedBuilder<T>::arrayptr() const {
    return array_.get();
  }

  template <typename T>
  int64_t IndexedBuilder<T>::length() const {
    return index_.length();
  }

  template <typename T>
  void IndexedBuilder<T>::clear() {
    index_.clear();
  }

  template <typename T>
  bool IndexedBuilder<T>::active() const {
    return false;
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::null() {
    index_.append(-1);
    hasnull_ = true;
    return that_;
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::boolean(bool x) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->boolean(x);
    return out;
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::integer(int64_t x) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->integer(x);
    return out;
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::real(double x) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->real(x);
    return out;
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::string(const char* x, int64_t length, const char* encoding) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->string(x, length, encoding);
    return out;
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::beginlist() {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginlist();
    return out;
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::begintuple(int64_t numfields) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::beginrecord(const char* name, bool check) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::field(const char* key, bool check) {
    throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
  }

  template <typename T>
  const std::shared_ptr<Builder> IndexedBuilder<T>::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  template class IndexedBuilder<Content>;

  const std::shared_ptr<Builder> IndexedGenericBuilder::fromnulls(const ArrayBuilderOptions& options, int64_t nullcount, const std::shared_ptr<Content>& array) {
    GrowableBuffer<int64_t> index = GrowableBuffer<int64_t>::full(options, -1, nullcount);
    std::shared_ptr<Builder> out;
    if (std::shared_ptr<IndexedArray32> ptr = std::dynamic_pointer_cast<IndexedArray32>(array)) {
      out = std::make_shared<IndexedI32Builder>(options, index, ptr, nullcount != 0);
    }
    else if (std::shared_ptr<IndexedArrayU32> ptr = std::dynamic_pointer_cast<IndexedArrayU32>(array)) {
      out = std::make_shared<IndexedIU32Builder>(options, index, ptr, nullcount != 0);
    }
    else if (std::shared_ptr<IndexedArray64> ptr = std::dynamic_pointer_cast<IndexedArray64>(array)) {
      out = std::make_shared<IndexedI64Builder>(options, index, ptr, nullcount != 0);
    }
    else if (std::shared_ptr<IndexedOptionArray32> ptr = std::dynamic_pointer_cast<IndexedOptionArray32>(array)) {
      out = std::make_shared<IndexedIO32Builder>(options, index, ptr, nullcount != 0);
    }
    else if (std::shared_ptr<IndexedOptionArray64> ptr = std::dynamic_pointer_cast<IndexedOptionArray64>(array)) {
      out = std::make_shared<IndexedIO64Builder>(options, index, ptr, nullcount != 0);
    }
    else {
      out = std::make_shared<IndexedGenericBuilder>(options, index, array, nullcount != 0);
    }
    out.get()->setthat(out);
    return out;
  }

  IndexedGenericBuilder::IndexedGenericBuilder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<Content>& array, bool hasnull)
      : IndexedBuilder<Content>(options, index, array, hasnull) { }

  const std::string IndexedGenericBuilder::classname() const {
    return "IndexedGenericBuilder";
  };

  const std::shared_ptr<Content> IndexedGenericBuilder::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    if (hasnull_) {
      return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, array_);
    }
    else {
      return std::make_shared<IndexedArray64>(Identities::none(), util::Parameters(), index, array_);
    }
  }

  const std::shared_ptr<Builder> IndexedGenericBuilder::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append(at);
    }
    else {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  template class IndexedBuilder<IndexedArray32>;

  IndexedI32Builder::IndexedI32Builder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedArray32>& array, bool hasnull)
      : IndexedBuilder<IndexedArray32>(options, index, array, hasnull) { }

  const std::string IndexedI32Builder::classname() const {
    return "IndexedI32Builder";
  };

  const std::shared_ptr<Content> IndexedI32Builder::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    if (hasnull_) {
      return std::make_shared<IndexedOptionArray64>(Identities::none(), array_.get()->content().get()->parameters(), index, array_.get()->content());
    }
    else {
      return std::make_shared<IndexedArray64>(Identities::none(), array_.get()->content().get()->parameters(), index, array_.get()->content());
    }
  }

  const std::shared_ptr<Builder> IndexedI32Builder::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append((int64_t)array_.get()->index_at_nowrap(at));
    }
    else {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  template class IndexedBuilder<IndexedArrayU32>;

  IndexedIU32Builder::IndexedIU32Builder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedArrayU32>& array, bool hasnull)
      : IndexedBuilder<IndexedArrayU32>(options, index, array, hasnull) { }

  const std::string IndexedIU32Builder::classname() const {
    return "IndexedIU32Builder";
  };

  const std::shared_ptr<Content> IndexedIU32Builder::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    if (hasnull_) {
      return std::make_shared<IndexedOptionArray64>(Identities::none(), array_.get()->content().get()->parameters(), index, array_.get()->content());
    }
    else {
      return std::make_shared<IndexedArray64>(Identities::none(), array_.get()->content().get()->parameters(), index, array_.get()->content());
    }
  }

  const std::shared_ptr<Builder> IndexedIU32Builder::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append((int64_t)array_.get()->index_at_nowrap(at));
    }
    else {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  template class IndexedBuilder<IndexedArray64>;

  IndexedI64Builder::IndexedI64Builder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedArray64>& array, bool hasnull)
      : IndexedBuilder<IndexedArray64>(options, index, array, hasnull) { }

  const std::string IndexedI64Builder::classname() const {
    return "IndexedI64Builder";
  };

  const std::shared_ptr<Content> IndexedI64Builder::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    if (hasnull_) {
      return std::make_shared<IndexedOptionArray64>(Identities::none(), array_.get()->content().get()->parameters(), index, array_.get()->content());
    }
    else {
      return std::make_shared<IndexedArray64>(Identities::none(), array_.get()->content().get()->parameters(), index, array_.get()->content());
    }
  }

  const std::shared_ptr<Builder> IndexedI64Builder::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append(array_.get()->index_at_nowrap(at));
    }
    else {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  template class IndexedBuilder<IndexedOptionArray32>;

  IndexedIO32Builder::IndexedIO32Builder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedOptionArray32>& array, bool hasnull)
      : IndexedBuilder<IndexedOptionArray32>(options, index, array, hasnull) { }

  const std::string IndexedIO32Builder::classname() const {
    return "IndexedIO32Builder";
  };

  const std::shared_ptr<Content> IndexedIO32Builder::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    return std::make_shared<IndexedOptionArray64>(Identities::none(), array_.get()->content().get()->parameters(), index, array_.get()->content());
  }

  const std::shared_ptr<Builder> IndexedIO32Builder::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append((int64_t)array_.get()->index_at_nowrap(at));
    }
    else {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  template class IndexedBuilder<IndexedOptionArray64>;

  IndexedIO64Builder::IndexedIO64Builder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedOptionArray64>& array, bool hasnull)
      : IndexedBuilder<IndexedOptionArray64>(options, index, array, hasnull) { }

  const std::string IndexedIO64Builder::classname() const {
    return "IndexedIO64Builder";
  };

  const std::shared_ptr<Content> IndexedIO64Builder::snapshot() const {
    Index64 index(index_.ptr(), 0, index_.length());
    return std::make_shared<IndexedOptionArray64>(Identities::none(), array_.get()->content().get()->parameters(), index, array_.get()->content());
  }

  const std::shared_ptr<Builder> IndexedIO64Builder::append(const std::shared_ptr<Content>& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append(array_.get()->index_at_nowrap(at));
    }
    else {
      std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
      out.get()->append(array, at);
      return out;
    }
    return that_;
  }

}
