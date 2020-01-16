// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/type/UnionType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/Slice.h"

#include "awkward/array/UnionArray.h"

namespace awkward {
  template <typename T, typename I>
  UnionArrayOf<T, I>::UnionArrayOf(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const IndexOf<T> tags, const IndexOf<I>& index, const std::vector<std::shared_ptr<Content>>& contents)
      : Content(identities, parameters)
      , tags_(tags)
      , index_(index)
      , contents_(contents) { }

  template <typename T, typename I>
  const IndexOf<T> UnionArrayOf<T, I>::tags() const {
    return tags_;
  }

  template <typename T, typename I>
  const IndexOf<I> UnionArrayOf<T, I>::index() const {
    return index_;
  }

  template <typename T, typename I>
  const std::vector<std::shared_ptr<Content>> UnionArrayOf<T, I>::contents() const {
    return contents_;
  }

  template <typename T, typename I>
  const std::string UnionArrayOf<T, I>::classname() const {
    throw std::runtime_error("UnionArray::classname");
  }

  template <typename T, typename I>
  void UnionArrayOf<T, I>::setidentities() {
    throw std::runtime_error("UnionArray::setidentities");
  }

  template <typename T, typename I>
  void UnionArrayOf<T, I>::setidentities(const std::shared_ptr<Identities>& identities) {
    throw std::runtime_error("UnionArray::setidentities(identities)");
  }

  template <typename T, typename I>
  const std::shared_ptr<Type> UnionArrayOf<T, I>::type() const {
    throw std::runtime_error("UnionArray::type");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::astype(const std::shared_ptr<Type>& type) const {
    throw std::runtime_error("UnionArray::astype");
  }

  template <typename T, typename I>
  const std::string UnionArrayOf<T, I>::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    throw std::runtime_error("UnionArray::tostring_part");
  }

  template <typename T, typename I>
  void UnionArrayOf<T, I>::tojson_part(ToJson& builder) const {
    throw std::runtime_error("UnionArray::tojson_part");
  }

  template <typename T, typename I>
  int64_t UnionArrayOf<T, I>::length() const {
    throw std::runtime_error("UnionArray::length");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::shallow_copy() const {
    throw std::runtime_error("UnionArray::shallow_copy");
  }

  template <typename T, typename I>
  void UnionArrayOf<T, I>::check_for_iteration() const {
    throw std::runtime_error("UnionArray::check_for_iteration");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_nothing() const {
    throw std::runtime_error("UnionArray::getitem_nothing");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_at(int64_t at) const {
    throw std::runtime_error("UnionArray::getitem_at");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_at_nowrap(int64_t at) const {
    throw std::runtime_error("UnionArray::getitem_at_nowrap");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_range(int64_t start, int64_t stop) const {
    throw std::runtime_error("UnionArray::getitem_range");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_range_nowrap(int64_t start, int64_t stop) const {
    throw std::runtime_error("UnionArray::getitem_range_nowrap");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_field(const std::string& key) const {
    throw std::runtime_error("UnionArray::getitem_field");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::runtime_error("UnionArray::getitem_fields");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("UnionArray::getitem_next");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::carry(const Index64& carry) const {
    throw std::runtime_error("UnionArray::carry");
  }

  template <typename T, typename I>
  const std::pair<int64_t, int64_t> UnionArrayOf<T, I>::minmax_depth() const {
    throw std::runtime_error("UnionArray::minmax_depth");
  }

  template <typename T, typename I>
  int64_t UnionArrayOf<T, I>::numfields() const {
    throw std::runtime_error("UnionArray::numfields");
  }

  template <typename T, typename I>
  int64_t UnionArrayOf<T, I>::fieldindex(const std::string& key) const {
    throw std::runtime_error("UnionArray::fieldindex");
  }

  template <typename T, typename I>
  const std::string UnionArrayOf<T, I>::key(int64_t fieldindex) const {
    throw std::runtime_error("UnionArray::key");
  }

  template <typename T, typename I>
  bool UnionArrayOf<T, I>::haskey(const std::string& key) const {
    throw std::runtime_error("UnionArray::haskey");
  }

  template <typename T, typename I>
  const std::vector<std::string> UnionArrayOf<T, I>::keys() const {
    throw std::runtime_error("UnionArray::keys");
  }

  template <typename T, typename I>
  const Index64 UnionArrayOf<T, I>::count64() const {
    throw std::runtime_error("UnionArray::count64");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::count(int64_t axis) const {
    throw std::runtime_error("UnionArray::count");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::flatten(int64_t axis) const {
    throw std::runtime_error("UnionArray::flatten");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("UnionArray::getitem_next(SliceAt)");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("UnionArray::getitem_next(SliceRange)");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("UnionArray::getitem_next(SliceArray64)");
  }

  template class UnionArrayOf<uint8_t, int32_t>;
  template class UnionArrayOf<uint8_t, uint32_t>;
  template class UnionArrayOf<uint8_t, int64_t>;
}
