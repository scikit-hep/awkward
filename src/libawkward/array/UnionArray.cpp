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
  int64_t UnionArrayOf<T, I>::numcontents() const {
    return (int64_t)contents_.size();
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::content(int64_t index) const {
    return contents_[(size_t)index];
  }

  template <typename T, typename I>
  const std::string UnionArrayOf<T, I>::classname() const {
    if (std::is_same<T, uint8_t>::value) {
      if (std::is_same<I, int32_t>::value) {
        return "UnionArrayU8_32";
      }
      else if (std::is_same<I, uint32_t>::value) {
        return "UnionArrayU8_U32";
      }
      else if (std::is_same<I, int64_t>::value) {
        return "UnionArrayU8_64";
      }
    }
    return "UnrecognizedUnionArray";
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
    std::vector<std::shared_ptr<Type>> types;
    for (auto item : contents_) {
      types.push_back(item.get()->type());
    }
    return std::make_shared<UnionType>(parameters_, types);
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::astype(const std::shared_ptr<Type>& type) const {
    if (UnionType* raw = dynamic_cast<UnionType*>(type.get())) {
      std::vector<std::shared_ptr<Content>> contents;
      for (int64_t i = 0;  i < raw->numtypes();  i++) {
        // FIXME: union equivalence could be defined much more flexibly than this, but do it later...
        if (i >= contents_.size()) {
          throw std::invalid_argument(classname() + std::string(" cannot be converted to type ") + type.get()->tostring() + std::string(" because the number of possibilities doesn't match"));
        }
        contents.push_back(contents_[(size_t)i].get()->astype(raw->type(i)));
      }
      return std::make_shared<UnionArrayOf<T, I>>(identities_, parameters_, tags_, index_, contents);
    }
    else {
      throw std::invalid_argument(classname() + std::string(" cannot be converted to type ") + type.get()->tostring());
    }
  }

  template <typename T, typename I>
  const std::string UnionArrayOf<T, I>::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << ">\n";
    if (identities_.get() != nullptr) {
      out << identities_.get()->tostring_part(indent + std::string("    "), "", "\n");
    }
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    for (size_t i = 0;  i < contents_.size();  i++) {
      out << indent << "    <content index=\"" << i << "\">\n";
      out << contents_[i].get()->tostring_part(indent + std::string("        "), "", "\n");
      out << indent << "    </content>\n";
    }
    out << indent << "</" << classname() << ">" << post;
    return out.str();
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
