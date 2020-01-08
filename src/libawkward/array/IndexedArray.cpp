// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/type/OptionType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/Slice.h"
#include "awkward/array/None.h"

#include "awkward/array/IndexedArray.h"

namespace awkward {
  template <typename T, bool ISOPTION>
  IndexedArrayOf<T, ISOPTION>::IndexedArrayOf(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const IndexOf<T>& index, const std::shared_ptr<Content>& content)
      : Content(identities, parameters)
      , index_(index)
      , content_(content) { }

  template <typename T, bool ISOPTION>
  const IndexOf<T> IndexedArrayOf<T, ISOPTION>::index() const {
    return index_;
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::content() const {
    return content_;
  }

  template <typename T, bool ISOPTION>
  bool IndexedArrayOf<T, ISOPTION>::isoption() const {
    return ISOPTION;
  }

  template <typename T, bool ISOPTION>
  const std::string IndexedArrayOf<T, ISOPTION>::classname() const {
    if (ISOPTION) {
      if (std::is_same<T, int32_t>::value) {
        return "IndexedOptionArray32";
      }
      else if (std::is_same<T, int64_t>::value) {
        return "IndexedOptionArray64";
      }
    }
    else {
      if (std::is_same<T, int32_t>::value) {
        return "IndexedArray32";
      }
      else if (std::is_same<T, uint32_t>::value) {
        return "IndexedArrayU32";
      }
      else if (std::is_same<T, int64_t>::value) {
        return "IndexedArray64";
      }
    }
    return "UnrecognizedIndexedArray";
  }

  template <typename T, bool ISOPTION>
  void IndexedArrayOf<T, ISOPTION>::setidentities(const std::shared_ptr<Identities>& identities) {
    throw std::runtime_error("FIXME: IndexedArrayOf<T, ISOPTION>::setidentities(identities)");
  }

  template <typename T, bool ISOPTION>
  void IndexedArrayOf<T, ISOPTION>::setidentities() {
    throw std::runtime_error("FIXME: IndexedArrayOf<T, ISOPTION>::setidentities()");
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Type> IndexedArrayOf<T, ISOPTION>::type() const {
    if (ISOPTION) {
      return std::make_shared<OptionType>(parameters_, content_.get()->type());
    }
    else {
      std::shared_ptr<Type> out = content_.get()->type();
      out.get()->setparameters(parameters_);
      return out;
    }
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::astype(const std::shared_ptr<Type>& type) const {
    if (ISOPTION) {
      if (OptionType* raw = dynamic_cast<OptionType*>(type.get())) {
        return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities_, type.get()->parameters(), index_, content_.get()->astype(raw->type()));
      }
      else {
        throw std::invalid_argument(classname() + std::string(" cannot be converted to type ") + type.get()->tostring());
      }
    }
    else {
      return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities_, parameters_, index_, content_.get()->astype(type));
    }
  }

  template <typename T, bool ISOPTION>
  const std::string IndexedArrayOf<T, ISOPTION>::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << ">\n";
    if (identities_.get() != nullptr) {
      out << identities_.get()->tostring_part(indent + std::string("    "), "", "\n");
    }
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    out << index_.tostring_part(indent + std::string("    "), "<index>", "</index>\n");
    out << content_.get()->tostring_part(indent + std::string("    "), "<content>", "</content>\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  template <typename T, bool ISOPTION>
  void IndexedArrayOf<T, ISOPTION>::tojson_part(ToJson& builder) const {
    int64_t len = length();
    builder.beginlist();
    for (int64_t i = 0;  i < len;  i++) {
      getitem_at_nowrap(i).get()->tojson_part(builder);
    }
    builder.endlist();
  }

  template <typename T, bool ISOPTION>
  int64_t IndexedArrayOf<T, ISOPTION>::length() const {
    return index_.length();
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::shallow_copy() const {
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities_, parameters_, index_, content_);
  }

  template <typename T, bool ISOPTION>
  void IndexedArrayOf<T, ISOPTION>::check_for_iteration() const {
    if (identities_.get() != nullptr  &&  identities_.get()->length() < index_.length()) {
      util::handle_error(failure("len(identities) < len(array)", kSliceNone, kSliceNone), identities_.get()->classname(), nullptr);
    }
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_nothing() const {
    return content_.get()->getitem_range_nowrap(0, 0);
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += index_.length();
    }
    if (!(0 <= regular_at  &&  regular_at < index_.length())) {
      util::handle_error(failure("index out of range", kSliceNone, at), classname(), identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_at_nowrap(int64_t at) const {
    int64_t index = (int64_t)index_.getitem_at_nowrap(at);
    if (index < 0) {
      if (ISOPTION) {
        return none;
      }
      else {
        util::handle_error(failure("index[i] < 0", kSliceNone, at), classname(), identities_.get());
      }
    }
    int64_t lencontent = content_.get()->length();
    if (index >= lencontent) {
      util::handle_error(failure("index[i] >= len(content)", kSliceNone, at), classname(), identities_.get());
    }
    return content_.get()->getitem_at_nowrap(index);
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), index_.length());
    if (identities_.get() != nullptr  &&  regular_stop > identities_.get()->length()) {
      util::handle_error(failure("index out of range", kSliceNone, stop), identities_.get()->classname(), nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_range_nowrap(int64_t start, int64_t stop) const {
    std::shared_ptr<Identities> identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities, parameters_, index_.getitem_range_nowrap(start, stop), content_);
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_field(const std::string& key) const {
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities_, util::Parameters(), index_, content_.get()->getitem_field(key));
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_fields(const std::vector<std::string>& keys) const {
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities_, util::Parameters(), index_, content_.get()->getitem_fields(keys));
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::carry(const Index64& carry) const {
    throw std::runtime_error("FIXME: IndexedArrayOf<T, ISOPTION>::carry");
  }

  template <typename T, bool ISOPTION>
  const std::pair<int64_t, int64_t> IndexedArrayOf<T, ISOPTION>::minmax_depth() const {
    return content_.get()->minmax_depth();
  }

  template <typename T, bool ISOPTION>
  int64_t IndexedArrayOf<T, ISOPTION>::numfields() const {
    return content_.get()->numfields();
  }

  template <typename T, bool ISOPTION>
  int64_t IndexedArrayOf<T, ISOPTION>::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  template <typename T, bool ISOPTION>
  const std::string IndexedArrayOf<T, ISOPTION>::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  template <typename T, bool ISOPTION>
  bool IndexedArrayOf<T, ISOPTION>::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  template <typename T, bool ISOPTION>
  const std::vector<std::string> IndexedArrayOf<T, ISOPTION>::keys() const {
    return content_.get()->keys();
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::flatten(int64_t axis) const {
    throw std::runtime_error("FIXME: IndexedArrayOf<T, ISOPTION>::flatten");
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("IndexedArrayOf<T, ISOPTION>::getitem_next");
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("IndexedArrayOf<T, ISOPTION>::getitem_next");
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("IndexedArrayOf<T, ISOPTION>::getitem_next");
  }

  template class IndexedArrayOf<int32_t, false>;
  template class IndexedArrayOf<uint32_t, false>;
  template class IndexedArrayOf<int64_t, false>;
  template class IndexedArrayOf<int32_t, true>;
  template class IndexedArrayOf<int64_t, true>;
}
