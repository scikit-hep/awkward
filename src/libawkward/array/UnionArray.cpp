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
  const IndexOf<I> UnionArrayOf<T, I>::regular_index(const IndexOf<T>& tags) {
    int64_t lentags = tags.length();
    IndexOf<I> outindex(lentags);
    struct Error err = util::awkward_unionarray_regular_index<T, I>(
      outindex.ptr().get(),
      tags.ptr().get(),
      tags.offset(),
      lentags);
    util::handle_error(err, "UnionArray", nullptr);
    return outindex;
  }

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
    if (!(0 <= index  &&  index < numcontents())) {
      throw std::invalid_argument(std::string("index ") + std::to_string(index) + std::string(" out of range for ") + classname() + std::string(" with ") + std::to_string(numcontents()) + std::string(" contents"));
    }
    return contents_[(size_t)index];
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::project(int64_t index) const {
    if (!(0 <= index  &&  index < numcontents())) {
      throw std::invalid_argument(std::string("index ") + std::to_string(index) + std::string(" out of range for ") + classname() + std::string(" with ") + std::to_string(numcontents()) + std::string(" contents"));
    }
    int64_t lentags = tags_.length();
    if (index_.length() < lentags) {
      util::handle_error(failure("len(index) < len(tags)", kSliceNone, kSliceNone), classname(), identities_.get());
    }
    int64_t lenout;
    Index64 tmpcarry(lentags);
    struct Error err = util::awkward_unionarray_project_64<T, I>(
      &lenout,
      tmpcarry.ptr().get(),
      tags_.ptr().get(),
      tags_.offset(),
      index_.ptr().get(),
      index_.offset(),
      lentags,
      index);
    util::handle_error(err, classname(), identities_.get());
    Index64 nextcarry(tmpcarry.ptr(), 0, lenout);
    return contents_[(size_t)index].get()->carry(nextcarry);
  }

  template <typename T, typename I>
  const std::string UnionArrayOf<T, I>::classname() const {
    if (std::is_same<T, int8_t>::value) {
      if (std::is_same<I, int32_t>::value) {
        return "UnionArray8_32";
      }
      else if (std::is_same<I, uint32_t>::value) {
        return "UnionArray8_U32";
      }
      else if (std::is_same<I, int64_t>::value) {
        return "UnionArray8_64";
      }
    }
    return "UnrecognizedUnionArray";
  }

  template <typename T, typename I>
  void UnionArrayOf<T, I>::setidentities() {
    if (length() <= kMaxInt32) {
      std::shared_ptr<Identities> newidentities = std::make_shared<Identities32>(Identities::newref(), Identities::FieldLoc(), 1, length());
      Identities32* rawidentities = reinterpret_cast<Identities32*>(newidentities.get());
      struct Error err = awkward_new_identities32(rawidentities->ptr().get(), length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
    else {
      std::shared_ptr<Identities> newidentities = std::make_shared<Identities64>(Identities::newref(), Identities::FieldLoc(), 1, length());
      Identities64* rawidentities = reinterpret_cast<Identities64*>(newidentities.get());
      struct Error err = awkward_new_identities64(rawidentities->ptr().get(), length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
  }

  template <typename T, typename I>
  void UnionArrayOf<T, I>::setidentities(const std::shared_ptr<Identities>& identities) {
    if (identities.get() == nullptr) {
      for (auto content : contents_) {
        content.get()->setidentities(identities);
      }
    }
    else {
      if (index_.length() < tags_.length()) {
        util::handle_error(failure("len(index) < len(tags)", kSliceNone, kSliceNone), classname(), identities_.get());
      }
      if (length() != identities.get()->length()) {
        util::handle_error(failure("content and its identities must have the same length", kSliceNone, kSliceNone), classname(), identities_.get());
      }
      for (size_t which = 0;  which < contents_.size();  which++) {
        std::shared_ptr<Content> content = contents_[which];
        std::shared_ptr<Identities> bigidentities = identities;
        if (content.get()->length() > kMaxInt32  ||  !std::is_same<I, int32_t>::value) {
          bigidentities = identities.get()->to64();
        }
        if (Identities32* rawidentities = dynamic_cast<Identities32*>(bigidentities.get())) {
          bool uniquecontents;
          std::shared_ptr<Identities> subidentities = std::make_shared<Identities32>(Identities::newref(), rawidentities->fieldloc(), rawidentities->width(), content.get()->length());
          Identities32* rawsubidentities = reinterpret_cast<Identities32*>(subidentities.get());
          struct Error err = util::awkward_identities32_from_unionarray<T, I>(
            &uniquecontents,
            rawsubidentities->ptr().get(),
            rawidentities->ptr().get(),
            tags_.ptr().get(),
            index_.ptr().get(),
            rawidentities->offset(),
            tags_.offset(),
            index_.offset(),
            content.get()->length(),
            length(),
            rawidentities->width(),
            (int64_t)which);
          util::handle_error(err, classname(), identities_.get());
          if (uniquecontents) {
            content.get()->setidentities(subidentities);
          }
          else {
            content.get()->setidentities(Identities::none());
          }
        }
        else if (Identities64* rawidentities = dynamic_cast<Identities64*>(bigidentities.get())) {
          bool uniquecontents;
          std::shared_ptr<Identities> subidentities = std::make_shared<Identities64>(Identities::newref(), rawidentities->fieldloc(), rawidentities->width(), content.get()->length());
          Identities64* rawsubidentities = reinterpret_cast<Identities64*>(subidentities.get());
          struct Error err = util::awkward_identities64_from_unionarray<T, I>(
            &uniquecontents,
            rawsubidentities->ptr().get(),
            rawidentities->ptr().get(),
            tags_.ptr().get(),
            index_.ptr().get(),
            rawidentities->offset(),
            tags_.offset(),
            index_.offset(),
            content.get()->length(),
            length(),
            rawidentities->width(),
            (int64_t)which);
          util::handle_error(err, classname(), identities_.get());
          if (uniquecontents) {
            content.get()->setidentities(subidentities);
          }
          else {
            content.get()->setidentities(Identities::none());
          }
        }
        else {
          throw std::runtime_error("unrecognized Identities specialization");
        }
      }
    }
    identities_ = identities;
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
        if (i >= (int64_t)contents_.size()) {
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
    out << tags_.tostring_part(indent + std::string("    "), "<tags>", "</tags>\n");
    out << index_.tostring_part(indent + std::string("    "), "<index>", "</index>\n");
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
    int64_t len = length();
    check_for_iteration();
    builder.beginlist();
    for (int64_t i = 0;  i < len;  i++) {
      getitem_at_nowrap(i).get()->tojson_part(builder);
    }
    builder.endlist();
  }

  template <typename T, typename I>
  void UnionArrayOf<T, I>::nbytes_part(std::map<size_t, int64_t>& largest) const {
    for (auto x : contents_) {
      x.get()->nbytes_part(largest);
    }
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  template <typename T, typename I>
  int64_t UnionArrayOf<T, I>::length() const {
    return tags_.length();
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::shallow_copy() const {
    return std::make_shared<UnionArrayOf<T, I>>(identities_, parameters_, tags_, index_, contents_);
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const {
    IndexOf<T> tags = copyindexes ? tags_.deep_copy() : tags_;
    IndexOf<I> index = copyindexes ? index_.deep_copy() : index_;
    std::vector<std::shared_ptr<Content>> contents;
    for (auto x : contents_) {
      contents.push_back(x.get()->deep_copy(copyarrays, copyindexes, copyidentities));
    }
    std::shared_ptr<Identities> identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<UnionArrayOf<T, I>>(identities, parameters_, tags, index, contents);
  }

  template <typename T, typename I>
  void UnionArrayOf<T, I>::check_for_iteration() const {
    if (index_.length() < tags_.length()) {
      util::handle_error(failure("len(index) < len(tags)", kSliceNone, kSliceNone), classname(), identities_.get());
    }
    if (identities_.get() != nullptr  &&  identities_.get()->length() < index_.length()) {
      util::handle_error(failure("len(identities) < len(array)", kSliceNone, kSliceNone), identities_.get()->classname(), nullptr);
    }
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_nothing() const {
    return getitem_range_nowrap(0, 0);
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    int64_t len = length();
    if (regular_at < 0) {
      regular_at += len;
    }
    if (!(0 <= regular_at  &&  regular_at < len)) {
      util::handle_error(failure("index out of range", kSliceNone, at), classname(), identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_at_nowrap(int64_t at) const {
    size_t tag = (size_t)tags_.getitem_at_nowrap(at);
    int64_t index = (int64_t)index_.getitem_at_nowrap(at);
    if (!(0 <= tag  &&  tag < contents_.size())) {
      util::handle_error(failure("not 0 <= tag[i] < numcontents", kSliceNone, at), classname(), identities_.get());
    }
    std::shared_ptr<Content> content = contents_[tag];
    if (!(0 <= index  &&  index < content.get()->length())) {
      util::handle_error(failure("index[i] > len(content(tag))", kSliceNone, at), classname(), identities_.get());
    }
    return content.get()->getitem_at_nowrap(index);
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), tags_.length());
    if (identities_.get() != nullptr  &&  regular_stop > identities_.get()->length()) {
      util::handle_error(failure("index out of range", kSliceNone, stop), identities_.get()->classname(), nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_range_nowrap(int64_t start, int64_t stop) const {
    std::shared_ptr<Identities> identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    return std::make_shared<UnionArrayOf<T, I>>(identities, parameters_, tags_.getitem_range_nowrap(start, stop), index_.getitem_range_nowrap(start, stop), contents_);
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_field(const std::string& key) const {
    std::vector<std::shared_ptr<Content>> contents;
    for (auto content : contents_) {
      contents.push_back(content.get()->getitem_field(key));
    }
    return std::make_shared<UnionArrayOf<T, I>>(identities_, util::Parameters(), tags_, index_, contents);
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_fields(const std::vector<std::string>& keys) const {
    std::vector<std::shared_ptr<Content>> contents;
    for (auto content : contents_) {
      contents.push_back(content.get()->getitem_fields(keys));
    }
    return std::make_shared<UnionArrayOf<T, I>>(identities_, util::Parameters(), tags_, index_, contents);
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const {
    if (head.get() == nullptr) {
      return shallow_copy();
    }
    else if (dynamic_cast<SliceAt*>(head.get())  ||  dynamic_cast<SliceRange*>(head.get())  ||  dynamic_cast<SliceArray64*>(head.get())) {
      std::vector<std::shared_ptr<Content>> outcontents;
      for (int64_t i = 0;  i < numcontents();  i++) {
        std::shared_ptr<Content> projection = project(i);
        outcontents.push_back(projection.get()->getitem_next(head, tail, advanced));
      }
      IndexOf<I> outindex = regular_index(tags_);
      return std::make_shared<UnionArrayOf<T, I>>(identities_, parameters_, tags_, outindex, outcontents);
    }
    else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
      return Content::getitem_next(*ellipsis, tail, advanced);
    }
    else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
      return Content::getitem_next(*newaxis, tail, advanced);
    }
    else if (SliceField* field = dynamic_cast<SliceField*>(head.get())) {
      return Content::getitem_next(*field, tail, advanced);
    }
    else if (SliceFields* fields = dynamic_cast<SliceFields*>(head.get())) {
      return Content::getitem_next(*fields, tail, advanced);
    }
    else {
      throw std::runtime_error("unrecognized slice type");
    }
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::carry(const Index64& carry) const {
    int64_t lentags = tags_.length();
    if (index_.length() < lentags) {
      util::handle_error(failure("len(index) < len(tags)", kSliceNone, kSliceNone), classname(), identities_.get());
    }
    int64_t lencarry = carry.length();
    IndexOf<T> nexttags(lencarry);
    struct Error err1 = util::awkward_index_carry_64<T>(
      nexttags.ptr().get(),
      tags_.ptr().get(),
      carry.ptr().get(),
      tags_.offset(),
      lentags,
      lencarry);
    util::handle_error(err1, classname(), identities_.get());
    IndexOf<I> nextindex(lencarry);
    struct Error err2 = util::awkward_index_carry_nocheck_64<I>(
      nextindex.ptr().get(),
      index_.ptr().get(),
      carry.ptr().get(),
      index_.offset(),
      lencarry);
    util::handle_error(err2, classname(), identities_.get());
    std::shared_ptr<Identities> identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }
    return std::make_shared<UnionArrayOf<T, I>>(identities, parameters_, nexttags, nextindex, contents_);
  }

  template <typename T, typename I>
  bool UnionArrayOf<T, I>::purelist_isregular() const {
    for (auto content : contents_) {
      if (!content.get()->purelist_isregular()) {
        return false;
      }
    }
    return true;
  }

  template <typename T, typename I>
  int64_t UnionArrayOf<T, I>::purelist_depth() const {
    bool first = true;
    int64_t out = -1;
    for (auto content : contents_) {
      if (first) {
        first = false;
        out = content.get()->purelist_depth();
      }
      else if (out != content.get()->purelist_depth()) {
        return -1;
      }
    }
    return out;
  }

  template <typename T, typename I>
  const std::pair<int64_t, int64_t> UnionArrayOf<T, I>::minmax_depth() const {
    if (contents_.empty()) {
      return std::pair<int64_t, int64_t>(0, 0);
    }
    int64_t min = kMaxInt64;
    int64_t max = 0;
    for (auto content : contents_) {
      std::pair<int64_t, int64_t> minmax = content.get()->minmax_depth();
      if (minmax.first < min) {
        min = minmax.first;
      }
      if (minmax.second > max) {
        max = minmax.second;
      }
    }
    return std::pair<int64_t, int64_t>(min, max);
  }

  template <typename T, typename I>
  int64_t UnionArrayOf<T, I>::numfields() const {
    return (int64_t)keys().size();
  }

  template <typename T, typename I>
  int64_t UnionArrayOf<T, I>::fieldindex(const std::string& key) const {
    throw std::invalid_argument("UnionArray breaks the one-to-one relationship between fieldindexes and keys");
  }

  template <typename T, typename I>
  const std::string UnionArrayOf<T, I>::key(int64_t fieldindex) const {
    throw std::invalid_argument("UnionArray breaks the one-to-one relationship between fieldindexes and keys");
  }

  template <typename T, typename I>
  bool UnionArrayOf<T, I>::haskey(const std::string& key) const {
    for (auto x : keys()) {
      if (x == key) {
        return true;
      }
    }
    return false;
  }

  template <typename T, typename I>
  const std::vector<std::string> UnionArrayOf<T, I>::keys() const {
    std::vector<std::string> out;
    if (contents_.empty()) {
      return out;
    }
    out = contents_[0].get()->keys();
    for (size_t i = 1;  i < contents_.size();  i++) {
      std::vector<std::string> tmp = contents_[i].get()->keys();
      for (int64_t j = (int64_t)out.size() - 1;  j >= 0;  j--) {
        bool found = false;
        for (size_t k = 0;  k < tmp.size();  k++) {
          if (tmp[k] == out[(size_t)j]) {
            found = true;
            break;
          }
        }
        if (!found) {
          out.erase(out.begin() + (size_t)j);
        }
      }
    }
    return out;
  }

  template <typename T, typename I>
  const Index64 UnionArrayOf<T, I>::count64() const {
    throw std::runtime_error("FIXME: UnionArray::count64");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::count(int64_t axis) const {
    throw std::runtime_error("FIXME: UnionArray::count");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::flatten(int64_t axis) const {
    std::vector<std::shared_ptr<Content>> contents;
    for (auto content : contents_) {
      contents.emplace_back(content.get()->flatten(axis));
    }
    return std::make_shared<UnionArrayOf<T, I>>(identities_, parameters_, tags_, index_, contents);
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: UnionArray::getitem_next(SliceAt)");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: UnionArray::getitem_next(SliceRange)");
  }

  template <typename T, typename I>
  const std::shared_ptr<Content> UnionArrayOf<T, I>::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: UnionArray::getitem_next(SliceArray64)");
  }

  template class UnionArrayOf<int8_t, int32_t>;
  template class UnionArrayOf<int8_t, uint32_t>;
  template class UnionArrayOf<int8_t, int64_t>;
}
