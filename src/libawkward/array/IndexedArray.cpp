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
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::project() const {
    if (ISOPTION) {
      int64_t numnull;
      struct Error err1 = util::awkward_indexedarray_numnull<T>(
        &numnull,
        index_.ptr().get(),
        index_.offset(),
        index_.length());
      util::handle_error(err1, classname(), identities_.get());

      Index64 nextcarry(length() - numnull);
      struct Error err2 = util::awkward_indexedarray_flatten_nextcarry_64<T>(
        nextcarry.ptr().get(),
        index_.ptr().get(),
        index_.offset(),
        index_.length(),
        content_.get()->length());
      util::handle_error(err2, classname(), identities_.get());

      return content_.get()->carry(nextcarry);
    }
    else {
      Index64 nextcarry(length());
      struct Error err = util::awkward_indexedarray_getitem_nextcarry_64<T>(
        nextcarry.ptr().get(),
        index_.ptr().get(),
        index_.offset(),
        index_.length(),
        content_.get()->length());
      util::handle_error(err, classname(), identities_.get());

      return content_.get()->carry(nextcarry);
    }
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::project(const Index8& mask) const {
    if (index_.length() != mask.length()) {
      throw std::invalid_argument(std::string("mask length (") + std::to_string(mask.length()) + std::string(") is not equal to ") + classname() + std::string(" length (") + std::to_string(index_.length()) + std::string(")"));
    }

    IndexOf<T> nextindex(index_.length());
    struct Error err = util::awkward_indexedarray_andmask_8<T>(
      nextindex.ptr().get(),
      mask.ptr().get(),
      mask.offset(),
      index_.ptr().get(),
      index_.offset(),
      index_.length());
    util::handle_error(err, classname(), identities_.get());

    IndexedArrayOf<T, true> next(identities_, parameters_, nextindex, content_);
    return next.project();
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
    if (identities.get() == nullptr) {
      content_.get()->setidentities(identities);
    }
    else {
      if (length() != identities.get()->length()) {
        util::handle_error(failure("content and its identities must have the same length", kSliceNone, kSliceNone), classname(), identities_.get());
      }
      std::shared_ptr<Identities> bigidentities = identities;
      if (content_.get()->length() > kMaxInt32  ||  !std::is_same<T, int32_t>::value) {
        bigidentities = identities.get()->to64();
      }
      if (Identities32* rawidentities = dynamic_cast<Identities32*>(bigidentities.get())) {
        bool uniquecontents;
        std::shared_ptr<Identities> subidentities = std::make_shared<Identities32>(Identities::newref(), rawidentities->fieldloc(), rawidentities->width(), content_.get()->length());
        Identities32* rawsubidentitites = reinterpret_cast<Identities32*>(subidentities.get());
        struct Error err = util::awkward_identities32_from_indexedarray<T>(
          &uniquecontents,
          rawsubidentitites->ptr().get(),
          rawidentities->ptr().get(),
          index_.ptr().get(),
          rawidentities->offset(),
          index_.offset(),
          content_.get()->length(),
          length(),
          rawidentities->width());
        util::handle_error(err, classname(), identities_.get());
        if (uniquecontents) {
          content_.get()->setidentities(subidentities);
        }
        else {
          content_.get()->setidentities(Identities::none());
        }
      }
      else if (Identities64* rawidentities = dynamic_cast<Identities64*>(bigidentities.get())) {
        bool uniquecontents;
        std::shared_ptr<Identities> subidentities = std::make_shared<Identities64>(Identities::newref(), rawidentities->fieldloc(), rawidentities->width(), content_.get()->length());
        Identities64* rawsubidentitites = reinterpret_cast<Identities64*>(subidentities.get());
        struct Error err = util::awkward_identities64_from_indexedarray<T>(
          &uniquecontents,
          rawsubidentitites->ptr().get(),
          rawidentities->ptr().get(),
          index_.ptr().get(),
          rawidentities->offset(),
          index_.offset(),
          content_.get()->length(),
          length(),
          rawidentities->width());
        util::handle_error(err, classname(), identities_.get());
        if (uniquecontents) {
          content_.get()->setidentities(subidentities);
        }
        else {
          content_.get()->setidentities(Identities::none());
        }
      }
      else {
        throw std::runtime_error("unrecognized Identities specialization");
      }
    }
    identities_ = identities;
  }

  template <typename T, bool ISOPTION>
  void IndexedArrayOf<T, ISOPTION>::setidentities() {
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
    check_for_iteration();
    builder.beginlist();
    for (int64_t i = 0;  i < len;  i++) {
      getitem_at_nowrap(i).get()->tojson_part(builder);
    }
    builder.endlist();
  }

  template <typename T, bool ISOPTION>
  void IndexedArrayOf<T, ISOPTION>::nbytes_part(std::map<size_t, int64_t>& largest) const {
    index_.nbytes_part(largest);
    content_.get()->nbytes_part(largest);
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
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
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const {
    IndexOf<T> index = copyindexes ? index_.deep_copy() : index_;
    std::shared_ptr<Content> content = content_.get()->deep_copy(copyarrays, copyindexes, copyidentities);
    std::shared_ptr<Identities> identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities, parameters_, index, content);
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
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const {
    if (head.get() == nullptr) {
      return shallow_copy();
    }
    else if (dynamic_cast<SliceAt*>(head.get())  ||  dynamic_cast<SliceRange*>(head.get())  ||  dynamic_cast<SliceArray64*>(head.get())) {
      if (ISOPTION) {
        int64_t numnull;
        struct Error err1 = util::awkward_indexedarray_numnull<T>(
          &numnull,
          index_.ptr().get(),
          index_.offset(),
          index_.length());
        util::handle_error(err1, classname(), identities_.get());

        Index64 nextcarry(length() - numnull);
        IndexOf<T> outindex(length());
        struct Error err2 = util::awkward_indexedarray_getitem_nextcarry_outindex_64<T>(
          nextcarry.ptr().get(),
          outindex.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          content_.get()->length());
        util::handle_error(err2, classname(), identities_.get());

        std::shared_ptr<Content> next = content_.get()->carry(nextcarry);
        std::shared_ptr<Content> out = next.get()->getitem_next(head, tail, advanced);
        return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities_, parameters_, outindex, out);
      }
      else {
        Index64 nextcarry(length());
        struct Error err = util::awkward_indexedarray_getitem_nextcarry_64<T>(
          nextcarry.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          content_.get()->length());
        util::handle_error(err, classname(), identities_.get());

        std::shared_ptr<Content> next = content_.get()->carry(nextcarry);
        return next.get()->getitem_next(head, tail, advanced);
      }
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

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::carry(const Index64& carry) const {
    IndexOf<T> nextindex(carry.length());
    struct Error err = util::awkward_indexedarray_getitem_carry_64<T>(
      nextindex.ptr().get(),
      index_.ptr().get(),
      carry.ptr().get(),
      index_.offset(),
      index_.length(),
      carry.length());
    util::handle_error(err, classname(), identities_.get());
    std::shared_ptr<Identities> identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities, parameters_, nextindex, content_);
  }

  template <typename T, bool ISOPTION>
  bool IndexedArrayOf<T, ISOPTION>::purelist_isregular() const {
    return content_.get()->purelist_isregular();
  }

  template <typename T, bool ISOPTION>
  int64_t IndexedArrayOf<T, ISOPTION>::purelist_depth() const {
    return content_.get()->purelist_depth();
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
  const Index64 IndexedArrayOf<T, ISOPTION>::count64() const {
    Index64 contentcount = content_.get()->count64();
    Index64 tocount(index_.length());
    struct Error err = util::awkward_indexedarray_count(
      tocount.ptr().get(),
      contentcount.ptr().get(),
      contentcount.length(),
      index_.ptr().get(),
      index_.length(),
      index_.offset());
    util::handle_error(err, classname(), identities_.get());
    return tocount;
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::count(int64_t axis) const {
    if (axis != 0) {
      throw std::runtime_error("FIXME: IndexedArray::count(axis != 0)");
    }
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities_, parameters_, index_, content_.get()->count(axis));
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::flatten(int64_t axis) const {
    if (axis < 0) {
      std::pair<int64_t, int64_t> minmax = minmax_depth();
      int64_t mindepth = minmax.first;
      int64_t maxdepth = minmax.second;
      int64_t depth = purelist_depth();
      if (mindepth == depth  &&  maxdepth == depth) {
        return flatten(-axis);
      }
      else {
        return content_.get()->flatten(axis);
      }
    }
    else if (axis == 0) {
      if (ISOPTION) {
        int64_t numnull;
        struct Error err1 = util::awkward_indexedarray_numnull<T>(
          &numnull,
          index_.ptr().get(),
          index_.offset(),
          index_.length());
          util::handle_error(err1, classname(), identities_.get());

        Index64 nextcarry(length() - numnull);
        struct Error err2 = util::awkward_indexedarray_flatten_nextcarry_64<T>(
          nextcarry.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          content_.get()->length());
          util::handle_error(err2, classname(), identities_.get());

        std::shared_ptr<Content> next = content_.get()->carry(nextcarry);
        return next.get()->flatten(axis);
      }
      else {
        Index64 nextcarry(length());
        struct Error err = util::awkward_indexedarray_getitem_nextcarry_64<T>(
          nextcarry.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          content_.get()->length());
        util::handle_error(err, classname(), identities_.get());

        std::shared_ptr<Content> next = content_.get()->carry(nextcarry);
        return next.get()->flatten(axis);
      }
    }
    else {
      return content_.get()->flatten(axis - 1);
    }
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: IndexedArray::getitem_next(SliceAt)");
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: IndexedArray::getitem_next(SliceRange)");
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: IndexedArray::getitem_next(SliceArray64)");
  }

  template class IndexedArrayOf<int32_t, false>;
  template class IndexedArrayOf<uint32_t, false>;
  template class IndexedArrayOf<int64_t, false>;
  template class IndexedArrayOf<int32_t, true>;
  template class IndexedArrayOf<int64_t, true>;
}
