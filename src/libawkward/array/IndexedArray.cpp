// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/operations.h"
#include "awkward/type/OptionType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/Slice.h"
#include "awkward/array/None.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/NumpyArray.h"

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

    Index64 nextindex(index_.length());
    struct Error err = util::awkward_indexedarray_overlay_mask8_to64<T>(
      nextindex.ptr().get(),
      mask.ptr().get(),
      mask.offset(),
      index_.ptr().get(),
      index_.offset(),
      index_.length());
    util::handle_error(err, classname(), identities_.get());

    IndexedOptionArray64 next(identities_, parameters_, nextindex, content_);
    return next.project();
  }

  template <typename T, bool ISOPTION>
  const Index8 IndexedArrayOf<T, ISOPTION>::bytemask() const {
    if (ISOPTION) {
      Index8 out(index_.length());
      struct Error err = util::awkward_indexedarray_mask8(
        out.ptr().get(),
        index_.ptr().get(),
        index_.offset(),
        index_.length());
      util::handle_error(err, classname(), identities_.get());
      return out;
    }
    else {
      Index8 out(index_.length());
      struct Error err = awkward_zero_mask8(
        out.ptr().get(),
        index_.length());
      util::handle_error(err, classname(), identities_.get());
      return out;
    }
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::simplify() const {
    if (ISOPTION) {
      if (IndexedArray32* rawcontent = dynamic_cast<IndexedArray32*>(content_.get())) {
        Index32 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = util::awkward_indexedarray_simplify32_to64(
          result.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          inner.ptr().get(),
          inner.offset(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_, parameters_, result, rawcontent->content());
      }
      else if (IndexedArrayU32* rawcontent = dynamic_cast<IndexedArrayU32*>(content_.get())) {
        IndexU32 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = util::awkward_indexedarray_simplifyU32_to64(
          result.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          inner.ptr().get(),
          inner.offset(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_, parameters_, result, rawcontent->content());
      }
      else if (IndexedArray64* rawcontent = dynamic_cast<IndexedArray64*>(content_.get())) {
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = util::awkward_indexedarray_simplify64_to64(
          result.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          inner.ptr().get(),
          inner.offset(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_, parameters_, result, rawcontent->content());
      }
      else if (IndexedOptionArray32* rawcontent = dynamic_cast<IndexedOptionArray32*>(content_.get())) {
        Index32 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = util::awkward_indexedarray_simplify32_to64(
          result.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          inner.ptr().get(),
          inner.offset(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_, parameters_, result, rawcontent->content());
      }
      else if (IndexedOptionArray64* rawcontent = dynamic_cast<IndexedOptionArray64*>(content_.get())) {
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = util::awkward_indexedarray_simplify64_to64(
          result.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          inner.ptr().get(),
          inner.offset(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_, parameters_, result, rawcontent->content());
      }
      else {
        return shallow_copy();
      }
    }
    else {
      if (IndexedArray32* rawcontent = dynamic_cast<IndexedArray32*>(content_.get())) {
        Index32 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = util::awkward_indexedarray_simplify32_to64(
          result.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          inner.ptr().get(),
          inner.offset(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedArray64>(identities_, parameters_, result, rawcontent->content());
      }
      else if (IndexedArrayU32* rawcontent = dynamic_cast<IndexedArrayU32*>(content_.get())) {
        IndexU32 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = util::awkward_indexedarray_simplifyU32_to64(
          result.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          inner.ptr().get(),
          inner.offset(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedArray64>(identities_, parameters_, result, rawcontent->content());
      }
      else if (IndexedArray64* rawcontent = dynamic_cast<IndexedArray64*>(content_.get())) {
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = util::awkward_indexedarray_simplify64_to64(
          result.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          inner.ptr().get(),
          inner.offset(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedArray64>(identities_, parameters_, result, rawcontent->content());
      }
      else if (IndexedOptionArray32* rawcontent = dynamic_cast<IndexedOptionArray32*>(content_.get())) {
        Index32 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = util::awkward_indexedarray_simplify32_to64(
          result.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          inner.ptr().get(),
          inner.offset(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_, parameters_, result, rawcontent->content());
      }
      else if (IndexedOptionArray64* rawcontent = dynamic_cast<IndexedOptionArray64*>(content_.get())) {
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = util::awkward_indexedarray_simplify64_to64(
          result.ptr().get(),
          index_.ptr().get(),
          index_.offset(),
          index_.length(),
          inner.ptr().get(),
          inner.offset(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_, parameters_, result, rawcontent->content());
      }
      else {
        return shallow_copy();
      }
    }
  }

  template <typename T, bool ISOPTION>
  T IndexedArrayOf<T, ISOPTION>::index_at_nowrap(int64_t at) const {
    return index_.getitem_at_nowrap(at);
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
  const std::shared_ptr<Type> IndexedArrayOf<T, ISOPTION>::type(const std::map<std::string, std::string>& typestrs) const {
    if (ISOPTION) {
      return std::make_shared<OptionType>(parameters_, util::gettypestr(parameters_, typestrs), content_.get()->type(typestrs));
    }
    else {
      std::shared_ptr<Type> out = content_.get()->type(typestrs);
      out.get()->setparameters(parameters_);
      return out;
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
    else if (dynamic_cast<SliceAt*>(head.get())  ||  dynamic_cast<SliceRange*>(head.get())  ||  dynamic_cast<SliceArray64*>(head.get())  ||  dynamic_cast<SliceJagged64*>(head.get())) {
      if (ISOPTION) {
        int64_t numnull;
        std::pair<Index64, IndexOf<T>> pair = nextcarry_outindex(numnull);
        Index64 nextcarry = pair.first;
        IndexOf<T> outindex = pair.second;

        std::shared_ptr<Content> next = content_.get()->carry(nextcarry);
        std::shared_ptr<Content> out = next.get()->getitem_next(head, tail, advanced);
        IndexedArrayOf<T, ISOPTION> out2(identities_, parameters_, outindex, out);
        return out2.simplify();
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
    else if (SliceMissing64* missing = dynamic_cast<SliceMissing64*>(head.get())) {
      return Content::getitem_next(*missing, tail, advanced);
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
  const std::string IndexedArrayOf<T, ISOPTION>::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
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
  const std::pair<bool, int64_t> IndexedArrayOf<T, ISOPTION>::branch_depth() const {
    return content_.get()->branch_depth();
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
  const std::string IndexedArrayOf<T, ISOPTION>::validityerror(const std::string& path) const {
    struct Error err = util::awkward_indexedarray_validity<T>(
      index_.ptr().get(),
      index_.offset(),
      index_.length(),
      content_.get()->length(),
      ISOPTION);
    if (err.str == nullptr) {
      return content_.get()->validityerror(path + std::string(".content"));
    }
    else {
      return std::string("at ") + path + std::string(" (") + classname() + std::string("): ") + std::string(err.str) + std::string(" at i=") + std::to_string(err.identity);
    }
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::num(int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      Index64 out(1);
      out.ptr().get()[0] = length();
      return NumpyArray(out).getitem_at_nowrap(0);
    }
    else if (ISOPTION) {
      int64_t numnull;
      std::pair<Index64, IndexOf<T>> pair = nextcarry_outindex(numnull);
      Index64 nextcarry = pair.first;
      IndexOf<T> outindex = pair.second;

      std::shared_ptr<Content> next = content_.get()->carry(nextcarry);
      std::shared_ptr<Content> out = next.get()->num(axis, depth);
      IndexedArrayOf<T, ISOPTION> out2(Identities::none(), util::Parameters(), outindex, out);
      return out2.simplify();
    }
    else {
      return project().get()->num(axis, depth);
    }
  }

  template <typename T, bool ISOPTION>
  const std::pair<Index64, std::shared_ptr<Content>> IndexedArrayOf<T, ISOPTION>::offsets_and_flattened(int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      throw std::invalid_argument("axis=0 not allowed for flatten");
    }
    else if (ISOPTION) {
      int64_t numnull;
      std::pair<Index64, IndexOf<T>> pair = nextcarry_outindex(numnull);
      Index64 nextcarry = pair.first;
      IndexOf<T> outindex = pair.second;

      std::shared_ptr<Content> next = content_.get()->carry(nextcarry);

      std::pair<Index64, std::shared_ptr<Content>> offsets_flattened = next.get()->offsets_and_flattened(axis, depth);
      Index64 offsets = offsets_flattened.first;
      std::shared_ptr<Content> flattened = offsets_flattened.second;

      if (offsets.length() == 0) {
        return std::pair<Index64, std::shared_ptr<Content>>(offsets, std::make_shared<IndexedArrayOf<T, ISOPTION>>(Identities::none(), util::Parameters(), index_, flattened));
      }
      else {
        Index64 outoffsets(offsets.length() + numnull);
        struct Error err = util::awkward_indexedarray_flatten_none2empty_64<T>(
          outoffsets.ptr().get(),
          outindex.ptr().get(),
          outindex.offset(),
          outindex.length(),
          offsets.ptr().get(),
          offsets.offset(),
          offsets.length());
        util::handle_error(err, classname(), identities_.get());
        return std::pair<Index64, std::shared_ptr<Content>>(outoffsets, flattened);
      }
    }
    else {
      return project().get()->offsets_and_flattened(axis, depth);
    }
  }

  template <typename T, bool ISOPTION>
  bool IndexedArrayOf<T, ISOPTION>::mergeable(const std::shared_ptr<Content>& other, bool mergebool) const {
    if (!parameters_equal(other.get()->parameters())) {
      return false;
    }

    if (dynamic_cast<EmptyArray*>(other.get())  ||
        dynamic_cast<UnionArray8_32*>(other.get())  ||
        dynamic_cast<UnionArray8_U32*>(other.get())  ||
        dynamic_cast<UnionArray8_64*>(other.get())) {
      return true;
    }

    if (IndexedArray32* rawother = dynamic_cast<IndexedArray32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (IndexedArrayU32* rawother = dynamic_cast<IndexedArrayU32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (IndexedArray64* rawother = dynamic_cast<IndexedArray64*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (IndexedOptionArray32* rawother = dynamic_cast<IndexedOptionArray32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (IndexedOptionArray64* rawother = dynamic_cast<IndexedOptionArray64*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else {
      return content_.get()->mergeable(other, mergebool);
    }
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::reverse_merge(const std::shared_ptr<Content>& other) const {
    int64_t theirlength = other.get()->length();
    int64_t mylength = length();
    Index64 index(theirlength + mylength);

    std::shared_ptr<Content> content = other.get()->merge(content_);
    struct Error err1 = awkward_indexedarray_fill_to64_count(
      index.ptr().get(),
      0,
      theirlength,
      0);
    util::handle_error(err1, classname(), identities_.get());

    int64_t mycontentlength = content_.get()->length();
    if (std::is_same<T, int32_t>::value) {
      struct Error err2 = awkward_indexedarray_fill_to64_from32(
        index.ptr().get(),
        theirlength,
        reinterpret_cast<int32_t*>(index_.ptr().get()),
        index_.offset(),
        mylength,
        mycontentlength);
      util::handle_error(err2, classname(), identities_.get());
    }
    else if (std::is_same<T, uint32_t>::value) {
      struct Error err2 = awkward_indexedarray_fill_to64_fromU32(
        index.ptr().get(),
        theirlength,
        reinterpret_cast<uint32_t*>(index_.ptr().get()),
        index_.offset(),
        mylength,
        mycontentlength);
      util::handle_error(err2, classname(), identities_.get());
    }
    if (std::is_same<T, int64_t>::value) {
      struct Error err2 = awkward_indexedarray_fill_to64_from64(
        index.ptr().get(),
        theirlength,
        reinterpret_cast<int64_t*>(index_.ptr().get()),
        index_.offset(),
        mylength,
        mycontentlength);
      util::handle_error(err2, classname(), identities_.get());
    }
    else {
      throw std::runtime_error("unrecognized IndexedArray specialization");
    }

    return std::make_shared<IndexedArrayOf<int64_t, ISOPTION>>(Identities::none(), util::Parameters(), index, content);
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::merge(const std::shared_ptr<Content>& other) const {
    if (!parameters_equal(other.get()->parameters())) {
      return merge_as_union(other);
    }

    if (dynamic_cast<EmptyArray*>(other.get())) {
      return shallow_copy();
    }
    else if (UnionArray8_32* rawother = dynamic_cast<UnionArray8_32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (UnionArray8_U32* rawother = dynamic_cast<UnionArray8_U32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (UnionArray8_64* rawother = dynamic_cast<UnionArray8_64*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }

    int64_t mylength = length();
    int64_t theirlength = other.get()->length();
    Index64 index(mylength + theirlength);

    if (std::is_same<T, int32_t>::value) {
      struct Error err = awkward_indexedarray_fill_to64_from32(
        index.ptr().get(),
        0,
        reinterpret_cast<int32_t*>(index_.ptr().get()),
        index_.offset(),
        mylength,
        0);
      util::handle_error(err, classname(), identities_.get());
    }
    else if (std::is_same<T, uint32_t>::value) {
      struct Error err = awkward_indexedarray_fill_to64_fromU32(
        index.ptr().get(),
        0,
        reinterpret_cast<uint32_t*>(index_.ptr().get()),
        index_.offset(),
        mylength,
        0);
      util::handle_error(err, classname(), identities_.get());
    }
    else if (std::is_same<T, int64_t>::value) {
      struct Error err = awkward_indexedarray_fill_to64_from64(
        index.ptr().get(),
        0,
        reinterpret_cast<int64_t*>(index_.ptr().get()),
        index_.offset(),
        mylength,
        0);
      util::handle_error(err, classname(), identities_.get());
    }
    else {
      throw std::runtime_error("unrecognized IndexedArray specialization");
    }

    int64_t mycontentlength = content_.get()->length();
    std::shared_ptr<Content> content;
    bool other_isoption = false;
    if (IndexedArray32* rawother = dynamic_cast<IndexedArray32*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      Index32 other_index = rawother->index();
      struct Error err = awkward_indexedarray_fill_to64_from32(
        index.ptr().get(),
        mylength,
        other_index.ptr().get(),
        other_index.offset(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(), rawother->identities().get());
    }
    else if (IndexedArrayU32* rawother = dynamic_cast<IndexedArrayU32*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      IndexU32 other_index = rawother->index();
      struct Error err = awkward_indexedarray_fill_to64_fromU32(
        index.ptr().get(),
        mylength,
        other_index.ptr().get(),
        other_index.offset(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(), rawother->identities().get());
    }
    else if (IndexedArray64* rawother = dynamic_cast<IndexedArray64*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      Index64 other_index = rawother->index();
      struct Error err = awkward_indexedarray_fill_to64_from64(
        index.ptr().get(),
        mylength,
        other_index.ptr().get(),
        other_index.offset(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(), rawother->identities().get());
    }
    else if (IndexedOptionArray32* rawother = dynamic_cast<IndexedOptionArray32*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      Index32 other_index = rawother->index();
      struct Error err = awkward_indexedarray_fill_to64_from32(
        index.ptr().get(),
        mylength,
        other_index.ptr().get(),
        other_index.offset(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(), rawother->identities().get());
      other_isoption = true;
    }
    else if (IndexedOptionArray64* rawother = dynamic_cast<IndexedOptionArray64*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      Index64 other_index = rawother->index();
      struct Error err = awkward_indexedarray_fill_to64_from64(
        index.ptr().get(),
        mylength,
        other_index.ptr().get(),
        other_index.offset(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(), rawother->identities().get());
      other_isoption = true;
    }
    else {
      content = content_.get()->merge(other);
      struct Error err = awkward_indexedarray_fill_to64_count(
        index.ptr().get(),
        mylength,
        theirlength,
        mycontentlength);
      util::handle_error(err, classname(), identities_.get());
    }

    if (ISOPTION  ||  other_isoption) {
      return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, content);
    }
    else {
      return std::make_shared<IndexedArray64>(Identities::none(), util::Parameters(), index, content);
    }
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<SliceItem> IndexedArrayOf<T, ISOPTION>::asslice() const {
    if (ISOPTION) {
      int64_t numnull;
      struct Error err1 = util::awkward_indexedarray_numnull<T>(
        &numnull,
        index_.ptr().get(),
        index_.offset(),
        index_.length());
      util::handle_error(err1, classname(), identities_.get());

      Index64 nextcarry(length() - numnull);
      Index64 outindex(length());
      struct Error err2 = util::awkward_indexedarray_getitem_nextcarry_outindex_mask_64<T>(
        nextcarry.ptr().get(),
        outindex.ptr().get(),
        index_.ptr().get(),
        index_.offset(),
        index_.length(),
        content_.get()->length());
      util::handle_error(err2, classname(), identities_.get());

      std::shared_ptr<Content> next = content_.get()->carry(nextcarry);

      std::shared_ptr<SliceItem> slicecontent = next.get()->asslice();
      if (SliceArray64* raw = dynamic_cast<SliceArray64*>(slicecontent.get())) {
        if (raw->frombool()) {
          Index64 nonzero(raw->index());
          Index8 originalmask(length());
          Index64 adjustedindex(nonzero.length() + numnull);
          Index64 adjustednonzero(nonzero.length());
          struct Error err3 = awkward_indexedarray_getitem_adjust_outindex_64(
            originalmask.ptr().get(),
            adjustedindex.ptr().get(),
            adjustednonzero.ptr().get(),
            outindex.ptr().get(),
            outindex.offset(),
            outindex.length(),
            nonzero.ptr().get(),
            nonzero.offset(),
            nonzero.length());
          util::handle_error(err3, classname(), nullptr);

          std::shared_ptr<SliceItem> outcontent = std::make_shared<SliceArray64>(adjustednonzero, raw->shape(), raw->strides(), true);
          return std::make_shared<SliceMissing64>(adjustedindex, originalmask, outcontent);
        }
      }
      return std::make_shared<SliceMissing64>(outindex, Index8(0), slicecontent);
    }
    else {
      return project().get()->asslice();
    }
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::rpad(int64_t target, int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      return rpad_axis0(target, false);
    }
    else if (toaxis == depth + 1) {
      if (ISOPTION) {
        Index8 mask = bytemask();
        Index64 index(mask.length());
        struct Error err = awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64(
          index.ptr().get(),
          mask.ptr().get(),
          mask.length());
        util::handle_error(err, classname(), identities_.get());

        std::shared_ptr<Content> next = project().get()->rpad(target, toaxis, depth);
        return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, next).get()->simplify();
      }
      else {
        return project().get()->rpad(target, toaxis, depth);
      }
    }
    else {
      return std::make_shared<IndexedArrayOf<T, ISOPTION>>(Identities::none(), parameters_, index_, content_.get()->rpad(target, toaxis, depth + 1));
    }
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::rpad_and_clip(int64_t target, int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      return rpad_axis0(target, true);
    }
    else if (toaxis == depth + 1) {
      if (ISOPTION) {
        Index8 mask = bytemask();
        Index64 index(mask.length());
        struct Error err = awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64(
          index.ptr().get(),
          mask.ptr().get(),
          mask.length());
        util::handle_error(err, classname(), identities_.get());

        std::shared_ptr<Content> next = project().get()->rpad_and_clip(target, toaxis, depth);
        return std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, next).get()->simplify();
      }
      else {
        return project().get()->rpad_and_clip(target, toaxis, depth);
      }
    }
    else {
      return std::make_shared<IndexedArrayOf<T, ISOPTION>>(Identities::none(), parameters_, index_, content_.get()->rpad_and_clip(target, toaxis, depth + 1));
    }
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::reduce_next(const Reducer& reducer, int64_t negaxis, const Index64& parents, int64_t outlength, bool mask, bool keepdims) const {
    int64_t numnull;
    struct Error err1 = util::awkward_indexedarray_numnull<T>(
      &numnull,
      index_.ptr().get(),
      index_.offset(),
      index_.length());
    util::handle_error(err1, classname(), identities_.get());

    Index64 nextparents(index_.length() - numnull);
    Index64 nextcarry(index_.length() - numnull);
    struct Error err2 = util::awkward_indexedarray_reduce_next_64<T>(
      nextcarry.ptr().get(),
      nextparents.ptr().get(),
      index_.ptr().get(),
      index_.offset(),
      parents.ptr().get(),
      parents.offset(),
      index_.length());
    util::handle_error(err2, classname(), identities_.get());

    std::shared_ptr<Content> next = content_.get()->carry(nextcarry);
    return next.get()->reduce_next(reducer, negaxis, nextparents, outlength, mask, keepdims);
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: IndexedArray::getitem_next(at)");
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: IndexedArray::getitem_next(range)");
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: IndexedArray::getitem_next(array)");
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceJagged64& jagged, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: IndexedArray::getitem_next(jagged)");
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceArray64& slicecontent, const Slice& tail) const {
    return getitem_next_jagged_generic<SliceArray64>(slicestarts, slicestops, slicecontent, tail);
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceMissing64& slicecontent, const Slice& tail) const {
    return getitem_next_jagged_generic<SliceMissing64>(slicestarts, slicestops, slicecontent, tail);
  }

  template <typename T, bool ISOPTION>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceJagged64& slicecontent, const Slice& tail) const {
    return getitem_next_jagged_generic<SliceJagged64>(slicestarts, slicestops, slicecontent, tail);
  }

  template <typename T, bool ISOPTION>
  template <typename S>
  const std::shared_ptr<Content> IndexedArrayOf<T, ISOPTION>::getitem_next_jagged_generic(const Index64& slicestarts, const Index64& slicestops, const S& slicecontent, const Slice& tail) const {
    if (ISOPTION) {
      int64_t numnull;
      std::pair<Index64, IndexOf<T>> pair = nextcarry_outindex(numnull);
      Index64 nextcarry = pair.first;
      IndexOf<T> outindex = pair.second;

      std::shared_ptr<Content> next = content_.get()->carry(nextcarry);
      std::shared_ptr<Content> out = next.get()->getitem_next_jagged(slicestarts, slicestops, slicecontent, tail);
      IndexedArrayOf<T, ISOPTION> out2(identities_, parameters_, outindex, out);
      return out2.simplify();
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
      return next.get()->getitem_next_jagged(slicestarts, slicestops, slicecontent, tail);
    }
  }

  template <typename T, bool ISOPTION>
  const std::pair<Index64, IndexOf<T>> IndexedArrayOf<T, ISOPTION>::nextcarry_outindex(int64_t& numnull) const {
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

    return std::pair<Index64, IndexOf<T>>(nextcarry, outindex);
  }

  template class IndexedArrayOf<int32_t, false>;
  template class IndexedArrayOf<uint32_t, false>;
  template class IndexedArrayOf<int64_t, false>;
  template class IndexedArrayOf<int32_t, true>;
  template class IndexedArrayOf<int64_t, true>;
}
