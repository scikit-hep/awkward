// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/operations.h"
#include "awkward/type/ListType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/Slice.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"

#include "awkward/array/ListArray.h"

namespace awkward {
  template <typename T>
  ListArrayOf<T>::ListArrayOf(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const IndexOf<T>& starts, const IndexOf<T>& stops, const std::shared_ptr<Content>& content)
      : Content(identities, parameters)
      , starts_(starts)
      , stops_(stops)
      , content_(content) {
    if (stops.length() < starts.length()) {
      throw std::invalid_argument("ListArray stops must not be shorter than its starts");
    }
  }

  template <typename T>
  const IndexOf<T> ListArrayOf<T>::starts() const {
    return starts_;
  }

  template <typename T>
  const IndexOf<T> ListArrayOf<T>::stops() const {
    return stops_;
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::content() const {
    return content_;
  }

  template <typename T>
  Index64 ListArrayOf<T>::compact_offsets64(bool start_at_zero) const {
    int64_t len = starts_.length();
    Index64 out(len + 1);
    struct Error err = util::awkward_listarray_compact_offsets64<T>(
      out.ptr().get(),
      starts_.ptr().get(),
      stops_.ptr().get(),
      starts_.offset(),
      stops_.offset(),
      len);
    util::handle_error(err, classname(), identities_.get());
    return out;
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::broadcast_tooffsets64(const Index64& offsets) const {
    if (offsets.length() == 0  ||  offsets.getitem_at_nowrap(0) != 0) {
      throw std::invalid_argument("broadcast_tooffsets64 can only be used with offsets that start at 0");
    }
    if (offsets.length() - 1 > starts_.length()) {
      throw std::invalid_argument(std::string("cannot broadcast ListArray of length ") + std::to_string(starts_.length()) + (" to length ") + std::to_string(offsets.length() - 1));
    }

    int64_t carrylen = offsets.getitem_at_nowrap(offsets.length() - 1);
    Index64 nextcarry(carrylen);
    struct Error err = util::awkward_listarray_broadcast_tooffsets64<T>(
      nextcarry.ptr().get(),
      offsets.ptr().get(),
      offsets.offset(),
      offsets.length(),
      starts_.ptr().get(),
      starts_.offset(),
      stops_.ptr().get(),
      stops_.offset(),
      content_.get()->length());
    util::handle_error(err, classname(), identities_.get());

    std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);

    std::shared_ptr<Identities> identities;
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(0, offsets.length() - 1);
    }
    return std::make_shared<ListOffsetArray64>(identities, parameters_, offsets, nextcontent);
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::toRegularArray() const {
    Index64 offsets = compact_offsets64(true);
    std::shared_ptr<Content> listoffsetarray64 = broadcast_tooffsets64(offsets);
    ListOffsetArray64* raw = dynamic_cast<ListOffsetArray64*>(listoffsetarray64.get());
    return raw->toRegularArray();
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::toListOffsetArray64(bool start_at_zero) const {
    Index64 offsets = compact_offsets64(start_at_zero);
    return broadcast_tooffsets64(offsets);
  }

  template <typename T>
  const std::string ListArrayOf<T>::classname() const {
    if (std::is_same<T, int32_t>::value) {
      return "ListArray32";
    }
    else if (std::is_same<T, uint32_t>::value) {
      return "ListArrayU32";
    }
    else if (std::is_same<T, int64_t>::value) {
      return "ListArray64";
    }
    else {
      return "UnrecognizedListArray";
    }
  }

  template <typename T>
  void ListArrayOf<T>::setidentities(const std::shared_ptr<Identities>& identities) {
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
        std::shared_ptr<Identities> subidentities = std::make_shared<Identities32>(Identities::newref(), rawidentities->fieldloc(), rawidentities->width() + 1, content_.get()->length());
        Identities32* rawsubidentities = reinterpret_cast<Identities32*>(subidentities.get());
        struct Error err = util::awkward_identities32_from_listarray<T>(
          &uniquecontents,
          rawsubidentities->ptr().get(),
          rawidentities->ptr().get(),
          starts_.ptr().get(),
          stops_.ptr().get(),
          rawidentities->offset(),
          starts_.offset(),
          stops_.offset(),
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
        std::shared_ptr<Identities> subidentities = std::make_shared<Identities64>(Identities::newref(), rawidentities->fieldloc(), rawidentities->width() + 1, content_.get()->length());
        Identities64* rawsubidentities = reinterpret_cast<Identities64*>(subidentities.get());
        struct Error err = util::awkward_identities64_from_listarray<T>(
          &uniquecontents,
          rawsubidentities->ptr().get(),
          rawidentities->ptr().get(),
          starts_.ptr().get(),
          stops_.ptr().get(),
          rawidentities->offset(),
          starts_.offset(),
          stops_.offset(),
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

  template <typename T>
  void ListArrayOf<T>::setidentities() {
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

  template <typename T>
  const std::shared_ptr<Type> ListArrayOf<T>::type(const std::map<std::string, std::string>& typestrs) const {
    return std::make_shared<ListType>(parameters_, util::gettypestr(parameters_, typestrs), content_.get()->type(typestrs));
  }

  template <typename T>
  const std::string ListArrayOf<T>::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << ">\n";
    if (identities_.get() != nullptr) {
      out << identities_.get()->tostring_part(indent + std::string("    "), "", "\n");
    }
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    out << starts_.tostring_part(indent + std::string("    "), "<starts>", "</starts>\n");
    out << stops_.tostring_part(indent + std::string("    "), "<stops>", "</stops>\n");
    out << content_.get()->tostring_part(indent + std::string("    "), "<content>", "</content>\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  template <typename T>
  void ListArrayOf<T>::tojson_part(ToJson& builder) const {
    int64_t len = length();
    check_for_iteration();
    builder.beginlist();
    for (int64_t i = 0;  i < len;  i++) {
      getitem_at_nowrap(i).get()->tojson_part(builder);
    }
    builder.endlist();
  }

  template <typename T>
  void ListArrayOf<T>::nbytes_part(std::map<size_t, int64_t>& largest) const {
    starts_.nbytes_part(largest);
    stops_.nbytes_part(largest);
    content_.get()->nbytes_part(largest);
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  template <typename T>
  int64_t ListArrayOf<T>::length() const {
    return starts_.length();
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::shallow_copy() const {
    return std::make_shared<ListArrayOf<T>>(identities_, parameters_, starts_, stops_, content_);
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const {
    IndexOf<T> starts = copyindexes ? starts_.deep_copy() : starts_;
    IndexOf<T> stops = copyindexes ? stops_.deep_copy() : stops_;
    std::shared_ptr<Content> content = content_.get()->deep_copy(copyarrays, copyindexes, copyidentities);
    std::shared_ptr<Identities> identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<ListArrayOf<T>>(identities, parameters_, starts, stops, content);
  }

  template <typename T>
  void ListArrayOf<T>::check_for_iteration() const {
    if (stops_.length() < starts_.length()) {
      util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
    }
    if (identities_.get() != nullptr  &&  identities_.get()->length() < starts_.length()) {
      util::handle_error(failure("len(identities) < len(array)", kSliceNone, kSliceNone), identities_.get()->classname(), nullptr);
    }
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_nothing() const {
    return content_.get()->getitem_range_nowrap(0, 0);
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += starts_.length();
    }
    if (!(0 <= regular_at  &&  regular_at < starts_.length())) {
      util::handle_error(failure("index out of range", kSliceNone, at), classname(), identities_.get());
    }
    if (regular_at >= stops_.length()) {
      util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_at_nowrap(int64_t at) const {
    int64_t start = (int64_t)starts_.getitem_at_nowrap(at);
    int64_t stop = (int64_t)stops_.getitem_at_nowrap(at);
    int64_t lencontent = content_.get()->length();
    if (start == stop) {
      start = stop = 0;
    }
    if (start < 0) {
      util::handle_error(failure("starts[i] < 0", kSliceNone, at), classname(), identities_.get());
    }
    if (start > stop) {
      util::handle_error(failure("starts[i] > stops[i]", kSliceNone, at), classname(), identities_.get());
    }
    if (stop > lencontent) {
      util::handle_error(failure("starts[i] != stops[i] and stops[i] > len(content)", kSliceNone, at), classname(), identities_.get());
    }
    return content_.get()->getitem_range_nowrap(start, stop);
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), starts_.length());
    if (regular_stop > stops_.length()) {
      util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
    }
    if (identities_.get() != nullptr  &&  regular_stop > identities_.get()->length()) {
      util::handle_error(failure("index out of range", kSliceNone, stop), identities_.get()->classname(), nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_range_nowrap(int64_t start, int64_t stop) const {
    std::shared_ptr<Identities> identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    return std::make_shared<ListArrayOf<T>>(identities, parameters_, starts_.getitem_range_nowrap(start, stop), stops_.getitem_range_nowrap(start, stop), content_);
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_field(const std::string& key) const {
    return std::make_shared<ListArrayOf<T>>(identities_, util::Parameters(), starts_, stops_, content_.get()->getitem_field(key));
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_fields(const std::vector<std::string>& keys) const {
    return std::make_shared<ListArrayOf<T>>(identities_, util::Parameters(), starts_, stops_, content_.get()->getitem_fields(keys));
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::carry(const Index64& carry) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
    }
    IndexOf<T> nextstarts(carry.length());
    IndexOf<T> nextstops(carry.length());
    struct Error err = util::awkward_listarray_getitem_carry_64<T>(
      nextstarts.ptr().get(),
      nextstops.ptr().get(),
      starts_.ptr().get(),
      stops_.ptr().get(),
      carry.ptr().get(),
      starts_.offset(),
      stops_.offset(),
      lenstarts,
      carry.length());
    util::handle_error(err, classname(), identities_.get());
    std::shared_ptr<Identities> identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }
    return std::make_shared<ListArrayOf<T>>(identities, parameters_, nextstarts, nextstops, content_);
  }

  template <typename T>
  const std::string ListArrayOf<T>::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  template <typename T>
  bool ListArrayOf<T>::purelist_isregular() const {
    return false;
  }

  template <typename T>
  int64_t ListArrayOf<T>::purelist_depth() const {
    return content_.get()->purelist_depth() + 1;
  }

  template <typename T>
  const std::pair<int64_t, int64_t> ListArrayOf<T>::minmax_depth() const {
    std::pair<int64_t, int64_t> content_depth = content_.get()->minmax_depth();
    return std::pair<int64_t, int64_t>(content_depth.first + 1, content_depth.second + 1);
  }

  template <typename T>
  const std::pair<bool, int64_t> ListArrayOf<T>::branch_depth() const {
    std::pair<bool, int64_t> content_depth = content_.get()->branch_depth();
    return std::pair<bool, int64_t>(content_depth.first, content_depth.second + 1);
  }

  template <typename T>
  int64_t ListArrayOf<T>::numfields() const {
    return content_.get()->numfields();
  }

  template <typename T>
  int64_t ListArrayOf<T>::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  template <typename T>
  const std::string ListArrayOf<T>::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  template <typename T>
  bool ListArrayOf<T>::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  template <typename T>
  const std::vector<std::string> ListArrayOf<T>::keys() const {
    return content_.get()->keys();
  }

  template <typename T>
  const std::string ListArrayOf<T>::validityerror(const std::string& path) const {
    struct Error err = util::awkward_listarray_validity<T>(
      starts_.ptr().get(),
      starts_.offset(),
      stops_.ptr().get(),
      stops_.offset(),
      starts_.length(),
      content_.get()->length());
    if (err.str == nullptr) {
      return content_.get()->validityerror(path + std::string(".content"));
    }
    else {
      return std::string("at ") + path + std::string(" (") + classname() + std::string("): ") + std::string(err.str) + std::string(" at i=") + std::to_string(err.identity);
    }
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::num(int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      Index64 out(1);
      out.ptr().get()[0] = length();
      return NumpyArray(out).getitem_at_nowrap(0);
    }
    else if (toaxis == depth + 1) {
      Index64 tonum(length());
      struct Error err = util::awkward_listarray_num_64<T>(
        tonum.ptr().get(),
        starts_.ptr().get(),
        starts_.offset(),
        stops_.ptr().get(),
        stops_.offset(),
        length());
      util::handle_error(err, classname(), identities_.get());
      return std::make_shared<NumpyArray>(tonum);
    }
    else {
      return toListOffsetArray64(true).get()->num(axis, depth);
    }
  }

  template <typename T>
  const std::pair<Index64, std::shared_ptr<Content>> ListArrayOf<T>::offsets_and_flattened(int64_t axis, int64_t depth) const {
    return toListOffsetArray64(true).get()->offsets_and_flattened(axis, depth);
  }

  template <typename T>
  bool ListArrayOf<T>::mergeable(const std::shared_ptr<Content>& other, bool mergebool) const {
    if (!parameters_equal(other.get()->parameters())) {
      return false;
    }

    if (dynamic_cast<EmptyArray*>(other.get())  ||
        dynamic_cast<UnionArray8_32*>(other.get())  ||
        dynamic_cast<UnionArray8_U32*>(other.get())  ||
        dynamic_cast<UnionArray8_64*>(other.get())) {
      return true;
    }
    else if (IndexedArray32* rawother = dynamic_cast<IndexedArray32*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (IndexedArrayU32* rawother = dynamic_cast<IndexedArrayU32*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (IndexedArray64* rawother = dynamic_cast<IndexedArray64*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (IndexedOptionArray32* rawother = dynamic_cast<IndexedOptionArray32*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (IndexedOptionArray64* rawother = dynamic_cast<IndexedOptionArray64*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (ByteMaskedArray* rawother = dynamic_cast<ByteMaskedArray*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (BitMaskedArray* rawother = dynamic_cast<BitMaskedArray*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (UnmaskedArray* rawother = dynamic_cast<UnmaskedArray*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }

    if (RegularArray* rawother = dynamic_cast<RegularArray*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ListArray32* rawother = dynamic_cast<ListArray32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ListArrayU32* rawother = dynamic_cast<ListArrayU32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ListArray64* rawother = dynamic_cast<ListArray64*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ListOffsetArray32* rawother = dynamic_cast<ListOffsetArray32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ListOffsetArrayU32* rawother = dynamic_cast<ListOffsetArrayU32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ListOffsetArray64* rawother = dynamic_cast<ListOffsetArray64*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else {
      return false;
    }
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::merge(const std::shared_ptr<Content>& other) const {
    if (!parameters_equal(other.get()->parameters())) {
      return merge_as_union(other);
    }

    if (dynamic_cast<EmptyArray*>(other.get())) {
      return shallow_copy();
    }
    else if (IndexedArray32* rawother = dynamic_cast<IndexedArray32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (IndexedArrayU32* rawother = dynamic_cast<IndexedArrayU32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (IndexedArray64* rawother = dynamic_cast<IndexedArray64*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (IndexedOptionArray32* rawother = dynamic_cast<IndexedOptionArray32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (IndexedOptionArray64* rawother = dynamic_cast<IndexedOptionArray64*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (ByteMaskedArray* rawother = dynamic_cast<ByteMaskedArray*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (BitMaskedArray* rawother = dynamic_cast<BitMaskedArray*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (UnmaskedArray* rawother = dynamic_cast<UnmaskedArray*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
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
    Index64 starts(mylength + theirlength);
    Index64 stops(mylength + theirlength);

    if (std::is_same<T, int32_t>::value) {
      struct Error err = awkward_listarray_fill_to64_from32(
        starts.ptr().get(),
        0,
        stops.ptr().get(),
        0,
        reinterpret_cast<int32_t*>(starts_.ptr().get()),
        starts_.offset(),
        reinterpret_cast<int32_t*>(stops_.ptr().get()),
        stops_.offset(),
        mylength,
        0);
      util::handle_error(err, classname(), identities_.get());
    }
    else if (std::is_same<T, uint32_t>::value) {
      struct Error err = awkward_listarray_fill_to64_fromU32(
        starts.ptr().get(),
        0,
        stops.ptr().get(),
        0,
        reinterpret_cast<uint32_t*>(starts_.ptr().get()),
        starts_.offset(),
        reinterpret_cast<uint32_t*>(stops_.ptr().get()),
        stops_.offset(),
        mylength,
        0);
      util::handle_error(err, classname(), identities_.get());
    }
    else if (std::is_same<T, int64_t>::value) {
      struct Error err = awkward_listarray_fill_to64_from64(
        starts.ptr().get(),
        0,
        stops.ptr().get(),
        0,
        reinterpret_cast<int64_t*>(starts_.ptr().get()),
        starts_.offset(),
        reinterpret_cast<int64_t*>(stops_.ptr().get()),
        stops_.offset(),
        mylength,
        0);
      util::handle_error(err, classname(), identities_.get());
    }
    else {
      throw std::runtime_error("unrecognized ListArray specialization");
    }

    int64_t mycontentlength = content_.get()->length();
    std::shared_ptr<Content> content;
    if (ListArray32* rawother = dynamic_cast<ListArray32*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      Index32 other_starts = rawother->starts();
      Index32 other_stops = rawother->stops();
      struct Error err = awkward_listarray_fill_to64_from32(
        starts.ptr().get(),
        mylength,
        stops.ptr().get(),
        mylength,
        other_starts.ptr().get(),
        other_starts.offset(),
        other_stops.ptr().get(),
        other_stops.offset(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(), rawother->identities().get());
    }
    else if (ListArrayU32* rawother = dynamic_cast<ListArrayU32*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      IndexU32 other_starts = rawother->starts();
      IndexU32 other_stops = rawother->stops();
      struct Error err = awkward_listarray_fill_to64_fromU32(
        starts.ptr().get(),
        mylength,
        stops.ptr().get(),
        mylength,
        other_starts.ptr().get(),
        other_starts.offset(),
        other_stops.ptr().get(),
        other_stops.offset(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(), rawother->identities().get());
    }
    else if (ListArray64* rawother = dynamic_cast<ListArray64*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      Index64 other_starts = rawother->starts();
      Index64 other_stops = rawother->stops();
      struct Error err = awkward_listarray_fill_to64_from64(
        starts.ptr().get(),
        mylength,
        stops.ptr().get(),
        mylength,
        other_starts.ptr().get(),
        other_starts.offset(),
        other_stops.ptr().get(),
        other_stops.offset(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(), rawother->identities().get());
    }
    else if (ListOffsetArray32* rawother = dynamic_cast<ListOffsetArray32*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      Index32 other_starts = rawother->starts();
      Index32 other_stops = rawother->stops();
      struct Error err = awkward_listarray_fill_to64_from32(
        starts.ptr().get(),
        mylength,
        stops.ptr().get(),
        mylength,
        other_starts.ptr().get(),
        other_starts.offset(),
        other_stops.ptr().get(),
        other_stops.offset(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(), rawother->identities().get());
    }
    else if (ListOffsetArrayU32* rawother = dynamic_cast<ListOffsetArrayU32*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      IndexU32 other_starts = rawother->starts();
      IndexU32 other_stops = rawother->stops();
      struct Error err = awkward_listarray_fill_to64_fromU32(
        starts.ptr().get(),
        mylength,
        stops.ptr().get(),
        mylength,
        other_starts.ptr().get(),
        other_starts.offset(),
        other_stops.ptr().get(),
        other_stops.offset(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(), rawother->identities().get());
    }
    else if (ListOffsetArray64* rawother = dynamic_cast<ListOffsetArray64*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      Index64 other_starts = rawother->starts();
      Index64 other_stops = rawother->stops();
      struct Error err = awkward_listarray_fill_to64_from64(
        starts.ptr().get(),
        mylength,
        stops.ptr().get(),
        mylength,
        other_starts.ptr().get(),
        other_starts.offset(),
        other_stops.ptr().get(),
        other_stops.offset(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(), rawother->identities().get());
    }
    else if (RegularArray* rawregulararray = dynamic_cast<RegularArray*>(other.get())) {
      std::shared_ptr<Content> listoffsetarray = rawregulararray->toListOffsetArray64(true);
      ListOffsetArray64* rawother = dynamic_cast<ListOffsetArray64*>(listoffsetarray.get());
      content = content_.get()->merge(rawother->content());
      Index64 other_starts = rawother->starts();
      Index64 other_stops = rawother->stops();
      struct Error err = awkward_listarray_fill_to64_from64(
        starts.ptr().get(),
        mylength,
        stops.ptr().get(),
        mylength,
        other_starts.ptr().get(),
        other_starts.offset(),
        other_stops.ptr().get(),
        other_stops.offset(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(), rawother->identities().get());
    }
    else {
      throw std::invalid_argument(std::string("cannot merge ") + classname() + std::string(" with ") + other.get()->classname());
    }

    return std::make_shared<ListArray64>(Identities::none(), util::Parameters(), starts, stops, content);
  }

  template <typename T>
  const std::shared_ptr<SliceItem> ListArrayOf<T>::asslice() const {
    return toListOffsetArray64(true).get()->asslice();
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::rpad(int64_t target, int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      return rpad_axis0(target, false);
    }
    else if (toaxis == depth + 1) {
      int64_t min = target;
      struct Error err1 = util::awkward_ListArray_min_range<T>(
        &min,
        starts_.ptr().get(),
        stops_.ptr().get(),
        starts_.length(),
        starts_.offset(),
        stops_.offset());
      util::handle_error(err1, classname(), identities_.get());
      if (target < min) {
        return shallow_copy();
      }
      else {
        int64_t tolength = 0;
        struct Error err2 = util::awkward_ListArray_rpad_and_clip_length_axis1<T>(
          &tolength,
          starts_.ptr().get(),
          stops_.ptr().get(),
          target,
          starts_.length(),
          starts_.offset(),
          stops_.offset()
        );
        util::handle_error(err2, classname(), identities_.get());

        Index64 index(tolength);
        IndexOf<T> starts(starts_.length());
        IndexOf<T> stops(starts_.length());
        struct Error err3 = util::awkward_ListArray_rpad_axis1_64<T>(
          index.ptr().get(),
          starts_.ptr().get(),
          stops_.ptr().get(),
          starts.ptr().get(),
          stops.ptr().get(),
          target,
          starts_.length(),
          starts_.offset(),
          stops_.offset());
        util::handle_error(err3, classname(), identities_.get());

        std::shared_ptr<IndexedOptionArray64> next = std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, content());
        return std::make_shared<ListArrayOf<T>>(Identities::none(), parameters_, starts, stops, next.get()->simplify());
      }
    }
    else {
      return std::make_shared<ListArrayOf<T>>(Identities::none(), parameters_, starts_, stops_, content_.get()->rpad(target, toaxis, depth + 1));
    }
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::rpad_and_clip(int64_t target, int64_t axis, int64_t depth) const {
    return toListOffsetArray64(true).get()->rpad_and_clip(target, axis, depth);
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::reduce_next(const Reducer& reducer, int64_t negaxis, const Index64& parents, int64_t outlength, bool mask, bool keepdims) const {
    return toListOffsetArray64(true).get()->reduce_next(reducer, negaxis, parents, outlength, mask, keepdims);
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::localindex(int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (axis == depth) {
      return localindex_axis0();
    }
    else if (axis == depth + 1) {
      Index64 offsets = compact_offsets64(true);
      int64_t innerlength = offsets.ptr().get()[offsets.offset() + offsets.length() - 1];
      Index64 localindex(innerlength);
      struct Error err = util::awkward_listarray_localindex_64(
        localindex.ptr().get(),
        offsets.ptr().get(),
        offsets.offset(),
        offsets.length() - 1);
      util::handle_error(err, classname(), identities_.get());
      return std::make_shared<ListOffsetArray64>(identities_, util::Parameters(), offsets, std::make_shared<NumpyArray>(localindex));
    }
    else {
      return std::make_shared<ListArrayOf<T>>(identities_, util::Parameters(), starts_, stops_, content_.get()->localindex(axis, depth + 1));
    }
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::choose(int64_t n, bool diagonal, const std::shared_ptr<util::RecordLookup>& recordlookup, const util::Parameters& parameters, int64_t axis, int64_t depth) const {
    if (n < 1) {
      throw std::invalid_argument("in choose, 'n' must be at least 1");
    }

    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      return choose_axis0(n, diagonal, recordlookup, parameters);
    }

    else if (toaxis == depth + 1) {
      int64_t totallen;
      Index64 offsets(length() + 1);
      struct Error err1 = util::awkward_listarray_choose_length_64<T>(
        &totallen,
        offsets.ptr().get(),
        n,
        diagonal,
        starts_.ptr().get(),
        starts_.offset(),
        stops_.ptr().get(),
        stops_.offset(),
        length());
      util::handle_error(err1, classname(), identities_.get());

      std::vector<std::shared_ptr<int64_t>> tocarry;
      std::vector<int64_t*> tocarryraw;
      for (int64_t j = 0;  j < n;  j++) {
        std::shared_ptr<int64_t> ptr(new int64_t[(size_t)totallen], util::array_deleter<int64_t>());
        tocarry.push_back(ptr);
        tocarryraw.push_back(ptr.get());
      }
      struct Error err2 = util::awkward_listarray_choose_64<T>(
        tocarryraw.data(),
        n,
        diagonal,
        starts_.ptr().get(),
        starts_.offset(),
        stops_.ptr().get(),
        stops_.offset(),
        length());
      util::handle_error(err2, classname(), identities_.get());

      std::vector<std::shared_ptr<Content>> contents;
      for (auto ptr : tocarry) {
        contents.push_back(content_.get()->carry(Index64(ptr, 0, totallen)));
      }
      std::shared_ptr<Content> recordarray = std::make_shared<RecordArray>(Identities::none(), parameters, contents, recordlookup);

      return std::make_shared<ListOffsetArray64>(identities_, util::Parameters(), offsets, recordarray);
    }

    else {
      std::shared_ptr<Content> compact = toListOffsetArray64(true);
      ListOffsetArray64* rawcompact = dynamic_cast<ListOffsetArray64*>(compact.get());
      std::shared_ptr<Content> next = rawcompact->content().get()->choose(n, diagonal, recordlookup, parameters, axis, depth + 1);
      return std::make_shared<ListOffsetArray64>(identities_, util::Parameters(), rawcompact->offsets(), next);
    }
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    if (advanced.length() != 0) {
      throw std::runtime_error("ListArray::getitem_next(SliceAt): advanced.length() != 0");
    }
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    Index64 nextcarry(lenstarts);
    struct Error err = util::awkward_listarray_getitem_next_at_64<T>(
      nextcarry.ptr().get(),
      starts_.ptr().get(),
      stops_.ptr().get(),
      lenstarts,
      starts_.offset(),
      stops_.offset(),
      at.at());
    util::handle_error(err, classname(), identities_.get());
    std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
    return nextcontent.get()->getitem_next(nexthead, nexttail, advanced);
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    int64_t start = range.start();
    int64_t stop = range.stop();
    int64_t step = range.step();
    if (step == Slice::none()) {
      step = 1;
    }
    int64_t carrylength;
    struct Error err1 = util::awkward_listarray_getitem_next_range_carrylength<T>(
      &carrylength,
      starts_.ptr().get(),
      stops_.ptr().get(),
      lenstarts,
      starts_.offset(),
      stops_.offset(),
      start,
      stop,
      step);
    util::handle_error(err1, classname(), identities_.get());

    IndexOf<T> nextoffsets(lenstarts + 1);
    Index64 nextcarry(carrylength);

    struct Error err2 = util::awkward_listarray_getitem_next_range_64<T>(
      nextoffsets.ptr().get(),
      nextcarry.ptr().get(),
      starts_.ptr().get(),
      stops_.ptr().get(),
      lenstarts,
      starts_.offset(),
      stops_.offset(),
      start,
      stop,
      step);
    util::handle_error(err2, classname(), identities_.get());
    std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);

    if (advanced.length() == 0) {
      return std::make_shared<ListOffsetArrayOf<T>>(identities_, parameters_, nextoffsets, nextcontent.get()->getitem_next(nexthead, nexttail, advanced));
    }
    else {
      int64_t total;
      struct Error err1 = util::awkward_listarray_getitem_next_range_counts_64<T>(
        &total,
        nextoffsets.ptr().get(),
        lenstarts);
      util::handle_error(err1, classname(), identities_.get());
      Index64 nextadvanced(total);
      struct Error err2 = util::awkward_listarray_getitem_next_range_spreadadvanced_64<T>(
        nextadvanced.ptr().get(),
        advanced.ptr().get(),
        nextoffsets.ptr().get(),
        lenstarts);
      util::handle_error(err2, classname(), identities_.get());
      return std::make_shared<ListOffsetArrayOf<T>>(identities_, parameters_, nextoffsets, nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced));
    }
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    Index64 flathead = array.ravel();
    if (advanced.length() == 0) {
      Index64 nextcarry(lenstarts*flathead.length());
      Index64 nextadvanced(lenstarts*flathead.length());
      struct Error err = util::awkward_listarray_getitem_next_array_64<T>(
        nextcarry.ptr().get(),
        nextadvanced.ptr().get(),
        starts_.ptr().get(),
        stops_.ptr().get(),
        flathead.ptr().get(),
        starts_.offset(),
        stops_.offset(),
        lenstarts,
        flathead.length(),
        content_.get()->length());
      util::handle_error(err, classname(), identities_.get());
      std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
      return getitem_next_array_wrap(nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced), array.shape());
    }
    else {
      Index64 nextcarry(lenstarts);
      Index64 nextadvanced(lenstarts);
      struct Error err = util::awkward_listarray_getitem_next_array_advanced_64<T>(
        nextcarry.ptr().get(),
        nextadvanced.ptr().get(),
        starts_.ptr().get(),
        stops_.ptr().get(),
        flathead.ptr().get(),
        advanced.ptr().get(),
        starts_.offset(),
        stops_.offset(),
        lenstarts,
        flathead.length(),
        content_.get()->length());
      util::handle_error(err, classname(), identities_.get());
      std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
      return nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced);
    }
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_next(const SliceJagged64& jagged, const Slice& tail, const Index64& advanced) const {
    if (advanced.length() != 0) {
      throw std::invalid_argument("cannot mix jagged slice with NumPy-style advanced indexing");
    }
    if (stops_.length() < starts_.length()) {
      util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    int64_t len = length();
    Index64 singleoffsets = jagged.offsets();
    Index64 multistarts(jagged.length()*len);
    Index64 multistops(jagged.length()*len);
    Index64 nextcarry(jagged.length()*len);
    struct Error err = util::awkward_listarray_getitem_jagged_expand_64(
      multistarts.ptr().get(),
      multistops.ptr().get(),
      singleoffsets.ptr().get(),
      nextcarry.ptr().get(),
      starts_.ptr().get(),
      starts_.offset(),
      stops_.ptr().get(),
      stops_.offset(),
      jagged.length(),
      len);
    util::handle_error(err, classname(), identities_.get());

    std::shared_ptr<Content> carried = content_.get()->carry(nextcarry);
    std::shared_ptr<Content> down = carried.get()->getitem_next_jagged(multistarts, multistops, jagged.content(), tail);

    return std::make_shared<RegularArray>(Identities::none(), util::Parameters(), down, jagged.length());
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceArray64& slicecontent, const Slice& tail) const {
    if (starts_.length() < slicestarts.length()) {
      util::handle_error(failure("jagged slice length differs from array length", kSliceNone, kSliceNone), classname(), identities_.get());
    }
    if (stops_.length() < starts_.length()) {
      util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    int64_t carrylen;
    struct Error err1 = awkward_listarray_getitem_jagged_carrylen_64(
      &carrylen,
      slicestarts.ptr().get(),
      slicestarts.offset(),
      slicestops.ptr().get(),
      slicestops.offset(),
      slicestarts.length());
    util::handle_error(err1, classname(), identities_.get());

    Index64 sliceindex = slicecontent.index();
    Index64 outoffsets(slicestarts.length() + 1);
    Index64 nextcarry(carrylen);
    struct Error err2 = util::awkward_listarray_getitem_jagged_apply_64<T>(
      outoffsets.ptr().get(),
      nextcarry.ptr().get(),
      slicestarts.ptr().get(),
      slicestarts.offset(),
      slicestops.ptr().get(),
      slicestops.offset(),
      slicestarts.length(),
      sliceindex.ptr().get(),
      sliceindex.offset(),
      sliceindex.length(),
      starts_.ptr().get(),
      starts_.offset(),
      stops_.ptr().get(),
      stops_.offset(),
      content_.get()->length());
    util::handle_error(err2, classname(), nullptr);

    std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
    std::shared_ptr<Content> outcontent = nextcontent.get()->getitem_next(tail.head(), tail.tail(), Index64(0));

    return std::make_shared<ListOffsetArray64>(Identities::none(), util::Parameters(), outoffsets, outcontent);
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceMissing64& slicecontent, const Slice& tail) const {
    if (starts_.length() < slicestarts.length()) {
      util::handle_error(failure("jagged slice length differs from array length", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    Index64 missing = slicecontent.index();
    int64_t numvalid;
    struct Error err1 = awkward_listarray_getitem_jagged_numvalid_64(
      &numvalid,
      slicestarts.ptr().get(),
      slicestarts.offset(),
      slicestops.ptr().get(),
      slicestops.offset(),
      slicestarts.length(),
      missing.ptr().get(),
      missing.offset(),
      missing.length());
    util::handle_error(err1, classname(), nullptr);

    Index64 nextcarry(numvalid);
    Index64 smalloffsets(slicestarts.length() + 1);
    Index64 largeoffsets(slicestarts.length() + 1);
    struct Error err2 = awkward_listarray_getitem_jagged_shrink_64(
      nextcarry.ptr().get(),
      smalloffsets.ptr().get(),
      largeoffsets.ptr().get(),
      slicestarts.ptr().get(),
      slicestarts.offset(),
      slicestops.ptr().get(),
      slicestops.offset(),
      slicestarts.length(),
      missing.ptr().get(),
      missing.offset());
    util::handle_error(err2, classname(), nullptr);

    std::shared_ptr<Content> out;
    if (dynamic_cast<SliceJagged64*>(slicecontent.content().get())) {
      std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
      std::shared_ptr<Content> next = std::make_shared<ListOffsetArray64>(Identities::none(), util::Parameters(), smalloffsets, nextcontent);
      out = next.get()->getitem_next_jagged(util::make_starts(smalloffsets), util::make_stops(smalloffsets), slicecontent.content(), tail);
    }
    else {
      out = Content::getitem_next_jagged(util::make_starts(smalloffsets), util::make_stops(smalloffsets), slicecontent.content(), tail);
    }

    if (ListOffsetArray64* raw = dynamic_cast<ListOffsetArray64*>(out.get())) {
      std::shared_ptr<Content> content = raw->content();
      IndexedOptionArray64 indexedoptionarray(Identities::none(), util::Parameters(), missing, content);
      return std::make_shared<ListOffsetArray64>(Identities::none(), util::Parameters(), largeoffsets, indexedoptionarray.simplify());
    }
    else {
      throw std::runtime_error(std::string("expected ListOffsetArray64 from ListArray::getitem_next_jagged, got ") + out.get()->classname());
    }
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceJagged64& slicecontent, const Slice& tail) const {
    if (starts_.length() < slicestarts.length()) {
      util::handle_error(failure("jagged slice length differs from array length", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    Index64 outoffsets(slicestarts.length() + 1);
    struct Error err = util::awkward_listarray_getitem_jagged_descend_64<T>(
      outoffsets.ptr().get(),
      slicestarts.ptr().get(),
      slicestarts.offset(),
      slicestops.ptr().get(),
      slicestops.offset(),
      slicestarts.length(),
      starts_.ptr().get(),
      starts_.offset(),
      stops_.ptr().get(),
      stops_.offset());
    util::handle_error(err, classname(), identities_.get());

    Index64 sliceoffsets = slicecontent.offsets();
    std::shared_ptr<Content> outcontent = content_.get()->getitem_next_jagged(util::make_starts(sliceoffsets), util::make_stops(sliceoffsets), slicecontent.content(), tail);

    return std::make_shared<ListOffsetArray64>(Identities::none(), util::Parameters(), outoffsets, outcontent);
  }

  template class ListArrayOf<int32_t>;
  template class ListArrayOf<uint32_t>;
  template class ListArrayOf<int64_t>;
}
