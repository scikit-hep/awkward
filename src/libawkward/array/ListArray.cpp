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

#include "awkward/array/ListArray.h"

namespace awkward {
  template <typename T>
  ListArrayOf<T>::ListArrayOf(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const IndexOf<T>& starts, const IndexOf<T>& stops, const std::shared_ptr<Content>& content)
      : Content(identities, parameters)
      , starts_(starts)
      , stops_(stops)
      , content_(content) { }

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
  Index64 ListArrayOf<T>::compact_offsets64() const {
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
    Index64 offsets = compact_offsets64();
    std::shared_ptr<Content> listoffsetarray64 = broadcast_tooffsets64(offsets);
    ListOffsetArray64* raw = dynamic_cast<ListOffsetArray64*>(listoffsetarray64.get());
    return raw->toRegularArray();
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
  const std::shared_ptr<Type> ListArrayOf<T>::type() const {
    return std::make_shared<ListType>(parameters_, content_.get()->type());
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::astype(const std::shared_ptr<Type>& type) const {
    if (ListType* raw = dynamic_cast<ListType*>(type.get())) {
      return std::make_shared<ListArrayOf<T>>(identities_, type.get()->parameters(), starts_, stops_, content_.get()->astype(raw->type()));
    }
    else {
      throw std::invalid_argument(classname() + std::string(" cannot be converted to type ") + type.get()->tostring());
    }
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
  const Index64 ListArrayOf<T>::count64() const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
    }
    Index64 tocount(starts_.length());
    struct Error err = util::awkward_listarray_count_64(
      tocount.ptr().get(),
      starts_.ptr().get(),
      stops_.ptr().get(),
      lenstarts,
      starts_.offset(),
      stops_.offset());
    util::handle_error(err, classname(), identities_.get());
    return tocount;
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::count(int64_t axis) const {
    if (axis != 0) {
      throw std::runtime_error("FIXME: ListArray::count(axis != 0)");
    }
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
    }
    IndexOf<T> tocount(starts_.length());
    struct Error err = util::awkward_listarray_count(
      tocount.ptr().get(),
      starts_.ptr().get(),
      stops_.ptr().get(),
      lenstarts,
      starts_.offset(),
      stops_.offset());
    util::handle_error(err, classname(), identities_.get());
    std::vector<ssize_t> shape({ (ssize_t)lenstarts });
    std::vector<ssize_t> strides({ (ssize_t)sizeof(T) });
    std::string format;
#ifdef _MSC_VER
    if (std::is_same<T, int32_t>::value) {
      format = "l";
    }
    else if (std::is_same<T, uint32_t>::value) {
      format = "L";
    }
    else if (std::is_same<T, int64_t>::value) {
      format = "q";
    }
#else
    if (std::is_same<T, int32_t>::value) {
      format = "i";
    }
    else if (std::is_same<T, uint32_t>::value) {
      format = "I";
    }
    else if (std::is_same<T, int64_t>::value) {
      format = "l";
    }
#endif
    else {
      throw std::runtime_error("unrecognized ListArray specialization");
    }
    return std::make_shared<NumpyArray>(Identities::none(), util::Parameters(), tocount.ptr(), shape, strides, 0, sizeof(T), format);
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::flatten(int64_t axis) const {
    if (axis < 0) {
      std::pair<int64_t, int64_t> minmax = minmax_depth();
      int64_t mindepth = minmax.first;
      int64_t maxdepth = minmax.second;
      int64_t depth = purelist_depth();
      if (mindepth == depth  &&  maxdepth == depth) {
        if (depth - 1 + axis < 0) {
          throw std::invalid_argument(std::string("ListArrayOf<T> cannot be flattened in axis ") + std::to_string(axis) + std::string(" because its depth is ") + std::to_string(depth));
        }
        return flatten(depth - 1 + axis);
      }
      else {
        return content_.get()->flatten(axis);
      }
    }
    else if (axis == 0) {
      int64_t lenstarts = starts_.length();
      if (stops_.length() < lenstarts) {
        util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
      }

      int64_t lenarray(0);
      struct Error err1 = util::awkward_listarray_flatten_length(
        &lenarray,
        starts_.ptr().get(),
        stops_.ptr().get(),
        lenstarts,
        starts_.offset(),
        stops_.offset());
      util::handle_error(err1, classname(), identities_.get());

      Index64 indxarray(lenarray);
      struct Error err2 = util::awkward_listarray_flatten_64<T>(
        indxarray.ptr().get(),
        starts_.ptr().get(),
        stops_.ptr().get(),
        lenstarts,
        starts_.offset(),
        stops_.offset());
      util::handle_error(err2, classname(), identities_.get());

      return content_.get()->carry(indxarray);
    }
    else if (axis == 1) {
      int64_t lenstarts = starts_.length();
      IndexOf<T> tostarts(lenstarts);
      IndexOf<T> tostops(lenstarts);

      Index64 count = count64();
      int64_t clength = count.length();
      if (clength != lenstarts) {
        throw std::runtime_error("scale index length must be equal to start and stop length");
      }
      struct Error err3 = util::awkward_listarray_flatten_scale_64<T>(
        tostarts.ptr().get(),
        tostops.ptr().get(),
        count.ptr().get(),
        starts_.ptr().get(),
        stops_.ptr().get(),
        lenstarts,
        starts_.offset(),
        stops_.offset());
      util::handle_error(err3, classname(), identities_.get());

      return std::make_shared<ListArrayOf<T>>(identities_, parameters_, tostarts, tostops, content_.get()->flatten(axis - 1));
    }
    else {
      return std::make_shared<ListArrayOf<T>>(identities_, parameters_, starts_, stops_, content_.get()->flatten(axis - 1));
    }
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      util::handle_error(failure("len(stops) < len(starts)", kSliceNone, kSliceNone), classname(), identities_.get());
    }

    assert(advanced.length() == 0);
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

  template class ListArrayOf<int32_t>;
  template class ListArrayOf<uint32_t>;
  template class ListArrayOf<int64_t>;
}
