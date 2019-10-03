// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/ListArray.h"

#include "awkward/ListOffsetArray.h"

namespace awkward {
  template <typename T>
  const std::string ListOffsetArrayOf<T>::classname() const {
    if (std::is_same<T, int32_t>::value) {
      return "ListOffsetArray32";
    }
    else if (std::is_same<T, int64_t>::value) {
      return "ListOffsetArray64";
    }
    else {
      return "UnrecognizedListOffsetArray";
    }
  }

  template <typename T>
  IndexOf<T> make_starts(const IndexOf<T>& offsets) {
    return IndexOf<T>(offsets.ptr(), offsets.offset(), offsets.length() - 1);
  }

  template <typename T>
  IndexOf<T> make_stops(const IndexOf<T>& offsets) {
    return IndexOf<T>(offsets.ptr(), offsets.offset() + 1, offsets.length() - 1);
  }

  template <>
  void ListOffsetArrayOf<int32_t>::setid(const std::shared_ptr<Identity> id) {
    if (id.get() == nullptr) {
      content_.get()->setid(id);
    }
    else {
      if (length() != id.get()->length()) {
        throw std::invalid_argument("content and its id must have the same length");
      }
      Index32 starts = make_starts(offsets_);
      Index32 stops = make_stops(offsets_);
      std::shared_ptr<Identity> bigid = id;
      if (content_.get()->length() > kMaxInt32) {
        bigid = id.get()->to64();
      }
      if (Identity32* rawid = dynamic_cast<Identity32*>(bigid.get())) {
        Identity32* rawsubid = new Identity32(Identity::newref(), rawid->fieldloc(), rawid->width() + 1, content_.get()->length());
        std::shared_ptr<Identity> subid(rawsubid);
        Error err = awkward_identity32_from_listarray32(
          rawsubid->ptr().get(),
          rawid->ptr().get(),
          starts.ptr().get(),
          stops.ptr().get(),
          rawid->offset(),
          starts.offset(),
          stops.offset(),
          content_.get()->length(),
          length(),
          rawid->width());
        util::handle_error(err, classname(), id_.get(), nullptr);
        content_.get()->setid(subid);
      }
      else if (Identity64* rawid = dynamic_cast<Identity64*>(bigid.get())) {
        Identity64* rawsubid = new Identity64(Identity::newref(), rawid->fieldloc(), rawid->width() + 1, content_.get()->length());
        std::shared_ptr<Identity> subid(rawsubid);
        Error err = awkward_identity64_from_listarray32(
          rawsubid->ptr().get(),
          rawid->ptr().get(),
          starts.ptr().get(),
          stops.ptr().get(),
          rawid->offset(),
          starts.offset(),
          stops.offset(),
          content_.get()->length(),
          length(),
          rawid->width());
        util::handle_error(err, classname(), id_.get(), nullptr);
        content_.get()->setid(subid);
      }
      else {
        throw std::runtime_error("unrecognized Identity specialization");
      }
    }
    id_ = id;
  }

  template <>
  void ListOffsetArrayOf<int64_t>::setid(const std::shared_ptr<Identity> id) {
    if (id.get() == nullptr) {
      content_.get()->setid(id);
    }
    else {
      if (length() != id.get()->length()) {
        throw std::invalid_argument("content and its id must have the same length");
      }
      Index64 starts = make_starts(offsets_);
      Index64 stops = make_stops(offsets_);
      std::shared_ptr<Identity> bigid = id.get()->to64();
      if (Identity64* rawid = dynamic_cast<Identity64*>(bigid.get())) {
        Identity64* rawsubid = new Identity64(Identity::newref(), rawid->fieldloc(), rawid->width() + 1, content_.get()->length());
        std::shared_ptr<Identity> subid(rawsubid);
        Error err = awkward_identity64_from_listarray64(
          rawsubid->ptr().get(),
          rawid->ptr().get(),
          starts.ptr().get(),
          stops.ptr().get(),
          rawid->offset(),
          starts.offset(),
          stops.offset(),
          content_.get()->length(),
          length(),
          rawid->width());
        util::handle_error(err, classname(), id_.get(), nullptr);
        content_.get()->setid(subid);
      }
      else {
        throw std::runtime_error("unrecognized Identity specialization");
      }
    }
    id_ = id;
  }

  template <typename T>
  void ListOffsetArrayOf<T>::setid() {
    if (length() <= kMaxInt32) {
      Identity32* rawid = new Identity32(Identity::newref(), Identity::FieldLoc(), 1, length());
      std::shared_ptr<Identity> newid(rawid);
      awkward_new_identity32(rawid->ptr().get(), length());
      setid(newid);
    }
    else {
      Identity64* rawid = new Identity64(Identity::newref(), Identity::FieldLoc(), 1, length());
      std::shared_ptr<Identity> newid(rawid);
      awkward_new_identity64(rawid->ptr().get(), length());
      setid(newid);
    }
  }

  template <typename T>
  const std::string ListOffsetArrayOf<T>::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << ">\n";
    if (id_.get() != nullptr) {
      out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
    }
    out << offsets_.tostring_part(indent + std::string("    "), "<offsets>", "</offsets>\n");
    out << content_.get()->tostring_part(indent + std::string("    "), "<content>", "</content>\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  template <typename T>
  int64_t ListOffsetArrayOf<T>::length() const {
    return offsets_.length() - 1;
  }

  template <typename T>
  const std::shared_ptr<Content> ListOffsetArrayOf<T>::shallow_copy() const {
    return std::shared_ptr<Content>(new ListOffsetArrayOf<T>(id_, offsets_, content_));
  }

  template <typename T>
  const std::shared_ptr<Content> ListOffsetArrayOf<T>::getitem_at(int64_t at) const {
    int64_t start = (int64_t)offsets_.getitem_at(at);
    int64_t stop = (int64_t)offsets_.getitem_at(at + 1);
    return content_.get()->getitem_range(start, stop);
  }

  template <typename T>
  const std::shared_ptr<Content> ListOffsetArrayOf<T>::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), offsets_.length() - 1);

    std::shared_ptr<Identity> id(nullptr);
    if (id_.get() != nullptr) {
      if (regular_stop > id_.get()->length()) {
        throw std::invalid_argument("index out of range for identity");
      }
      id = id_.get()->getitem_range(regular_start, regular_stop);
    }

    return std::shared_ptr<Content>(new ListOffsetArrayOf<T>(id, offsets_.getitem_range(regular_start, regular_stop + 1), content_));
  }

  template <>
  const std::shared_ptr<Content> ListOffsetArrayOf<int32_t>::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const {
    int64_t lenstarts = offsets_.length() - 1;

    if (head.get() == nullptr) {
      return shallow_copy();
    }

    else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
      assert(advanced.length() == 0);
      Index32 starts = make_starts(offsets_);
      Index32 stops = make_stops(offsets_);
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      Index64 nextcarry(lenstarts);
      Error err = awkward_listarray32_getitem_next_at_64(
        nextcarry.ptr().get(),
        starts.ptr().get(),
        stops.ptr().get(),
        lenstarts,
        starts.offset(),
        stops.offset(),
        at->at());
      std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
      return nextcontent.get()->getitem_next(nexthead, nexttail, advanced);
    }

    else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
      Index32 starts = make_starts(offsets_);
      Index32 stops = make_stops(offsets_);
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      int64_t start = range->start();
      int64_t stop = range->stop();
      int64_t step = range->step();
      if (step == Slice::none()) {
        step = 1;
      }
      int64_t carrylength;
      awkward_listarray32_getitem_next_range_carrylength(
        &carrylength,
        starts.ptr().get(),
        stops.ptr().get(),
        lenstarts,
        starts.offset(),
        stops.offset(),
        start,
        stop,
        step);

      Index32 nextoffsets(lenstarts + 1);
      Index64 nextcarry(carrylength);

      awkward_listarray32_getitem_next_range_64(
        nextoffsets.ptr().get(),
        nextcarry.ptr().get(),
        starts.ptr().get(),
        stops.ptr().get(),
        lenstarts,
        starts.offset(),
        stops.offset(),
        start,
        stop,
        step);
      std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);

      if (advanced.length() == 0) {
        return std::shared_ptr<Content>(new ListOffsetArrayOf<int32_t>(id_, nextoffsets, nextcontent.get()->getitem_next(nexthead, nexttail, advanced)));
      }
      else {
        int64_t total;
        awkward_listarray32_getitem_next_range_counts_64(
          &total,
          nextoffsets.ptr().get(),
          lenstarts);
        Index64 nextadvanced(total);
        awkward_listarray32_getitem_next_range_spreadadvanced_64(
          nextadvanced.ptr().get(),
          advanced.ptr().get(),
          nextoffsets.ptr().get(),
          lenstarts);
        return std::shared_ptr<Content>(new ListOffsetArrayOf<int32_t>(id_, nextoffsets, nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced)));
      }
    }

    else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
      return getitem_ellipsis(tail, advanced);
    }

    else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
      return getitem_newaxis(tail, advanced);
    }

    else if (SliceArray64* array = dynamic_cast<SliceArray64*>(head.get())) {
      Index32 starts = make_starts(offsets_);
      Index32 stops = make_stops(offsets_);
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      Index64 flathead = array->ravel();
      if (advanced.length() == 0) {
        Index64 nextcarry(lenstarts*flathead.length());
        Index64 nextadvanced(lenstarts*flathead.length());
        Index32 nextoffsets(lenstarts + 1);
        Index32 nextstops(lenstarts);
        Error err = awkward_listarray32_getitem_next_array_64(
          nextoffsets.ptr().get(),
          nextcarry.ptr().get(),
          nextadvanced.ptr().get(),
          starts.ptr().get(),
          stops.ptr().get(),
          flathead.ptr().get(),
          starts.offset(),
          stops.offset(),
          lenstarts,
          flathead.length(),
          content_.get()->length());
        util::handle_error(err, classname(), id_.get(), nullptr);
        std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
        // FIXME: if the head is not flat, you'll need to wrap the ListArray output in a RegularArray
        return std::shared_ptr<Content>(new ListOffsetArrayOf<int32_t>(id_, nextoffsets, nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced)));
      }
      else {
        Index64 nextcarry(lenstarts);
        Index64 nextadvanced(lenstarts);
        Error err = awkward_listarray32_getitem_next_array_advanced_64(
          nextcarry.ptr().get(),
          nextadvanced.ptr().get(),
          starts.ptr().get(),
          stops.ptr().get(),
          flathead.ptr().get(),
          advanced.ptr().get(),
          starts.offset(),
          stops.offset(),
          lenstarts,
          flathead.length(),
          content_.get()->length());
        util::handle_error(err, classname(), id_.get(), nullptr);
        std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
        return nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced);
      }
    }

    else {
      throw std::runtime_error("unrecognized slice item type");
    }
  }

  template <>
  const std::shared_ptr<Content> ListOffsetArrayOf<int64_t>::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const {
    int64_t lenstarts = offsets_.length() - 1;

    if (head.get() == nullptr) {
      return shallow_copy();
    }

    else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
      assert(advanced.length() == 0);
      Index64 starts = make_starts(offsets_);
      Index64 stops = make_stops(offsets_);
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      Index64 nextcarry(lenstarts);
      Error err = awkward_listarray64_getitem_next_at_64(
        nextcarry.ptr().get(),
        starts.ptr().get(),
        stops.ptr().get(),
        lenstarts,
        starts.offset(),
        stops.offset(),
        at->at());
      std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
      return nextcontent.get()->getitem_next(nexthead, nexttail, advanced);
    }

    else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
      Index64 starts = make_starts(offsets_);
      Index64 stops = make_stops(offsets_);
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      int64_t start = range->start();
      int64_t stop = range->stop();
      int64_t step = range->step();
      if (step == Slice::none()) {
        step = 1;
      }
      int64_t carrylength;
      awkward_listarray64_getitem_next_range_carrylength(
        &carrylength,
        starts.ptr().get(),
        stops.ptr().get(),
        lenstarts,
        starts.offset(),
        stops.offset(),
        start,
        stop,
        step);

      Index64 nextoffsets(lenstarts + 1);
      Index64 nextcarry(carrylength);

      awkward_listarray64_getitem_next_range_64(
        nextoffsets.ptr().get(),
        nextcarry.ptr().get(),
        starts.ptr().get(),
        stops.ptr().get(),
        lenstarts,
        starts.offset(),
        stops.offset(),
        start,
        stop,
        step);
      std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);

      if (advanced.length() == 0) {
        return std::shared_ptr<Content>(new ListOffsetArrayOf<int64_t>(id_, nextoffsets, nextcontent.get()->getitem_next(nexthead, nexttail, advanced)));
      }
      else {
        int64_t total;
        awkward_listarray64_getitem_next_range_counts_64(
          &total,
          nextoffsets.ptr().get(),
          lenstarts);
        Index64 nextadvanced(total);
        awkward_listarray64_getitem_next_range_spreadadvanced_64(
          nextadvanced.ptr().get(),
          advanced.ptr().get(),
          nextoffsets.ptr().get(),
          lenstarts);
        return std::shared_ptr<Content>(new ListOffsetArrayOf<int64_t>(id_, nextoffsets, nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced)));
      }
    }

    else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
      return getitem_ellipsis(tail, advanced);
    }

    else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
      return getitem_newaxis(tail, advanced);
    }

    else if (SliceArray64* array = dynamic_cast<SliceArray64*>(head.get())) {
      Index64 starts = make_starts(offsets_);
      Index64 stops = make_stops(offsets_);
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      Index64 flathead = array->ravel();
      if (advanced.length() == 0) {
        Index64 nextcarry(lenstarts*flathead.length());
        Index64 nextadvanced(lenstarts*flathead.length());
        Index64 nextoffsets(lenstarts + 1);
        Index64 nextstops(lenstarts);
        Error err = awkward_listarray64_getitem_next_array_64(
          nextoffsets.ptr().get(),
          nextcarry.ptr().get(),
          nextadvanced.ptr().get(),
          starts.ptr().get(),
          stops.ptr().get(),
          flathead.ptr().get(),
          starts.offset(),
          stops.offset(),
          lenstarts,
          flathead.length(),
          content_.get()->length());
        util::handle_error(err, classname(), id_.get(), nullptr);
        std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
        // FIXME: if the head is not flat, you'll need to wrap the ListArray output in a RegularArray
        return std::shared_ptr<Content>(new ListOffsetArrayOf<int64_t>(id_, nextoffsets, nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced)));
      }
      else {
        Index64 nextcarry(lenstarts);
        Index64 nextadvanced(lenstarts);
        Error err = awkward_listarray64_getitem_next_array_advanced_64(
          nextcarry.ptr().get(),
          nextadvanced.ptr().get(),
          starts.ptr().get(),
          stops.ptr().get(),
          flathead.ptr().get(),
          advanced.ptr().get(),
          starts.offset(),
          stops.offset(),
          lenstarts,
          flathead.length(),
          content_.get()->length());
        util::handle_error(err, classname(), id_.get(), nullptr);
        std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
        return nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced);
      }
    }

    else {
      throw std::runtime_error("unrecognized slice item type");
    }
  }

  template <>
  const std::shared_ptr<Content> ListOffsetArrayOf<int32_t>::carry(const Index64& carry) const {
    Index32 starts = make_starts(offsets_);
    Index32 stops = make_stops(offsets_);
    Index32 nextstarts(carry.length());
    Index32 nextstops(carry.length());
    awkward_listarray32_getitem_carry_64(
      nextstarts.ptr().get(),
      nextstops.ptr().get(),
      starts.ptr().get(),
      stops.ptr().get(),
      carry.ptr().get(),
      starts.offset(),
      stops.offset(),
      carry.length());
    std::shared_ptr<Identity> id(nullptr);
    if (id_.get() != nullptr) {
      id = id_.get()->getitem_carry_64(carry);
    }
    return std::shared_ptr<Content>(new ListArrayOf<int32_t>(id, nextstarts, nextstops, content_));
  }

  template <>
  const std::shared_ptr<Content> ListOffsetArrayOf<int64_t>::carry(const Index64& carry) const {
    Index64 starts = make_starts(offsets_);
    Index64 stops = make_stops(offsets_);
    Index64 nextstarts(carry.length());
    Index64 nextstops(carry.length());
    awkward_listarray64_getitem_carry_64(
      nextstarts.ptr().get(),
      nextstops.ptr().get(),
      starts.ptr().get(),
      stops.ptr().get(),
      carry.ptr().get(),
      starts.offset(),
      stops.offset(),
      carry.length());
    std::shared_ptr<Identity> id(nullptr);
    if (id_.get() != nullptr) {
      id = id_.get()->getitem_carry_64(carry);
    }
    return std::shared_ptr<Content>(new ListArrayOf<int64_t>(id, nextstarts, nextstops, content_));
  }

  template <typename T>
  const std::pair<int64_t, int64_t> ListOffsetArrayOf<T>::minmax_depth() const {
    std::pair<int64_t, int64_t> content_depth = content_.get()->minmax_depth();
    return std::pair<int64_t, int64_t>(content_depth.first + 1, content_depth.second + 1);
  }

  template class ListOffsetArrayOf<int32_t>;
  template class ListOffsetArrayOf<int64_t>;
}
