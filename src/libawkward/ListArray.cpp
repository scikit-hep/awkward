// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/Slice.h"
#include "awkward/ListOffsetArray.h"

#include "awkward/ListArray.h"

namespace awkward {
  template <typename T>
  const std::string ListArrayOf<T>::classname() const {
    if (std::is_same<T, int32_t>::value) {
      return "ListArray32";
    }
    else if (std::is_same<T, int64_t>::value) {
      return "ListArray64";
    }
    else {
      return "UnrecognizedListArray";
    }
  }

  template <>
  void ListArrayOf<int32_t>::setid(const std::shared_ptr<Identity> id) {
    if (id.get() == nullptr) {
      content_.get()->setid(id);
    }
    else {
      if (length() != id.get()->length()) {
        throw std::invalid_argument("content and its id must have the same length");
      }
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
          starts_.ptr().get(),
          stops_.ptr().get(),
          rawid->offset(),
          starts_.offset(),
          stops_.offset(),
          content_.get()->length(),
          length(),
          rawid->width());
        util::handle_error(err, classname(), id_.get());
        content_.get()->setid(subid);
      }
      else if (Identity64* rawid = dynamic_cast<Identity64*>(bigid.get())) {
        Identity64* rawsubid = new Identity64(Identity::newref(), rawid->fieldloc(), rawid->width() + 1, content_.get()->length());
        std::shared_ptr<Identity> subid(rawsubid);
        Error err = awkward_identity64_from_listarray32(
          rawsubid->ptr().get(),
          rawid->ptr().get(),
          starts_.ptr().get(),
          stops_.ptr().get(),
          rawid->offset(),
          starts_.offset(),
          stops_.offset(),
          content_.get()->length(),
          length(),
          rawid->width());
        util::handle_error(err, classname(), id_.get());
        content_.get()->setid(subid);
      }
      else {
        throw std::runtime_error("unrecognized Identity specialization");
      }
    }
    id_ = id;
  }

  template <>
  void ListArrayOf<int64_t>::setid(const std::shared_ptr<Identity> id) {
    if (id.get() == nullptr) {
      content_.get()->setid(id);
    }
    else {
      if (length() != id.get()->length()) {
        throw std::invalid_argument("content and its id must have the same length");
      }
      std::shared_ptr<Identity> bigid = id.get()->to64();
      if (Identity64* rawid = dynamic_cast<Identity64*>(bigid.get())) {
        Identity64* rawsubid = new Identity64(Identity::newref(), rawid->fieldloc(), rawid->width() + 1, content_.get()->length());
        std::shared_ptr<Identity> subid(rawsubid);
        Error err = awkward_identity64_from_listarray64(
          rawsubid->ptr().get(),
          rawid->ptr().get(),
          starts_.ptr().get(),
          stops_.ptr().get(),
          rawid->offset(),
          starts_.offset(),
          stops_.offset(),
          content_.get()->length(),
          length(),
          rawid->width());
        util::handle_error(err, classname(), id_.get());
        content_.get()->setid(subid);
      }
      else {
        throw std::runtime_error("unrecognized Identity specialization");
      }
    }
    id_ = id;
  }

  template <typename T>
  void ListArrayOf<T>::setid() {
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
  const std::string ListArrayOf<T>::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << ">\n";
    if (id_.get() != nullptr) {
      out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
    }
    out << starts_.tostring_part(indent + std::string("    "), "<starts>", "</starts>\n");
    out << stops_.tostring_part(indent + std::string("    "), "<stops>", "</stops>\n");
    out << content_.get()->tostring_part(indent + std::string("    "), "<content>", "</content>\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  template <typename T>
  int64_t ListArrayOf<T>::length() const {
    return starts_.length();
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::shallow_copy() const {
    return std::shared_ptr<Content>(new ListArrayOf<T>(id_, starts_, stops_, content_));
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += starts_.length();
    }
    if (regular_at < 0  ||  regular_at >= starts_.length()) {
      util::handle_error(failure(kSliceNone, at, "index out of range"), classname(), id_.get());
    }
    if (regular_at >= stops_.length()) {
      throw std::invalid_argument(std::string("in ") + classname() + std::string(", len(stops) < len(starts)"));
    }
    return content_.get()->getitem_range(starts_.getitem_at(regular_at), stops_.getitem_at(regular_at));
  }

  template <typename T>
  const std::shared_ptr<Content> ListArrayOf<T>::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), starts_.length());
    if (regular_stop > stops_.length()) {
      throw std::invalid_argument(std::string("in ") + classname() + std::string(", len(stops) < len(starts)"));
    }

    std::shared_ptr<Identity> id(nullptr);
    if (id_.get() != nullptr) {
      if (regular_stop > id_.get()->length()) {
        util::handle_error(failure(kSliceNone, stop, "index out of range"), id_.get()->classname(), nullptr);
      }
      id = id_.get()->getitem_range(regular_start, regular_stop);
    }

    return std::shared_ptr<Content>(new ListArrayOf<T>(id, starts_.getitem_range(regular_start, regular_stop), stops_.getitem_range(regular_start, regular_stop), content_));
  }

  template <>
  const std::shared_ptr<Content> ListArrayOf<int32_t>::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      throw std::invalid_argument(std::string("in ") + classname() + std::string(", len(stops) < len(starts)"));
    }

    if (head.get() == nullptr) {
      return shallow_copy();
    }

    else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
      assert(advanced.length() == 0);
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      Index64 nextcarry(lenstarts);
      Error err = awkward_listarray32_getitem_next_at_64(
        nextcarry.ptr().get(),
        starts_.ptr().get(),
        stops_.ptr().get(),
        lenstarts,
        starts_.offset(),
        stops_.offset(),
        at->at());
      std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
      return nextcontent.get()->getitem_next(nexthead, nexttail, advanced);
    }

    else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
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
        starts_.ptr().get(),
        stops_.ptr().get(),
        lenstarts,
        starts_.offset(),
        stops_.offset(),
        start,
        stop,
        step);

      Index32 nextoffsets(lenstarts + 1);
      Index64 nextcarry(carrylength);

      awkward_listarray32_getitem_next_range_64(
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
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      Index64 flathead = array->ravel();
      if (advanced.length() == 0) {
        Index64 nextcarry(lenstarts*flathead.length());
        Index64 nextadvanced(lenstarts*flathead.length());
        Index32 nextoffsets(lenstarts + 1);
        Error err = awkward_listarray32_getitem_next_array_64(
          nextoffsets.ptr().get(),
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
        util::handle_error(err, classname(), id_.get());
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
          starts_.ptr().get(),
          stops_.ptr().get(),
          flathead.ptr().get(),
          advanced.ptr().get(),
          starts_.offset(),
          stops_.offset(),
          lenstarts,
          flathead.length(),
          content_.get()->length());
        util::handle_error(err, classname(), id_.get());
        std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
        return nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced);
      }
    }

    else {
      throw std::runtime_error("unrecognized slice item type");
    }
  }

  template <>
  const std::shared_ptr<Content> ListArrayOf<int64_t>::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      throw std::invalid_argument(std::string("in ") + classname() + std::string(", len(stops) < len(starts)"));
    }

    if (head.get() == nullptr) {
      return shallow_copy();
    }

    else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
      assert(advanced.length() == 0);
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      Index64 nextcarry(lenstarts);
      Error err = awkward_listarray64_getitem_next_at_64(
        nextcarry.ptr().get(),
        starts_.ptr().get(),
        stops_.ptr().get(),
        lenstarts,
        starts_.offset(),
        stops_.offset(),
        at->at());
      std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
      return nextcontent.get()->getitem_next(nexthead, nexttail, advanced);
    }

    else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
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
        starts_.ptr().get(),
        stops_.ptr().get(),
        lenstarts,
        starts_.offset(),
        stops_.offset(),
        start,
        stop,
        step);

      Index64 nextoffsets(lenstarts + 1);
      Index64 nextcarry(carrylength);

      awkward_listarray64_getitem_next_range_64(
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
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      Index64 flathead = array->ravel();
      if (advanced.length() == 0) {
        Index64 nextcarry(lenstarts*flathead.length());
        Index64 nextadvanced(lenstarts*flathead.length());
        Index64 nextoffsets(lenstarts + 1);
        Error err = awkward_listarray64_getitem_next_array_64(
          nextoffsets.ptr().get(),
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
        util::handle_error(err, classname(), id_.get());
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
          starts_.ptr().get(),
          stops_.ptr().get(),
          flathead.ptr().get(),
          advanced.ptr().get(),
          starts_.offset(),
          stops_.offset(),
          lenstarts,
          flathead.length(),
          content_.get()->length());
        util::handle_error(err, classname(), id_.get());
        std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
        return nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced);
      }
    }

    else {
      throw std::runtime_error("unrecognized slice item type");
    }
  }

  template <>
  const std::shared_ptr<Content> ListArrayOf<int32_t>::carry(const Index64& carry) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      throw std::invalid_argument(std::string("in ") + classname() + std::string(", len(stops) < len(starts)"));
    }
    Index32 nextstarts(carry.length());
    Index32 nextstops(carry.length());
    Error err = awkward_listarray32_getitem_carry_64(
      nextstarts.ptr().get(),
      nextstops.ptr().get(),
      starts_.ptr().get(),
      stops_.ptr().get(),
      carry.ptr().get(),
      starts_.offset(),
      stops_.offset(),
      lenstarts,
      carry.length());
    util::handle_error(err, classname(), id_.get());
    std::shared_ptr<Identity> id(nullptr);
    if (id_.get() != nullptr) {
      id = id_.get()->getitem_carry_64(carry);
    }
    return std::shared_ptr<Content>(new ListArrayOf<int32_t>(id, nextstarts, nextstops, content_));
  }

  template <>
  const std::shared_ptr<Content> ListArrayOf<int64_t>::carry(const Index64& carry) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      throw std::invalid_argument(std::string("in ") + classname() + std::string(", len(stops) < len(starts)"));
    }
    Index64 nextstarts(carry.length());
    Index64 nextstops(carry.length());
    Error err = awkward_listarray64_getitem_carry_64(
      nextstarts.ptr().get(),
      nextstops.ptr().get(),
      starts_.ptr().get(),
      stops_.ptr().get(),
      carry.ptr().get(),
      starts_.offset(),
      stops_.offset(),
      lenstarts,
      carry.length());
    util::handle_error(err, classname(), id_.get());
    std::shared_ptr<Identity> id(nullptr);
    if (id_.get() != nullptr) {
      id = id_.get()->getitem_carry_64(carry);
    }
    return std::shared_ptr<Content>(new ListArrayOf<int64_t>(id, nextstarts, nextstops, content_));
  }

  template <typename T>
  const std::pair<int64_t, int64_t> ListArrayOf<T>::minmax_depth() const {
    std::pair<int64_t, int64_t> content_depth = content_.get()->minmax_depth();
    return std::pair<int64_t, int64_t>(content_depth.first + 1, content_depth.second + 1);
  }

  template class ListArrayOf<int32_t>;
  template class ListArrayOf<int64_t>;
}
