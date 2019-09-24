// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/cpu-kernels/getitem.h"

#include "awkward/Slice.h"
#include "awkward/ListOffsetArray.h"

#include "awkward/ListArray.h"

using namespace awkward;

template <typename T>
void ListArrayOf<T>::setid() {
  throw std::runtime_error("FIXME");
}

template <typename T>
void ListArrayOf<T>::setid(const std::shared_ptr<Identity> id) {
  throw std::runtime_error("FIXME");
}

template <typename T>
const std::string ListArrayOf<T>::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  std::string name = "Unrecognized ListArray";
  if (std::is_same<T, int32_t>::value) {
    name = "ListArray32";
  }
  else if (std::is_same<T, int64_t>::value) {
    name = "ListArray64";
  }
  out << indent << pre << "<" << name << ">\n";
  if (id_.get() != nullptr) {
    out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
  }
  out << starts_.tostring_part(indent + std::string("    "), "<starts>", "</starts>\n");
  out << stops_.tostring_part(indent + std::string("    "), "<stops>", "</stops>\n");
  out << content_.get()->tostring_part(indent + std::string("    "), "<content>", "</content>\n");
  out << indent << "</" << name << ">" << post;
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
    throw std::invalid_argument("index out of range");
  }
  if (regular_at >= stops_.length()) {
    throw std::invalid_argument("len(stops) < len(starts) in ListArray");
  }
  return content_.get()->getitem_range(starts_.getitem_at(regular_at), stops_.getitem_at(regular_at));
}

template <typename T>
const std::shared_ptr<Content> ListArrayOf<T>::getitem_range(int64_t start, int64_t stop) const {
  int64_t regular_start = start;
  int64_t regular_stop = stop;
  awkward_regularize_rangeslice(regular_start, regular_stop, true, start != Slice::none(), stop != Slice::none(), starts_.length());
  if (regular_stop > stops_.length()) {
    throw std::invalid_argument("len(stops) < len(starts) in ListArray");
  }

  std::shared_ptr<Identity> id(nullptr);
  if (id_.get() != nullptr) {
    if (regular_stop > id_.get()->length()) {
      throw std::invalid_argument("index out of range for identity");
    }
    id = id_.get()->getitem_range(regular_start, regular_stop);
  }

  return std::shared_ptr<Content>(new ListArrayOf<T>(id, starts_.getitem_range(regular_start, regular_stop), stops_.getitem_range(regular_start, regular_stop), content_));
}

template <typename T>
const std::shared_ptr<Content> ListArrayOf<T>::getitem(const Slice& where) const {
  // FIXME: find a better way to wrap these
  Index64 nextstarts(1);
  Index64 nextstops(1);
  *nextstarts.ptr().get() = 0;
  *nextstops.ptr().get() = length();
  ListArrayOf<int64_t> next(std::shared_ptr<Identity>(nullptr), nextstarts, nextstops, shallow_copy());

  std::shared_ptr<SliceItem> nexthead = where.head();
  Slice nexttail = where.tail();
  Index64 nextadvanced(0);
  std::shared_ptr<Content> out = next.getitem_next(nexthead, nexttail, nextadvanced);

  return dynamic_cast<ListArrayOf<int64_t>*>(out.get())->content();
}

template <typename T>
const std::shared_ptr<Content> ListArrayOf<T>::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const {
  if (head.get() == nullptr) {
        throw std::runtime_error("ListArray[null]");
  }

  else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
    throw std::runtime_error("ListArray[at]");
  }

  else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
    throw std::runtime_error("ListArray[range]");
  }

  else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
    throw std::runtime_error("ListArray[ellipsis]");
  }

  else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
    throw std::runtime_error("ListArray[newaxis]");
  }

  else if (SliceArray64* array = dynamic_cast<SliceArray64*>(head.get())) {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      throw std::invalid_argument("len(stops) < len(starts)");
    }

    Index64 flathead = array->ravel();

    if (advanced.length() == 0) {
      Index64 nextcarry(lenstarts*flathead.length());
      Index64 nextadvanced(lenstarts*flathead.length());
      if (std::is_same<T, int32_t>::value) {
        Index32 nextstarts(lenstarts);
        Index32 nextstops(lenstarts);
        Error err = awkward_listarray32_getitem_next_array_64(
          nextstarts.ptr().get(),
          nextstops.ptr().get(),
          nextcarry.ptr().get(),
          nextadvanced.ptr().get(),
          reinterpret_cast<int32_t*>(starts_.ptr().get()),
          reinterpret_cast<int32_t*>(stops_.ptr().get()),
          flathead.ptr().get(),
          starts_.offset(),
          stops_.offset(),
          lenstarts,
          flathead.length(),
          content_.get()->length());
        HANDLE_ERROR(err)
        return std::shared_ptr<Content>(new ListArrayOf<int32_t>(id_, nextstarts, nextstops, content_.get()->carry(nextcarry)));
      }
      else if (std::is_same<T, int64_t>::value) {
        Index64 nextstarts(lenstarts);
        Index64 nextstops(lenstarts);
        Error err = awkward_listarray64_getitem_next_array_64(
          nextstarts.ptr().get(),
          nextstops.ptr().get(),
          nextcarry.ptr().get(),
          nextadvanced.ptr().get(),
          reinterpret_cast<int64_t*>(starts_.ptr().get()),
          reinterpret_cast<int64_t*>(stops_.ptr().get()),
          flathead.ptr().get(),
          starts_.offset(),
          stops_.offset(),
          lenstarts,
          flathead.length(),
          content_.get()->length());
        HANDLE_ERROR(err)
        return std::shared_ptr<Content>(new ListArrayOf<int64_t>(id_, nextstarts, nextstops, content_.get()->carry(nextcarry)));
      }
      else {
        throw std::runtime_error("unrecognized ListArray specialization");
      }
    }
    else {
      throw std::runtime_error("FIXME");
    }
  }

  else {
    throw std::runtime_error("unrecognized slice item type");
  }
}

template <typename T>
const std::shared_ptr<Content> ListArrayOf<T>::carry(const Index64& carry) const {
  if (stops_.length() < starts_.length()) {
    throw std::invalid_argument("len(stops) < len(starts)");
  }
  if (std::is_same<T, int32_t>::value) {
    Index32 nextstarts(carry.length());
    Index32 nextstops(carry.length());
    awkward_listarray32_getitem_carry_64(
      nextstarts.ptr().get(),
      nextstops.ptr().get(),
      reinterpret_cast<int32_t*>(starts_.ptr().get()),
      reinterpret_cast<int32_t*>(stops_.ptr().get()),
      carry.ptr().get(),
      starts_.offset(),
      stops_.offset(),
      carry.length());
    std::shared_ptr<Identity> id(nullptr);
    if (id_.get() != nullptr) {
      id = id_.get()->getitem_carry_64(carry);
    }
    return std::shared_ptr<Content>(new ListArrayOf<int32_t>(id, nextstarts, nextstops, content_));
  }
  else if (std::is_same<T, int64_t>::value) {
    Index64 nextstarts(carry.length());
    Index64 nextstops(carry.length());
    awkward_listarray64_getitem_carry_64(
      nextstarts.ptr().get(),
      nextstops.ptr().get(),
      reinterpret_cast<int64_t*>(starts_.ptr().get()),
      reinterpret_cast<int64_t*>(stops_.ptr().get()),
      carry.ptr().get(),
      starts_.offset(),
      stops_.offset(),
      carry.length());
    std::shared_ptr<Identity> id(nullptr);
    if (id_.get() != nullptr) {
      id = id_.get()->getitem_carry_64(carry);
    }
    return std::shared_ptr<Content>(new ListArrayOf<int64_t>(id, nextstarts, nextstops, content_));
  }
  else {
    throw std::runtime_error("unrecognized ListArray specialization");
  }
}

template <typename T>
const std::pair<int64_t, int64_t> ListArrayOf<T>::minmax_depth() const {
  std::pair<int64_t, int64_t> content_depth = content_.get()->minmax_depth();
  return std::pair<int64_t, int64_t>(content_depth.first + 1, content_depth.second + 1);
}

namespace awkward {
  template class ListArrayOf<int32_t>;
  template class ListArrayOf<int64_t>;
}
