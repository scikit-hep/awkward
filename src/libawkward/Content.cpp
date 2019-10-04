// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/ListArray.h"

#include "awkward/Content.h"

namespace awkward {
  const std::string Content::tostring() const {
    return tostring_part("", "", "");
  }

  const std::shared_ptr<Content> Content::getitem(const Slice& where) const {
    Index64 nextstarts(1);
    Index64 nextstops(1);
    *nextstarts.ptr().get() = 0;
    *nextstops.ptr().get() = length();
    ListArrayOf<int64_t> next(std::shared_ptr<Identity>(nullptr), nextstarts, nextstops, shallow_copy());

    std::shared_ptr<SliceItem> nexthead = where.head();
    Slice nexttail = where.tail();
    Index64 nextadvanced(0);
    std::shared_ptr<Content> out = next.getitem_next(nexthead, nexttail, nextadvanced);
    return out.get()->getitem_at(0);
  }

  const std::shared_ptr<Content> Content::getitem_ellipsis(const Slice& tail, const Index64& advanced) const {
    std::pair<int64_t, int64_t> minmax = minmax_depth();
    int64_t mindepth = minmax.first;
    int64_t maxdepth = minmax.second;

    if (tail.length() == 0  ||  (mindepth - 1 == tail.dimlength()  &&  maxdepth - 1 == tail.dimlength())) {
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      return getitem_next(nexthead, nexttail, advanced);
    }
    else if (mindepth - 1 == tail.dimlength()  ||  maxdepth - 1 == tail.dimlength()) {
      throw std::invalid_argument("ellipsis (...) can't be used on a data structure of different depths");
    }
    else {
      std::vector<std::shared_ptr<SliceItem>> tailitems = tail.items();
      std::vector<std::shared_ptr<SliceItem>> items = { std::shared_ptr<SliceItem>(new SliceEllipsis()) };
      items.insert(items.end(), tailitems.begin(), tailitems.end());
      std::shared_ptr<SliceItem> nexthead(new SliceRange(Slice::none(), Slice::none(), 1));
      Slice nexttail(items);
      return getitem_next(nexthead, nexttail, advanced);
    }
  }

  const std::shared_ptr<Content> Content::getitem_newaxis(const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: insert a RegularArray of 1 here");
  }
}
