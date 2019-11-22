// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/array/RegularArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/type/ArrayType.h"

#include "awkward/Content.h"

namespace awkward {
  bool Content::isscalar() const {
    return false;
  }

  const ArrayType Content::type() const {
    return ArrayType(type_part(), length());
  }

  const std::string Content::tostring() const {
    return tostring_part("", "", "");
  }

  const std::string Content::tojson(bool pretty, int64_t maxdecimals) const {
    if (pretty) {
      ToJsonPrettyString builder(maxdecimals);
      tojson_part(builder);
      return builder.tostring();
    }
    else {
      ToJsonString builder(maxdecimals);
      tojson_part(builder);
      return builder.tostring();
    }
  }

  void Content::tojson(FILE* destination, bool pretty, int64_t maxdecimals, int64_t buffersize) const {
    if (pretty) {
      ToJsonPrettyFile builder(destination, maxdecimals, buffersize);
      builder.beginlist();
      tojson_part(builder);
      builder.endlist();
    }
    else {
      ToJsonFile builder(destination, maxdecimals, buffersize);
      builder.beginlist();
      tojson_part(builder);
      builder.endlist();
    }
  }

  const std::shared_ptr<Content> Content::getitem(const Slice& where) const {
    std::shared_ptr<Content> next(new RegularArray(Identity::none(), shallow_copy(), length()));

    std::shared_ptr<SliceItem> nexthead = where.head();
    Slice nexttail = where.tail();
    Index64 nextadvanced(0);
    std::shared_ptr<Content> out = next.get()->getitem_next(nexthead, nexttail, nextadvanced);

    if (out.get()->length() == 0) {
      return out.get()->getitem_nothing();
    }
    else {
      return out.get()->getitem_at_nowrap(0);
    }
  }

  const std::shared_ptr<Content> Content::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const {
    if (head.get() == nullptr) {
      return shallow_copy();
    }
    else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
      return getitem_next(*at, tail, advanced);
    }
    else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
      return getitem_next(*range, tail, advanced);
    }
    else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
      return getitem_next(*ellipsis, tail, advanced);
    }
    else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
      return getitem_next(*newaxis, tail, advanced);
    }
    else if (SliceArray64* array = dynamic_cast<SliceArray64*>(head.get())) {
      return getitem_next(*array, tail, advanced);
    }
    else {
      throw std::runtime_error("unrecognized slice type");
    }
  }

  const std::shared_ptr<Content> Content::getitem_next(const SliceEllipsis& ellipsis, const Slice& tail, const Index64& advanced) const {
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

  const std::shared_ptr<Content> Content::getitem_next(const SliceNewAxis& newaxis, const Slice& tail, const Index64& advanced) const {
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    return std::shared_ptr<Content>(new RegularArray(Identity::none(), getitem_next(nexthead, nexttail, advanced), 1));
  }

  const std::shared_ptr<Content> Content::getitem_next_array_wrap(const std::shared_ptr<Content> outcontent, const std::vector<int64_t>& shape) const {
    std::shared_ptr<Content> out(new RegularArray(Identity::none(), outcontent, (int64_t)shape[shape.size() - 1]));
    for (int64_t i = (int64_t)shape.size() - 2;  i >= 0;  i--) {
      out = std::shared_ptr<Content>(new RegularArray(Identity::none(), out, (int64_t)shape[i]));
    }
    return out;
  }

}
