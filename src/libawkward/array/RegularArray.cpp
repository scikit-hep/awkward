// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/operations.h"
#include "awkward/type/RegularType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ListOffsetArray.h"

#include "awkward/array/RegularArray.h"

namespace awkward {
  RegularArray::RegularArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const std::shared_ptr<Content>& content, int64_t size)
      : Content(identities, parameters)
      , content_(content)
      , size_(size) { }

  const std::shared_ptr<Content> RegularArray::content() const {
    return content_;
  }

  int64_t RegularArray::size() const {
    return size_;
  }

  Index64 RegularArray::compact_offsets64() const {
    int64_t len = length();
    Index64 out(len + 1);
    struct Error err = awkward_regulararray_compact_offsets64(
      out.ptr().get(),
      len,
      size_);
    util::handle_error(err, classname(), identities_.get());
    return out;
  }

  const std::shared_ptr<Content> RegularArray::broadcast_tooffsets64(const Index64& offsets) const {
    if (offsets.length() == 0  ||  offsets.getitem_at_nowrap(0) != 0) {
      throw std::invalid_argument("broadcast_tooffsets64 can only be used with offsets that start at 0");
    }

    int64_t len = length();
    if (offsets.length() - 1 != len) {
      throw std::invalid_argument(std::string("cannot broadcast RegularArray of length ") + std::to_string(len) + (" to length ") + std::to_string(offsets.length() - 1));
    }

    std::shared_ptr<Identities> identities;
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(0, offsets.length() - 1);
    }

    if (size_ == 1) {
      int64_t carrylen = offsets.getitem_at_nowrap(offsets.length() - 1);
      Index64 nextcarry(carrylen);
      struct Error err = awkward_regulararray_broadcast_tooffsets64_size1(
        nextcarry.ptr().get(),
        offsets.ptr().get(),
        offsets.offset(),
        offsets.length());
      util::handle_error(err, classname(), identities_.get());
      std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
      return std::make_shared<ListOffsetArray64>(identities, parameters_, offsets, nextcontent);
    }
    else {
      struct Error err = awkward_regulararray_broadcast_tooffsets64(
        offsets.ptr().get(),
        offsets.offset(),
        offsets.length(),
        size_);
      util::handle_error(err, classname(), identities_.get());
      return std::make_shared<ListOffsetArray64>(identities, parameters_, offsets, content_);
    }
  }

  const std::string RegularArray::classname() const {
    return "RegularArray";
  }

  void RegularArray::setidentities(const std::shared_ptr<Identities>& identities) {
    if (identities.get() == nullptr) {
      content_.get()->setidentities(identities);
    }
    else {
      if (length() != identities.get()->length()) {
        util::handle_error(failure("content and its identities must have the same length", kSliceNone, kSliceNone), classname(), identities_.get());
      }
      std::shared_ptr<Identities> bigidentities = identities;
      if (content_.get()->length() > kMaxInt32) {
        bigidentities = identities.get()->to64();
      }
      if (Identities32* rawidentities = dynamic_cast<Identities32*>(bigidentities.get())) {
        std::shared_ptr<Identities> subidentities = std::make_shared<Identities32>(Identities::newref(), rawidentities->fieldloc(), rawidentities->width() + 1, content_.get()->length());
        Identities32* rawsubidentities = reinterpret_cast<Identities32*>(subidentities.get());
        struct Error err = awkward_identities32_from_regulararray(
          rawsubidentities->ptr().get(),
          rawidentities->ptr().get(),
          rawidentities->offset(),
          size_,
          content_.get()->length(),
          length(),
          rawidentities->width());
        util::handle_error(err, classname(), identities_.get());
        content_.get()->setidentities(subidentities);
      }
      else if (Identities64* rawidentities = dynamic_cast<Identities64*>(bigidentities.get())) {
        std::shared_ptr<Identities> subidentities = std::make_shared<Identities64>(Identities::newref(), rawidentities->fieldloc(), rawidentities->width() + 1, content_.get()->length());
        Identities64* rawsubidentities = reinterpret_cast<Identities64*>(subidentities.get());
        struct Error err = awkward_identities64_from_regulararray(
          rawsubidentities->ptr().get(),
          rawidentities->ptr().get(),
          rawidentities->offset(),
          size_,
          content_.get()->length(),
          length(),
          rawidentities->width());
        util::handle_error(err, classname(), identities_.get());
        content_.get()->setidentities(subidentities);
      }
      else {
        throw std::runtime_error("unrecognized Identities specialization");
      }
    }
    identities_ = identities;
  }

  void RegularArray::setidentities() {
    if (length() < kMaxInt32) {
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

  const std::shared_ptr<Type> RegularArray::type() const {
    return std::make_shared<RegularType>(parameters_, content_.get()->type(), size_);
  }

  const std::shared_ptr<Content> RegularArray::astype(const std::shared_ptr<Type>& type) const {
    if (RegularType* raw = dynamic_cast<RegularType*>(type.get())) {
      if (raw->size() == size_) {
        return std::make_shared<RegularArray>(identities_, type.get()->parameters(), content_.get()->astype(raw->type()), size_);
      }
      else {
        throw std::invalid_argument(classname() + std::string(" cannot be converted to type ") + type.get()->tostring() + std::string(" because sizes do not match"));
      }
    }
    else {
      throw std::invalid_argument(classname() + std::string(" cannot be converted to type ") + type.get()->tostring());
    }
  }

  const std::string RegularArray::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << " size=\"" << size_ << "\">\n";
    if (identities_.get() != nullptr) {
      out << identities_.get()->tostring_part(indent + std::string("    "), "", "\n");
    }
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    out << content_.get()->tostring_part(indent + std::string("    "), "<content>", "</content>\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  void RegularArray::tojson_part(ToJson& builder) const {
    int64_t len = length();
    check_for_iteration();
    builder.beginlist();
    for (int64_t i = 0;  i < len;  i++) {
      getitem_at_nowrap(i).get()->tojson_part(builder);
    }
    builder.endlist();
  }

  void RegularArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    content_.get()->nbytes_part(largest);
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  int64_t RegularArray::length() const {
    return size_ == 0 ? 0 : content_.get()->length() / size_;   // floor of length / size
  }

  const std::shared_ptr<Content> RegularArray::shallow_copy() const {
    return std::make_shared<RegularArray>(identities_, parameters_, content_, size_);
  }

  const std::shared_ptr<Content> RegularArray::deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const {
    std::shared_ptr<Content> content = content_.get()->deep_copy(copyarrays, copyindexes, copyidentities);
    std::shared_ptr<Identities> identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<RegularArray>(identities, parameters_, content, size_);
  }

  void RegularArray::check_for_iteration() const {
    if (identities_.get() != nullptr  && identities_.get()->length() < length()) {
      util::handle_error(failure("len(identities) < len(array)", kSliceNone, kSliceNone), identities_.get()->classname(), nullptr);
    }
  }

  const std::shared_ptr<Content> RegularArray::getitem_nothing() const {
    return content_.get()->getitem_range_nowrap(0, 0);
  }

  const std::shared_ptr<Content> RegularArray::getitem_at(int64_t at) const {
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

  const std::shared_ptr<Content> RegularArray::getitem_at_nowrap(int64_t at) const {
    return content_.get()->getitem_range_nowrap(at*size_, (at + 1)*size_);
  }

  const std::shared_ptr<Content> RegularArray::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), length());
    if (identities_.get() != nullptr  &&  regular_stop > identities_.get()->length()) {
      util::handle_error(failure("index out of range", kSliceNone, stop), identities_.get()->classname(), nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  const std::shared_ptr<Content> RegularArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    std::shared_ptr<Identities> identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    return std::make_shared<RegularArray>(identities_, parameters_, content_.get()->getitem_range_nowrap(start*size_, stop*size_), size_);
  }

  const std::shared_ptr<Content> RegularArray::getitem_field(const std::string& key) const {
    return std::make_shared<RegularArray>(identities_, util::Parameters(), content_.get()->getitem_field(key), size_);
  }

  const std::shared_ptr<Content> RegularArray::getitem_fields(const std::vector<std::string>& keys) const {
    return std::make_shared<RegularArray>(identities_, util::Parameters(), content_.get()->getitem_fields(keys), size_);
  }

  const std::shared_ptr<Content> RegularArray::carry(const Index64& carry) const {
    Index64 nextcarry(carry.length()*size_);

    struct Error err = awkward_regulararray_getitem_carry_64(
      nextcarry.ptr().get(),
      carry.ptr().get(),
      carry.length(),
      size_);
    util::handle_error(err, classname(), identities_.get());

    std::shared_ptr<Identities> identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }
    return std::make_shared<RegularArray>(identities, parameters_, content_.get()->carry(nextcarry), size_);
  }

  bool RegularArray::purelist_isregular() const {
    return content_.get()->purelist_isregular();
  }

  int64_t RegularArray::purelist_depth() const {
    return content_.get()->purelist_depth() + 1;
  }

  const std::pair<int64_t, int64_t> RegularArray::minmax_depth() const {
    std::pair<int64_t, int64_t> content_depth = content_.get()->minmax_depth();
    return std::pair<int64_t, int64_t>(content_depth.first + 1, content_depth.second + 1);
  }

  int64_t RegularArray::numfields() const {
    return content_.get()->numfields();
  }

  int64_t RegularArray::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string RegularArray::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool RegularArray::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string> RegularArray::keys() const {
    return content_.get()->keys();
  }

  const Index64 RegularArray::count64() const {
    int64_t len = length();
    Index64 tocount(len);
    struct Error err = awkward_regulararray_count(
      tocount.ptr().get(),
      size_,
      len);
    util::handle_error(err, classname(), identities_.get());
    return tocount;
  }

  const std::shared_ptr<Content> RegularArray::count(int64_t axis) const {
    if (axis != 0) {
      throw std::runtime_error("FIXME: RegularArray::count(axis != 0)");
    }
    Index64 tocount = count64();
    std::vector<ssize_t> shape({ (ssize_t)tocount.length() });
    std::vector<ssize_t> strides({ (ssize_t)sizeof(int64_t) });
#ifdef _MSC_VER
    std::string format = "q";
#else
    std::string format = "l";
#endif
    return std::make_shared<NumpyArray>(Identities::none(), util::Parameters(), tocount.ptr(), shape, strides, 0, sizeof(int64_t), format);
  }

  const std::shared_ptr<Content> RegularArray::flatten(int64_t axis) const {
    if (axis < 0) {
      std::pair<int64_t, int64_t> minmax = minmax_depth();
      int64_t mindepth = minmax.first;
      int64_t maxdepth = minmax.second;
      int64_t depth = purelist_depth();
      if (mindepth == depth  &&  maxdepth == depth) {
        if (depth - 1 + axis < 0) {
          throw std::invalid_argument(std::string("RegularArray cannot be flattened in axis ") + std::to_string(axis) + std::string(" because its depth is ") + std::to_string(depth));
        }
        return flatten(depth - 1 + axis);
      }
      else {
        return content_.get()->flatten(axis);
      }
    }
    else if (axis == 0) {
      if (content_.get()->length() % size_ != 0) {
        return content_.get()->getitem_range_nowrap(0, length()*size_);
      }
      else {
        return content_;
      }
    }
    else {
      Index64 count = count64();
      Index64 ccount = content_.get()->count64();
      Index64 offsets(length() + 1);
      offsets.ptr().get()[0] = 0;
      for (ssize_t i = 0; i < length(); i++) {
        int64_t l = 0;
        for (int64_t j = 0; j < count.ptr().get()[i]; j++) {
          l += ccount.ptr().get()[j + i*size_];
        }
        offsets.ptr().get()[i + 1] = l + offsets.ptr().get()[i];
      }

      std::shared_ptr<Content> nextcontent = content_.get()->flatten(axis - 1);

      return std::make_shared<ListOffsetArray64>(identities_, parameters_, offsets, nextcontent);
    }
  }

  const std::shared_ptr<Content> RegularArray::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    assert(advanced.length() == 0);

    int64_t len = length();
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    Index64 nextcarry(len);

    struct Error err = awkward_regulararray_getitem_next_at_64(
      nextcarry.ptr().get(),
      at.at(),
      len,
      size_);
    util::handle_error(err, classname(), identities_.get());

    std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
    return nextcontent.get()->getitem_next(nexthead, nexttail, advanced);
  }

  const std::shared_ptr<Content> RegularArray::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    int64_t len = length();
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();

    assert(range.step() != 0);
    int64_t regular_start = range.start();
    int64_t regular_stop = range.stop();
    int64_t regular_step = std::abs(range.step());
    awkward_regularize_rangeslice(&regular_start, &regular_stop, range.step() > 0, range.start() != Slice::none(), range.stop() != Slice::none(), size_);
    int64_t nextsize = 0;
    if (range.step() > 0  &&  regular_stop - regular_start > 0) {
      int64_t diff = regular_stop - regular_start;
      nextsize = diff / regular_step;
      if (diff % regular_step != 0) {
        nextsize++;
      }
    }
    else if (range.step() < 0  &&  regular_stop - regular_start < 0) {
      int64_t diff = regular_start - regular_stop;
      nextsize = diff / regular_step;
      if (diff % regular_step != 0) {
        nextsize++;
      }
    }

    Index64 nextcarry(len*nextsize);

    struct Error err = awkward_regulararray_getitem_next_range_64(
      nextcarry.ptr().get(),
      regular_start,
      range.step(),
      len,
      size_,
      nextsize);
    util::handle_error(err, classname(), identities_.get());

    std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);

    if (advanced.length() == 0) {
      return std::make_shared<RegularArray>(identities_, parameters_, nextcontent.get()->getitem_next(nexthead, nexttail, advanced), nextsize);
    }
    else {
      Index64 nextadvanced(len*nextsize);

      struct Error err = awkward_regulararray_getitem_next_range_spreadadvanced_64(
        nextadvanced.ptr().get(),
        advanced.ptr().get(),
        len,
        nextsize);
      util::handle_error(err, classname(), identities_.get());

      return std::make_shared<RegularArray>(identities_, parameters_, nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced), nextsize);
    }
  }

  const std::shared_ptr<Content> RegularArray::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    int64_t len = length();
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    Index64 flathead = array.ravel();
    Index64 regular_flathead(flathead.length());

    struct Error err = awkward_regulararray_getitem_next_array_regularize_64(
      regular_flathead.ptr().get(),
      flathead.ptr().get(),
      flathead.length(),
      size_);
    util::handle_error(err, classname(), identities_.get());

    if (advanced.length() == 0) {
      Index64 nextcarry(len*flathead.length());
      Index64 nextadvanced(len*flathead.length());

      struct Error err = awkward_regulararray_getitem_next_array_64(
        nextcarry.ptr().get(),
        nextadvanced.ptr().get(),
        regular_flathead.ptr().get(),
        len,
        regular_flathead.length(),
        size_);
      util::handle_error(err, classname(), identities_.get());

      std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);

      return getitem_next_array_wrap(nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced), array.shape());
    }
    else {
      Index64 nextcarry(len);
      Index64 nextadvanced(len);

      struct Error err = awkward_regulararray_getitem_next_array_advanced_64(
        nextcarry.ptr().get(),
        nextadvanced.ptr().get(),
        advanced.ptr().get(),
        regular_flathead.ptr().get(),
        len,
        regular_flathead.length(),
        size_);
      util::handle_error(err, classname(), identities_.get());

      std::shared_ptr<Content> nextcontent = content_.get()->carry(nextcarry);
      return nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced);
    }
  }

}
