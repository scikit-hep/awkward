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
#include "awkward/array/UnmaskedArray.h"

#include "awkward/array/BitMaskedArray.h"

namespace awkward {
  BitMaskedArray::BitMaskedArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const IndexU8& mask, const std::shared_ptr<Content>& content, bool validwhen, int64_t length, bool lsb_order)
      : Content(identities, parameters)
      , mask_(mask)
      , content_(content)
      , validwhen_(validwhen != 0)
      , length_(length)
      , lsb_order_(lsb_order) {
    int64_t bitlength = ((length / 8) + ((length % 8) != 0));
    if (mask.length() < bitlength) {
      throw std::invalid_argument("BitMaskedArray mask must not be shorter than its ceil(length / 8.0)");
    }
    if (content.get()->length() < length) {
      throw std::invalid_argument("BitMaskedArray content must not be shorter than its length");
    }
  }

  const IndexU8 BitMaskedArray::mask() const {
    return mask_;
  }

  const std::shared_ptr<Content> BitMaskedArray::content() const {
    return content_;
  }

  bool BitMaskedArray::validwhen() const {
    return validwhen_;
  }

  bool BitMaskedArray::lsb_order() const {
    return lsb_order_;
  }

  const std::shared_ptr<Content> BitMaskedArray::project() const {
    return toByteMaskedArray().get()->project();
  }

  const std::shared_ptr<Content> BitMaskedArray::project(const Index8& mask) const {
    return toByteMaskedArray().get()->project(mask);
  }

  const Index8 BitMaskedArray::bytemask() const {
    Index8 bytemask(mask_.length() * 8);
    struct Error err = awkward_bitmaskedarray_to_bytemaskedarray(
      bytemask.ptr().get(),
      mask_.ptr().get(),
      mask_.offset(),
      mask_.length(),
      false,
      lsb_order_);
    util::handle_error(err, classname(), identities_.get());
    return bytemask.getitem_range_nowrap(0, length_);
  }

  const std::shared_ptr<Content> BitMaskedArray::simplify_optiontype() const {
    if (dynamic_cast<IndexedArray32*>(content_.get())        ||
        dynamic_cast<IndexedArrayU32*>(content_.get())       ||
        dynamic_cast<IndexedArray64*>(content_.get())        ||
        dynamic_cast<IndexedOptionArray32*>(content_.get())  ||
        dynamic_cast<IndexedOptionArray64*>(content_.get())  ||
        dynamic_cast<ByteMaskedArray*>(content_.get())       ||
        dynamic_cast<BitMaskedArray*>(content_.get())        ||
        dynamic_cast<UnmaskedArray*>(content_.get())) {
      std::shared_ptr<Content> step1 = toIndexedOptionArray64();
      IndexedOptionArray64* step2 = dynamic_cast<IndexedOptionArray64*>(step1.get());
      return step2->simplify_optiontype();
    }
  }

  const std::shared_ptr<ByteMaskedArray> BitMaskedArray::toByteMaskedArray() const {
    Index8 bytemask(mask_.length() * 8);
    struct Error err = awkward_bitmaskedarray_to_bytemaskedarray(
      bytemask.ptr().get(),
      mask_.ptr().get(),
      mask_.offset(),
      mask_.length(),
      validwhen_,
      lsb_order_);
    util::handle_error(err, classname(), identities_.get());
    return std::make_shared<ByteMaskedArray>(identities_, parameters_, bytemask.getitem_range_nowrap(0, length_), content_, validwhen_);
  }

  const std::shared_ptr<IndexedOptionArray64> BitMaskedArray::toIndexedOptionArray64() const {
    Index64 index(mask_.length() * 8);
    struct Error err = awkward_bitmaskedarray_to_indexedoptionarray_64(
      index.ptr().get(),
      mask_.ptr().get(),
      mask_.offset(),
      mask_.length(),
      validwhen_,
      lsb_order_);
    util::handle_error(err, classname(), identities_.get());
    return std::make_shared<IndexedOptionArray64>(identities_, parameters_, index.getitem_range_nowrap(0, length_), content_);
  }

  const std::string BitMaskedArray::classname() const {
    return "BitMaskedArray";
  }

  void BitMaskedArray::setidentities(const std::shared_ptr<Identities>& identities) {
    if (identities.get() == nullptr) {
      content_.get()->setidentities(identities);
    }
    else {
      if (length() != identities.get()->length()) {
        util::handle_error(failure("content and its identities must have the same length", kSliceNone, kSliceNone), classname(), identities_.get());
      }
      if (Identities32* rawidentities = dynamic_cast<Identities32*>(identities.get())) {
        std::shared_ptr<Identities32> subidentities = std::make_shared<Identities32>(Identities::newref(), rawidentities->fieldloc(), rawidentities->width(), content_.get()->length());
        Identities32* rawsubidentities = reinterpret_cast<Identities32*>(subidentities.get());
        struct Error err = awkward_identities32_extend(
          rawsubidentities->ptr().get(),
          rawidentities->ptr().get(),
          rawidentities->offset(),
          rawidentities->length(),
          content_.get()->length());
        util::handle_error(err, classname(), identities_.get());
        content_.get()->setidentities(subidentities);
      }
      else if (Identities64* rawidentities = dynamic_cast<Identities64*>(identities.get())) {
        std::shared_ptr<Identities64> subidentities = std::make_shared<Identities64>(Identities::newref(), rawidentities->fieldloc(), rawidentities->width(), content_.get()->length());
        Identities64* rawsubidentities = reinterpret_cast<Identities64*>(subidentities.get());
        struct Error err = awkward_identities64_extend(
          rawsubidentities->ptr().get(),
          rawidentities->ptr().get(),
          rawidentities->offset(),
          rawidentities->length(),
          content_.get()->length());
        util::handle_error(err, classname(), identities_.get());
        content_.get()->setidentities(subidentities);
      }
      else {
        throw std::runtime_error("unrecognized Identities specialization");
      }
    }
    identities_ = identities;
  }

  void BitMaskedArray::setidentities() {
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

  const std::shared_ptr<Type> BitMaskedArray::type(const std::map<std::string, std::string>& typestrs) const {
    return std::make_shared<OptionType>(parameters_, util::gettypestr(parameters_, typestrs), content_.get()->type(typestrs));
  }

  const std::string BitMaskedArray::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << " validwhen=\"" << (validwhen_ ? "true" : "false") << "\" length=\"" << length_ << "\" lsb_order=\"" << (lsb_order_ ? "true" : "false") << "\">\n";
    if (identities_.get() != nullptr) {
      out << identities_.get()->tostring_part(indent + std::string("    "), "", "\n");
    }
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    out << mask_.tostring_part(indent + std::string("    "), "<mask>", "</mask>\n");
    out << content_.get()->tostring_part(indent + std::string("    "), "<content>", "</content>\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  void BitMaskedArray::tojson_part(ToJson& builder) const {
    int64_t len = length();
    check_for_iteration();
    builder.beginlist();
    for (int64_t i = 0;  i < len;  i++) {
      getitem_at_nowrap(i).get()->tojson_part(builder);
    }
    builder.endlist();
  }

  void BitMaskedArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    mask_.nbytes_part(largest);
    content_.get()->nbytes_part(largest);
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  int64_t BitMaskedArray::length() const {
    return length_;
  }

  const std::shared_ptr<Content> BitMaskedArray::shallow_copy() const {
    return std::make_shared<BitMaskedArray>(identities_, parameters_, mask_, content_, validwhen_, length_, lsb_order_);
  }

  const std::shared_ptr<Content> BitMaskedArray::deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const {
    IndexU8 mask = copyindexes ? mask_.deep_copy() : mask_;
    std::shared_ptr<Content> content = content_.get()->deep_copy(copyarrays, copyindexes, copyidentities);
    std::shared_ptr<Identities> identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<BitMaskedArray>(identities, parameters_, mask, content, validwhen_, length_, lsb_order_);
  }

  void BitMaskedArray::check_for_iteration() const {
    if (identities_.get() != nullptr  &&  identities_.get()->length() < length()) {
      util::handle_error(failure("len(identities) < len(array)", kSliceNone, kSliceNone), identities_.get()->classname(), nullptr);
    }
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_nothing() const {
    return content_.get()->getitem_range_nowrap(0, 0);
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length();
    }
    if (!(0 <= regular_at  &&  regular_at < length())) {
      util::handle_error(failure("index out of range", kSliceNone, at), classname(), identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_at_nowrap(int64_t at) const {
    int64_t bitat = at / 8;
    int64_t shift = at % 8;
    uint8_t byte = mask_.getitem_at_nowrap(bitat);
    uint8_t asbool = (lsb_order_ ? ((byte >> ((uint8_t)shift)) & ((uint8_t)1)) : ((byte << ((uint8_t)shift)) & ((uint8_t)128)));
    if ((asbool != 0) == validwhen_) {
      return content_.get()->getitem_at_nowrap(at);
    }
    else {
      return none;
    }
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), length());
    if (identities_.get() != nullptr  &&  regular_stop > identities_.get()->length()) {
      util::handle_error(failure("index out of range", kSliceNone, stop), identities_.get()->classname(), nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    int64_t bitstart = start / 8;
    int64_t remainder = start % 8;
    if (remainder == 0) {
      std::shared_ptr<Identities> identities(nullptr);
      if (identities_.get() != nullptr) {
        identities = identities_.get()->getitem_range_nowrap(start, stop);
      }
      int64_t length = stop - start;
      int64_t bitlength = length / 8;
      int64_t remainder = length % 8;
      int64_t bitstop = bitstart + (bitlength + (remainder != 0));
      return std::make_shared<BitMaskedArray>(identities, parameters_, mask_.getitem_range_nowrap(bitstart, bitstop), content_.get()->getitem_range_nowrap(start, stop), validwhen_, length, lsb_order_);
    }
    else {
      return toByteMaskedArray().get()->getitem_range_nowrap(start, stop);
    }
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_field(const std::string& key) const {
    return std::make_shared<BitMaskedArray>(identities_, util::Parameters(), mask_, content_.get()->getitem_field(key), validwhen_, length_, lsb_order_);
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_fields(const std::vector<std::string>& keys) const {
    return std::make_shared<BitMaskedArray>(identities_, util::Parameters(), mask_, content_.get()->getitem_fields(keys), validwhen_, length_, lsb_order_);
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const {
    return toByteMaskedArray().get()->getitem_next(head, tail, advanced);
  }

  const std::shared_ptr<Content> BitMaskedArray::carry(const Index64& carry) const {
    return toByteMaskedArray().get()->carry(carry);
  }

  const std::string BitMaskedArray::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  bool BitMaskedArray::purelist_isregular() const {
    return content_.get()->purelist_isregular();
  }

  int64_t BitMaskedArray::purelist_depth() const {
    return content_.get()->purelist_depth();
  }

  const std::pair<int64_t, int64_t> BitMaskedArray::minmax_depth() const {
    return content_.get()->minmax_depth();
  }

  const std::pair<bool, int64_t> BitMaskedArray::branch_depth() const {
    return content_.get()->branch_depth();
  }

  int64_t BitMaskedArray::numfields() const {
    return content_.get()->numfields();
  }

  int64_t BitMaskedArray::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string BitMaskedArray::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool BitMaskedArray::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string> BitMaskedArray::keys() const {
    return content_.get()->keys();
  }

  const std::string BitMaskedArray::validityerror(const std::string& path) const {
    return content_.get()->validityerror(path + std::string(".content"));
  }

  const std::shared_ptr<Content> BitMaskedArray::shallow_simplify() const {
    return simplify_optiontype();
  }

  const std::shared_ptr<Content> BitMaskedArray::num(int64_t axis, int64_t depth) const {
    return toByteMaskedArray().get()->num(axis, depth);
  }

  const std::pair<Index64, std::shared_ptr<Content>> BitMaskedArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    return toByteMaskedArray().get()->offsets_and_flattened(axis, depth);
  }

  bool BitMaskedArray::mergeable(const std::shared_ptr<Content>& other, bool mergebool) const {
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
    else if (ByteMaskedArray* rawother = dynamic_cast<ByteMaskedArray*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (BitMaskedArray* rawother = dynamic_cast<BitMaskedArray*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (UnmaskedArray* rawother = dynamic_cast<UnmaskedArray*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else {
      return content_.get()->mergeable(other, mergebool);
    }
  }

  const std::shared_ptr<Content> BitMaskedArray::reverse_merge(const std::shared_ptr<Content>& other) const {
    std::shared_ptr<Content> indexedoptionarray = toIndexedOptionArray64();
    IndexedOptionArray64* raw = dynamic_cast<IndexedOptionArray64*>(indexedoptionarray.get());
    return raw->reverse_merge(other);
  }

  const std::shared_ptr<Content> BitMaskedArray::merge(const std::shared_ptr<Content>& other) const {
    return toIndexedOptionArray64().get()->merge(other);
  }

  const std::shared_ptr<SliceItem> BitMaskedArray::asslice() const {
    return toIndexedOptionArray64().get()->asslice();
  }

  const std::shared_ptr<Content> BitMaskedArray::fillna(const std::shared_ptr<Content>& value) const {
    return toIndexedOptionArray64().get()->fillna(value);
  }

  const std::shared_ptr<Content> BitMaskedArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    return toByteMaskedArray().get()->rpad(target, axis, depth);
  }

  const std::shared_ptr<Content> BitMaskedArray::rpad_and_clip(int64_t target, int64_t axis, int64_t depth) const {
    return toByteMaskedArray().get()->rpad_and_clip(target, axis, depth);
  }

  const std::shared_ptr<Content> BitMaskedArray::reduce_next(const Reducer& reducer, int64_t negaxis, const Index64& starts, const Index64& parents, int64_t outlength, bool mask, bool keepdims) const {
    return toByteMaskedArray().get()->reduce_next(reducer, negaxis, starts, parents, outlength, mask, keepdims);
  }

  const std::shared_ptr<Content> BitMaskedArray::localindex(int64_t axis, int64_t depth) const {
    return toByteMaskedArray().get()->localindex(axis, depth);
  }

  const std::shared_ptr<Content> BitMaskedArray::choose(int64_t n, bool diagonal, const std::shared_ptr<util::RecordLookup>& recordlookup, const util::Parameters& parameters, int64_t axis, int64_t depth) const {
    return toByteMaskedArray().get()->choose(n, diagonal, recordlookup, parameters, axis, depth);
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: BitMaskedArray::getitem_next(at)");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: BitMaskedArray::getitem_next(range)");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: BitMaskedArray::getitem_next(array)");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next(const SliceJagged64& jagged, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: BitMaskedArray::getitem_next(jagged)");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceArray64& slicecontent, const Slice& tail) const {
    return toByteMaskedArray().get()->getitem_next_jagged(slicestarts, slicestops, slicecontent, tail);
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceMissing64& slicecontent, const Slice& tail) const {
    return toByteMaskedArray().get()->getitem_next_jagged(slicestarts, slicestops, slicecontent, tail);
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceJagged64& slicecontent, const Slice& tail) const {
    return toByteMaskedArray().get()->getitem_next_jagged(slicestarts, slicestops, slicecontent, tail);
  }

}
