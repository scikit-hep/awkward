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
#include "awkward/array/ByteMaskedArray.h"

#include "awkward/array/BitMaskedArray.h"

namespace awkward {
  BitMaskedArray::BitMaskedArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const IndexU8& mask, const std::shared_ptr<Content>& content, bool validwhen, int64_t length, bool lsb_order)
      : Content(identities, parameters)
      , mask_(mask)
      , content_(content)
      , validwhen_(validwhen)
      , length_(length)
      , lsb_order_(lsb_order)
      , asByteMaskedArray_(nullptr) { }

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

  const std::shared_ptr<Content> BitMaskedArray::toByteMaskedArray() const {
    throw std::runtime_error("FIXME: BitMaskedArray::toByteMaskedArray");
  }

  const std::shared_ptr<Content> BitMaskedArray::project() const {
    throw std::runtime_error("FIXME: BitMaskedArray::project");
  }

  const std::shared_ptr<Content> BitMaskedArray::project(const Index8& mask) const {
    throw std::runtime_error("FIXME: BitMaskedArray::project(mask)");
  }

  const Index8 BitMaskedArray::bytemask() const {
    throw std::runtime_error("FIXME: BitMaskedArray::bytemask");
  }

  const std::shared_ptr<Content> BitMaskedArray::simplify() const {
    throw std::runtime_error("FIXME: BitMaskedArray::simplify");
  }

  const std::string BitMaskedArray::classname() const {
    return "BitMaskedArray";
  }

  void BitMaskedArray::setidentities(const std::shared_ptr<Identities>& identities) {
    throw std::runtime_error("FIXME: BitMaskedArray::setidentities(identities)");
  }

  void BitMaskedArray::setidentities() {
    throw std::runtime_error("FIXME: BitMaskedArray::setidentities");
  }

  const std::shared_ptr<Type> BitMaskedArray::type(const std::map<std::string, std::string>& typestrs) const {
    return std::make_shared<OptionType>(parameters_, util::gettypestr(parameters_, typestrs), content_.get()->type(typestrs));
  }

  const std::string BitMaskedArray::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    throw std::runtime_error("FIXME: BitMaskedArray::tostring_part");
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
    return mask_.length();
  }

  const std::shared_ptr<Content> BitMaskedArray::shallow_copy() const {
    throw std::runtime_error("FIXME: BitMaskedArray::shallow_copy");
  }

  const std::shared_ptr<Content> BitMaskedArray::deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const {
    throw std::runtime_error("FIXME: BitMaskedArray::deep_copy");
  }

  void BitMaskedArray::check_for_iteration() const {
    throw std::runtime_error("FIXME: BitMaskedArray::check_for_iteration");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_nothing() const {
    throw std::runtime_error("FIXME: BitMaskedArray::getitem_nothing");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_at(int64_t at) const {
    throw std::runtime_error("FIXME: BitMaskedArray::getitem_at");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_at_nowrap(int64_t at) const {
    throw std::runtime_error("FIXME: BitMaskedArray::getitem_at_nowrap");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_range(int64_t start, int64_t stop) const {
    throw std::runtime_error("FIXME: BitMaskedArray::getitem_range");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    throw std::runtime_error("FIXME: BitMaskedArray::getitem_range_nowrap");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_field(const std::string& key) const {
    throw std::runtime_error("FIXME: BitMaskedArray::getitem_field");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::runtime_error("FIXME: BitMaskedArray::getitem_fields");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: BitMaskedArray::getitem_next");
  }

  const std::shared_ptr<Content> BitMaskedArray::carry(const Index64& carry) const {
    throw std::runtime_error("FIXME: BitMaskedArray::carry");
  }

  const std::string BitMaskedArray::purelist_parameter(const std::string& key) const {
    throw std::runtime_error("FIXME: BitMaskedArray::purelist_parameter");
  }

  bool BitMaskedArray::purelist_isregular() const {
    throw std::runtime_error("FIXME: BitMaskedArray::purelist_isregular");
  }

  int64_t BitMaskedArray::purelist_depth() const {
    throw std::runtime_error("FIXME: BitMaskedArray::purelist_depth");
  }

  const std::pair<int64_t, int64_t> BitMaskedArray::minmax_depth() const {
    throw std::runtime_error("FIXME: BitMaskedArray::minmax_depth");
  }

  const std::pair<bool, int64_t> BitMaskedArray::branch_depth() const {
    throw std::runtime_error("FIXME: BitMaskedArray::branch_depth");
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
    throw std::runtime_error("FIXME: BitMaskedArray::validityerror");
  }

  const std::shared_ptr<Content> BitMaskedArray::num(int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: BitMaskedArray::num");
  }

  const std::pair<Index64, std::shared_ptr<Content>> BitMaskedArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: BitMaskedArray::offsets_and_flattened");
  }

  bool BitMaskedArray::mergeable(const std::shared_ptr<Content>& other, bool mergebool) const {
    throw std::runtime_error("FIXME: BitMaskedArray::mergeable");
  }

  const std::shared_ptr<Content> BitMaskedArray::reverse_merge(const std::shared_ptr<Content>& other) const {
    throw std::runtime_error("FIXME: BitMaskedArray::reverse_merge");
  }

  const std::shared_ptr<Content> BitMaskedArray::merge(const std::shared_ptr<Content>& other) const {
    throw std::runtime_error("FIXME: BitMaskedArray::merge");
  }

  const std::shared_ptr<SliceItem> BitMaskedArray::asslice() const {
    throw std::runtime_error("FIXME: BitMaskedArray::asslice");
  }

  const std::shared_ptr<Content> BitMaskedArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: BitMaskedArray::rpad");
  }

  const std::shared_ptr<Content> BitMaskedArray::rpad_and_clip(int64_t target, int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: BitMaskedArray::rpad_and_clip");
  }

  const std::shared_ptr<Content> BitMaskedArray::reduce_next(const Reducer& reducer, int64_t negaxis, const Index64& parents, int64_t outlength, bool mask, bool keepdims) const {
    throw std::runtime_error("FIXME: BitMaskedArray::reduce_next");
  }

  const std::shared_ptr<Content> BitMaskedArray::localindex(int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: BitMaskedArray::localindex");
  }

  const std::shared_ptr<Content> BitMaskedArray::choose(int64_t n, bool diagonal, const std::shared_ptr<util::RecordLookup>& recordlookup, const util::Parameters& parameters, int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: BitMaskedArray::choose");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: should this be an undefined operation: IndexedArray::getitem_next(at)");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: should this be an undefined operation: IndexedArray::getitem_next(range)");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: should this be an undefined operation: IndexedArray::getitem_next(array)");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next(const SliceJagged64& jagged, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: should this be an undefined operation: IndexedArray::getitem_next(jagged)");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceArray64& slicecontent, const Slice& tail) const {
    throw std::runtime_error("FIXME: BitMaskedArray::getitem_next_jagged(array)");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceMissing64& slicecontent, const Slice& tail) const {
    throw std::runtime_error("FIXME: BitMaskedArray::getitem_next_jagged(missing)");
  }

  const std::shared_ptr<Content> BitMaskedArray::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceJagged64& slicecontent, const Slice& tail) const {
    throw std::runtime_error("FIXME: BitMaskedArray::getitem_next_jagged(jagged)");
  }

  template <typename S>
  const std::shared_ptr<Content> BitMaskedArray::getitem_next_jagged_generic(const Index64& slicestarts, const Index64& slicestops, const S& slicecontent, const Slice& tail) const {
    throw std::runtime_error("FIXME: BitMaskedArray::getitem_next_jagged_generic");
  }

}
