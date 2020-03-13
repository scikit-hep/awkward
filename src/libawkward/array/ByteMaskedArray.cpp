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

namespace awkward {
  ByteMaskedArray::ByteMaskedArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const Index8& mask, const std::shared_ptr<Content>& content, bool validwhen)
      : Content(identities, parameters)
      , mask_(mask)
      , content_(content)
      , validwhen_(validwhen) { }

  const Index8 ByteMaskedArray::mask() const {
    return mask_;
  }

  const std::shared_ptr<Content> ByteMaskedArray::content() const {
    return content_;
  }

  bool ByteMaskedArray::validwhen() const {
    return validwhen_;
  }

  const std::shared_ptr<Content> ByteMaskedArray::project() const {
    throw std::runtime_error("FIXME: ByteMaskedArray::project");
  }

  const std::shared_ptr<Content> ByteMaskedArray::project(const Index8& mask) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::project(mask)");
  }

  const Index8 ByteMaskedArray::bytemask() const {
    throw std::runtime_error("FIXME: ByteMaskedArray::bytemask");
  }

  const std::shared_ptr<Content> ByteMaskedArray::simplify() const {
    throw std::runtime_error("FIXME: ByteMaskedArray::simplify");
  }

  const std::string ByteMaskedArray::classname() const {
    return "ByteMaskedArray";
  }

  void ByteMaskedArray::setidentities(const std::shared_ptr<Identities>& identities) {
    throw std::runtime_error("FIXME: ByteMaskedArray::setidentities(identities)");
  }

  void ByteMaskedArray::setidentities() {
    throw std::runtime_error("FIXME: ByteMaskedArray::setidentities");
  }

  const std::shared_ptr<Type> ByteMaskedArray::type(const std::map<std::string, std::string>& typestrs) const {
    return std::make_shared<OptionType>(parameters_, util::gettypestr(parameters_, typestrs), content_.get()->type(typestrs));
  }

  const std::string ByteMaskedArray::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::tostring_part");
  }

  void ByteMaskedArray::tojson_part(ToJson& builder) const {
    int64_t len = length();
    check_for_iteration();
    builder.beginlist();
    for (int64_t i = 0;  i < len;  i++) {
      getitem_at_nowrap(i).get()->tojson_part(builder);
    }
    builder.endlist();
  }

  void ByteMaskedArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    mask_.nbytes_part(largest);
    content_.get()->nbytes_part(largest);
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  int64_t ByteMaskedArray::length() const {
    return mask_.length();
  }

  const std::shared_ptr<Content> ByteMaskedArray::shallow_copy() const {
    throw std::runtime_error("FIXME: ByteMaskedArray::shallow_copy");
  }

  const std::shared_ptr<Content> ByteMaskedArray::deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::deep_copy");
  }

  void ByteMaskedArray::check_for_iteration() const {
    throw std::runtime_error("FIXME: ByteMaskedArray::check_for_iteration");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_nothing() const {
    throw std::runtime_error("FIXME: ByteMaskedArray::getitem_nothing");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_at(int64_t at) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::getitem_at");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_at_nowrap(int64_t at) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::getitem_at_nowrap");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_range(int64_t start, int64_t stop) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::getitem_range");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::getitem_range_nowrap");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_field(const std::string& key) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::getitem_field");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::getitem_fields");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::getitem_next");
  }

  const std::shared_ptr<Content> ByteMaskedArray::carry(const Index64& carry) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::carry");
  }

  const std::string ByteMaskedArray::purelist_parameter(const std::string& key) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::purelist_parameter");
  }

  bool ByteMaskedArray::purelist_isregular() const {
    throw std::runtime_error("FIXME: ByteMaskedArray::purelist_isregular");
  }

  int64_t ByteMaskedArray::purelist_depth() const {
    throw std::runtime_error("FIXME: ByteMaskedArray::purelist_depth");
  }

  const std::pair<int64_t, int64_t> ByteMaskedArray::minmax_depth() const {
    throw std::runtime_error("FIXME: ByteMaskedArray::minmax_depth");
  }

  const std::pair<bool, int64_t> ByteMaskedArray::branch_depth() const {
    throw std::runtime_error("FIXME: ByteMaskedArray::branch_depth");
  }

  int64_t ByteMaskedArray::numfields() const {
    return content_.get()->numfields();
  }

  int64_t ByteMaskedArray::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string ByteMaskedArray::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool ByteMaskedArray::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string> ByteMaskedArray::keys() const {
    return content_.get()->keys();
  }

  const std::string ByteMaskedArray::validityerror(const std::string& path) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::validityerror");
  }

  const std::shared_ptr<Content> ByteMaskedArray::num(int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::num");
  }

  const std::pair<Index64, std::shared_ptr<Content>> ByteMaskedArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::offsets_and_flattened");
  }

  bool ByteMaskedArray::mergeable(const std::shared_ptr<Content>& other, bool mergebool) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::mergeable");
  }

  const std::shared_ptr<Content> ByteMaskedArray::reverse_merge(const std::shared_ptr<Content>& other) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::reverse_merge");
  }

  const std::shared_ptr<Content> ByteMaskedArray::merge(const std::shared_ptr<Content>& other) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::merge");
  }

  const std::shared_ptr<SliceItem> ByteMaskedArray::asslice() const {
    throw std::runtime_error("FIXME: ByteMaskedArray::asslice");
  }

  const std::shared_ptr<Content> ByteMaskedArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::rpad");
  }

  const std::shared_ptr<Content> ByteMaskedArray::rpad_and_clip(int64_t target, int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::rpad_and_clip");
  }

  const std::shared_ptr<Content> ByteMaskedArray::reduce_next(const Reducer& reducer, int64_t negaxis, const Index64& parents, int64_t outlength, bool mask, bool keepdims) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::reduce_next");
  }

  const std::shared_ptr<Content> ByteMaskedArray::localindex(int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::localindex");
  }

  const std::shared_ptr<Content> ByteMaskedArray::choose(int64_t n, bool diagonal, const std::shared_ptr<util::RecordLookup>& recordlookup, const util::Parameters& parameters, int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::choose");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: should this be an undefined operation: IndexedArray::getitem_next(at)");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: should this be an undefined operation: IndexedArray::getitem_next(range)");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: should this be an undefined operation: IndexedArray::getitem_next(array)");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_next(const SliceJagged64& jagged, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: should this be an undefined operation: IndexedArray::getitem_next(jagged)");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceArray64& slicecontent, const Slice& tail) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::getitem_next_jagged(array)");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceMissing64& slicecontent, const Slice& tail) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::getitem_next_jagged(missing)");
  }

  const std::shared_ptr<Content> ByteMaskedArray::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceJagged64& slicecontent, const Slice& tail) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::getitem_next_jagged(jagged)");
  }

  template <typename S>
  const std::shared_ptr<Content> ByteMaskedArray::getitem_next_jagged_generic(const Index64& slicestarts, const Index64& slicestops, const S& slicecontent, const Slice& tail) const {
    throw std::runtime_error("FIXME: ByteMaskedArray::getitem_next_jagged_generic");
  }

}
