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

#include "awkward/array/UnmaskedArray.h"

namespace awkward {
  UnmaskedArray::UnmaskedArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const std::shared_ptr<Content>& content)
      : Content(identities, parameters)
      , content_(content) { }

  const std::shared_ptr<Content> UnmaskedArray::content() const {
    return content_;
  }

  const std::shared_ptr<Content> UnmaskedArray::project() const {
    throw std::runtime_error("FIXME: UnmaskedArray::project");
  }

  const std::shared_ptr<Content> UnmaskedArray::project(const Index8& mask) const {
    throw std::runtime_error("FIXME: UnmaskedArray::project(mask)");
  }

  const Index8 UnmaskedArray::bytemask() const {
    throw std::runtime_error("FIXME: UnmaskedArray::bytemask");
  }

  const std::shared_ptr<Content> UnmaskedArray::simplify() const {
    throw std::runtime_error("FIXME: UnmaskedArray::simplify");
  }

  const std::shared_ptr<Content> UnmaskedArray::toIndexedOptionArray64() const {
    throw std::runtime_error("FIXME: UnmaskedArray::toIndexedOptionArray64");
  }

  const std::string UnmaskedArray::classname() const {
    return "UnmaskedArray";
  }

  void UnmaskedArray::setidentities(const std::shared_ptr<Identities>& identities) {
    throw std::runtime_error("FIXME: UnmaskedArray::setidentities(identities)");
  }

  void UnmaskedArray::setidentities() {
    throw std::runtime_error("FIXME: UnmaskedArray::setidentities");
  }

  const std::shared_ptr<Type> UnmaskedArray::type(const std::map<std::string, std::string>& typestrs) const {
    return std::make_shared<OptionType>(parameters_, util::gettypestr(parameters_, typestrs), content_.get()->type(typestrs));
  }

  const std::string UnmaskedArray::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    throw std::runtime_error("FIXME: UnmaskedArray::tostring_part");
  }

  void UnmaskedArray::tojson_part(ToJson& builder) const {
    content_.get()->tojson_part(builder);
  }

  void UnmaskedArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    content_.get()->nbytes_part(largest);
  }

  int64_t UnmaskedArray::length() const {
    return content_.get()->length();
  }

  const std::shared_ptr<Content> UnmaskedArray::shallow_copy() const {
    throw std::runtime_error("FIXME: UnmaskedArray::shallow_copy");
  }

  const std::shared_ptr<Content> UnmaskedArray::deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const {
    throw std::runtime_error("FIXME: UnmaskedArray::deep_copy");
  }

  void UnmaskedArray::check_for_iteration() const {
    throw std::runtime_error("FIXME: UnmaskedArray::check_for_iteration");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_nothing() const {
    throw std::runtime_error("FIXME: UnmaskedArray::getitem_nothing");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_at(int64_t at) const {
    throw std::runtime_error("FIXME: UnmaskedArray::getitem_at");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_at_nowrap(int64_t at) const {
    throw std::runtime_error("FIXME: UnmaskedArray::getitem_at_nowrap");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_range(int64_t start, int64_t stop) const {
    throw std::runtime_error("FIXME: UnmaskedArray::getitem_range");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    throw std::runtime_error("FIXME: UnmaskedArray::getitem_range_nowrap");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_field(const std::string& key) const {
    throw std::runtime_error("FIXME: UnmaskedArray::getitem_field");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::runtime_error("FIXME: UnmaskedArray::getitem_fields");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: UnmaskedArray::getitem_next");
  }

  const std::shared_ptr<Content> UnmaskedArray::carry(const Index64& carry) const {
    throw std::runtime_error("FIXME: UnmaskedArray::carry");
  }

  const std::string UnmaskedArray::purelist_parameter(const std::string& key) const {
    throw std::runtime_error("FIXME: UnmaskedArray::purelist_parameter");
  }

  bool UnmaskedArray::purelist_isregular() const {
    throw std::runtime_error("FIXME: UnmaskedArray::purelist_isregular");
  }

  int64_t UnmaskedArray::purelist_depth() const {
    throw std::runtime_error("FIXME: UnmaskedArray::purelist_depth");
  }

  const std::pair<int64_t, int64_t> UnmaskedArray::minmax_depth() const {
    throw std::runtime_error("FIXME: UnmaskedArray::minmax_depth");
  }

  const std::pair<bool, int64_t> UnmaskedArray::branch_depth() const {
    throw std::runtime_error("FIXME: UnmaskedArray::branch_depth");
  }

  int64_t UnmaskedArray::numfields() const {
    return content_.get()->numfields();
  }

  int64_t UnmaskedArray::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string UnmaskedArray::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool UnmaskedArray::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string> UnmaskedArray::keys() const {
    return content_.get()->keys();
  }

  const std::string UnmaskedArray::validityerror(const std::string& path) const {
    throw std::runtime_error("FIXME: UnmaskedArray::validityerror");
  }

  const std::shared_ptr<Content> UnmaskedArray::num(int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: UnmaskedArray::num");
  }

  const std::pair<Index64, std::shared_ptr<Content>> UnmaskedArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: UnmaskedArray::offsets_and_flattened");
  }

  bool UnmaskedArray::mergeable(const std::shared_ptr<Content>& other, bool mergebool) const {
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

  const std::shared_ptr<Content> UnmaskedArray::reverse_merge(const std::shared_ptr<Content>& other) const {
    std::shared_ptr<Content> indexedoptionarray = toIndexedOptionArray64();
    IndexedOptionArray64* raw = dynamic_cast<IndexedOptionArray64*>(indexedoptionarray.get());
    return raw->reverse_merge(other);
  }

  const std::shared_ptr<Content> UnmaskedArray::merge(const std::shared_ptr<Content>& other) const {
    return toIndexedOptionArray64().get()->merge(other);
  }

  const std::shared_ptr<SliceItem> UnmaskedArray::asslice() const {
    throw std::runtime_error("FIXME: UnmaskedArray::asslice");
  }

  const std::shared_ptr<Content> UnmaskedArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: UnmaskedArray::rpad");
  }

  const std::shared_ptr<Content> UnmaskedArray::rpad_and_clip(int64_t target, int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: UnmaskedArray::rpad_and_clip");
  }

  const std::shared_ptr<Content> UnmaskedArray::reduce_next(const Reducer& reducer, int64_t negaxis, const Index64& parents, int64_t outlength, bool mask, bool keepdims) const {
    throw std::runtime_error("FIXME: UnmaskedArray::reduce_next");
  }

  const std::shared_ptr<Content> UnmaskedArray::localindex(int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: UnmaskedArray::localindex");
  }

  const std::shared_ptr<Content> UnmaskedArray::choose(int64_t n, bool diagonal, const std::shared_ptr<util::RecordLookup>& recordlookup, const util::Parameters& parameters, int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: UnmaskedArray::choose");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: should this be an undefined operation: IndexedArray::getitem_next(at)");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: should this be an undefined operation: IndexedArray::getitem_next(range)");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: should this be an undefined operation: IndexedArray::getitem_next(array)");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_next(const SliceJagged64& jagged, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: should this be an undefined operation: IndexedArray::getitem_next(jagged)");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceArray64& slicecontent, const Slice& tail) const {
    throw std::runtime_error("FIXME: UnmaskedArray::getitem_next_jagged(array)");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceMissing64& slicecontent, const Slice& tail) const {
    throw std::runtime_error("FIXME: UnmaskedArray::getitem_next_jagged(missing)");
  }

  const std::shared_ptr<Content> UnmaskedArray::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceJagged64& slicecontent, const Slice& tail) const {
    throw std::runtime_error("FIXME: UnmaskedArray::getitem_next_jagged(jagged)");
  }

  template <typename S>
  const std::shared_ptr<Content> UnmaskedArray::getitem_next_jagged_generic(const Index64& slicestarts, const Index64& slicestops, const S& slicecontent, const Slice& tail) const {
    throw std::runtime_error("FIXME: UnmaskedArray::getitem_next_jagged_generic");
  }

}
