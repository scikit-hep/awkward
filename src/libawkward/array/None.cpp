// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/type/UnknownType.h"
#include "awkward/type/ArrayType.h"

#include "awkward/array/None.h"

namespace awkward {
  None::None()
      : Content(Identities::none(), util::Parameters()) { }

  bool
  None::isscalar() const {
    return true;
  }

  const std::string
  None::classname() const {
    return "None";
  }

  void
  None::setidentities(const IdentitiesPtr& identities) {
    throw std::runtime_error(
      "undefined operation: None::setidentities(identities)");
  }

  void
  None::setidentities() {
    throw std::runtime_error("undefined operation: None::setidentities()");
  }

  const TypePtr
  None::type(const util::TypeStrs& typestrs) const {
    throw std::runtime_error("undefined operation: None::type");
  }

  const FormPtr
  None::form(bool materialize) const {
    throw std::runtime_error("undefined operation: None::form");
  }

  bool
  None::has_virtual_form() const {
    throw std::runtime_error("undefined operation: None::has_virtual_form");
  }

  bool
  None::has_virtual_length() const {
    throw std::runtime_error("undefined operation: None::has_virtual_length");
  }

  const std::string
  None::tostring_part(const std::string& indent,
                      const std::string& pre,
                      const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << "/>" << post;
    return out.str();
  }

  void
  None::tojson_part(ToJson& builder,
                    bool include_beginendlist) const {
    builder.null();
  }

  void
  None::nbytes_part(std::map<size_t, int64_t>& largest) const {
    throw std::runtime_error("undefined operation: None::nbytes_part");
  }

  int64_t
  None::length() const {
    return -1;
  }

  const ContentPtr
  None::shallow_copy() const {
    return std::make_shared<None>();
  }

  const ContentPtr
  None::deep_copy(bool copyarrays,
                  bool copyindexes,
                  bool copyidentities) const {
    return std::make_shared<None>();
  }

  void
  None::check_for_iteration() const { }

  const ContentPtr
  None::getitem_nothing() const {
    throw std::runtime_error("undefined operation: None::getitem_nothing");
  }

  const ContentPtr
  None::getitem_at(int64_t at) const {
    throw std::runtime_error("undefined operation: None::getitem_at");
  }

  const ContentPtr
  None::getitem_at_nowrap(int64_t at) const {
    throw std::runtime_error("undefined operation: None::getitem_at_nowrap");
  }

  const ContentPtr
  None::getitem_range(int64_t start, int64_t stop) const {
    throw std::runtime_error("undefined operation: None::getitem_range");
  }

  const ContentPtr
  None::getitem_range_nowrap(int64_t start, int64_t stop) const {
    throw std::runtime_error(
      "undefined operation: None::getitem_range_nowrap");
  }

  const ContentPtr
  None::getitem_field(const std::string& key) const {
    throw std::runtime_error("undefined operation: None::getitem_field");
  }

  const ContentPtr
  None::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::runtime_error("undefined operation: None::getitem_fields");
  }

  const ContentPtr
  None::carry(const Index64& carry) const {
    throw std::runtime_error("undefined operation: None::carry");
  }

  int64_t
  None::numfields() const {
    throw std::runtime_error("undefined operatino: None::numfields");
  }

  int64_t
  None::fieldindex(const std::string& key) const {
    throw std::runtime_error("undefined operatino: None::fieldindex");
  }

  const std::string
  None::key(int64_t fieldindex) const {
    throw std::runtime_error("undefined operatino: None::key");
  }

  bool
  None::haskey(const std::string& key) const {
    throw std::runtime_error("undefined operatino: None::haskey");
  }

  const std::vector<std::string>
  None::keys() const {
    throw std::runtime_error("undefined operatino: None::keys");
  }

  const std::string
  None::validityerror(const std::string& path) const {
    throw std::runtime_error("undefined operation: None::validityerror");
  }

  const ContentPtr
  None::shallow_simplify() const {
    throw std::runtime_error("undefined operation: None::shallow_simplify");
  }

  const ContentPtr
  None::num(int64_t axis, int64_t depth) const {
    throw std::runtime_error("undefined operation: None::num");
  }

  const std::pair<Index64, ContentPtr>
  None::offsets_and_flattened(int64_t axis, int64_t depth) const {
    throw std::runtime_error(
      "undefined operation: None::offsets_and_flattened");
  }

  bool
  None::mergeable(const ContentPtr& other, bool mergebool) const {
    throw std::runtime_error("undefined operation: None::mergeable");
  }

  const ContentPtr
  None::merge(const ContentPtr& other) const {
    throw std::runtime_error("undefined operation: None::merge");
  }

  const SliceItemPtr
  None::asslice() const {
    throw std::runtime_error("undefined opteration: None::asslice");
  }

  const ContentPtr
  None::fillna(const ContentPtr& value) const {
    throw std::runtime_error("undefined opteration: None::fillna");
  }

  const ContentPtr
  None::rpad(int64_t length, int64_t axis, int64_t depth) const {
    throw std::runtime_error("undefined operation: None::rpad");
  }

  const ContentPtr
  None::rpad_and_clip(int64_t length, int64_t axis, int64_t depth) const {
    throw std::runtime_error("undefined operation: None::rpad_and_clip");
  }

  const ContentPtr
  None::reduce_next(const Reducer& reducer,
                    int64_t negaxis,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength,
                    bool mask,
                    bool keepdims) const {
    throw std::runtime_error("undefined operation: None::reduce_next");
  }

  const ContentPtr
  None::localindex(int64_t axis, int64_t depth) const {
    throw std::runtime_error("undefined operation: None:localindex");
  }

  const ContentPtr
  None::combinations(int64_t n,
                     bool replacement,
                     const util::RecordLookupPtr& recordlookup,
                     const util::Parameters& parameters,
                     int64_t axis,
                     int64_t depth) const {
    throw std::runtime_error("undefined operation: None::combinations");
  }

  const ContentPtr
  None::getitem_next(const SliceAt& at,
                     const Slice& tail,
                     const Index64& advanced) const {
    throw std::runtime_error("undefined operation: None::getitem_next(at)");
  }

  const ContentPtr
  None::getitem_next(const SliceRange& range,
                     const Slice& tail,
                     const Index64& advanced) const {
    throw std::runtime_error("undefined operation: None::getitem_next(range)");
  }

  const ContentPtr
  None::getitem_next(const SliceArray64& array,
                     const Slice& tail,
                     const Index64& advanced) const {
    throw std::runtime_error("undefined operation: None::getitem_next(array)");
  }

  const ContentPtr
  None::getitem_next(const SliceField& field,
                     const Slice& tail,
                     const Index64& advanced) const {
    throw std::runtime_error("undefined operation: None::getitem_next(field)");
  }

  const ContentPtr
  None::getitem_next(const SliceFields& fields,
                     const Slice& tail,
                     const Index64& advanced) const {
    throw std::runtime_error(
      "undefined operation: None::getitem_next(fields)");
  }

  const ContentPtr
  None::getitem_next(const SliceJagged64& jagged,
                     const Slice& tail,
                     const Index64& advanced) const {
    throw std::runtime_error(
      "undefined operation: None::getitem_next(jagged)");
  }

  const ContentPtr
  None::getitem_next_jagged(const Index64& slicestarts,
                            const Index64& slicestops,
                            const SliceArray64& slicecontent,
                            const Slice& tail) const {
    throw std::runtime_error(
      "undefined operation: None::getitem_next_jagged(array)");
  }

  const ContentPtr
  None::getitem_next_jagged(const Index64& slicestarts,
                            const Index64& slicestops,
                            const SliceMissing64& slicecontent,
                            const Slice& tail) const {
    throw std::runtime_error(
      "undefined operation: None::getitem_next_jagged(missing)");
  }

  const ContentPtr
  None::getitem_next_jagged(const Index64& slicestarts,
                            const Index64& slicestops,
                            const SliceJagged64& slicecontent,
                            const Slice& tail) const {
    throw std::runtime_error(
      "undefined operation: None::getitem_next_jagged(jagged)");
  }

  const ContentPtr none = std::make_shared<None>();
}
