// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/None.cpp", line)

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
      std::string("undefined operation: None::setidentities(identities)")
      + FILENAME(__LINE__));
  }

  void
  None::setidentities() {
    throw std::runtime_error(
      std::string("undefined operation: None::setidentities()")
      + FILENAME(__LINE__));
  }

  const TypePtr
  None::type(const util::TypeStrs& typestrs) const {
    throw std::runtime_error(
      std::string("undefined operation: None::type")
      + FILENAME(__LINE__));
  }

  const FormPtr
  None::form(bool materialize) const {
    throw std::runtime_error(
      std::string("undefined operation: None::form")
      + FILENAME(__LINE__));
  }

  bool
  None::has_virtual_form() const {
    throw std::runtime_error(
      std::string("undefined operation: None::has_virtual_form")
      + FILENAME(__LINE__));
  }

  bool
  None::has_virtual_length() const {
    throw std::runtime_error(
      std::string("undefined operation: None::has_virtual_length")
      + FILENAME(__LINE__));
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
    throw std::runtime_error(
      std::string("undefined operation: None::nbytes_part")
      + FILENAME(__LINE__));
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
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_nothing")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_at(int64_t at) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_at")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_at_nowrap(int64_t at) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_at_nowrap")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_range(int64_t start, int64_t stop) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_range")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_range_nowrap(int64_t start, int64_t stop) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_range_nowrap")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_field(const std::string& key) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_field")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_fields")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::carry(const Index64& carry, bool allow_lazy) const {
    throw std::runtime_error(
      std::string("undefined operation: None::carry")
      + FILENAME(__LINE__));
  }

  int64_t
  None::numfields() const {
    throw std::runtime_error(
      std::string("undefined operatino: None::numfields")
      + FILENAME(__LINE__));
  }

  int64_t
  None::fieldindex(const std::string& key) const {
    throw std::runtime_error(
      std::string("undefined operatino: None::fieldindex")
      + FILENAME(__LINE__));
  }

  const std::string
  None::key(int64_t fieldindex) const {
    throw std::runtime_error(
      std::string("undefined operatino: None::key")
      + FILENAME(__LINE__));
  }

  bool
  None::haskey(const std::string& key) const {
    throw std::runtime_error(
      std::string("undefined operatino: None::haskey")
      + FILENAME(__LINE__));
  }

  const std::vector<std::string>
  None::keys() const {
    throw std::runtime_error(
      std::string("undefined operatino: None::keys")
      + FILENAME(__LINE__));
  }

  const std::string
  None::validityerror(const std::string& path) const {
    throw std::runtime_error(
      std::string("undefined operation: None::validityerror")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::shallow_simplify() const {
    throw std::runtime_error(
      std::string("undefined operation: None::shallow_simplify")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::num(int64_t axis, int64_t depth) const {
    throw std::runtime_error(
      std::string("undefined operation: None::num")
      + FILENAME(__LINE__));
  }

  const std::pair<Index64, ContentPtr>
  None::offsets_and_flattened(int64_t axis, int64_t depth) const {
    throw std::runtime_error(
      std::string("undefined operation: None::offsets_and_flattened")
      + FILENAME(__LINE__));
  }

  bool
  None::mergeable(const ContentPtr& other, bool mergebool) const {
    throw std::runtime_error(
      std::string("undefined operation: None::mergeable")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::merge(const ContentPtr& other) const {
    throw std::runtime_error(
      std::string("undefined operation: None::merge")
      + FILENAME(__LINE__));
  }

  const SliceItemPtr
  None::asslice() const {
    throw std::runtime_error(
      std::string("undefined opteration: None::asslice")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::fillna(const ContentPtr& value) const {
    throw std::runtime_error(
      std::string("undefined opteration: None::fillna")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::rpad(int64_t length, int64_t axis, int64_t depth) const {
    throw std::runtime_error(
      std::string("undefined operation: None::rpad")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::rpad_and_clip(int64_t length, int64_t axis, int64_t depth) const {
    throw std::runtime_error(
      std::string("undefined operation: None::rpad_and_clip")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::reduce_next(const Reducer& reducer,
                    int64_t negaxis,
                    const Index64& starts,
                    const Index64& shifts,
                    const Index64& parents,
                    int64_t outlength,
                    bool mask,
                    bool keepdims) const {
    throw std::runtime_error(
      std::string("undefined operation: None::reduce_next")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::localindex(int64_t axis, int64_t depth) const {
    throw std::runtime_error(
      std::string("undefined operation: None:localindex")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::combinations(int64_t n,
                     bool replacement,
                     const util::RecordLookupPtr& recordlookup,
                     const util::Parameters& parameters,
                     int64_t axis,
                     int64_t depth) const {
    throw std::runtime_error(
      std::string("undefined operation: None::combinations")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::sort_next(int64_t negaxis,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength,
                  bool ascending,
                  bool stable,
                  bool keepdims) const {
    throw std::runtime_error(
      std::string("undefined operation: None::sort_next")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::argsort_next(int64_t negaxis,
                     const Index64& starts,
                     const Index64& parents,
                     int64_t outlength,
                     bool ascending,
                     bool stable,
                     bool keepdims) const {
    throw std::runtime_error(
      std::string("undefined operation: None::argsort_next")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_next(const SliceAt& at,
                     const Slice& tail,
                     const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_next(at)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_next(const SliceRange& range,
                     const Slice& tail,
                     const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_next(range)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_next(const SliceArray64& array,
                     const Slice& tail,
                     const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_next(array)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_next(const SliceField& field,
                     const Slice& tail,
                     const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_next(field)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_next(const SliceFields& fields,
                     const Slice& tail,
                     const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_next(fields)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_next(const SliceJagged64& jagged,
                     const Slice& tail,
                     const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_next(jagged)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_next_jagged(const Index64& slicestarts,
                            const Index64& slicestops,
                            const SliceArray64& slicecontent,
                            const Slice& tail) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_next_jagged(array)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_next_jagged(const Index64& slicestarts,
                            const Index64& slicestops,
                            const SliceMissing64& slicecontent,
                            const Slice& tail) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_next_jagged(missing)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::getitem_next_jagged(const Index64& slicestarts,
                            const Index64& slicestops,
                            const SliceJagged64& slicecontent,
                            const Slice& tail) const {
    throw std::runtime_error(
      std::string("undefined operation: None::getitem_next_jagged(jagged)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::copy_to(kernel::lib ptr_lib) const {
    throw std::runtime_error(
      std::string("undefined operation: None::copy_to(ptr_lib)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  None::numbers_to_type(const std::string& name) const {
    throw std::runtime_error(
      std::string("undefined operation: None::numbers_to_type")
      + FILENAME(__LINE__));
  }

  const ContentPtr none = std::make_shared<None>();
}
