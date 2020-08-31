// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/Record.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/Record.cpp", line)

#include <sstream>
#include <memory>

#include "awkward/kernels/identities.h"
#include "awkward/kernels/getitem.h"
#include "awkward/type/RecordType.h"
#include "awkward/type/ArrayType.h"

#include "awkward/array/Record.h"

namespace awkward {
  Record::Record(const std::shared_ptr<const RecordArray> array, int64_t at)
      : Content(Identities::none(), array.get()->parameters())
      , array_(array)
      , at_(at) {
    if (!(0 <= at  &&  at < array.get()->length())) {
      throw std::invalid_argument(
        std::string("at=") + std::to_string(at)
        + std::string(" is out of range for recordarray")
        + FILENAME(__LINE__));
    }
  }

  const std::shared_ptr<const RecordArray>
  Record::array() const {
    return array_;
  }

  int64_t
  Record::at() const {
    return at_;
  }

  const ContentPtrVec
  Record::contents() const {
    ContentPtrVec out;
    for (auto item : array_.get()->contents()) {
      out.push_back(item.get()->getitem_at_nowrap(at_));
    }
    return out;
  }

  const util::RecordLookupPtr
  Record::recordlookup() const {
    return array_.get()->recordlookup();
  }

  bool
  Record::istuple() const {
    return array_.get()->istuple();
  }

  bool
  Record::isscalar() const {
    return true;
  }

  const std::string
  Record::classname() const {
    return "Record";
  }

  const IdentitiesPtr
  Record::identities() const {
    IdentitiesPtr recidentities = array_.get()->identities();
    if (recidentities.get() == nullptr) {
      return recidentities;
    }
    else {
      return recidentities.get()->getitem_range_nowrap(at_, at_ + 1);
    }
  }

  void
  Record::setidentities() {
    throw std::runtime_error(
      std::string("undefined operation: Record::setidentities")
      + FILENAME(__LINE__));
  }

  void
  Record::setidentities(const IdentitiesPtr& identities) {
    throw std::runtime_error(
      std::string("undefined operation: Record::setidentities")
      + FILENAME(__LINE__));
  }

  const TypePtr
  Record::type(const util::TypeStrs& typestrs) const {
    return form(true).get()->type(typestrs);
  }

  const FormPtr
  Record::form(bool materialize) const {
    return array_.get()->form(materialize);
  }

  bool
  Record::has_virtual_form() const {
    return array_.get()->has_virtual_form();
  }

  bool
  Record::has_virtual_length() const {
    return array_.get()->has_virtual_length();
  }

  const std::string
  Record::tostring_part(const std::string& indent,
                        const std::string& pre,
                        const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << " at=\"" << at_ << "\">\n";
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    out << array_.get()->tostring_part(indent + std::string("    "), "", "\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  void
  Record::tojson_part(ToJson& builder, bool include_beginendlist) const {
    size_t cols = (size_t)numfields();
    util::RecordLookupPtr keys = array_.get()->recordlookup();
    if (istuple()) {
      keys = std::make_shared<util::RecordLookup>();
      for (size_t j = 0;  j < cols;  j++) {
        keys.get()->push_back(std::to_string(j));
      }
    }
    ContentPtrVec contents = array_.get()->contents();
    builder.beginrecord();
    for (size_t j = 0;  j < cols;  j++) {
      builder.field(keys.get()->at(j).c_str());
      contents[j].get()->getitem_at_nowrap(at_).get()->tojson_part(builder,
                                                                   true);
    }
    builder.endrecord();
  }

  void
  Record::nbytes_part(std::map<size_t, int64_t>& largest) const {
    return array_.get()->nbytes_part(largest);
  }

  int64_t
  Record::length() const {
    return -1;   // just like NumpyArray with ndim == 0, which is also a scalar
  }

  const ContentPtr
  Record::shallow_copy() const {
    return std::make_shared<Record>(array_, at_);
  }

  const ContentPtr
  Record::deep_copy(bool copyarrays,
                    bool copyindexes,
                    bool copyidentities) const {
    ContentPtr out = array_.get()->deep_copy(copyarrays,
                                             copyindexes,
                                             copyidentities);
    return std::make_shared<Record>(
      std::dynamic_pointer_cast<RecordArray>(out), at_);
  }

  void
  Record::check_for_iteration() const {
    if (array_.get()->identities().get() != nullptr  &&
        array_.get()->identities().get()->length() != 1) {
      util::handle_error(
        failure("len(identities) != 1 for scalar Record",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        array_.get()->identities().get()->classname(),
        nullptr);
    }
  }

  const ContentPtr
  Record::getitem_nothing() const {
    throw std::runtime_error(
      std::string("undefined operation: Record::getitem_nothing")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_at(int64_t at) const {
    throw std::invalid_argument(
      std::string("scalar Record can only be sliced by field name (string); "
                  "try ") + util::quote(std::to_string(at))
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_at_nowrap(int64_t at) const {
    throw std::invalid_argument(
      std::string("scalar Record can only be sliced by field name (string); "
                  "try ") + util::quote(std::to_string(at))
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_range(int64_t start, int64_t stop) const {
    throw std::invalid_argument(
      std::string("scalar Record can only be sliced by field name (string)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_range_nowrap(int64_t start, int64_t stop) const {
    throw std::invalid_argument(
      std::string("scalar Record can only be sliced by field name (string)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_field(const std::string& key) const {
    return array_.get()->field(key).get()->getitem_at_nowrap(at_);
  }

  const ContentPtr
  Record::getitem_fields(const std::vector<std::string>& keys) const {
    ContentPtr recordarray = array_.get()->getitem_fields(keys);
    return recordarray.get()->getitem_at_nowrap(at_);
  }

  const ContentPtr
  Record::carry(const Index64& carry, bool allow_lazy) const {
    throw std::runtime_error(
      std::string("undefined operation: Record::carry") + FILENAME(__LINE__));
  }

  int64_t
  Record::purelist_depth() const {
    return 0;
  }

  const std::pair<int64_t, int64_t>
  Record::minmax_depth() const {
    std::pair<int64_t, int64_t> out = array_.get()->minmax_depth();
    return std::pair<int64_t, int64_t>(out.first - 1, out.second - 1);
  }

  const std::pair<bool, int64_t>
  Record::branch_depth() const {
    std::pair<bool, int64_t> out = array_.get()->branch_depth();
    return std::pair<bool, int64_t>(out.first, out.second - 1);
  }

  int64_t
  Record::numfields() const {
    return array_.get()->numfields();
  }

  int64_t
  Record::fieldindex(const std::string& key) const {
    return array_.get()->fieldindex(key);
  }

  const std::string
  Record::key(int64_t fieldindex) const {
    return array_.get()->key(fieldindex);
  }

  bool
  Record::haskey(const std::string& key) const {
    return array_.get()->haskey(key);
  }

  const std::vector<std::string>
  Record::keys() const {
    return array_.get()->keys();
  }

  const std::string
  Record::validityerror(const std::string& path) const {
    return array_.get()->validityerror(path + std::string(".array"));
  }

  const ContentPtr
  Record::shallow_simplify() const {
    return shallow_copy();
  }

  const ContentPtr
  Record::num(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      throw std::invalid_argument(
        std::string("cannot call 'num' with an 'axis' of 0 on a Record")
        + FILENAME(__LINE__));
    }
    else {
      ContentPtr singleton = array_.get()->getitem_range_nowrap(at_, at_ + 1);
      return singleton.get()->num(posaxis, depth).get()->getitem_at_nowrap(0);
    }
  }

  const std::pair<Index64, ContentPtr>
  Record::offsets_and_flattened(int64_t axis, int64_t depth) const {
    throw std::invalid_argument(
      std::string("Record cannot be flattened because it is not an array")
      + FILENAME(__LINE__));
  }

  bool
  Record::mergeable(const ContentPtr& other, bool mergebool) const {
    throw std::invalid_argument(
      std::string("Record cannot be merged because it is not an array")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::merge(const ContentPtr& other) const {
    throw std::invalid_argument(
      std::string("Record cannot be merged because it is not an array")
      + FILENAME(__LINE__));
  }

  const SliceItemPtr
  Record::asslice() const {
    throw std::invalid_argument(
      std::string("cannot use a record as a slice") + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::fillna(const ContentPtr& value) const {
    return array_.get()                                // get RecordArray
           ->getitem_range_nowrap(at_, at_ + 1).get()  // of just this element
           ->fillna(value).get()                       // fillna
           ->getitem_at_nowrap(0);                     // turn into a scalar
  }

  const ContentPtr
  Record::rpad(int64_t length, int64_t axis, int64_t depth) const {
    throw std::invalid_argument(
      std::string("Record cannot be padded because it is not an array")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::rpad_and_clip(int64_t length, int64_t axis, int64_t depth) const {
    throw std::invalid_argument(
      std::string("Record cannot be padded because it is not an array")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::reduce_next(const Reducer& reducer,
                      int64_t negaxis,
                      const Index64& starts,
                      const Index64& shifts,
                      const Index64& parents,
                      int64_t outlength,
                      bool mask,
                      bool keepdims) const {
    ContentPtr trimmed = array_.get()->getitem_range_nowrap(at_, at_ + 1);
    return trimmed.get()->reduce_next(reducer,
                                      negaxis,
                                      starts,
                                      shifts,
                                      parents,
                                      outlength,
                                      mask,
                                      keepdims);
  }

  const ContentPtr
  Record::localindex(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      throw std::invalid_argument(
        std::string("cannot call 'localindex' with an 'axis' of 0 on a Record")
        + FILENAME(__LINE__));
    }
    else {
      ContentPtr singleton = array_.get()->getitem_range_nowrap(at_, at_ + 1);
      return singleton.get()
             ->localindex(posaxis, depth).get()
             ->getitem_at_nowrap(0);
    }
  }

  const ContentPtr
  Record::combinations(int64_t n,
                       bool replacement,
                       const util::RecordLookupPtr& recordlookup,
                       const util::Parameters& parameters,
                       int64_t axis,
                       int64_t depth) const {
    if (n < 1) {
      throw std::invalid_argument(
        std::string("in combinations, 'n' must be at least 1") + FILENAME(__LINE__));
    }
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      throw std::invalid_argument(
        std::string("cannot call 'combinations' with an 'axis' of 0 on a Record")
        + FILENAME(__LINE__));
    }
    else {
      ContentPtr singleton = array_.get()->getitem_range_nowrap(at_, at_ + 1);
      return singleton.get()->combinations(n,
                                           replacement,
                                           recordlookup,
                                           parameters,
                                           posaxis,
                                           depth).get()->getitem_at_nowrap(0);
    }
  }

  const ContentPtr
  Record::field(int64_t fieldindex) const {
    return array_.get()->field(fieldindex).get()->getitem_at_nowrap(at_);
  }

  const ContentPtr
  Record::field(const std::string& key) const {
    return array_.get()->field(key).get()->getitem_at_nowrap(at_);
  }

  const ContentPtrVec
  Record::fields() const {
    ContentPtrVec out;
    int64_t cols = numfields();
    for (int64_t j = 0;  j < cols;  j++) {
      out.push_back(array_.get()->field(j).get()->getitem_at_nowrap(at_));
    }
    return out;
  }

  const std::vector<std::pair<std::string, ContentPtr>>
  Record::fielditems() const {
    std::vector<std::pair<std::string, ContentPtr>> out;
    util::RecordLookupPtr keys = array_.get()->recordlookup();
    if (istuple()) {
      int64_t cols = numfields();
      for (int64_t j = 0;  j < cols;  j++) {
        out.push_back(std::pair<std::string, ContentPtr>(
          std::to_string(j),
          array_.get()->field(j).get()->getitem_at_nowrap(at_)));
      }
    }
    else {
      int64_t cols = numfields();
      for (int64_t j = 0;  j < cols;  j++) {
        out.push_back(std::pair<std::string, ContentPtr>(
          keys.get()->at((size_t)j),
          array_.get()->field(j).get()->getitem_at_nowrap(at_)));
      }
    }
    return out;
  }

  const std::shared_ptr<Record>
  Record::astuple() const {
    return std::make_shared<Record>(array_.get()->astuple(), at_);
  }

  const ContentPtr
  Record::sort_next(int64_t negaxis,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength,
                    bool ascending,
                    bool stable,
                    bool keepdims) const {
    ContentPtr next = array_.get()->getitem_at_nowrap(at_);
    return next.get()->sort_next(negaxis,
                                 starts,
                                 parents,
                                 outlength,
                                 ascending,
                                 stable,
                                 keepdims);
  }

  const ContentPtr
  Record::argsort_next(int64_t negaxis,
                       const Index64& starts,
                       const Index64& parents,
                       int64_t outlength,
                       bool ascending,
                       bool stable,
                       bool keepdims) const {
    ContentPtr next = array_.get()->getitem_at_nowrap(at_);
    return next.get()->argsort_next(negaxis,
                                    starts,
                                    parents,
                                    outlength,
                                    ascending,
                                    stable,
                                    keepdims);
  }

  const ContentPtr
  Record::getitem(const Slice& where) const {
    ContentPtr next = array_.get()->getitem_range_nowrap(at_, at_ + 1);

    SliceItemPtr nexthead = where.head();
    Slice nexttail = where.tail();
    Index64 nextadvanced(0);
    ContentPtr out = next.get()->getitem_next(nexthead,
                                              nexttail,
                                              nextadvanced);

    if (out.get()->length() == 0) {
      return out.get()->getitem_nothing();
    }
    else {
      return out.get()->getitem_at_nowrap(0);
    }
  }

  const ContentPtr
  Record::getitem_next(const SliceAt& at,
                       const Slice& tail,
                       const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: Record::getitem_next(at)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_next(const SliceRange& range,
                       const Slice& tail,
                       const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: Record::getitem_next(range)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_next(const SliceArray64& array,
                       const Slice& tail,
                       const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: Record::getitem_next(array)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_next(const SliceField& field,
                       const Slice& tail,
                       const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: Record::getitem_next(field)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_next(const SliceFields& fields,
                       const Slice& tail,
                       const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: Record::getitem_next(fields)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_next(const SliceJagged64& jagged,
                       const Slice& tail,
                       const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: Record::getitem_next(jagged)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_next_jagged(const Index64& slicestarts,
                              const Index64& slicestops,
                              const SliceArray64& slicecontent,
                              const Slice& tail) const {
    throw std::runtime_error(
      std::string("undefined operation: Record::getitem_next_jagged(array)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_next_jagged(const Index64& slicestarts,
                              const Index64& slicestops,
                              const SliceMissing64& slicecontent,
                              const Slice& tail) const {
    throw std::runtime_error(
      std::string("undefined operation: Record::getitem_next_jagged(missing)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::getitem_next_jagged(const Index64& slicestarts,
                              const Index64& slicestops,
                              const SliceJagged64& slicecontent,
                              const Slice& tail) const {
    throw std::runtime_error(
      std::string("undefined operation: Record::getitem_next_jagged(jagged)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  Record::copy_to(kernel::lib ptr_lib) const {
    ContentPtr array = array_.get()->copy_to(ptr_lib);
    return std::make_shared<Record>(
             std::dynamic_pointer_cast<RecordArray>(array), at_);
  }

  const ContentPtr
  Record::numbers_to_type(const std::string& name) const {
    ContentPtr out = array_.get()->numbers_to_type(name);
    return std::make_shared<Record>(
      std::dynamic_pointer_cast<RecordArray>(out), at_);
  }

}
