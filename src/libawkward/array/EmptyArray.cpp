// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/EmptyArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/EmptyArray.cpp", line)

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/kernels/operations.h"
#include "awkward/type/UnknownType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/RegularArray.h"

#include "awkward/array/EmptyArray.h"

namespace awkward {
  ////////// EmptyForm

  EmptyForm::EmptyForm(bool has_identities,
                       const util::Parameters& parameters,
                       const FormKey& form_key)
      : Form(has_identities, parameters, form_key) { }

  const TypePtr
  EmptyForm::type(const util::TypeStrs& typestrs) const {
    return std::make_shared<UnknownType>(
               parameters_,
               util::gettypestr(parameters_, typestrs));
  }

  void
  EmptyForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    builder.string("EmptyArray");
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  EmptyForm::shallow_copy() const {
    return std::make_shared<EmptyForm>(has_identities_,
                                       parameters_,
                                       form_key_);
  }

  const std::string
  EmptyForm::purelist_parameter(const std::string& key) const {
    return parameter(key);
  }

  bool
  EmptyForm::purelist_isregular() const {
    return true;
  }

  int64_t
  EmptyForm::purelist_depth() const {
    return 1;
  }

  const std::pair<int64_t, int64_t>
  EmptyForm::minmax_depth() const {
    return std::pair<int64_t, int64_t>(1, 1);
  }

  const std::pair<bool, int64_t>
  EmptyForm::branch_depth() const {
    return std::pair<bool, int64_t>(false, 1);
  }

  int64_t
  EmptyForm::numfields() const {
    return -1;
  }

  int64_t
  EmptyForm::fieldindex(const std::string& key) const {
    throw std::invalid_argument(
      std::string("key ") + util::quote(key)
      + std::string(" does not exist (data might not be records)")
      + FILENAME(__LINE__));
  }

  const std::string
  EmptyForm::key(int64_t fieldindex) const {
    throw std::invalid_argument(
      std::string("fieldindex \"") + std::to_string(fieldindex)
      + std::string("\" does not exist (data might not be records)")
      + FILENAME(__LINE__));
  }

  bool
  EmptyForm::haskey(const std::string& key) const {
    return false;
  }

  const std::vector<std::string>
  EmptyForm::keys() const {
    return std::vector<std::string>();
  }

  bool
  EmptyForm::equal(const FormPtr& other,
                   bool check_identities,
                   bool check_parameters,
                   bool check_form_key,
                   bool compatibility_check) const {
    if (check_identities  &&
        has_identities_ != other.get()->has_identities()) {
      return false;
    }
    if (check_parameters  &&
        !util::parameters_equal(parameters_, other.get()->parameters())) {
      return false;
    }
    if (check_form_key  &&
        !form_key_equals(other.get()->form_key())) {
      return false;
    }
    if (EmptyForm* t = dynamic_cast<EmptyForm*>(other.get())) {
      return true;
    }
    else {
      return false;
    }
  }

  const FormPtr
  EmptyForm::getitem_field(const std::string& key) const {
    throw std::invalid_argument(
      std::string("key ") + util::quote(key)
      + std::string(" does not exist (data might not be records)"));
  }

  ////////// EmptyArray

  EmptyArray::EmptyArray(const IdentitiesPtr& identities,
                         const util::Parameters& parameters)
      : Content(identities, parameters) { }

  const std::shared_ptr<NumpyArray>
  EmptyArray::toNumpyArray(const std::string& format,
                           ssize_t itemsize,
                           util::dtype dtype) const {
    std::shared_ptr<void> ptr(new uint8_t[0], kernel::array_deleter<uint8_t>());
    std::vector<ssize_t> shape({ 0 });
    std::vector<ssize_t> strides({ itemsize });
    return std::make_shared<NumpyArray>(identities_,
                                        parameters_,
                                        ptr,
                                        shape,
                                        strides,
                                        0,
                                        itemsize,
                                        format,
                                        dtype,
                                        kernel::lib::cpu);
  }

  const std::string
  EmptyArray::classname() const {
    return "EmptyArray";
  }

  void
  EmptyArray::setidentities(const IdentitiesPtr& identities) {
    if (identities.get() != nullptr  &&
        length() != identities.get()->length()) {
      util::handle_error(
        failure("content and its identities must have the same length",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    identities_ = identities;
  }

  void
  EmptyArray::setidentities() { }

  const TypePtr
  EmptyArray::type(const util::TypeStrs& typestrs) const {
    return form(true).get()->type(typestrs);
  }

  const FormPtr
  EmptyArray::form(bool materialize) const {
    return std::make_shared<EmptyForm>(identities_.get() != nullptr,
                                       parameters_,
                                       FormKey(nullptr));
  }

  bool
  EmptyArray::has_virtual_form() const {
    return false;
  }

  bool
  EmptyArray::has_virtual_length() const {
    return false;
  }

  const std::string
  EmptyArray::tostring_part(const std::string& indent,
                            const std::string& pre,
                            const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname();
    if (identities_.get() == nullptr  &&  parameters_.empty()) {
      out << "/>" << post;
    }
    else {
      out << ">\n";
      if (identities_.get() != nullptr) {
        out << identities_.get()->tostring_part(
                 indent + std::string("    "), "", "\n")
            << indent << "</" << classname() << ">" << post;
      }
      if (!parameters_.empty()) {
        out << parameters_tostring(indent + std::string("    "), "", "\n");
      }
      out << indent << "</" << classname() << ">" << post;
    }
    return out.str();
  }

  void
  EmptyArray::tojson_part(ToJson& builder,
                          bool include_beginendlist) const {
    check_for_iteration();
    if (include_beginendlist) {
      builder.beginlist();
      builder.endlist();
    }
  }

  void
  EmptyArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  int64_t
  EmptyArray::length() const {
    return 0;
  }

  const ContentPtr
  EmptyArray::shallow_copy() const {
    return std::make_shared<EmptyArray>(identities_, parameters_);
  }

  const ContentPtr
  EmptyArray::deep_copy(bool copyarrays,
                        bool copyindexes,
                        bool copyidentities) const {
    IdentitiesPtr identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<EmptyArray>(identities, parameters_);
  }

  void
  EmptyArray::check_for_iteration() const { }

  const ContentPtr
  EmptyArray::getitem_nothing() const {
    return shallow_copy();
  }

  const ContentPtr
  EmptyArray::getitem_at(int64_t at) const {
    util::handle_error(
      failure("index out of range",
              kSliceNone,
              at,
              FILENAME_C(__LINE__)),
      classname(),
      identities_.get());
    return ContentPtr(nullptr);  // make Windows compiler happy
  }

  const ContentPtr
  EmptyArray::getitem_at_nowrap(int64_t at) const {
    util::handle_error(
      failure("index out of range", kSliceNone, at, FILENAME_C(__LINE__)),
      classname(),
      identities_.get());
    return ContentPtr(nullptr);  // make Windows compiler happy
  }

  const ContentPtr
  EmptyArray::getitem_range(int64_t start, int64_t stop) const {
    return shallow_copy();
  }

  const ContentPtr
  EmptyArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    return shallow_copy();
  }

  const ContentPtr
  EmptyArray::getitem_field(const std::string& key) const {
    throw std::invalid_argument(
      std::string("cannot slice ") + classname()
      + std::string(" by field name") + FILENAME(__LINE__));
  }

  const ContentPtr
  EmptyArray::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::invalid_argument(
      std::string("cannot slice ") + classname()
      + std::string(" by field names") + FILENAME(__LINE__));
  }

  const ContentPtr
  EmptyArray::carry(const Index64& carry, bool allow_lazy) const {
    return shallow_copy();
  }

  int64_t
  EmptyArray::numfields() const {
    return -1;
  }

  int64_t
  EmptyArray::fieldindex(const std::string& key) const {
    throw std::invalid_argument(
      std::string("key ") + util::quote(key)
      + std::string(" does not exist (data might not be records)")
      + FILENAME(__LINE__));
  }

  const std::string
  EmptyArray::key(int64_t fieldindex) const {
    throw std::invalid_argument(
      std::string("fieldindex \"") + std::to_string(fieldindex)
      + std::string("\" does not exist (data might not be records)")
      + FILENAME(__LINE__));
  }

  bool
  EmptyArray::haskey(const std::string& key) const {
    return false;
  }

  const std::vector<std::string>
  EmptyArray::keys() const {
    return std::vector<std::string>();
  }

  const std::string
  EmptyArray::validityerror(const std::string& path) const {
    return std::string();
  }

  const ContentPtr
  EmptyArray::shallow_simplify() const {
    return shallow_copy();
  }

  const ContentPtr
  EmptyArray::num(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      Index64 out(1);
      out.setitem_at_nowrap(0, length());
      return NumpyArray(out).getitem_at_nowrap(0);
    }
    else {
      return std::make_shared<NumpyArray>(Index64(0));
    }
  }

  const std::pair<Index64, ContentPtr>
  EmptyArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      throw std::invalid_argument(
        std::string("axis=0 not allowed for flatten") + FILENAME(__LINE__));
    }
    else {
      Index64 offsets(1);
      offsets.setitem_at_nowrap(0, 0);
      return std::pair<Index64, ContentPtr>(
        offsets,
        std::make_shared<EmptyArray>(Identities::none(), util::Parameters()));
    }
  }

  bool
  EmptyArray::mergeable(const ContentPtr& other, bool mergebool) const {
    return true;
  }

  const ContentPtr
  EmptyArray::merge(const ContentPtr& other) const {
    return other;
  }

  const SliceItemPtr
  EmptyArray::asslice() const {
    Index64 index(0);
    std::vector<int64_t> shape({ 0 });
    std::vector<int64_t> strides({ 1 });
    return std::make_shared<SliceArray64>(index, shape, strides, false);
  }

  const ContentPtr
  EmptyArray::fillna(const ContentPtr& value) const {
    return std::make_shared<EmptyArray>(Identities::none(),
                                        util::Parameters());
  }

  const ContentPtr
  EmptyArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis != depth) {
      throw std::invalid_argument(
        std::string("axis exceeds the depth of this array") + FILENAME(__LINE__));
    }
    else {
      return rpad_and_clip(target, posaxis, depth);
    }
  }

  const ContentPtr
  EmptyArray::rpad_and_clip(int64_t target,
                            int64_t axis,
                            int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis != depth) {
      throw std::invalid_argument(
        std::string("axis exceeds the depth of this array") + FILENAME(__LINE__));
    }
    else {
      return rpad_axis0(target, true);
    }
  }

  const ContentPtr
  EmptyArray::reduce_next(const Reducer& reducer,
                          int64_t negaxis,
                          const Index64& starts,
                          const Index64& shifts,
                          const Index64& parents,
                          int64_t outlength,
                          bool mask,
                          bool keepdims) const {
    util::dtype dtype = reducer.preferred_dtype();
    std::string format = util::dtype_to_format(dtype);
    int64_t itemsize = util::dtype_to_itemsize(dtype);
    ContentPtr asnumpy = toNumpyArray(format, itemsize, dtype);
    return asnumpy.get()->reduce_next(reducer,
                                      negaxis,
                                      starts,
                                      shifts,
                                      parents,
                                      outlength,
                                      mask,
                                      keepdims);
  }

  const ContentPtr
  EmptyArray::localindex(int64_t axis, int64_t depth) const {
    return std::make_shared<NumpyArray>(Index64(0));
  }

  const ContentPtr
  EmptyArray::combinations(int64_t n,
                           bool replacement,
                           const util::RecordLookupPtr& recordlookup,
                           const util::Parameters& parameters,
                           int64_t axis,
                           int64_t depth) const {
    if (n < 1) {
      throw std::invalid_argument(
        std::string("in combinations, 'n' must be at least 1") + FILENAME(__LINE__));
    }
    return std::make_shared<EmptyArray>(identities_, util::Parameters());
  }

  const ContentPtr
  EmptyArray::sort_next(int64_t negaxis,
                        const Index64& starts,
                        const Index64& parents,
                        int64_t outlength,
                        bool ascending,
                        bool stable,
                        bool keepdims) const {
    ContentPtr asnumpy = toNumpyArray("d", 8, util::dtype::float64);
    return asnumpy.get()->sort_next(negaxis,
                                    starts,
                                    parents,
                                    outlength,
                                    ascending,
                                    stable,
                                    keepdims);
  }

  const ContentPtr
  EmptyArray::argsort_next(int64_t negaxis,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength,
                           bool ascending,
                           bool stable,
                           bool keepdims) const {
    ContentPtr asnumpy = toNumpyArray("d", 8, util::dtype::float64);
    ContentPtr out = asnumpy.get()->argsort_next(negaxis,
                                                 starts,
                                                 parents,
                                                 outlength,
                                                 ascending,
                                                 stable,
                                                 keepdims);
    return out;
  }

  const ContentPtr
  EmptyArray::getitem_next(const SliceAt& at,
                           const Slice& tail,
                           const Index64& advanced) const {
    util::handle_error(
      failure("too many dimensions in slice",
              kSliceNone,
              kSliceNone,
              FILENAME_C(__LINE__)),
      classname(),
      identities_.get());
    return ContentPtr(nullptr);  // make Windows compiler happy
  }

  const ContentPtr
  EmptyArray::getitem_next(const SliceRange& range,
                           const Slice& tail,
                           const Index64& advanced) const {
    util::handle_error(
      failure("too many dimensions in slice",
              kSliceNone,
              kSliceNone,
              FILENAME_C(__LINE__)),
      classname(),
      identities_.get());
    return ContentPtr(nullptr);  // make Windows compiler happy
  }

  const ContentPtr
  EmptyArray::getitem_next(const SliceArray64& array,
                           const Slice& tail,
                           const Index64& advanced) const {
    util::handle_error(
      failure("too many dimensions in slice",
              kSliceNone,
              kSliceNone,
              FILENAME_C(__LINE__)),
      classname(),
      identities_.get());
    return ContentPtr(nullptr);  // make Windows compiler happy
  }

  const ContentPtr
  EmptyArray::getitem_next(const SliceField& field,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::invalid_argument(
      std::string("cannot slice ") + classname()
      + std::string(" by a field name because it has no fields")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  EmptyArray::getitem_next(const SliceFields& fields,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::invalid_argument(
      std::string("cannot slice ") + classname()
      + std::string(" by field names because it has no fields")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  EmptyArray::getitem_next(const SliceJagged64& jagged,
                           const Slice& tail,
                           const Index64& advanced) const {
    if (advanced.length() != 0) {
      throw std::invalid_argument(
        std::string("cannot mix jagged slice with NumPy-style advanced indexing")
        + FILENAME(__LINE__));
    }
    throw std::runtime_error(
      std::string("FIXME: EmptyArray::getitem_next(jagged)") + FILENAME(__LINE__));
  }

  const ContentPtr
  EmptyArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceArray64& slicecontent,
                                  const Slice& tail) const {
    throw std::runtime_error(
      std::string("undefined operation: EmptyArray::getitem_next_jagged(array)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  EmptyArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceMissing64& slicecontent,
                                  const Slice& tail) const {
    throw std::runtime_error(
      std::string("undefined operation: EmptyArray::getitem_next_jagged(missing)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  EmptyArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceJagged64& slicecontent,
                                  const Slice& tail) const {
    throw std::runtime_error(
      std::string("undefined operation: EmptyArray::getitem_next_jagged(jagged)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  EmptyArray::copy_to(kernel::lib ptr_lib) const {
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->copy_to(ptr_lib);
    }
    return std::make_shared<EmptyArray>(identities,
                                        parameters_);
  }

  const ContentPtr
  EmptyArray::numbers_to_type(const std::string& name) const {
    return std::make_shared<EmptyArray>(identities(),
                                        parameters());
  }

}
