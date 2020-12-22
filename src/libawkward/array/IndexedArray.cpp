// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/IndexedArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/IndexedArray.cpp", line)

#include <sstream>
#include <type_traits>

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"
#include "awkward/type/OptionType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/Slice.h"
#include "awkward/array/None.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/VirtualArray.h"

#define AWKWARD_INDEXEDARRAY_NO_EXTERN_TEMPLATE
#include "awkward/array/IndexedArray.h"

namespace {
  std::vector<std::string> valid_parameters = { "\"categorical\"" };
  std::vector<std::string> non_valid_parameters = {};
}

namespace awkward {
  ////////// IndexedForm

  IndexedForm::IndexedForm(bool has_identities,
                           const util::Parameters& parameters,
                           const FormKey& form_key,
                           Index::Form index,
                           const FormPtr& content)
      : Form(has_identities, parameters, form_key)
      , index_(index)
      , content_(content) { }

  Index::Form
  IndexedForm::index() const {
    return index_;
  }

  const FormPtr
  IndexedForm::content() const {
    return content_;
  }

  const TypePtr
  IndexedForm::type(const util::TypeStrs& typestrs) const {
    TypePtr out = content_.get()->type(typestrs);
    if (out.get()->parameters().empty()  &&  !parameters_.empty()) {
      out.get()->setparameters(parameters_);
      if (parameter_equals("__array__", "\"categorical\"")) {
        out.get()->setparameter("__array__", "null");
        out.get()->setparameter("__categorical__", "true");
      }
    }
    else if (!out.get()->parameters().empty()  &&  !parameters_.empty()) {
      for (auto p : parameters_) {
        if (p.first != std::string("__array__")) {
          out.get()->setparameter(p.first, p.second);
        }
      }
      if (parameter_equals("__array__", "\"categorical\"")) {
        out.get()->setparameter("__categorical__", "true");
      }
    }
    return out;
  }

  void
  IndexedForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    if (index_ == Index::Form::i32) {
      builder.string("IndexedArray32");
    }
    else if (index_ == Index::Form::u32) {
      builder.string("IndexedArrayU32");
    }
    else if (index_ == Index::Form::i64) {
      builder.string("IndexedArray64");
    }
    else {
      builder.string("UnrecognizedIndexedArray");
    }
    builder.field("index");
    builder.string(Index::form2str(index_));
    builder.field("content");
    content_.get()->tojson_part(builder, verbose);
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  IndexedForm::shallow_copy() const {
    return std::make_shared<IndexedForm>(has_identities_,
                                         parameters_,
                                         form_key_,
                                         index_,
                                         content_);
  }

  const std::string
  IndexedForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  bool
  IndexedForm::purelist_isregular() const {
    return content_.get()->purelist_isregular();
  }

  int64_t
  IndexedForm::purelist_depth() const {
    return content_.get()->purelist_depth();
  }

  bool
  IndexedForm::dimension_optiontype() const {
    return false;
  }

  const std::pair<int64_t, int64_t>
  IndexedForm::minmax_depth() const {
    return content_.get()->minmax_depth();
  }

  const std::pair<bool, int64_t>
  IndexedForm::branch_depth() const {
    return content_.get()->branch_depth();
  }

  int64_t
  IndexedForm::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  IndexedForm::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  IndexedForm::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  IndexedForm::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  IndexedForm::keys() const {
    return content_.get()->keys();
  }

  bool
  IndexedForm::equal(const FormPtr& other,
                     bool check_identities,
                     bool check_parameters,
                     bool check_form_key,
                     bool compatibility_check) const {
    if (compatibility_check) {
      if (VirtualForm* raw = dynamic_cast<VirtualForm*>(other.get())) {
        if (raw->form().get() != nullptr) {
          return equal(raw->form(),
                       check_identities,
                       check_parameters,
                       check_form_key,
                       compatibility_check);
        }
      }
    }

    if (check_identities  &&
        has_identities_ != other.get()->has_identities()) {
      return false;
    }
    if (check_parameters  &&
        !util::parameters_equal(parameters_, other.get()->parameters(), false)) {
      return false;
    }
    if (check_form_key  &&
        !form_key_equals(other.get()->form_key())) {
      return false;
    }
    if (IndexedForm* t = dynamic_cast<IndexedForm*>(other.get())) {
      return (index_ == t->index()  &&
              content_.get()->equal(t->content(),
                                    check_identities,
                                    check_parameters,
                                    check_form_key,
                                    compatibility_check));
    }
    else {
      return false;
    }
  }

  const FormPtr
  IndexedForm::getitem_field(const std::string& key) const {
    return std::make_shared<IndexedForm>(
      has_identities_,
      util::Parameters(),
      FormKey(nullptr),
      index_,
      content_.get()->getitem_field(key));
  }

  ////////// IndexedOptionForm

  IndexedOptionForm::IndexedOptionForm(bool has_identities,
                                       const util::Parameters& parameters,
                                       const FormKey& form_key,
                                       Index::Form index,
                                       const FormPtr& content)
      : Form(has_identities, parameters, form_key)
      , index_(index)
      , content_(content) { }

  Index::Form
  IndexedOptionForm::index() const {
    return index_;
  }

  const FormPtr
  IndexedOptionForm::content() const {
    return content_;
  }

  const TypePtr
  IndexedOptionForm::type(const util::TypeStrs& typestrs) const {
    TypePtr out = std::make_shared<OptionType>(
                    parameters_,
                    util::gettypestr(parameters_, typestrs),
                    content_.get()->type(typestrs));
    if (out.get()->parameter_equals("__array__", "\"categorical\"")) {
      out.get()->setparameter("__array__", "null");
      out.get()->setparameter("__categorical__", "true");
    }
    return out;
  }

  void
  IndexedOptionForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    if (index_ == Index::Form::i32) {
      builder.string("IndexedOptionArray32");
    }
    else if (index_ == Index::Form::i64) {
      builder.string("IndexedOptionArray64");
    }
    else {
      builder.string("UnrecognizedIndexedOptionArray");
    }
    builder.field("index");
    builder.string(Index::form2str(index_));
    builder.field("content");
    content_.get()->tojson_part(builder, verbose);
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  IndexedOptionForm::shallow_copy() const {
    return std::make_shared<IndexedOptionForm>(has_identities_,
                                               parameters_,
                                               form_key_,
                                               index_,
                                               content_);
  }

  const std::string
  IndexedOptionForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  bool
  IndexedOptionForm::purelist_isregular() const {
    return content_.get()->purelist_isregular();
  }

  int64_t
  IndexedOptionForm::purelist_depth() const {
    return content_.get()->purelist_depth();
  }

  bool
  IndexedOptionForm::dimension_optiontype() const {
    return true;
  }

  const std::pair<int64_t, int64_t>
  IndexedOptionForm::minmax_depth() const {
    return content_.get()->minmax_depth();
  }

  const std::pair<bool, int64_t>
  IndexedOptionForm::branch_depth() const {
    return content_.get()->branch_depth();
  }

  int64_t
  IndexedOptionForm::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  IndexedOptionForm::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  IndexedOptionForm::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  IndexedOptionForm::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  IndexedOptionForm::keys() const {
    return content_.get()->keys();
  }

  bool
  IndexedOptionForm::equal(const FormPtr& other,
                           bool check_identities,
                           bool check_parameters,
                           bool check_form_key,
                           bool compatibility_check) const {
    if (compatibility_check) {
      if (VirtualForm* raw = dynamic_cast<VirtualForm*>(other.get())) {
        if (raw->form().get() != nullptr) {
          return equal(raw->form(),
                       check_identities,
                       check_parameters,
                       check_form_key,
                       compatibility_check);
        }
      }
    }

    if (check_identities  &&
        has_identities_ != other.get()->has_identities()) {
      return false;
    }
    if (check_parameters  &&
        !util::parameters_equal(parameters_, other.get()->parameters(), false)) {
      return false;
    }
    if (check_form_key  &&
        !form_key_equals(other.get()->form_key())) {
      return false;
    }
    if (IndexedOptionForm* t = dynamic_cast<IndexedOptionForm*>(other.get())) {
      return (index_ == t->index()  &&
              content_.get()->equal(t->content(),
                                    check_identities,
                                    check_parameters,
                                    check_form_key,
                                    compatibility_check));
    }
    else {
      return false;
    }
  }

  const FormPtr
  IndexedOptionForm::getitem_field(const std::string& key) const {
    return std::make_shared<IndexedOptionForm>(
      has_identities_,
      util::Parameters(),
      FormKey(nullptr),
      index_,
      content_.get()->getitem_field(key));
  }

  ////////// IndexedArray

  template <typename T, bool ISOPTION>
  IndexedArrayOf<T, ISOPTION>::IndexedArrayOf(
    const IdentitiesPtr& identities,
    const util::Parameters& parameters,
    const IndexOf<T>& index,
    const ContentPtr& content)
      : Content(identities, parameters)
      , index_(index)
      , content_(content) { }

  template <typename T, bool ISOPTION>
  const IndexOf<T>
  IndexedArrayOf<T, ISOPTION>::index() const {
    return index_;
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::content() const {
    return content_;
  }

  template <typename T, bool ISOPTION>
  bool
  IndexedArrayOf<T, ISOPTION>::isoption() const {
    return ISOPTION;
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::project() const {
    if (ISOPTION) {
      int64_t numnull;
      struct Error err1 = kernel::IndexedArray_numnull<T>(
        kernel::lib::cpu,   // DERIVE
        &numnull,
        index_.data(),
        index_.length());
      util::handle_error(err1, classname(), identities_.get());

      Index64 nextcarry(length() - numnull);
      struct Error err2 = kernel::IndexedArray_flatten_nextcarry_64<T>(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        index_.data(),
        index_.length(),
        content_.get()->length());
      util::handle_error(err2, classname(), identities_.get());

      return content_.get()->carry(nextcarry, false);
    }
    else {
      Index64 nextcarry(length());
      struct Error err = kernel::IndexedArray_getitem_nextcarry_64<T>(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        index_.data(),
        index_.length(),
        content_.get()->length());
      util::handle_error(err, classname(), identities_.get());

      return content_.get()->carry(nextcarry, false);
    }
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::project(const Index8& mask) const {
    if (index_.length() != mask.length()) {
      throw std::invalid_argument(
        std::string("mask length (") + std::to_string(mask.length())
        + std::string(") is not equal to ") + classname()
        + std::string(" length (") + std::to_string(index_.length())
        + std::string(")") + FILENAME(__LINE__));
    }

    Index64 nextindex(index_.length());
    struct Error err = kernel::IndexedArray_overlay_mask8_to64<T>(
      kernel::lib::cpu,   // DERIVE
      nextindex.data(),
      mask.data(),
      index_.data(),
      index_.length());
    util::handle_error(err, classname(), identities_.get());

    IndexedOptionArray64 next(identities_, parameters_, nextindex, content_);
    return next.project();
  }

  template <typename T, bool ISOPTION>
  const Index8
  IndexedArrayOf<T, ISOPTION>::bytemask() const {
    if (ISOPTION) {
      Index8 out(index_.length());
      struct Error err = kernel::IndexedArray_mask8(
        kernel::lib::cpu,   // DERIVE
        out.data(),
        index_.data(),
        index_.length());
      util::handle_error(err, classname(), identities_.get());
      return out;
    }
    else {
      Index8 out(index_.length());
      struct Error err = kernel::zero_mask8(
        kernel::lib::cpu,   // DERIVE
        out.data(),
        index_.length());
      util::handle_error(err, classname(), identities_.get());
      return out;
    }
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::simplify_optiontype() const {
    if (ISOPTION) {
      if (IndexedArray32* rawcontent =
          dynamic_cast<IndexedArray32*>(content_.get())) {
        Index32 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify32_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else if (IndexedArrayU32* rawcontent =
               dynamic_cast<IndexedArrayU32*>(content_.get())) {
        IndexU32 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplifyU32_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else if (IndexedArray64* rawcontent =
               dynamic_cast<IndexedArray64*>(content_.get())) {
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify64_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else if (IndexedOptionArray32* rawcontent =
               dynamic_cast<IndexedOptionArray32*>(content_.get())) {
        Index32 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify32_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else if (IndexedOptionArray64* rawcontent =
               dynamic_cast<IndexedOptionArray64*>(content_.get())) {
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify64_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else if (ByteMaskedArray* step1 =
               dynamic_cast<ByteMaskedArray*>(content_.get())) {
        ContentPtr step2 = step1->toIndexedOptionArray64();
        IndexedOptionArray64* rawcontent =
          dynamic_cast<IndexedOptionArray64*>(step2.get());
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify64_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else if (BitMaskedArray* step1 =
               dynamic_cast<BitMaskedArray*>(content_.get())) {
        ContentPtr step2 = step1->toIndexedOptionArray64();
        IndexedOptionArray64* rawcontent =
          dynamic_cast<IndexedOptionArray64*>(step2.get());
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify64_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else if (UnmaskedArray* step1 =
               dynamic_cast<UnmaskedArray*>(content_.get())) {
        ContentPtr step2 = step1->toIndexedOptionArray64();
        IndexedOptionArray64* rawcontent =
          dynamic_cast<IndexedOptionArray64*>(step2.get());
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify64_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else {
        return shallow_copy();
      }
    }
    else {
      if (IndexedArray32* rawcontent =
          dynamic_cast<IndexedArray32*>(content_.get())) {
        Index32 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify32_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedArray64>(identities_,
                                                parameters_,
                                                result,
                                                rawcontent->content());
      }
      else if (IndexedArrayU32* rawcontent =
               dynamic_cast<IndexedArrayU32*>(content_.get())) {
        IndexU32 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplifyU32_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedArray64>(identities_,
                                                parameters_,
                                                result,
                                                rawcontent->content());
      }
      else if (IndexedArray64* rawcontent =
               dynamic_cast<IndexedArray64*>(content_.get())) {
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify64_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedArray64>(identities_,
                                                parameters_,
                                                result,
                                                rawcontent->content());
      }
      else if (IndexedOptionArray32* rawcontent =
               dynamic_cast<IndexedOptionArray32*>(content_.get())) {
        Index32 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify32_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else if (IndexedOptionArray64* rawcontent =
               dynamic_cast<IndexedOptionArray64*>(content_.get())) {
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify64_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else if (ByteMaskedArray* step1 =
               dynamic_cast<ByteMaskedArray*>(content_.get())) {
        ContentPtr step2 = step1->toIndexedOptionArray64();
        IndexedOptionArray64* rawcontent =
          dynamic_cast<IndexedOptionArray64*>(step2.get());
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify64_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else if (BitMaskedArray* step1 =
               dynamic_cast<BitMaskedArray*>(content_.get())) {
        ContentPtr step2 = step1->toIndexedOptionArray64();
        IndexedOptionArray64* rawcontent =
          dynamic_cast<IndexedOptionArray64*>(step2.get());
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify64_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else if (UnmaskedArray* step1 =
               dynamic_cast<UnmaskedArray*>(content_.get())) {
        ContentPtr step2 = step1->toIndexedOptionArray64();
        IndexedOptionArray64* rawcontent =
          dynamic_cast<IndexedOptionArray64*>(step2.get());
        Index64 inner = rawcontent->index();
        Index64 result(index_.length());
        struct Error err = kernel::IndexedArray_simplify64_to64(
          kernel::lib::cpu,   // DERIVE
          result.data(),
          index_.data(),
          index_.length(),
          inner.data(),
          inner.length());
        util::handle_error(err, classname(), identities_.get());
        return std::make_shared<IndexedOptionArray64>(identities_,
                                                      parameters_,
                                                      result,
                                                      rawcontent->content());
      }
      else {
        return shallow_copy();
      }
    }
  }

  template <typename T, bool ISOPTION>
  T
  IndexedArrayOf<T, ISOPTION>::index_at_nowrap(int64_t at) const {
    return index_.getitem_at_nowrap(at);
  }

  template <typename T, bool ISOPTION>
  const std::string
  IndexedArrayOf<T, ISOPTION>::classname() const {
    if (ISOPTION) {
      if (std::is_same<T, int32_t>::value) {
        return "IndexedOptionArray32";
      }
      else if (std::is_same<T, int64_t>::value) {
        return "IndexedOptionArray64";
      }
    }
    else {
      if (std::is_same<T, int32_t>::value) {
        return "IndexedArray32";
      }
      else if (std::is_same<T, uint32_t>::value) {
        return "IndexedArrayU32";
      }
      else if (std::is_same<T, int64_t>::value) {
        return "IndexedArray64";
      }
    }
    return "UnrecognizedIndexedArray";
  }

  template <typename T, bool ISOPTION>
  void
  IndexedArrayOf<T, ISOPTION>::setidentities(const IdentitiesPtr& identities) {
    if (identities.get() == nullptr) {
      content_.get()->setidentities(identities);
    }
    else {
      if (length() != identities.get()->length()) {
        util::handle_error(
          failure("content and its identities must have the same length",
                  kSliceNone,
                  kSliceNone,
                  FILENAME_C(__LINE__)),
          classname(),
          identities_.get());
      }
      IdentitiesPtr bigidentities = identities;
      if (content_.get()->length() > kMaxInt32  ||
          !std::is_same<T, int32_t>::value) {
        bigidentities = identities.get()->to64();
      }
      if (Identities32* rawidentities =
          dynamic_cast<Identities32*>(bigidentities.get())) {
        bool uniquecontents;
        IdentitiesPtr subidentities =
          std::make_shared<Identities32>(Identities::newref(),
                                         rawidentities->fieldloc(),
                                         rawidentities->width(),
                                         content_.get()->length());
        Identities32* rawsubidentitites =
          reinterpret_cast<Identities32*>(subidentities.get());
        struct Error err = kernel::Identities_from_IndexedArray<int32_t, T>(
          kernel::lib::cpu,   // DERIVE
          &uniquecontents,
          rawsubidentitites->data(),
          rawidentities->data(),
          index_.data(),
          content_.get()->length(),
          length(),
          rawidentities->width());
        util::handle_error(err, classname(), identities_.get());
        if (uniquecontents) {
          content_.get()->setidentities(subidentities);
        }
        else {
          content_.get()->setidentities(Identities::none());
        }
      }
      else if (Identities64* rawidentities =
               dynamic_cast<Identities64*>(bigidentities.get())) {
        bool uniquecontents;
        IdentitiesPtr subidentities =
          std::make_shared<Identities64>(Identities::newref(),
                                         rawidentities->fieldloc(),
                                         rawidentities->width(),
                                         content_.get()->length());
        Identities64* rawsubidentitites =
          reinterpret_cast<Identities64*>(subidentities.get());
        struct Error err = kernel::Identities_from_IndexedArray<int64_t, T>(
          kernel::lib::cpu,   // DERIVE
          &uniquecontents,
          rawsubidentitites->data(),
          rawidentities->data(),
          index_.data(),
          content_.get()->length(),
          length(),
          rawidentities->width());
        util::handle_error(err, classname(), identities_.get());
        if (uniquecontents) {
          content_.get()->setidentities(subidentities);
        }
        else {
          content_.get()->setidentities(Identities::none());
        }
      }
      else {
        throw std::runtime_error(
          std::string("unrecognized Identities specialization") + FILENAME(__LINE__));
      }
    }
    identities_ = identities;
  }

  template <typename T, bool ISOPTION>
  void
  IndexedArrayOf<T, ISOPTION>::setidentities() {
    if (length() <= kMaxInt32) {
      IdentitiesPtr newidentities =
        std::make_shared<Identities32>(Identities::newref(),
                                       Identities::FieldLoc(),
                                       1,
                                       length());
      Identities32* rawidentities =
        reinterpret_cast<Identities32*>(newidentities.get());
      struct Error err = kernel::new_Identities<int32_t>(
        kernel::lib::cpu,   // DERIVE
        rawidentities->data(),
        length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
    else {
      IdentitiesPtr newidentities =
        std::make_shared<Identities64>(Identities::newref(),
                                       Identities::FieldLoc(),
                                       1,
                                       length());
      Identities64* rawidentities =
        reinterpret_cast<Identities64*>(newidentities.get());
      struct Error err = kernel::new_Identities<int64_t>(
        kernel::lib::cpu,   // DERIVE
        rawidentities->data(),
        length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
  }

  template <typename T, bool ISOPTION>
  const TypePtr
  IndexedArrayOf<T, ISOPTION>::type(const util::TypeStrs& typestrs) const {
    return form(true).get()->type(typestrs);
  }

  template <typename T, bool ISOPTION>
  const FormPtr
  IndexedArrayOf<T, ISOPTION>::form(bool materialize) const {
    if (ISOPTION) {
      return std::make_shared<IndexedOptionForm>(
                                           identities_.get() != nullptr,
                                           parameters_,
                                           FormKey(nullptr),
                                           index_.form(),
                                           content_.get()->form(materialize));
    }
    else {
      return std::make_shared<IndexedForm>(identities_.get() != nullptr,
                                           parameters_,
                                           FormKey(nullptr),
                                           index_.form(),
                                           content_.get()->form(materialize));
    }
  }

  template <typename T, bool ISOPTION>
  kernel::lib
  IndexedArrayOf<T, ISOPTION>::kernels() const {
    return kernels_compare(index_.ptr_lib(), content_);
  }

  template <typename T, bool ISOPTION>
  void
  IndexedArrayOf<T, ISOPTION>::caches(std::vector<ArrayCachePtr>& out) const {
    content_.get()->caches(out);
  }

  template <typename T, bool ISOPTION>
  const std::string
  IndexedArrayOf<T, ISOPTION>::tostring_part(const std::string& indent,
                                             const std::string& pre,
                                             const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << ">\n";
    if (identities_.get() != nullptr) {
      out << identities_.get()->tostring_part(
               indent + std::string("    "), "", "\n");
    }
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    out << index_.tostring_part(
             indent + std::string("    "), "<index>", "</index>\n");
    out << content_.get()->tostring_part(
             indent + std::string("    "), "<content>", "</content>\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  template <typename T, bool ISOPTION>
  void
  IndexedArrayOf<T, ISOPTION>::tojson_part(ToJson& builder,
                                           bool include_beginendlist) const {
    int64_t len = length();
    check_for_iteration();
    if (include_beginendlist) {
      builder.beginlist();
    }
    for (int64_t i = 0;  i < len;  i++) {
      getitem_at_nowrap(i).get()->tojson_part(builder, true);
    }
    if (include_beginendlist) {
      builder.endlist();
    }
  }

  template <typename T, bool ISOPTION>
  void
  IndexedArrayOf<T, ISOPTION>::nbytes_part(std::map<size_t,
                                           int64_t>& largest) const {
    index_.nbytes_part(largest);
    content_.get()->nbytes_part(largest);
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  template <typename T, bool ISOPTION>
  int64_t
  IndexedArrayOf<T, ISOPTION>::length() const {
    return index_.length();
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::shallow_copy() const {
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities_,
                                                         parameters_,
                                                         index_,
                                                         content_);
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::deep_copy(bool copyarrays,
                                         bool copyindexes,
                                         bool copyidentities) const {
    IndexOf<T> index = copyindexes ? index_.deep_copy() : index_;
    ContentPtr content = content_.get()->deep_copy(copyarrays,
                                                   copyindexes,
                                                   copyidentities);
    IdentitiesPtr identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities,
                                                         parameters_,
                                                         index,
                                                         content);
  }

  template <typename T, bool ISOPTION>
  void
  IndexedArrayOf<T, ISOPTION>::check_for_iteration() const {
    if (identities_.get() != nullptr  &&
        identities_.get()->length() < index_.length()) {
      util::handle_error(
        failure("len(identities) < len(array)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        identities_.get()->classname(),
        nullptr);
    }
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_nothing() const {
    return content_.get()->getitem_range_nowrap(0, 0);
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += index_.length();
    }
    if (!(0 <= regular_at  &&  regular_at < index_.length())) {
      util::handle_error(
        failure("index out of range", kSliceNone, at, FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_at_nowrap(int64_t at) const {
    int64_t index = (int64_t)index_.getitem_at_nowrap(at);
    if (index < 0) {
      if (ISOPTION) {
        return none;
      }
      else {
        util::handle_error(
          failure("index[i] < 0", kSliceNone, at, FILENAME_C(__LINE__)),
          classname(),
          identities_.get());
      }
    }
    int64_t lencontent = content_.get()->length();
    if (index >= lencontent) {
      util::handle_error(
        failure("index[i] >= len(content)",
                kSliceNone,
                at,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    return content_.get()->getitem_at_nowrap(index);
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_range(int64_t start,
                                             int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    kernel::regularize_rangeslice(&regular_start, &regular_stop,
      true, start != Slice::none(), stop != Slice::none(), index_.length());
    if (identities_.get() != nullptr  &&
        regular_stop > identities_.get()->length()) {
      util::handle_error(
        failure("index out of range",
                kSliceNone,
                stop,
                FILENAME_C(__LINE__)),
        identities_.get()->classname(),
        nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_range_nowrap(int64_t start,
                                                    int64_t stop) const {
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(
      identities,
      parameters_,
      index_.getitem_range_nowrap(start, stop),
      content_);
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_field(const std::string& key) const {
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(
      identities_,
      util::Parameters(),
      index_,
      content_.get()->getitem_field(key));
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_fields(
    const std::vector<std::string>& keys) const {
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(
      identities_,
      util::Parameters(),
      index_,
      content_.get()->getitem_fields(keys));
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceItemPtr& head,
                                            const Slice& tail,
                                            const Index64& advanced) const {
    if (head.get() == nullptr) {
      return shallow_copy();
    }
    else if (dynamic_cast<SliceAt*>(head.get())  ||
             dynamic_cast<SliceRange*>(head.get())  ||
             dynamic_cast<SliceArray64*>(head.get())  ||
             dynamic_cast<SliceJagged64*>(head.get())) {
      if (ISOPTION) {
        int64_t numnull;
        std::pair<Index64, IndexOf<T>> pair = nextcarry_outindex(numnull);
        Index64 nextcarry = pair.first;
        IndexOf<T> outindex = pair.second;

        ContentPtr next = content_.get()->carry(nextcarry, true);
        ContentPtr out = next.get()->getitem_next(head, tail, advanced);
        IndexedArrayOf<T, ISOPTION> out2(identities_,
                                         parameters_,
                                         outindex,
                                         out);
        return out2.simplify_optiontype();
      }
      else {
        Index64 nextcarry(length());
        struct Error err = kernel::IndexedArray_getitem_nextcarry_64<T>(
          kernel::lib::cpu,   // DERIVE
          nextcarry.data(),
          index_.data(),
          index_.length(),
          content_.get()->length());
        util::handle_error(err, classname(), identities_.get());

        // must be an eager carry (allow_lazy = false) to avoid infinite loop
        ContentPtr next = content_.get()->carry(nextcarry, false);
        return next.get()->getitem_next(head, tail, advanced);
      }
    }
    else if (SliceEllipsis* ellipsis =
             dynamic_cast<SliceEllipsis*>(head.get())) {
      return Content::getitem_next(*ellipsis, tail, advanced);
    }
    else if (SliceNewAxis* newaxis =
             dynamic_cast<SliceNewAxis*>(head.get())) {
      return Content::getitem_next(*newaxis, tail, advanced);
    }
    else if (SliceField* field =
             dynamic_cast<SliceField*>(head.get())) {
      return Content::getitem_next(*field, tail, advanced);
    }
    else if (SliceFields* fields =
             dynamic_cast<SliceFields*>(head.get())) {
      return Content::getitem_next(*fields, tail, advanced);
    }
    else if (SliceMissing64* missing =
             dynamic_cast<SliceMissing64*>(head.get())) {
      return Content::getitem_next(*missing, tail, advanced);
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized slice type") + FILENAME(__LINE__));
    }
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::carry(const Index64& carry, bool allow_lazy) const {
    IndexOf<T> nextindex(carry.length());
    struct Error err = kernel::IndexedArray_getitem_carry_64<T>(
      kernel::lib::cpu,   // DERIVE
      nextindex.data(),
      index_.data(),
      carry.data(),
      index_.length(),
      carry.length());
    util::handle_error(err, classname(), identities_.get());
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities,
                                                         parameters_,
                                                         nextindex,
                                                         content_);
  }

  template <typename T, bool ISOPTION>
  int64_t
  IndexedArrayOf<T, ISOPTION>::numfields() const {
    return content_.get()->numfields();
  }

  template <typename T, bool ISOPTION>
  int64_t
  IndexedArrayOf<T, ISOPTION>::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  template <typename T, bool ISOPTION>
  const std::string
  IndexedArrayOf<T, ISOPTION>::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  template <typename T, bool ISOPTION>
  bool
  IndexedArrayOf<T, ISOPTION>::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  template <typename T, bool ISOPTION>
  const std::vector<std::string>
  IndexedArrayOf<T, ISOPTION>::keys() const {
    return content_.get()->keys();
  }

  template <typename T, bool ISOPTION>
  const std::string
  IndexedArrayOf<T, ISOPTION>::validityerror(const std::string& path) const {
    if (parameters_.size() != 0) {
      bool result = std::any_of(valid_parameters.begin(), valid_parameters.end(),
        [&](const std::string& i){
          return (parameter_equals("__array__", i)) ? true : false;
        });
      if (result) {
        if (parameter_equals("__array__", "\"categorical\"")  &&  !is_unique()) {
          return (std::string("at ") + path + std::string(" (") + classname()
                  + std::string("): not unique __array__ can not be ")
                  + util::parameter_asstring(parameters_, "__array__")
                  + FILENAME(__LINE__));
        }
      }
      result = std::any_of(non_valid_parameters.begin(), non_valid_parameters.end(),
        [&](const std::string& i){
          return (parameter_equals("__array__", i)) ? true : false;
        });
      if (result) {
        return (std::string("at ") + path + std::string(" (") + classname()
                + std::string("): __array__ can not be ")
                + util::parameter_asstring(parameters_, "__array__")
                + FILENAME(__LINE__));
      }
    }
    struct Error err = kernel::IndexedArray_validity<T>(
      kernel::lib::cpu,   // DERIVE
      index_.data(),
      index_.length(),
      content_.get()->length(),
      ISOPTION);
    if (err.str == nullptr) {
      return content_.get()->validityerror(path + std::string(".content"));
    }
    else {
      return (std::string("at ") + path + std::string(" (") + classname()
              + std::string("): ") + std::string(err.str)
              + std::string(" at i=") + std::to_string(err.identity)
              + std::string(err.filename == nullptr ? "" : err.filename));
    }
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::shallow_simplify() const {
    return simplify_optiontype();
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::num(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      Index64 out(1);
      out.setitem_at_nowrap(0, length());
      return NumpyArray(out).getitem_at_nowrap(0);
    }
    else if (ISOPTION) {
      int64_t numnull;
      std::pair<Index64, IndexOf<T>> pair = nextcarry_outindex(numnull);
      Index64 nextcarry = pair.first;
      IndexOf<T> outindex = pair.second;

      ContentPtr next = content_.get()->carry(nextcarry, false);
      ContentPtr out = next.get()->num(posaxis, depth);
      IndexedArrayOf<T, ISOPTION> out2(Identities::none(),
                                       util::Parameters(),
                                       outindex,
                                       out);
      return out2.simplify_optiontype();
    }
    else {
      return project().get()->num(posaxis, depth);
    }
  }

  template <typename T, bool ISOPTION>
  const std::pair<Index64, ContentPtr>
  IndexedArrayOf<T, ISOPTION>::offsets_and_flattened(int64_t axis,
                                                     int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      throw std::invalid_argument(
        std::string("axis=0 not allowed for flatten") + FILENAME(__LINE__));
    }
    else if (ISOPTION) {
      int64_t numnull;
      std::pair<Index64, IndexOf<T>> pair = nextcarry_outindex(numnull);
      Index64 nextcarry = pair.first;
      IndexOf<T> outindex = pair.second;

      ContentPtr next = content_.get()->carry(nextcarry, false);

      std::pair<Index64, ContentPtr> offsets_flattened =
        next.get()->offsets_and_flattened(posaxis, depth);
      Index64 offsets = offsets_flattened.first;
      ContentPtr flattened = offsets_flattened.second;

      if (offsets.length() == 0) {
        return std::pair<Index64, ContentPtr>(
          offsets,
          std::make_shared<IndexedArrayOf<T, ISOPTION>>(Identities::none(),
                                                        util::Parameters(),
                                                        outindex,
                                                        flattened));
      }
      else {
        Index64 outoffsets(offsets.length() + numnull);
        struct Error err = kernel::IndexedArray_flatten_none2empty_64<T>(
          kernel::lib::cpu,   // DERIVE
          outoffsets.data(),
          outindex.data(),
          outindex.length(),
          offsets.data(),
          offsets.length());
        util::handle_error(err, classname(), identities_.get());
        return std::pair<Index64, ContentPtr>(outoffsets, flattened);
      }
    }
    else {
      return project().get()->offsets_and_flattened(posaxis, depth);
    }
  }

  template <typename T, bool ISOPTION>
  bool
  IndexedArrayOf<T, ISOPTION>::mergeable(const ContentPtr& other,
                                         bool mergebool) const {
    if (VirtualArray* raw = dynamic_cast<VirtualArray*>(other.get())) {
      return mergeable(raw->array(), mergebool);
    }

    if (!parameters_equal(other.get()->parameters(), false)) {
      return false;
    }

    if (dynamic_cast<EmptyArray*>(other.get())  ||
        dynamic_cast<UnionArray8_32*>(other.get())  ||
        dynamic_cast<UnionArray8_U32*>(other.get())  ||
        dynamic_cast<UnionArray8_64*>(other.get())) {
      return true;
    }

    if (IndexedArray32* rawother =
        dynamic_cast<IndexedArray32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (IndexedArrayU32* rawother =
             dynamic_cast<IndexedArrayU32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (IndexedArray64* rawother =
             dynamic_cast<IndexedArray64*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (IndexedOptionArray32* rawother =
             dynamic_cast<IndexedOptionArray32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (IndexedOptionArray64* rawother =
             dynamic_cast<IndexedOptionArray64*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ByteMaskedArray* rawother =
             dynamic_cast<ByteMaskedArray*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (BitMaskedArray* rawother =
             dynamic_cast<BitMaskedArray*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (UnmaskedArray* rawother =
             dynamic_cast<UnmaskedArray*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else {
      return content_.get()->mergeable(other, mergebool);
    }
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::reverse_merge(const ContentPtr& other) const {
    if (VirtualArray* raw = dynamic_cast<VirtualArray*>(other.get())) {
      return reverse_merge(raw->array());
    }

    int64_t theirlength = other.get()->length();
    int64_t mylength = length();
    Index64 index(theirlength + mylength);

    ContentPtr content = other.get()->merge(content_);
    struct Error err1 = kernel::IndexedArray_fill_to64_count(
      kernel::lib::cpu,   // DERIVE
      index.data(),
      0,
      theirlength,
      0);
    util::handle_error(err1, classname(), identities_.get());

    if (std::is_same<T, int32_t>::value) {
      struct Error err2 = kernel::IndexedArray_fill<int32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        theirlength,
        reinterpret_cast<int32_t*>(index_.data()),
        mylength,
        theirlength);
      util::handle_error(err2, classname(), identities_.get());
    }
    else if (std::is_same<T, uint32_t>::value) {
      struct Error err2 = kernel::IndexedArray_fill<uint32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        theirlength,
        reinterpret_cast<uint32_t*>(index_.data()),
        mylength,
        theirlength);
      util::handle_error(err2, classname(), identities_.get());
    }
    if (std::is_same<T, int64_t>::value) {
      struct Error err2 = kernel::IndexedArray_fill<int64_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        theirlength,
        reinterpret_cast<int64_t*>(index_.data()),
        mylength,
        theirlength);
      util::handle_error(err2, classname(), identities_.get());
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized IndexedArray specialization") + FILENAME(__LINE__));
    }

    util::Parameters parameters(parameters_);
    util::merge_parameters(parameters, other.get()->parameters());

    return std::make_shared<IndexedArrayOf<int64_t, ISOPTION>>(
      Identities::none(),
      parameters,
      index,
      content);
  }

  template <typename T, bool ISOPTION>
  const std::pair<ContentPtrVec, ContentPtrVec>
  IndexedArrayOf<T, ISOPTION>::merging_strategy(const ContentPtrVec& others) const {
    if (others.empty()) {
      throw std::invalid_argument(
        std::string("to merge this array with 'others', at least one other "
                    "must be provided") + FILENAME(__LINE__));
    }

    ContentPtrVec head;
    ContentPtrVec tail;

    head.push_back(shallow_copy());

    size_t i = 0;
    for (;  i < others.size();  i++) {
      ContentPtr other = others[i];
      if (dynamic_cast<UnionArray8_32*>(other.get())  ||
          dynamic_cast<UnionArray8_U32*>(other.get())  ||
          dynamic_cast<UnionArray8_64*>(other.get())) {
        break;
      }
      else if (VirtualArray* raw = dynamic_cast<VirtualArray*>(other.get())) {
        head.push_back(raw->array());
      }
      else {
        head.push_back(other);
      }
    }

    for (;  i < others.size();  i++) {
      ContentPtr other = others[i];
      tail.push_back(other);
    }

    return std::pair<ContentPtrVec, ContentPtrVec>(head, tail);
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::mergemany(const ContentPtrVec& others) const {
    if (others.empty()) {
      return shallow_copy();
    }

    std::pair<ContentPtrVec, ContentPtrVec> head_tail = merging_strategy(others);
    ContentPtrVec head = head_tail.first;
    ContentPtrVec tail = head_tail.second;

    int64_t total_length = 0;
    for (auto array : head) {
      total_length += array.get()->length();
    }

    kernel::lib ptr_lib = kernel::lib::cpu;   // DERIVE

    util::Parameters parameters(parameters_);
    bool is_option = false;
    ContentPtrVec contents;
    int64_t contentlength_so_far = 0;
    int64_t length_so_far = 0;
    Index64 nextindex(total_length);
    for (auto array : head) {
      util::merge_parameters(parameters, array.get()->parameters());

      if (ByteMaskedArray* raw = dynamic_cast<ByteMaskedArray*>(array.get())) {
        array = raw->toIndexedOptionArray64();
      }
      else if (BitMaskedArray* raw = dynamic_cast<BitMaskedArray*>(array.get())) {
        array = raw->toIndexedOptionArray64();
      }
      else if (UnmaskedArray* raw = dynamic_cast<UnmaskedArray*>(array.get())) {
        array = raw->toIndexedOptionArray64();
      }

      if (IndexedArray32* raw = dynamic_cast<IndexedArray32*>(array.get())) {
        is_option = false;
        contents.push_back(raw->content());
        Index32 array_index = raw->index();
        struct Error err = kernel::IndexedArray_fill<int32_t, int64_t>(
          ptr_lib,
          nextindex.data(),
          length_so_far,
          array_index.data(),
          array.get()->length(),
          contentlength_so_far);
        util::handle_error(err, array.get()->classname(), array.get()->identities().get());
        contentlength_so_far += raw->content().get()->length();
        length_so_far += array.get()->length();
      }
      else if (IndexedArrayU32* raw = dynamic_cast<IndexedArrayU32*>(array.get())) {
        is_option = false;
        contents.push_back(raw->content());
        IndexU32 array_index = raw->index();
        struct Error err = kernel::IndexedArray_fill<uint32_t, int64_t>(
          ptr_lib,
          nextindex.data(),
          length_so_far,
          array_index.data(),
          array.get()->length(),
          contentlength_so_far);
        util::handle_error(err, array.get()->classname(), array.get()->identities().get());
        contentlength_so_far += raw->content().get()->length();
        length_so_far += array.get()->length();
      }
      else if (IndexedArray64* raw = dynamic_cast<IndexedArray64*>(array.get())) {
        is_option = false;
        contents.push_back(raw->content());
        Index64 array_index = raw->index();
        struct Error err = kernel::IndexedArray_fill<int64_t, int64_t>(
          ptr_lib,
          nextindex.data(),
          length_so_far,
          array_index.data(),
          array.get()->length(),
          contentlength_so_far);
        util::handle_error(err, array.get()->classname(), array.get()->identities().get());
        contentlength_so_far += raw->content().get()->length();
        length_so_far += array.get()->length();
      }
      else if (IndexedOptionArray32* raw = dynamic_cast<IndexedOptionArray32*>(array.get())) {
        is_option = true;
        contents.push_back(raw->content());
        Index32 array_index = raw->index();
        struct Error err = kernel::IndexedArray_fill<int32_t, int64_t>(
          ptr_lib,
          nextindex.data(),
          length_so_far,
          array_index.data(),
          array.get()->length(),
          contentlength_so_far);
        util::handle_error(err, array.get()->classname(), array.get()->identities().get());
        contentlength_so_far += raw->content().get()->length();
        length_so_far += array.get()->length();
      }
      else if (IndexedOptionArray64* raw = dynamic_cast<IndexedOptionArray64*>(array.get())) {
        is_option = true;
        contents.push_back(raw->content());
        Index64 array_index = raw->index();
        struct Error err = kernel::IndexedArray_fill<int64_t, int64_t>(
          ptr_lib,
          nextindex.data(),
          length_so_far,
          array_index.data(),
          array.get()->length(),
          contentlength_so_far);
        util::handle_error(err, array.get()->classname(), array.get()->identities().get());
        contentlength_so_far += raw->content().get()->length();
        length_so_far += array.get()->length();
      }
      else if (EmptyArray* raw = dynamic_cast<EmptyArray*>(array.get())) {
        ;
      }
      else {
        contents.push_back(array);
        struct Error err = kernel::IndexedArray_fill_to64_count(
          ptr_lib,
          nextindex.data(),
          length_so_far,
          array.get()->length(),
          contentlength_so_far);
        util::handle_error(err, array.get()->classname(), array.get()->identities().get());
        contentlength_so_far += array.get()->length();
        length_so_far += array.get()->length();
      }
    }

    ContentPtrVec tail_contents(contents.begin() + 1, contents.end());
    ContentPtr nextcontent = contents[0].get()->mergemany(tail_contents);

    ContentPtr next(nullptr);
    if (is_option) {
      next = std::make_shared<IndexedOptionArray64>(Identities::none(),
                                                    parameters,
                                                    nextindex,
                                                    nextcontent);
    }
    else {
      next = std::make_shared<IndexedArray64>(Identities::none(),
                                              parameters,
                                              nextindex,
                                              nextcontent);
    }

    if (tail.empty()) {
      return next;
    }

    ContentPtr reversed = tail[0].get()->reverse_merge(next);
    if (tail.size() == 1) {
      return reversed;
    }
    else {
      return reversed.get()->mergemany(ContentPtrVec(tail.begin() + 1, tail.end()));
    }

    throw std::runtime_error(
      std::string("not implemented: ") + classname() + std::string("::mergemany"));
  }

  template <typename T, bool ISOPTION>
  const SliceItemPtr
  IndexedArrayOf<T, ISOPTION>::asslice() const {
    if (ISOPTION) {
      int64_t numnull;
      struct Error err1 = kernel::IndexedArray_numnull<T>(
        kernel::lib::cpu,   // DERIVE
        &numnull,
        index_.data(),
        index_.length());
      util::handle_error(err1, classname(), identities_.get());

      Index64 nextcarry(length() - numnull);
      Index64 outindex(length());
      struct Error err2 = kernel::IndexedArray_getitem_nextcarry_outindex_mask_64<T>(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        outindex.data(),
        index_.data(),
        index_.length(),
        content_.get()->length());
      util::handle_error(err2, classname(), identities_.get());

      ContentPtr next = content_.get()->carry(nextcarry, false);

      SliceItemPtr slicecontent = next.get()->asslice();
      if (SliceArray64* raw =
          dynamic_cast<SliceArray64*>(slicecontent.get())) {
        if (raw->frombool()) {
          Index64 nonzero(raw->index());
          Index8 originalmask(length());
          Index64 adjustedindex(nonzero.length() + numnull);
          Index64 adjustednonzero(nonzero.length());
          struct Error err3 = kernel::IndexedArray_getitem_adjust_outindex_64(
            kernel::lib::cpu,   // DERIVE
            originalmask.data(),
            adjustedindex.data(),
            adjustednonzero.data(),
            outindex.data(),
            outindex.length(),
            nonzero.data(),
            nonzero.length());
          util::handle_error(err3, classname(), nullptr);

          SliceItemPtr outcontent =
            std::make_shared<SliceArray64>(adjustednonzero,
                                           raw->shape(),
                                           raw->strides(),
                                           true);
          return std::make_shared<SliceMissing64>(adjustedindex,
                                                  originalmask,
                                                  outcontent);
        }
      }
      return std::make_shared<SliceMissing64>(outindex,
                                              Index8(0),
                                              slicecontent);
    }
    else {
      return project().get()->asslice();
    }
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::fillna(const ContentPtr& value) const {
    if (value.get()->length() != 1) {
      throw std::invalid_argument(
        std::string("fillna value length (")
        + std::to_string(value.get()->length())
        + std::string(") is not equal to 1") + FILENAME(__LINE__));
    }
    if (ISOPTION) {
      ContentPtrVec contents;
      contents.emplace_back(content());
      contents.emplace_back(value);

      Index8 tags = bytemask();
      Index64 index(tags.length());
      struct Error err = kernel::UnionArray_fillna_64<T>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        index_.data(),
        tags.length());
      util::handle_error(err, classname(), identities_.get());

      std::shared_ptr<UnionArray8_64> out =
        std::make_shared<UnionArray8_64>(Identities::none(),
                                         parameters_,
                                         tags,
                                         index,
                                         contents);
      return out.get()->simplify_uniontype(true);
    }
    else {
      return std::make_shared<IndexedArrayOf<T, ISOPTION>>(
        Identities::none(),
        parameters_,
        index_,
        content_.get()->fillna(value));
    }
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::rpad(int64_t target,
                                    int64_t axis,
                                    int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return rpad_axis0(target, false);
    }
    else if (posaxis == depth + 1) {
      if (ISOPTION) {
        Index8 mask = bytemask();
        Index64 index(mask.length());
        struct Error err = kernel::IndexedOptionArray_rpad_and_clip_mask_axis1_64(
          kernel::lib::cpu,   // DERIVE
          index.data(),
          mask.data(),
          mask.length());
        util::handle_error(err, classname(), identities_.get());

        ContentPtr next = project().get()->rpad(target, posaxis, depth);
        return std::make_shared<IndexedOptionArray64>(
          Identities::none(),
          util::Parameters(),
          index,
          next).get()->simplify_optiontype();
      }
      else {
        return project().get()->rpad(target, posaxis, depth);
      }
    }
    else {
      return std::make_shared<IndexedArrayOf<T, ISOPTION>>(
        Identities::none(),
        parameters_,
        index_,
        content_.get()->rpad(target, posaxis, depth));
    }
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::rpad_and_clip(int64_t target,
                                             int64_t axis,
                                             int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return rpad_axis0(target, true);
    }
    else if (posaxis == depth + 1) {
      if (ISOPTION) {
        Index8 mask = bytemask();
        Index64 index(mask.length());
        struct Error err = kernel::IndexedOptionArray_rpad_and_clip_mask_axis1_64(
          kernel::lib::cpu,   // DERIVE
          index.data(),
          mask.data(),
          mask.length());
        util::handle_error(err, classname(), identities_.get());

        ContentPtr next =
          project().get()->rpad_and_clip(target, posaxis, depth);
        return std::make_shared<IndexedOptionArray64>(
          Identities::none(),
          util::Parameters(),
          index,
          next).get()->simplify_optiontype();
      }
      else {
        return project().get()->rpad_and_clip(target, posaxis, depth);
      }
    }
    else {
      return std::make_shared<IndexedArrayOf<T, ISOPTION>>(
        Identities::none(),
        parameters_,
        index_,
        content_.get()->rpad_and_clip(target, posaxis, depth));
    }
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::reduce_next(const Reducer& reducer,
                                           int64_t negaxis,
                                           const Index64& starts,
                                           const Index64& shifts,
                                           const Index64& parents,
                                           int64_t outlength,
                                           bool mask,
                                           bool keepdims) const {
    int64_t numnull;
    struct Error err1 = kernel::IndexedArray_numnull<T>(
      kernel::lib::cpu,   // DERIVE
      &numnull,
      index_.data(),
      index_.length());
    util::handle_error(err1, classname(), identities_.get());

    Index64 nextparents(index_.length() - numnull);
    Index64 nextcarry(index_.length() - numnull);
    Index64 outindex(index_.length());
    struct Error err2 = kernel::IndexedArray_reduce_next_64<T>(
      kernel::lib::cpu,   // DERIVE
      nextcarry.data(),
      nextparents.data(),
      outindex.data(),
      index_.data(),
      parents.data(),
      index_.length());
    util::handle_error(err2, classname(), identities_.get());

    std::pair<bool, int64_t> branchdepth = branch_depth();
    bool make_shifts = (isoption()  &&
                        reducer.returns_positions()  &&
                        !branchdepth.first  && negaxis == branchdepth.second);

    Index64 nextshifts(make_shifts ? index_.length() - numnull : 0);
    if (make_shifts) {
      if (shifts.length() == 0) {
        struct Error err3 =
            kernel::IndexedArray_reduce_next_nonlocal_nextshifts_64<T>(
          kernel::lib::cpu,   // DERIVE
          nextshifts.data(),
          index_.data(),
          index_.length());
        util::handle_error(err3, classname(), identities_.get());
      }
      else {
        struct Error err3 =
            kernel::IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64<T>(
          kernel::lib::cpu,   // DERIVE
          nextshifts.data(),
          index_.data(),
          index_.length(),
          shifts.data());
        util::handle_error(err3, classname(), identities_.get());
      }
    }

    ContentPtr next = content_.get()->carry(nextcarry, false);
    if (ISOPTION) {
      if (RegularArray* raw = dynamic_cast<RegularArray*>(next.get())) {
        next = raw->toListOffsetArray64(true);
      }
    }

    ContentPtr out = next.get()->reduce_next(reducer,
                                             negaxis,
                                             starts,
                                             nextshifts,
                                             nextparents,
                                             outlength,
                                             mask,
                                             keepdims);

    if (!branchdepth.first  &&  negaxis == branchdepth.second) {
      return out;
    }
    else {
      if (RegularArray* raw =
          dynamic_cast<RegularArray*>(out.get())) {
        out = raw->toListOffsetArray64(true);
      }
      if (ListOffsetArray64* raw =
          dynamic_cast<ListOffsetArray64*>(out.get())) {
        Index64 outoffsets(starts.length() + 1);
        if (starts.length() > 0  &&  starts.getitem_at_nowrap(0) != 0) {
          throw std::runtime_error(
            std::string("reduce_next with unbranching depth > negaxis expects a "
                        "ListOffsetArray64 whose offsets start at zero ")
            + FILENAME(__LINE__));
        }
        struct Error err4 = kernel::IndexedArray_reduce_next_fix_offsets_64(
          kernel::lib::cpu,   // DERIVE
          outoffsets.data(),
          starts.data(),
          starts.length(),
          outindex.length());
        util::handle_error(err4, classname(), identities_.get());

        return std::make_shared<ListOffsetArray64>(
          raw->identities(),
          raw->parameters(),
          outoffsets,
          std::make_shared<IndexedOptionArray64>(Identities::none(),
                                                 util::Parameters(),
                                                 outindex,
                                                 raw->content()));
      }
      else {
        throw std::runtime_error(
          std::string("reduce_next with unbranching depth > negaxis is only "
                      "expected to return RegularArray or ListOffsetArray64; "
                      "instead, it returned ") + out.get()->classname()
          + FILENAME(__LINE__));
      }
    }

    return out;
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::localindex(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return localindex_axis0();
    }
    else {
      if (ISOPTION) {
        int64_t numnull;
        std::pair<Index64, IndexOf<T>> pair = nextcarry_outindex(numnull);
        Index64 nextcarry = pair.first;
        IndexOf<T> outindex = pair.second;

        ContentPtr next = content_.get()->carry(nextcarry, false);
        ContentPtr out = next.get()->localindex(posaxis, depth);
        IndexedArrayOf<T, ISOPTION> out2(identities_,
                                         util::Parameters(),
                                         outindex, out);
        return out2.simplify_optiontype();
      }
      else {
        return project().get()->localindex(posaxis, depth);
      }
    }
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::combinations(
    int64_t n,
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
      return combinations_axis0(n, replacement, recordlookup, parameters);
    }
    else {
      if (ISOPTION) {
        int64_t numnull;
        std::pair<Index64, IndexOf<T>> pair = nextcarry_outindex(numnull);
        Index64 nextcarry = pair.first;
        IndexOf<T> outindex = pair.second;

        ContentPtr next = content_.get()->carry(nextcarry, true);
        ContentPtr out = next.get()->combinations(n,
                                                  replacement,
                                                  recordlookup,
                                                  parameters,
                                                  posaxis,
                                                  depth);
        IndexedArrayOf<T, ISOPTION> out2(identities_,
                                         util::Parameters(),
                                         outindex,
                                         out);
        return out2.simplify_optiontype();
      }
      else {
        return project().get()->combinations(n,
                                             replacement,
                                             recordlookup,
                                             parameters,
                                             posaxis,
                                             depth);
      }
    }
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::sort_next(int64_t negaxis,
                                         const Index64& starts,
                                         const Index64& parents,
                                         int64_t outlength,
                                         bool ascending,
                                         bool stable,
                                         bool keepdims) const {
    int64_t index_length = index_.length();
    int64_t parents_length = parents.length();

    int64_t starts_length = starts.length();
    int64_t numnull(0);
    struct Error err1 = kernel::IndexedArray_numnull<T>(
      kernel::lib::cpu,   // DERIVE
      &numnull,
      index_.data(),
      index_length);
    util::handle_error(err1, classname(), identities_.get());

    int64_t next_length = (numnull > 0) ? index_length - numnull : index_length;
    Index64 nextparents(next_length);
    Index64 nextcarry(next_length);
    Index64 outindex(index_length);
    struct Error err2 = kernel::IndexedArray_reduce_next_64<T>(
      kernel::lib::cpu,   // DERIVE
      nextcarry.data(),
      nextparents.data(),
      outindex.data(),
      index_.data(),
      parents.data(),
      index_length);
    util::handle_error(err2, classname(), identities_.get());

    ContentPtr next = content_.get()->carry(nextcarry, false);

    bool inject_nones = false;
    std::pair<bool, int64_t> branchdepth = branch_depth();
    if (numnull > 0  &&  !branchdepth.first  &&  negaxis != branchdepth.second) {
      keepdims = false;
      inject_nones = true;
    }
    ContentPtr out = next.get()->sort_next(negaxis,
                                           starts,
                                           nextparents,
                                           outlength,
                                           ascending,
                                           stable,
                                           keepdims);

    Index64 nextoutindex(parents_length);
    struct Error err3 = kernel::IndexedArray_local_preparenext_64(
        kernel::lib::cpu,   // DERIVE
        nextoutindex.data(),
        starts.data(),
        parents.data(),
        parents_length,
        nextparents.data(),
        next_length);
    util::handle_error(err3, classname(), identities_.get());

    out = std::make_shared<IndexedArrayOf<int64_t, ISOPTION>>(
            Identities::none(),
            parameters_,
            nextoutindex,
            out);

    if (inject_nones) {
      out = std::make_shared<RegularArray>(Identities::none(),
                                           util::Parameters(),
                                           out,
                                           parents_length,
                                           0);
    }

    if (!branchdepth.first  &&  negaxis == branchdepth.second) {
      return out;
    }
    else {
      if (RegularArray* raw =
        dynamic_cast<RegularArray*>(out.get())) {
        out = raw->toListOffsetArray64(true);
      }
      if (ListOffsetArray64* raw =
        dynamic_cast<ListOffsetArray64*>(out.get())) {
        Index64 outoffsets(starts.length() + 1);
        if (starts.length() > 0  &&  starts.getitem_at_nowrap(0) != 0) {
          throw std::runtime_error(
            std::string("sort_next with unbranching depth > negaxis expects a "
                        "ListOffsetArray64 whose offsets start at zero")
            + FILENAME(__LINE__));
        }
        struct Error err4 = kernel::IndexedArray_reduce_next_fix_offsets_64(
          kernel::lib::cpu,   // DERIVE
          outoffsets.data(),
          starts.data(),
          starts_length,
          outindex.length());
        util::handle_error(err4, classname(), identities_.get());

        return std::make_shared<ListOffsetArray64>(
          raw->identities(),
          raw->parameters(),
          outoffsets,
          std::make_shared<IndexedArrayOf<int64_t, ISOPTION>>(
            Identities::none(),
            parameters_,
            outindex,
            raw->content()));
      }
      if (IndexedArrayOf<int64_t, ISOPTION>* raw =
        dynamic_cast<IndexedArrayOf<int64_t, ISOPTION>*>(out.get())) {
          return out;
      }
      else {
        throw std::runtime_error(
          std::string("sort_next with unbranching depth > negaxis is only "
                      "expected to return RegularArray or ListOffsetArray64 or "
                      "IndexedArrayOf<int64_t, ISOPTION>; "
                      "instead, it returned ") + out.get()->classname()
          + FILENAME(__LINE__));
      }
    }

    return out;
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::argsort_next(int64_t negaxis,
                                            const Index64& starts,
                                            const Index64& parents,
                                            int64_t outlength,
                                            bool ascending,
                                            bool stable,
                                            bool keepdims) const {
    int64_t index_length = index_.length();
    int64_t parents_length = parents.length();

    int64_t starts_length = starts.length();
    int64_t numnull(0);
    struct Error err1 = kernel::IndexedArray_numnull<T>(
      kernel::lib::cpu,   // DERIVE
      &numnull,
      index_.data(),
      index_length);
    util::handle_error(err1, classname(), identities_.get());

    int64_t next_length = (numnull > 0) ? index_length - numnull : index_length;
    Index64 nextparents(next_length);
    Index64 nextcarry(next_length);
    Index64 outindex(index_length);
    struct Error err2 = kernel::IndexedArray_reduce_next_64<T>(
      kernel::lib::cpu,   // DERIVE
      nextcarry.data(),
      nextparents.data(),
      outindex.data(),
      index_.data(),
      parents.data(),
      index_length);
    util::handle_error(err2, classname(), identities_.get());

    ContentPtr next = content_.get()->carry(nextcarry, false);

    bool inject_nones = false;
    std::pair<bool, int64_t> branchdepth = branch_depth();
    if (numnull > 0  &&  !branchdepth.first  &&  negaxis != branchdepth.second) {
      keepdims = false;
      inject_nones = true;
    }
    ContentPtr out = next.get()->argsort_next(negaxis,
                                              starts,
                                              nextparents,
                                              outlength,
                                              ascending,
                                              stable,
                                              keepdims);

    Index64 nextoutindex(parents_length);
    struct Error err3 = kernel::IndexedArray_local_preparenext_64(
      kernel::lib::cpu,   // DERIVE
      nextoutindex.data(),
      starts.data(),
      parents.data(),
      parents_length,
      nextparents.data(),
      next_length);
    util::handle_error(err3, classname(), identities_.get());

    out = std::make_shared<IndexedArrayOf<int64_t, ISOPTION>>(
        Identities::none(),
        util::Parameters(),
        nextoutindex,
        out);

    if (inject_nones) {
      out = std::make_shared<RegularArray>(Identities::none(),
                                           util::Parameters(),
                                           out,
                                           parents_length,
                                           0);
    }

    if (!branchdepth.first  &&  negaxis == branchdepth.second) {
      return out;
    }
    else {
      if (RegularArray* raw =
        dynamic_cast<RegularArray*>(out.get())) {
          out = raw->toListOffsetArray64(true);
      }
      if (ListOffsetArray64* raw =
        dynamic_cast<ListOffsetArray64*>(out.get())) {
        Index64 outoffsets(starts.length() + 1);
        if (starts.length() > 0  &&  starts.getitem_at_nowrap(0) != 0) {
          throw std::runtime_error(
            std::string("argsort_next with unbranching depth > negaxis expects a "
                        "ListOffsetArray64 whose offsets start at zero")
            + FILENAME(__LINE__));
        }
        struct Error err4 = kernel::IndexedArray_reduce_next_fix_offsets_64(
          kernel::lib::cpu,   // DERIVE
          outoffsets.data(),
          starts.data(),
          starts_length,
          outindex.length());
        util::handle_error(err4, classname(), identities_.get());

        return std::make_shared<ListOffsetArray64>(
          raw->identities(),
          raw->parameters(),
          outoffsets,
          std::make_shared<IndexedArrayOf<int64_t, ISOPTION>>(
            Identities::none(),
            util::Parameters(),
            outindex,
            raw->content()));
      }
      if (IndexedArrayOf<int64_t, ISOPTION>* raw =
        dynamic_cast<IndexedArrayOf<int64_t, ISOPTION>*>(out.get())) {
          return out;
      }
      else {
        throw std::runtime_error(
          std::string("argsort_next with unbranching depth > negaxis is only "
                      "expected to return RegularArray or ListOffsetArray64 or "
                      "IndexedArrayOf<int64_t, ISOPTION>; "
                      "instead, it returned ") + out.get()->classname()
          + FILENAME(__LINE__));
      }
    }

    return out;
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T,
                 ISOPTION>::getitem_next(const SliceAt& at,
                                         const Slice& tail,
                                         const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: IndexedArray::getitem_next(at)")
      + FILENAME(__LINE__));
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceRange& range,
                                            const Slice& tail,
                                            const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: IndexedArray::getitem_next(range)")
      + FILENAME(__LINE__));
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceArray64& array,
                                            const Slice& tail,
                                            const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: IndexedArray::getitem_next(array)")
      + FILENAME(__LINE__));
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_next(const SliceJagged64& jagged,
                                            const Slice& tail,
                                            const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: IndexedArray::getitem_next(jagged)")
      + FILENAME(__LINE__));
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_next_jagged(
    const Index64& slicestarts,
    const Index64& slicestops,
    const SliceArray64& slicecontent,
    const Slice& tail) const {
    return getitem_next_jagged_generic<SliceArray64>(slicestarts,
                                                     slicestops,
                                                     slicecontent,
                                                     tail);
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_next_jagged(
    const Index64& slicestarts,
    const Index64& slicestops,
    const SliceMissing64& slicecontent,
    const Slice& tail) const {
    return getitem_next_jagged_generic<SliceMissing64>(slicestarts,
                                                       slicestops,
                                                       slicecontent,
                                                       tail);
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_next_jagged(
    const Index64& slicestarts,
    const Index64& slicestops,
    const SliceJagged64& slicecontent,
    const Slice& tail) const {
    return getitem_next_jagged_generic<SliceJagged64>(slicestarts,
                                                      slicestops,
                                                      slicecontent,
                                                      tail);
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::copy_to(kernel::lib ptr_lib) const {
    IndexOf<T> index = index_.copy_to(ptr_lib);
    ContentPtr content = content_.get()->copy_to(ptr_lib);
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->copy_to(ptr_lib);
    }
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities,
                                                         parameters_,
                                                         index,
                                                         content);

  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::numbers_to_type(const std::string& name) const {
    IndexOf<T> index = index_.deep_copy();
    ContentPtr content = content_.get()->numbers_to_type(name);
    IdentitiesPtr identities = identities_;
    if (identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<IndexedArrayOf<T, ISOPTION>>(identities,
                                                         parameters_,
                                                         index,
                                                         content);
  }

  template <typename T, bool ISOPTION>
  template <typename S>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::getitem_next_jagged_generic(
    const Index64& slicestarts,
    const Index64& slicestops,
    const S& slicecontent,
    const Slice& tail) const {
    if (ISOPTION) {
      int64_t numnull;
      std::pair<Index64, IndexOf<T>> pair = nextcarry_outindex(numnull);
      Index64 nextcarry = pair.first;
      IndexOf<T> outindex = pair.second;

      Index64 reducedstarts(length() - numnull);
      Index64 reducedstops(length() - numnull);
      struct Error err = kernel::MaskedArray_getitem_next_jagged_project<T>(
        kernel::lib::cpu,   // DERIVE
        outindex.data(),
        slicestarts.data(),
        slicestops.data(),
        reducedstarts.data(),
        reducedstops.data(),
        length());
      util::handle_error(err, classname(), identities_.get());

      ContentPtr next = content_.get()->carry(nextcarry, true);
      ContentPtr out = next.get()->getitem_next_jagged(
          reducedstarts, reducedstops, slicecontent, tail);
      IndexedArrayOf<T, ISOPTION> out2(identities_, parameters_, outindex, out);
      return out2.simplify_optiontype();
    }
    else {
      Index64 nextcarry(length());
      struct Error err = kernel::IndexedArray_getitem_nextcarry_64<T>(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        index_.data(),
        index_.length(),
        content_.get()->length());
      util::handle_error(err, classname(), identities_.get());

      // an eager carry (allow_lazy = false) to avoid infinite loop (unproven)
      ContentPtr next = content_.get()->carry(nextcarry, false);
      return next.get()->getitem_next_jagged(slicestarts,
                                             slicestops,
                                             slicecontent,
                                             tail);
    }
  }

  template <typename T, bool ISOPTION>
  const std::pair<Index64, IndexOf<T>>
  IndexedArrayOf<T, ISOPTION>::nextcarry_outindex(int64_t& numnull) const {
    struct Error err1 = kernel::IndexedArray_numnull<T>(
      kernel::lib::cpu,   // DERIVE
      &numnull,
      index_.data(),
      index_.length());
    util::handle_error(err1, classname(), identities_.get());

    Index64 nextcarry(length() - numnull);
    IndexOf<T> outindex(length());
    struct Error err2 =
      kernel::IndexedArray_getitem_nextcarry_outindex_64<T>(
      kernel::lib::cpu,   // DERIVE
      nextcarry.data(),
      outindex.data(),
      index_.data(),
      index_.length(),
      content_.get()->length());
    util::handle_error(err2, classname(), identities_.get());

    return std::pair<Index64, IndexOf<T>>(nextcarry, outindex);
  }

  template <typename T, bool ISOPTION>
  bool
  IndexedArrayOf<T, ISOPTION>::is_unique() const {
    Index64 start(1);
    start.setitem_at_nowrap(0, index().offset());
    Index64 stop(1);
    stop.setitem_at_nowrap(0, index().length());
    return is_subrange_equal(start, stop);
  }

  template <typename T, bool ISOPTION>
  const ContentPtr
  IndexedArrayOf<T, ISOPTION>::unique() const {
    throw std::runtime_error(
      std::string("FIXME: operation not yet implemented: IndexedArrayOf<T, ISOPTION>::unique")
      + FILENAME(__LINE__));
  }

  template <typename T, bool ISOPTION>
  bool
  IndexedArrayOf<T, ISOPTION>::is_subrange_equal(const Index64& starts, const Index64& stops) const {
    if (starts.length() != stops.length()) {
      throw std::invalid_argument(
        std::string("IndexedArrayOf<T, ISOPTION> starts length must be equal to stops length")
        + FILENAME(__LINE__));
    }

    Index64 nextstarts(starts.length());
    Index64 nextstops(stops.length());

    int64_t subranges_length = 0;
    struct Error err1 = kernel::IndexedArray_ranges_next_64<T>(
      kernel::lib::cpu,   // DERIVE
      index_.data(),
      starts.data(),
      stops.data(),
      starts.length(),
      nextstarts.data(),
      nextstops.data(),
      &subranges_length);
    util::handle_error(err1, classname(), identities_.get());

    Index64 nextcarry(subranges_length);
    struct Error err2 = kernel::IndexedArray_ranges_carry_next_64<T>(
      kernel::lib::cpu,   // DERIVE
      index_.data(),
      starts.data(),
      stops.data(),
      starts.length(),
      nextcarry.data());
    util::handle_error(err2, classname(), identities_.get());

    ContentPtr next = content_.get()->carry(nextcarry, false);
    if (nextstarts.length() > 1) {
      return next.get()->is_subrange_equal(nextstarts, nextstops);
    }
    else {
      return next.get()->is_unique();
    }
  }

  // IndexedArrayOf<int64_t, true> has to be first, or ld on darwin
  // will hide the typeinfo symbol
  template class EXPORT_TEMPLATE_INST IndexedArrayOf<int64_t, true>;

  template class EXPORT_TEMPLATE_INST IndexedArrayOf<int32_t, false>;
  template class EXPORT_TEMPLATE_INST IndexedArrayOf<uint32_t, false>;
  template class EXPORT_TEMPLATE_INST IndexedArrayOf<int64_t, false>;
  template class EXPORT_TEMPLATE_INST IndexedArrayOf<int32_t, true>;
}
