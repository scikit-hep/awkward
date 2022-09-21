// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/IndexedArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/IndexedArray.cpp", line)

#include <sstream>
#include <type_traits>

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"


#include "awkward/Reducer.h"
#include "awkward/Slice.h"
#include "awkward/io/json.h"

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


#define AWKWARD_INDEXEDARRAY_NO_EXTERN_TEMPLATE
#include "awkward/array/IndexedArray.h"

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

  const FormPtr
  IndexedForm::with_form_key(const FormKey& form_key) const {
    return std::make_shared<IndexedForm>(has_identities_,
                                         parameters_,
                                         form_key,
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
  IndexedForm::istuple() const {
    return content_.get()->istuple();
  }

  bool
  IndexedForm::equal(const FormPtr& other,
                     bool check_identities,
                     bool check_parameters,
                     bool check_form_key,
                     bool compatibility_check) const {
    if (compatibility_check) {

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
    IndexedForm step1(has_identities_,
                      util::Parameters(),
                      FormKey(nullptr),
                      index_,
                      content_.get()->getitem_field(key));
    return step1.simplify_optiontype();
  }

  const FormPtr
  IndexedForm::getitem_fields(const std::vector<std::string>& keys) const {
    IndexedForm step1(has_identities_,
                      util::Parameters(),
                      FormKey(nullptr),
                      index_,
                      content_.get()->getitem_fields(keys));
    return step1.simplify_optiontype();
  }

  const FormPtr
  IndexedForm::simplify_optiontype() const {
    if (IndexedForm* rawcontent = dynamic_cast<IndexedForm*>(content_.get())) {
      return std::make_shared<IndexedForm>(
        has_identities_,
        parameters_,
        form_key_,
        Index::Form::i64,
        rawcontent->content());
    }
    else if (IndexedOptionForm* rawcontent = dynamic_cast<IndexedOptionForm*>(content_.get())) {
      return std::make_shared<IndexedOptionForm>(
        has_identities_,
        parameters_,
        form_key_,
        Index::Form::i64,
        rawcontent->content());
    }
    else if (ByteMaskedForm* rawcontent = dynamic_cast<ByteMaskedForm*>(content_.get())) {
      return std::make_shared<IndexedOptionForm>(
        has_identities_,
        parameters_,
        form_key_,
        Index::Form::i64,
        rawcontent->content());
    }
    else if (BitMaskedForm* rawcontent = dynamic_cast<BitMaskedForm*>(content_.get())) {
      return std::make_shared<IndexedOptionForm>(
        has_identities_,
        parameters_,
        form_key_,
        Index::Form::i64,
        rawcontent->content());
    }
    else if (UnmaskedForm* rawcontent = dynamic_cast<UnmaskedForm*>(content_.get())) {
      return std::make_shared<IndexedOptionForm>(
        has_identities_,
        parameters_,
        form_key_,
        Index::Form::i64,
        rawcontent->content());
    }
    else {
      return shallow_copy();
    }
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

  const FormPtr
  IndexedOptionForm::with_form_key(const FormKey& form_key) const {
    return std::make_shared<IndexedOptionForm>(has_identities_,
                                               parameters_,
                                               form_key,
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
  IndexedOptionForm::istuple() const {
    return content_.get()->istuple();
  }

  bool
  IndexedOptionForm::equal(const FormPtr& other,
                           bool check_identities,
                           bool check_parameters,
                           bool check_form_key,
                           bool compatibility_check) const {
    if (compatibility_check) {

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
    IndexedOptionForm step1(has_identities_,
                            util::Parameters(),
                            FormKey(nullptr),
                            index_,
                            content_.get()->getitem_field(key));
    return step1.simplify_optiontype();
  }

  const FormPtr
  IndexedOptionForm::getitem_fields(const std::vector<std::string>& keys) const {
    IndexedOptionForm step1(has_identities_,
                            util::Parameters(),
                            FormKey(nullptr),
                            index_,
                            content_.get()->getitem_fields(keys));
    return step1.simplify_optiontype();
  }

  const FormPtr
  IndexedOptionForm::simplify_optiontype() const {
    if (IndexedForm* rawcontent = dynamic_cast<IndexedForm*>(content_.get())) {
      return std::make_shared<IndexedOptionForm>(
        has_identities_,
        parameters_,
        form_key_,
        Index::Form::i64,
        rawcontent->content());
    }
    else if (IndexedOptionForm* rawcontent = dynamic_cast<IndexedOptionForm*>(content_.get())) {
      return std::make_shared<IndexedOptionForm>(
        has_identities_,
        parameters_,
        form_key_,
        Index::Form::i64,
        rawcontent->content());
    }
    else if (ByteMaskedForm* rawcontent = dynamic_cast<ByteMaskedForm*>(content_.get())) {
      return std::make_shared<IndexedOptionForm>(
        has_identities_,
        parameters_,
        form_key_,
        Index::Form::i64,
        rawcontent->content());
    }
    else if (BitMaskedForm* rawcontent = dynamic_cast<BitMaskedForm*>(content_.get())) {
      return std::make_shared<IndexedOptionForm>(
        has_identities_,
        parameters_,
        form_key_,
        Index::Form::i64,
        rawcontent->content());
    }
    else if (UnmaskedForm* rawcontent = dynamic_cast<UnmaskedForm*>(content_.get())) {
      return std::make_shared<IndexedOptionForm>(
        has_identities_,
        parameters_,
        form_key_,
        Index::Form::i64,
        rawcontent->content());
    }
    else {
      return shallow_copy();
    }
  }

}
