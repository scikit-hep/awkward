// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/ListOffsetArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/ListOffsetArray.cpp", line)

#include <algorithm>
#include <numeric>
#include <sstream>
#include <type_traits>

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"


#include "awkward/Reducer.h"

#include "awkward/io/json.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"


#define AWKWARD_LISTOFFSETARRAY_NO_EXTERN_TEMPLATE
#include "awkward/array/ListOffsetArray.h"

namespace awkward {
  ////////// ListOffsetForm

  ListOffsetForm::ListOffsetForm(bool has_identities,
                                 const util::Parameters& parameters,
                                 const FormKey& form_key,
                                 Index::Form offsets,
                                 const FormPtr& content)
      : Form(has_identities, parameters, form_key)
      , offsets_(offsets)
      , content_(content) { }

  Index::Form
  ListOffsetForm::offsets() const {
    return offsets_;
  }

  const FormPtr
  ListOffsetForm::content() const {
    return content_;
  }

  void
  ListOffsetForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    if (offsets_ == Index::Form::i32) {
      builder.string("ListOffsetArray32");
    }
    else if (offsets_ == Index::Form::u32) {
      builder.string("ListOffsetArrayU32");
    }
    else if (offsets_ == Index::Form::i64) {
      builder.string("ListOffsetArray64");
    }
    else {
      builder.string("UnrecognizedListOffsetArray");
    }
    builder.field("offsets");
    builder.string(Index::form2str(offsets_));
    builder.field("content");
    content_.get()->tojson_part(builder, verbose);
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  ListOffsetForm::shallow_copy() const {
    return std::make_shared<ListOffsetForm>(has_identities_,
                                            parameters_,
                                            form_key_,
                                            offsets_,
                                            content_);
  }

  const FormPtr
  ListOffsetForm::with_form_key(const FormKey& form_key) const {
    return std::make_shared<ListOffsetForm>(has_identities_,
                                            parameters_,
                                            form_key,
                                            offsets_,
                                            content_);
  }

  const std::string
  ListOffsetForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  bool
  ListOffsetForm::purelist_isregular() const {
    return false;
  }

  int64_t
  ListOffsetForm::purelist_depth() const {
    if (parameter_equals("__array__", "\"string\"")  ||
        parameter_equals("__array__", "\"bytestring\"")) {
      return 1;
    }
    return content_.get()->purelist_depth() + 1;
  }

  bool
  ListOffsetForm::dimension_optiontype() const {
    return false;
  }

  const std::pair<int64_t, int64_t>
  ListOffsetForm::minmax_depth() const {
    if (parameter_equals("__array__", "\"string\"")  ||
        parameter_equals("__array__", "\"bytestring\"")) {
      return std::pair<int64_t, int64_t>(1, 1);
    }
    std::pair<int64_t, int64_t> content_depth = content_.get()->minmax_depth();
    return std::pair<int64_t, int64_t>(content_depth.first + 1,
                                       content_depth.second + 1);
  }

  const std::pair<bool, int64_t>
  ListOffsetForm::branch_depth() const {
    if (parameter_equals("__array__", "\"string\"")  ||
        parameter_equals("__array__", "\"bytestring\"")) {
      return std::pair<bool, int64_t>(false, 1);
    }
    std::pair<bool, int64_t> content_depth = content_.get()->branch_depth();
    return std::pair<bool, int64_t>(content_depth.first,
                                    content_depth.second + 1);
  }

  int64_t
  ListOffsetForm::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  ListOffsetForm::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  ListOffsetForm::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  ListOffsetForm::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  ListOffsetForm::keys() const {
    return content_.get()->keys();
  }

  bool
  ListOffsetForm::istuple() const {
    return content_.get()->istuple();
  }

  bool
  ListOffsetForm::equal(const FormPtr& other,
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
    if (ListOffsetForm* t = dynamic_cast<ListOffsetForm*>(other.get())) {
      return (offsets_ == t->offsets()  &&
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
  ListOffsetForm::getitem_field(const std::string& key) const {
    return std::make_shared<ListOffsetForm>(
      has_identities_,
      util::Parameters(),
      FormKey(nullptr),
      offsets_,
      content_.get()->getitem_field(key));
  }

  const FormPtr
  ListOffsetForm::getitem_fields(const std::vector<std::string>& keys) const {
    return std::make_shared<ListOffsetForm>(
      has_identities_,
      util::Parameters(),
      FormKey(nullptr),
      offsets_,
      content_.get()->getitem_fields(keys));
  }
}
