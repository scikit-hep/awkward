// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/RegularArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/RegularArray.cpp", line)

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"

#include "awkward/Reducer.h"
#include "awkward/io/json.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"

#include "awkward/array/RegularArray.h"

namespace awkward {
  ////////// RegularForm

  RegularForm::RegularForm(bool has_identities,
                           const util::Parameters& parameters,
                           const FormKey& form_key,
                           const FormPtr& content,
                           int64_t size)
      : Form(has_identities, parameters, form_key)
      , content_(content)
      , size_(size) { }

  const FormPtr
  RegularForm::content() const {
    return content_;
  }

  int64_t
  RegularForm::size() const {
    return size_;
  }

  void
  RegularForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    builder.string("RegularArray");
    builder.field("content");
    content_.get()->tojson_part(builder, verbose);
    builder.field("size");
    builder.integer(size_);
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  RegularForm::shallow_copy() const {
    return std::make_shared<RegularForm>(has_identities_,
                                         parameters_,
                                         form_key_,
                                         content_,
                                         size_);
  }

  const FormPtr
  RegularForm::with_form_key(const FormKey& form_key) const {
    return std::make_shared<RegularForm>(has_identities_,
                                         parameters_,
                                         form_key,
                                         content_,
                                         size_);
  }

  const std::string
  RegularForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  bool
  RegularForm::purelist_isregular() const {
    return content_.get()->purelist_isregular();
  }

  int64_t
  RegularForm::purelist_depth() const {
    if (parameter_equals("__array__", "\"string\"")  ||
        parameter_equals("__array__", "\"bytestring\"")) {
      return 1;
    }
    return content_.get()->purelist_depth() + 1;
  }

  bool
  RegularForm::dimension_optiontype() const {
    return false;
  }

  const std::pair<int64_t, int64_t>
  RegularForm::minmax_depth() const {
    if (parameter_equals("__array__", "\"string\"")  ||
        parameter_equals("__array__", "\"bytestring\"")) {
      return std::pair<int64_t, int64_t>(1, 1);
    }
    std::pair<int64_t, int64_t> content_depth = content_.get()->minmax_depth();
    return std::pair<int64_t, int64_t>(content_depth.first + 1,
                                       content_depth.second + 1);
  }

  const std::pair<bool, int64_t>
  RegularForm::branch_depth() const {
    if (parameter_equals("__array__", "\"string\"")  ||
        parameter_equals("__array__", "\"bytestring\"")) {
      return std::pair<bool, int64_t>(false, 1);
    }
    std::pair<bool, int64_t> content_depth = content_.get()->branch_depth();
    return std::pair<bool, int64_t>(content_depth.first,
                                    content_depth.second + 1);
  }

  int64_t
  RegularForm::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  RegularForm::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  RegularForm::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  RegularForm::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  RegularForm::keys() const {
    return content_.get()->keys();
  }

  bool
  RegularForm::istuple() const {
    return content_.get()->istuple();
  }

  bool
  RegularForm::equal(const FormPtr& other,
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
    if (RegularForm* t = dynamic_cast<RegularForm*>(other.get())) {
      return (content_.get()->equal(t->content(),
                                    check_identities,
                                    check_parameters,
                                    check_form_key,
                                    compatibility_check)  &&
              size_ == t->size());
    }
    else {
      return false;
    }
  }

  const FormPtr
  RegularForm::getitem_field(const std::string& key) const {
    return std::make_shared<RegularForm>(
      has_identities_,
      util::Parameters(),
      FormKey(nullptr),
      content_.get()->getitem_field(key),
      size_);
  }

  const FormPtr
  RegularForm::getitem_fields(const std::vector<std::string>& keys) const {
    return std::make_shared<RegularForm>(
      has_identities_,
      util::Parameters(),
      FormKey(nullptr),
      content_.get()->getitem_fields(keys),
      size_);
  }
}
