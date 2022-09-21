// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/UnmaskedArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/UnmaskedArray.cpp", line)

#include <sstream>
#include <type_traits>

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"

#include "awkward/Reducer.h"

#include "awkward/io/json.h"

#include "awkward/array/EmptyArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"

#include "awkward/array/RegularArray.h"

#include "awkward/array/UnmaskedArray.h"

namespace awkward {
  ////////// UnmaskedForm

  UnmaskedForm::UnmaskedForm(bool has_identities,
                             const util::Parameters& parameters,
                             const FormKey& form_key,
                             const FormPtr& content)
      : Form(has_identities, parameters, form_key)
      , content_(content) { }

  const FormPtr
  UnmaskedForm::content() const {
    return content_;
  }

  void
  UnmaskedForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    builder.string("UnmaskedArray");
    builder.field("content");
    content_.get()->tojson_part(builder, verbose);
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  UnmaskedForm::shallow_copy() const {
    return std::make_shared<UnmaskedForm>(has_identities_,
                                          parameters_,
                                          form_key_,
                                          content_);
  }

  const FormPtr
  UnmaskedForm::with_form_key(const FormKey& form_key) const {
    return std::make_shared<UnmaskedForm>(has_identities_,
                                          parameters_,
                                          form_key,
                                          content_);
  }

  const std::string
  UnmaskedForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  bool
  UnmaskedForm::purelist_isregular() const {
    return content_.get()->purelist_isregular();
  }

  int64_t
  UnmaskedForm::purelist_depth() const {
    return content_.get()->purelist_depth();
  }

  bool
  UnmaskedForm::dimension_optiontype() const {
    return true;
  }

  const std::pair<int64_t, int64_t>
  UnmaskedForm::minmax_depth() const {
    return content_.get()->minmax_depth();
  }

  const std::pair<bool, int64_t>
  UnmaskedForm::branch_depth() const {
    return content_.get()->branch_depth();
  }

  int64_t
  UnmaskedForm::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  UnmaskedForm::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  UnmaskedForm::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  UnmaskedForm::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  UnmaskedForm::keys() const {
    return content_.get()->keys();
  }

  bool
  UnmaskedForm::istuple() const {
    return content_.get()->istuple();
  }

  bool
  UnmaskedForm::equal(const FormPtr& other,
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
    if (UnmaskedForm* t = dynamic_cast<UnmaskedForm*>(other.get())) {
      return (content_.get()->equal(t->content(),
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
  UnmaskedForm::getitem_field(const std::string& key) const {
    UnmaskedForm step1(has_identities_,
                       util::Parameters(),
                       FormKey(nullptr),
                       content_.get()->getitem_field(key));
    return step1.simplify_optiontype();
  }

  const FormPtr
  UnmaskedForm::getitem_fields(const std::vector<std::string>& keys) const {
    UnmaskedForm step1(has_identities_,
                       util::Parameters(),
                       FormKey(nullptr),
                       content_.get()->getitem_fields(keys));
    return step1.simplify_optiontype();
  }

  const FormPtr
  UnmaskedForm::simplify_optiontype() const {
    if (dynamic_cast<IndexedForm*>(content_.get())         ||
        dynamic_cast<IndexedOptionForm*>(content_.get())   ||
        dynamic_cast<ByteMaskedForm*>(content_.get())      ||
        dynamic_cast<BitMaskedForm*>(content_.get())       ||
        dynamic_cast<UnmaskedForm*>(content_.get())) {
      return content_;
    }
    else {
      return shallow_copy();
    }
  }
}
