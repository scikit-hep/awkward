// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/EmptyArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/EmptyArray.cpp", line)

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/kernels.h"

#include "awkward/Reducer.h"
#include "awkward/io/json.h"
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

  const FormPtr
  EmptyForm::with_form_key(const FormKey& form_key) const {
    return std::make_shared<EmptyForm>(has_identities_,
                                       parameters_,
                                       form_key);
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

  bool
  EmptyForm::dimension_optiontype() const {
    return false;
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
  EmptyForm::istuple() const {
    return false;
  }

  bool
  EmptyForm::equal(const FormPtr& other,
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

  const FormPtr
  EmptyForm::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::invalid_argument(
      std::string("requested keys do not exist (data might not be records)"));
  }
}
