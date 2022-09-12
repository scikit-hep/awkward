// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/NumpyArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/NumpyArray.cpp", line)

#include <algorithm>
#include <complex>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"



#include "awkward/io/json.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"

#include "awkward/util.h"
#include "awkward/datetime_util.h"

#include "awkward/array/NumpyArray.h"

namespace awkward {
  ////////// NumpyForm

  NumpyForm::NumpyForm(bool has_identities,
                       const util::Parameters& parameters,
                       const FormKey& form_key,
                       const std::vector<int64_t>& inner_shape,
                       int64_t itemsize,
                       const std::string& format,
                       util::dtype dtype)
      : Form(has_identities, parameters, form_key)
      , inner_shape_(inner_shape)
      , itemsize_(itemsize)
      , format_(format)
      , dtype_(dtype) { }

  const std::vector<int64_t>
  NumpyForm::inner_shape() const {
    return inner_shape_;
  }

  int64_t
  NumpyForm::itemsize() const {
    return itemsize_;
  }

  const std::string
  NumpyForm::format() const {
    return format_;
  }

  util::dtype
  NumpyForm::dtype() const {
    return dtype_;
  }

  const std::string
  NumpyForm::primitive() const {
    return util::dtype_to_name(dtype_);
  }

  const std::string
  NumpyForm::tostring() const {
    ToJsonPrettyString builder(-1);
    tojson_part(builder, false, true);
    return builder.tostring();
  }

  const std::string
  NumpyForm::tojson(bool pretty, bool verbose) const {
    if (pretty) {
      ToJsonPrettyString builder(-1);
      tojson_part(builder, verbose, true);
      return builder.tostring();
    }
    else {
      ToJsonString builder(-1);
      tojson_part(builder, verbose, true);
      return builder.tostring();
    }
  }

  void
  NumpyForm::tojson_part(ToJson& builder, bool verbose) const {
    return tojson_part(builder, verbose, false);
  }

  void
  NumpyForm::tojson_part(ToJson& builder, bool verbose, bool toplevel) const {
    std::string p = primitive();
    if (verbose  ||
        toplevel  ||
        p.empty()  ||
        !inner_shape_.empty() ||
        has_identities_  ||
        !parameters_.empty()  ||
        form_key_.get() != nullptr) {
      builder.beginrecord();
      builder.field("class");
      builder.string("NumpyArray");
      if (verbose  ||  !inner_shape_.empty()) {
        builder.field("inner_shape");
        builder.beginlist();
        for (auto x : inner_shape_) {
          builder.integer(x);
        }
        builder.endlist();
      }
      builder.field("itemsize");
      builder.integer(itemsize_);
      builder.field("format");
      builder.string(format_);
      if (!p.empty()) {
        builder.field("primitive");
        builder.string(p);
      }
      else if (verbose) {
        builder.field("primitive");
        builder.null();
      }
      identities_tojson(builder, verbose);
      parameters_tojson(builder, verbose);
      form_key_tojson(builder, verbose);
      builder.endrecord();
    }
    else {
      builder.string(p.c_str(), (int64_t)p.length());
    }
  }

  const FormPtr
  NumpyForm::shallow_copy() const {
    return std::make_shared<NumpyForm>(has_identities_,
                                       parameters_,
                                       form_key_,
                                       inner_shape_,
                                       itemsize_,
                                       format_,
                                       dtype_);
  }

  const FormPtr
  NumpyForm::with_form_key(const FormKey& form_key) const {
    return std::make_shared<NumpyForm>(has_identities_,
                                       parameters_,
                                       form_key,
                                       inner_shape_,
                                       itemsize_,
                                       format_,
                                       dtype_);
  }

  const std::string
  NumpyForm::purelist_parameter(const std::string& key) const {
    return parameter(key);
  }

  bool
  NumpyForm::purelist_isregular() const {
    return true;
  }

  int64_t
  NumpyForm::purelist_depth() const {
    return (int64_t)inner_shape_.size() + 1;
  }

  bool
  NumpyForm::dimension_optiontype() const {
    return false;
  }

  const std::pair<int64_t, int64_t>
  NumpyForm::minmax_depth() const {
    return std::pair<int64_t, int64_t>((int64_t)inner_shape_.size() + 1,
                                       (int64_t)inner_shape_.size() + 1);
  }

  const std::pair<bool, int64_t>
  NumpyForm::branch_depth() const {
    return std::pair<bool, int64_t>(false, (int64_t)inner_shape_.size() + 1);
  }

  int64_t
  NumpyForm::numfields() const {
    return -1;
  }

  int64_t
  NumpyForm::fieldindex(const std::string& key) const {
    throw std::invalid_argument(
      std::string("key ") + util::quote(key)
      + std::string(" does not exist (data are not records)")
      + FILENAME(__LINE__));
  }

  const std::string
  NumpyForm::key(int64_t fieldindex) const {
    throw std::invalid_argument(
      std::string("fieldindex \"") + std::to_string(fieldindex)
      + std::string("\" does not exist (data are not records)")
      + FILENAME(__LINE__));
  }

  bool
  NumpyForm::haskey(const std::string& key) const {
    return false;
  }

  const std::vector<std::string>
  NumpyForm::keys() const {
    return std::vector<std::string>();
  }

  bool
  NumpyForm::istuple() const {
    return false;
  }

  bool
  NumpyForm::equal(const FormPtr& other,
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
    if (NumpyForm* t = dynamic_cast<NumpyForm*>(other.get())) {
      return (inner_shape_ == t->inner_shape()  &&  format_ == t->format());
    }
    else {
      return false;
    }
  }

  const FormPtr
  NumpyForm::getitem_field(const std::string& key) const {
    throw std::invalid_argument(
      std::string("key ") + util::quote(key)
      + std::string(" does not exist (data are not records)"));
  }

  const FormPtr
  NumpyForm::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::invalid_argument(
      std::string("requested keys do not exist (data are not records)"));
  }
}
