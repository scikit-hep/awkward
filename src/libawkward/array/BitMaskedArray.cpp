// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/BitMaskedArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/BitMaskedArray.cpp", line)

#include <sstream>
#include <type_traits>

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"


#include "awkward/Reducer.h"
#include "awkward/Slice.h"
#include "awkward/io/json.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"

#include "awkward/array/BitMaskedArray.h"

namespace awkward {
  ////////// BitMaskedForm

  BitMaskedForm::BitMaskedForm(bool has_identities,
                               const util::Parameters& parameters,
                               const FormKey& form_key,
                               Index::Form mask,
                               const FormPtr& content,
                               bool valid_when,
                               bool lsb_order)
      : Form(has_identities, parameters, form_key)
      , mask_(mask)
      , content_(content)
      , valid_when_(valid_when)
      , lsb_order_(lsb_order) { }

  Index::Form
  BitMaskedForm::mask() const {
    return mask_;
  }

  const FormPtr
  BitMaskedForm::content() const {
    return content_;
  }

  bool
  BitMaskedForm::valid_when() const {
    return valid_when_;
  }

  bool
  BitMaskedForm::lsb_order() const {
    return lsb_order_;
  }

  void
  BitMaskedForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    builder.string("BitMaskedArray");
    builder.field("mask");
    builder.string(Index::form2str(mask_));
    builder.field("content");
    content_.get()->tojson_part(builder, verbose);
    builder.field("valid_when");
    builder.boolean(valid_when_);
    builder.field("lsb_order");
    builder.boolean(lsb_order_);
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  BitMaskedForm::shallow_copy() const {
    return std::make_shared<BitMaskedForm>(has_identities_,
                                           parameters_,
                                           form_key_,
                                           mask_,
                                           content_,
                                           valid_when_,
                                           lsb_order_);
  }

  const FormPtr
  BitMaskedForm::with_form_key(const FormKey& form_key) const {
    return std::make_shared<BitMaskedForm>(has_identities_,
                                           parameters_,
                                           form_key,
                                           mask_,
                                           content_,
                                           valid_when_,
                                           lsb_order_);
  }

  const std::string
  BitMaskedForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  bool
  BitMaskedForm::purelist_isregular() const {
    return content_.get()->purelist_isregular();
  }

  int64_t
  BitMaskedForm::purelist_depth() const {
    return content_.get()->purelist_depth();
  }

  bool
  BitMaskedForm::dimension_optiontype() const {
    return true;
  }

  const std::pair<int64_t, int64_t>
  BitMaskedForm::minmax_depth() const {
    return content_.get()->minmax_depth();
  }

  const std::pair<bool, int64_t>
  BitMaskedForm::branch_depth() const {
    return content_.get()->branch_depth();
  }

  int64_t
  BitMaskedForm::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  BitMaskedForm::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  BitMaskedForm::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  BitMaskedForm::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  BitMaskedForm::keys() const {
    return content_.get()->keys();
  }

  bool
  BitMaskedForm::istuple() const {
    return content_.get()->istuple();
  }

  bool
  BitMaskedForm::equal(const FormPtr& other,
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
    if (BitMaskedForm* t = dynamic_cast<BitMaskedForm*>(other.get())) {
      return (mask_ == t->mask()  &&
              content_.get()->equal(t->content(),
                                    check_identities,
                                    check_parameters,
                                    check_form_key,
                                    compatibility_check)  &&
              valid_when_ == t->valid_when()  &&
              lsb_order_ == t->lsb_order());
    }
    else {
      return false;
    }
  }

  const FormPtr
  BitMaskedForm::getitem_range() const {
    return std::make_shared<ByteMaskedForm>(
      has_identities_,
      parameters_,
      FormKey(nullptr),
      Index::Form::i8,
      content_.get()->getitem_range(),
      valid_when_);
  }

  const FormPtr
  BitMaskedForm::getitem_field(const std::string& key) const {
    BitMaskedForm step1(has_identities_,
                        util::Parameters(),
                        FormKey(nullptr),
                        mask_,
                        content_.get()->getitem_field(key),
                        valid_when_,
                        lsb_order_);
    return step1.simplify_optiontype();
  }

  const FormPtr
  BitMaskedForm::getitem_fields(const std::vector<std::string>& keys) const {
    BitMaskedForm step1(has_identities_,
                        util::Parameters(),
                        FormKey(nullptr),
                        mask_,
                        content_.get()->getitem_fields(keys),
                        valid_when_,
                        lsb_order_);
    return step1.simplify_optiontype();
  }

  const FormPtr
  BitMaskedForm::simplify_optiontype() const {
    if (dynamic_cast<IndexedForm*>(content_.get())         ||
        dynamic_cast<IndexedOptionForm*>(content_.get())   ||
        dynamic_cast<ByteMaskedForm*>(content_.get())      ||
        dynamic_cast<BitMaskedForm*>(content_.get())       ||
        dynamic_cast<UnmaskedForm*>(content_.get())) {
      IndexedOptionForm step1(has_identities_,
                              parameters_,
                              form_key_,
                              Index::Form::i64,
                              content_);
      return step1.simplify_optiontype();
    }
    else {
      return shallow_copy();
    }
  }
}
