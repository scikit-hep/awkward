// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/RegularArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/RegularArray.h"

namespace awkward {

  ///
  RegularArrayBuilder::RegularArrayBuilder(const RegularFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) { }

  const std::string
  RegularArrayBuilder::classname() const {
    return "RegularArrayBuilder";
  }

  const ContentPtr
  RegularArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    ContentPtr out;
    if(content_ != nullptr) {
      int64_t length = 0; // outputs.len(); // FIXME
      out = std::make_shared<RegularArray>(Identities::none(),
                                           form_.get()->parameters(),
                                           content_.get()->snapshot(outputs),
                                           length); // FIXME
    }
    return out;
  }

  const FormPtr
  RegularArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  RegularArrayBuilder::vm_output() const {
    return std::string("\n");
  }

  const std::string
  RegularArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(vm_func_name())
      .append("\n");
  }

  const std::string
  RegularArrayBuilder::vm_func_name() const {
    return std::string(*form_key_)
      .append("-reg");
  }

}
