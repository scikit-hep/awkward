// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/UnmaskedArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/EmptyArray.h"

namespace awkward {

  ///
  UnmaskedArrayBuilder::UnmaskedArrayBuilder(const UnmaskedFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-mask");
  }

  const std::string
  UnmaskedArrayBuilder::classname() const {
    return "UnmaskedArrayBuilder";
  }

  const ContentPtr
  UnmaskedArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    // FIXME
      return std::make_shared<EmptyArray>(Identities::none(),
                                          form_.get()->parameters());
  }

  const FormPtr
  UnmaskedArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  UnmaskedArrayBuilder::vm_output() const {
    return std::string("output ")
      .append(vm_output_data_)
      .append("\n");
  }

  const std::string
  UnmaskedArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(*form_key_)
      .append("-")
      .append("unmasked\n");
  }

  const std::string
  UnmaskedArrayBuilder::vm_func_name() const {
    return std::string(*form_key_)
      .append("-unmasked");
  }

}
