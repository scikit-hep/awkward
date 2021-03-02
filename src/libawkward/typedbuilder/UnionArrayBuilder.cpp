// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/UnionArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/EmptyArray.h"

namespace awkward {

  ///
  UnionArrayBuilder::UnionArrayBuilder(const UnionFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-tags");
 }

  const std::string
  UnionArrayBuilder::classname() const {
    return "UnionArrayBuilder";
  }

  const ContentPtr
  UnionArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    // FIXME:
    return std::make_shared<EmptyArray>(Identities::none(),
                                        form_.get()->parameters());
      // return std::make_shared<UnionArray8_64>(Identities::none(),
      //                                         form_.get()->parameters());
  }

  const FormPtr
  UnionArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  UnionArrayBuilder::vm_output() const {
    return std::string("output ")
      .append(vm_output_data_)
      .append("\n");
  }

  const std::string
  UnionArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(*form_key_)
      .append("-")
      .append("union\n");
  }

  const std::string
  UnionArrayBuilder::vm_func_name() const {
    return std::string(*form_key_)
      .append("-union");
  }

}
