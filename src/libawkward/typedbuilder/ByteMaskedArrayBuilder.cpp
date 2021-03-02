// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/ByteMaskedArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/ByteMaskedArray.h"

namespace awkward {

  ///
  ByteMaskedArrayBuilder::ByteMaskedArrayBuilder(const ByteMaskedFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(TypedArrayBuilder::formBuilderFromA(form.get()->content())) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-mask");
  }

  const std::string
  ByteMaskedArrayBuilder::classname() const {
    return "ByteMaskedArrayBuilder";
  }

  const ContentPtr
  ByteMaskedArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    ContentPtr out;
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      out = std::make_shared<ByteMaskedArray>(Identities::none(),
                                              form_.get()->parameters(),
                                              search->second.get()->toIndex8(),
                                              content_.get()->snapshot(outputs),
                                              form_.get()->valid_when());
    }
    return out;
  }

  const FormPtr
  ByteMaskedArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  ByteMaskedArrayBuilder::vm_output() const {
    return std::string("output ")
      .append(vm_output_data_)
      .append("\n");
  }

  const std::string
  ByteMaskedArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(vm_func_name())
      .append(";\n");
  }

  const std::string
  ByteMaskedArrayBuilder::vm_func_name() const {
    std::string out;
    out.append(*form_key_)
      .append("-")
      .append("bytemask");
    return out;
  }

}
