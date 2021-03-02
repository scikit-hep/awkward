// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/VirtualArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/VirtualArray.h"
#include "awkward/array/EmptyArray.h"

namespace awkward {

  ///
  VirtualArrayBuilder::VirtualArrayBuilder(const VirtualFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) { }

  const std::string
  VirtualArrayBuilder::classname() const {
    return "VirtualArrayBuilder";
  }

  const ContentPtr
  VirtualArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    // FIXME:
    return std::make_shared<EmptyArray>(Identities::none(),
                                        form_.get()->parameters());
  }

  const FormPtr
  VirtualArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  VirtualArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  VirtualArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  VirtualArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

}
