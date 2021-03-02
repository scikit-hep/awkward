// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/NumpyArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/NumpyArray.h"

namespace awkward {

  ///
  NumpyArrayBuilder::NumpyArrayBuilder(const NumpyFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-data");

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(dtype_to_name(form_.get()->dtype())).append("\n");

    vm_func_name_ = std::string(*form_key_)
      .append("-")
      .append(dtype_to_name(form_.get()->dtype()));

    vm_func_ = std::string(": ").append(vm_func_name()).append("\n")
      .append(dtype_to_state(form_.get()->dtype()))
      .append(" = if").append("\n")
      .append("0 data seek").append("\n")
      .append("data ").append(dtype_to_vm_format(form_.get()->dtype()))
      .append("-> ").append(vm_output_data_).append("\n")
      .append("else").append("\n")
      .append("halt").append("\n")
      .append("then").append("\n")
      .append(";").append("\n");
  }

  const std::string
  NumpyArrayBuilder::classname() const {
    return "NumpyArrayBuilder";
  }

  const ContentPtr
  NumpyArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      return search->second.get()->toNumpyArray();
    }
    throw std::invalid_argument(
        std::string("Snapshot of a ") + classname()
        + std::string(" needs data")
        + FILENAME(__LINE__));
  }

  const FormPtr
  NumpyArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  NumpyArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  NumpyArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  NumpyArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  NumpyArrayBuilder::vm_from_stack() const {
    return std::string();
  }

}
