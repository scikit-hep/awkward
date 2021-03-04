// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/BitMaskedArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/BitMaskedArray.h"

namespace awkward {

  /// @brief
  BitMaskedArrayBuilder::BitMaskedArrayBuilder(const BitMaskedFormPtr& form,
                                               const std::string attribute,
                                               const std::string partition)
    : form_(form),
      form_key_(!form.get()->form_key() ?
        std::make_shared<std::string>(std::string("node-id")
        + std::to_string(TypedArrayBuilder::next_id()))
        : form.get()->form_key()),
      attribute_(attribute),
      partition_(partition),
      content_(TypedArrayBuilder::formBuilderFromA(form.get()->content())) {
    vm_output_data_ = std::string("part")
      .append(partition_).append("-")
      .append(*form_key_).append("-")
      .append(attribute_);

    vm_func_name_ = std::string(*form_key_).append("-").append("bitmask");

    vm_func_ = std::string(": ")
      .append(vm_func_name_).append("\n")
      .append(";").append("\n");

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append("FIXME-type").append("\n");
  }

  const std::string
  BitMaskedArrayBuilder::classname() const {
    return "BitMaskedArrayBuilder";
  }

  const ContentPtr
  BitMaskedArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    ContentPtr out;
    int64_t length = 0; // FIXME
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      out = std::make_shared<BitMaskedArray>(Identities::none(),
                                             form_.get()->parameters(),
                                             search->second.get()->toIndexU8(),
                                             content_.get()->snapshot(outputs),
                                             form_.get()->valid_when(),
                                             length, // FIXME
                                             form_.get()->lsb_order());
    }
    return out;
  }

  const FormPtr
  BitMaskedArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  BitMaskedArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  BitMaskedArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  BitMaskedArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  BitMaskedArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  BitMaskedArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

}
