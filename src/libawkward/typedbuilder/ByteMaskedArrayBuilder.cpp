// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/ByteMaskedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/ByteMaskedArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/ByteMaskedArray.h"

namespace awkward {

  ///
  ByteMaskedArrayBuilder::ByteMaskedArrayBuilder(const ByteMaskedFormPtr& form,
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
    vm_func_name_ = std::string(*form_key_).append("-").append(attribute_);

    vm_func_type_ = content_.get()->vm_func_type();

    vm_func_.append(content_.get()->vm_func())
      .append(": ")
      .append(vm_func_name_).append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append(";").append("\n");

    vm_output_ = content_.get()->vm_output();
    vm_error_ = content_.get()->vm_error();
  }

  const std::string
  ByteMaskedArrayBuilder::classname() const {
    return "ByteMaskedArrayBuilder";
  }

  const ContentPtr
  ByteMaskedArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    // FIXME: how to define a mask? is it needed?
    return content_.get()->snapshot(outputs);
  }

  const FormPtr
  ByteMaskedArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  ByteMaskedArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  ByteMaskedArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }

  const std::string
  ByteMaskedArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  ByteMaskedArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  ByteMaskedArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  ByteMaskedArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  const std::string
  ByteMaskedArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  ByteMaskedArrayBuilder::boolean(bool x, TypedArrayBuilder* builder) {
    content_.get()->boolean(x, builder);
  }

  void
  ByteMaskedArrayBuilder::int64(int64_t x, TypedArrayBuilder* builder) {
    content_.get()->int64(x, builder);
  }

  void
  ByteMaskedArrayBuilder::float64(double x, TypedArrayBuilder* builder) {
    content_.get()->float64(x, builder);
  }

  void
  ByteMaskedArrayBuilder::complex(std::complex<double> x, TypedArrayBuilder* builder) {
    content_.get()->complex(x, builder);
  }

  void
  ByteMaskedArrayBuilder::bytestring(const std::string& x, TypedArrayBuilder* builder) {
    content_.get()->bytestring(x, builder);
  }

  void
  ByteMaskedArrayBuilder::string(const std::string& x, TypedArrayBuilder* builder) {
    content_.get()->string(x, builder);
  }

  void
  ByteMaskedArrayBuilder::begin_list(TypedArrayBuilder* builder) {
    content_.get()->begin_list(builder);
  }

  void
  ByteMaskedArrayBuilder::end_list(TypedArrayBuilder* builder) {
    content_.get()->end_list(builder);
  }

}
