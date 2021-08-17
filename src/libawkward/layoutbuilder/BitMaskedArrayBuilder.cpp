// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/BitMaskedArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/BitMaskedArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  /// @bclass BitMaskedArrayBuilder
  ///
  /// @brief BitMaskedArray builder from a Bit Masked Json Form
  BitMaskedArrayBuilder::BitMaskedArrayBuilder(FormBuilderPtr content,
                                               const util::Parameters& parameters,
                                               const std::string& form_key,
                                               const std::string attribute,
                                               const std::string partition)
    : content_(content),
      parameters_(parameters) {
    vm_func_name_ = std::string(form_key).append("-").append(attribute);

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
  BitMaskedArrayBuilder::classname() const {
    return "BitMaskedArrayBuilder";
  }

  const std::string
  BitMaskedArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  BitMaskedArrayBuilder::vm_output_data() const {
    return vm_output_data_;
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

  const std::string
  BitMaskedArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  BitMaskedArrayBuilder::boolean(bool x, LayoutBuilder* builder) {
    content_.get()->boolean(x, builder);
  }

  void
  BitMaskedArrayBuilder::int64(int64_t x, LayoutBuilder* builder) {
    content_.get()->int64(x, builder);
  }

  void
  BitMaskedArrayBuilder::float64(double x, LayoutBuilder* builder) {
    content_.get()->float64(x, builder);
  }

  void
  BitMaskedArrayBuilder::complex(std::complex<double> x, LayoutBuilder* builder) {
    content_.get()->complex(x, builder);
  }

  void
  BitMaskedArrayBuilder::bytestring(const std::string& x, LayoutBuilder* builder) {
    content_.get()->bytestring(x, builder);
  }

  void
  BitMaskedArrayBuilder::string(const std::string& x, LayoutBuilder* builder) {
    content_.get()->string(x, builder);
  }

  void
  BitMaskedArrayBuilder::begin_list(LayoutBuilder* builder) {
    content_.get()->begin_list(builder);
  }

  void
  BitMaskedArrayBuilder::end_list(LayoutBuilder* builder) {
    content_.get()->end_list(builder);
  }

}
