// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/EmptyArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/EmptyArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  /// @class EmptyArrayBuilder
  ///
  /// @brief EmptyArray builder from a Empty Json Form
  EmptyArrayBuilder::EmptyArrayBuilder(const util::Parameters& parameters)
    : parameters_(parameters),
      vm_empty_command_("( This does nothing. )\n"),
      vm_error_("s\" EmptyArray Builder error\"") { }

  const std::string
  EmptyArrayBuilder::classname() const {
    return "EmptyArrayBuilder";
  }

  const std::string
  EmptyArrayBuilder::vm_output() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_output_data() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_func() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_func_name() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_func_type() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_from_stack() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  EmptyArrayBuilder::boolean(bool x, LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'bool'"));
  }

  void
  EmptyArrayBuilder::int64(int64_t x, LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'int64'"));
  }

  void
  EmptyArrayBuilder::float64(double x, LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'float64'"));
  }

  void
  EmptyArrayBuilder::complex(std::complex<double> x, LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'complex'"));
  }

  void
  EmptyArrayBuilder::bytestring(const std::string& x, LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'bytestring'"));
  }

  void
  EmptyArrayBuilder::string(const std::string& x, LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'string'"));
  }

  void
  EmptyArrayBuilder::begin_list(LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'begin_list'"));
  }

  void
  EmptyArrayBuilder::end_list(LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'end_list'"));
  }

}
