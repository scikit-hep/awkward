// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_FORMBUILDER_H_
#define AWKWARD_FORMBUILDER_H_

#include <complex>
#include <map>
#include <memory>
#include <string>
#include "awkward/forth/ForthMachine.h"
#include "awkward/forth/ForthOutputBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {

  using ForthOutputBufferMap = std::map<std::string, std::shared_ptr<ForthOutputBuffer>>;

  template <typename T, typename I> class LayoutBuilder;

  template<typename T, typename I>
  using LayoutBuilderPtr = LayoutBuilder<T, I>*;

  /// @class FormBuilder
  ///
  /// @brief Abstract base class for nodes within a LayoutBuilder
  /// Every builder will have an output buffer based on the
  /// key_format="part{partition}-{form_key}-{attribute}"
  ///
  template<typename T, typename I>
  class LIBAWKWARD_EXPORT_SYMBOL FormBuilder {
  public:
    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~FormBuilder();

    /// @brief User-friendly name of this class.
    virtual const std::string
      classname() const = 0;

    /// @brief Copy the current snapshot into the BuffersContainer and
    /// return a Form as a std::string (JSON).
    virtual const std::string
      to_buffers(BuffersContainer& container, const ForthOutputBufferMap& outputs) const = 0;

    /// @brief The builder's output buffer length.
    virtual ssize_t
      len(const ForthOutputBufferMap& outputs) const = 0;

    virtual bool
      is_complex() const {
        return false;
      }

    // /// @brief The Form describing the array.
    // virtual const FormPtr
    //   form() const = 0;

    /// @brief AwkwardForth virtual machine instructions of the data outputs.
    virtual const std::string
      vm_output() const = 0;

    /// @brief AwkwardForth virtual machine data output key.
    virtual const std::string
      vm_output_data() const = 0;

    /// @brief AwkwardForth virtual machine instructions of the array builder function.
    virtual const std::string
      vm_func() const = 0;

    /// @brief The array builder VM function name.
    virtual const std::string
      vm_func_name() const = 0;

    /// @brief The array builder VM function type.
    virtual const std::string
      vm_func_type() const = 0;

    /// @brief AwkwardForth virtual machine instructions to retrieve the data from
    /// the VM stack.
    virtual const std::string
      vm_from_stack() const = 0;

    /// @brief Error messages in the AwkwardForth virtual machine instructions.
    virtual const std::string
      vm_error() const = 0;

    /// @brief Adds an integer value `x` to the accumulated data.
    virtual void
      tag(int8_t x) {
        throw std::runtime_error(
          std::string("FormBuilder 'tag' is not implemented yet"));
      }

    /// @brief Adds a boolean value `x` to the accumulated data.
    virtual void
      boolean(bool x, LayoutBuilderPtr<T, I> builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'boolean' is not implemented yet"));
      }

    /// @brief Adds an integer value `x` to the accumulated data.
    virtual void
      int64(int64_t x, LayoutBuilderPtr<T, I> builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'int64' is not implemented yet"));
      }

    /// @brief Adds a real value `x` to the accumulated data.
    virtual void
      float64(double x, LayoutBuilderPtr<T, I> builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'float64' is not implemented yet"));
      }

    /// @brief Adds a complex value `x` to the accumulated data.
    virtual void
      complex(std::complex<double> x, LayoutBuilderPtr<T, I> builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'complex' is not implemented yet"));
      }

    /// @brief Adds an unencoded bytestring `x` in STL format to the
    /// accumulated data.
    virtual void
      bytestring(const std::string& x, LayoutBuilderPtr<T, I> builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'bytestring' is not implemented yet"));
      }

    /// @brief Adds a UTF-8 encoded bytestring `x` in STL format to the
    /// accumulated data.
    virtual void
      string(const std::string& x, LayoutBuilderPtr<T, I> builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'string' is not implemented yet"));
      }

    /// @brief Begins building a nested list.
    virtual void
      begin_list(LayoutBuilderPtr<T, I> builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'begin_list' is not implemented yet"));
      }

    /// @brief Ends a nested list.
    virtual void
      end_list(LayoutBuilderPtr<T, I> builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'end_list' is not implemented yet"));
      }

    /// @brief If `true`, this node has started but has not finished a
    /// multi-step command (e.g. `begin_list ... end_list`).
    virtual bool
      active() {
        return false;
      }

    /// @brief FIXME: find if it's already implemented in utils
    virtual std::string
    parameters_as_string(const util::Parameters& parameters) const {
      std::stringstream p;
      if (!parameters.empty()) {
        p << "\"parameters\": {";
        for (auto const &pair: parameters) {
          p << "\"" << pair.first << "\": " << pair.second << " ";
        }
        p << "}, ";
      }
      return p.str();
    }

  };

  template <typename T, typename I>
  using FormBuilderPtr = std::shared_ptr<FormBuilder<T, I>>;

}

#endif // AWKWARD_FORMBUILDER_H_
