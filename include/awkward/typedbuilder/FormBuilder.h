// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_FORMBUILDER_H_
#define AWKWARD_FORMBUILDER_H_

#include <complex>
#include <map>
#include <memory>
#include <string>
#include "awkward/forth/ForthMachine.h"
#include "awkward/forth/ForthOutputBuffer.h"

namespace awkward {

  class Content;
  using ContentPtr = std::shared_ptr<Content>;
  using ForthOutputBufferMap = std::map<std::string, std::shared_ptr<ForthOutputBuffer>>;
  class TypedArrayBuilder;

  typedef void (TypedArrayBuilder::*int64)(int64_t x);
  typedef void (TypedArrayBuilder::*add_int64)(int64_t x);

  /// @class FormBuilder
  ///
  /// @brief Abstract base class for nodes within a TypedArrayBuilder
  /// Every builder will have an output buffer based on the
  /// key_format="part{partition}-{form_key}-{attribute}"
  ///
  class LIBAWKWARD_EXPORT_SYMBOL FormBuilder {
  public:
    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~FormBuilder();

    /// @brief User-friendly name of this class.
    virtual const std::string
      classname() const = 0;

    /// @brief Turns the accumulated data into a Content array.
    virtual const ContentPtr
      snapshot(const ForthOutputBufferMap& outputs) const = 0;

    /// @brief
    virtual const FormPtr
      form() const = 0;

    /// @brief
    virtual const std::string
      vm_output() const = 0;

    /// @brief
    virtual const std::string
      vm_output_data() const = 0;

    /// @brief
    virtual const std::string
      vm_func() const = 0;

    /// @brief
    virtual const std::string
      vm_func_name() const = 0;

    /// @brief
    virtual const std::string
      vm_func_type() const = 0;

    /// @brief
    virtual const std::string
      vm_from_stack() const = 0;

    /// @brief
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
      boolean(bool x, TypedArrayBuilder* builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'boolean' is not implemented yet"));
      }

    /// @brief Adds an integer value `x` to the accumulated data.
    virtual void
      int64(int64_t x, TypedArrayBuilder* builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'int64' is not implemented yet"));
      }

    /// @brief Adds a real value `x` to the accumulated data.
    virtual void
      float64(double x, TypedArrayBuilder* builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'float64' is not implemented yet"));
      }

    /// @brief Adds a complex value `x` to the accumulated data.
    virtual void
      complex(std::complex<double> x, TypedArrayBuilder* builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'complex' is not implemented yet"));
      }

    /// @brief Adds an unencoded bytestring `x` in STL format to the
    /// accumulated data.
    virtual void
      bytestring(const std::string& x, TypedArrayBuilder* builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'bytestring' is not implemented yet"));
      }

    /// @brief Adds a UTF-8 encoded bytestring `x` in STL format to the
    /// accumulated data.
    virtual void
      string(const std::string& x, TypedArrayBuilder* builder) {
        throw std::runtime_error(
          std::string("FormBuilder 'string' is not implemented yet"));
      }

  };
}

#endif // AWKWARD_FORMBUILDER_H_
