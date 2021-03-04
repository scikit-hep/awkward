// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_FORMBUILDER_H_
#define AWKWARD_FORMBUILDER_H_

#include <map>
#include <memory>
#include <string>
#include "awkward/forth/ForthOutputBuffer.h"

namespace awkward {

  class Content;
  using ContentPtr = std::shared_ptr<Content>;
  using ForthOutputBufferMap = std::map<std::string, std::shared_ptr<ForthOutputBuffer>>;

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
  };
}

#endif // AWKWARD_FORMBUILDER_H_
