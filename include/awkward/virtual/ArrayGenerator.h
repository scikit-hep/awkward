// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ARRAYGENERATOR_H_
#define AWKWARD_ARRAYGENERATOR_H_

#include "awkward/Content.h"

namespace awkward {
  /// @class ArrayGenerator
  ///
  /// @brief Abstract superclass to generat arrays for VirtualArray, definining
  /// the interface.
  ///
  /// The main implementation, PyArrayGenerator, is passed through pybind11 to
  /// Python to work with Python functions and lambdas, but in principle, pure
  /// C++ generators could be written.
  class EXPORT_SYMBOL ArrayGenerator {
  public:
    /// @brief Called by subclasses to set the #form of an ArrayGenerator.
    ///
    /// The #form can be `nullptr`, in which case the generated array can have
    /// any Form and any Type, but doing so would cause VirtualArray to
    /// materialize in more circumstances, undermining its usefulness.
    ArrayGenerator(const FormPtr& form);

    /// @brief The Form the generated array is expected to take; may be
    /// `nullptr`.
    const FormPtr
      form() const;

    /// @brief Creates an array but does not check it against the #form.
    virtual const ContentPtr
      generate() const = 0;

    /// @brief Creates an array and checks it against the #form.
    const ContentPtr
      generate_and_check() const;

  protected:
    const FormPtr form_;
  };

  using ArrayGeneratorPtr = std::shared_ptr<ArrayGenerator>;

}

#endif // AWKWARD_ARRAYGENERATOR_H_
