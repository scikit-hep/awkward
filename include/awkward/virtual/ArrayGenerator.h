// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ARRAYGENERATOR_H_
#define AWKWARD_ARRAYGENERATOR_H_

#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  ////////// ArrayGenerator

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
    ///
    /// Similarly, the #length can be specified to avoid materializing a
    /// VirtualArray when its length must be known. Any negative value,
    /// such as -1, is interpreted as "unknown."
    ArrayGenerator(const FormPtr& form, int64_t length);

    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~ArrayGenerator();

    /// @brief The Form the generated array is expected to take; may be
    /// `nullptr`.
    const FormPtr
      form() const;

    /// @brief The length the generated array is expected to have; may
    /// be negative to indicate that the length is unknown.
    int64_t
      length() const;

    /// @brief Creates an array but does not check it against the #form.
    virtual const ContentPtr
      generate() const = 0;

    /// @brief Creates an array and checks it against the #form.
    const ContentPtr
      generate_and_check() const;

    virtual const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const = 0;

  protected:
    const FormPtr form_;
    int64_t length_;
  };

  using ArrayGeneratorPtr = std::shared_ptr<ArrayGenerator>;

  ////////// SliceGenerator

  /// @class SliceGenerator
  ///
  /// @brief Generator for lazy slicing. Used to avoid materializing a
  /// VirtualArray before its content is needed (in case its content is
  /// never needed).
  class EXPORT_SYMBOL SliceGenerator: public ArrayGenerator {
  public:
    SliceGenerator(const FormPtr& form,
                   int64_t length,
                   const ContentPtr& content,
                   const Slice& slice);

    const ContentPtr
      content() const;

    const Slice
      slice() const;

    const ContentPtr
      generate() const override;

    const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const override;

  protected:
    const ContentPtr content_;
    const Slice slice_;
  };
}

#endif // AWKWARD_ARRAYGENERATOR_H_
