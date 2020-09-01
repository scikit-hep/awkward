// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ITERATOR_H_
#define AWKWARD_ITERATOR_H_

#include "awkward/common.h"
#include "awkward/Content.h"

namespace awkward {
  /// @class Iterator
  ///
  /// @brief Represents the current state of iteration over a Content array.
  ///
  /// An Iterator is characterized by its #content (an input parameter) and
  /// where it is #at (in-place mutable integer).
  ///
  /// It can only be modified by calling #next.
  class LIBAWKWARD_EXPORT_SYMBOL Iterator {
  public:
    /// @brief Creates an Iterator from a full set of parameters.
    ///
    /// @param content The array to iterate over.
    Iterator(const ContentPtr& content);

    /// @brief The array to iterate over.
    const ContentPtr
      content() const;

    /// @brief The current position of the Iterator.
    const int64_t
      at() const;

    /// @brief If `true`, the Iterator has reached the end of the array and
    /// calling #next again would raise an error. If `false`, the Iterator
    /// can still be moved forward.
    const bool
      isdone() const;

    /// @brief Return the current item and then move the pointer to the next.
    const ContentPtr
      next();

    /// @brief Internal function to build an output string for #tostring.
    ///
    /// @param indent Indentation depth as a string of spaces.
    /// @param pre Prefix string, usually an opening XML tag.
    /// @param post Postfix string, usually a closing XML tag and carriage
    /// return.
    const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const;

    /// @brief Returns a string representation of this array (single-line XML).
    const std::string
      tostring() const;

  private:
    /// @brief See #content.
    const ContentPtr content_;
    /// @brief See #at.
    int64_t at_;
  };
}

#endif // AWKWARD_ITERATOR_H_
