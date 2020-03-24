// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLE_H_
#define AWKWARD_FILLABLE_H_

#include <string>
#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"

namespace awkward {
  class Builder;
  using BuilderPtr = std::shared_ptr<Builder>;

  /// @class Builder
  ///
  /// @brief Abstract base class for nodes within an ArrayBuilder that
  /// cumulatively discover an array's type and fill it.
  class EXPORT_SYMBOL Builder {
  public:
    /// @brief Empty destructor; required for some C++ reason.
    virtual ~Builder();

    /// @brief User-friendly name of this class.
    virtual const std::string
      classname() const = 0;

    /// @brief Current length of the accumulated array.
    virtual int64_t
      length() const = 0;

    /// @brief Removes all accumulated data without resetting the type
    /// knowledge.
    virtual void
      clear() = 0;

    /// @brief Turns the accumulated data into a Content array.
    virtual const ContentPtr
      snapshot() const = 0;

    /// @brief If `true`, this node has started but has not finished a
    /// multi-step command (e.g. `beginX ... endX`).
    virtual bool
      active() const = 0;

    /// @brief Adds a `null` value to the accumulated data.
    virtual const BuilderPtr
      null() = 0;

    /// @brief Adds a boolean value `x` to the accumulated data.
    virtual const BuilderPtr
      boolean(bool x) = 0;

    /// @brief Adds an integer value `x` to the accumulated data.
    virtual const BuilderPtr
      integer(int64_t x) = 0;

    /// @brief Adds a real value `x` to the accumulated data.
    virtual const BuilderPtr
      real(double x) = 0;

    /// @brief Adds a string value `x` with a given `length` and `encoding`
    /// to the accumulated data.
    ///
    /// @note Currently, only `encoding =`
    ///
    ///    - `nullptr` (no encoding; a bytestring)
    ///    - `"utf-8"` (variable-length Unicode 8-bit encoding)
    ///
    /// are supported.
    virtual const BuilderPtr
      string(const char* x, int64_t length, const char* encoding) = 0;

    /// @brief Begins building a nested list.
    virtual const BuilderPtr
      beginlist() = 0;

    /// @brief Ends a nested list.
    virtual const BuilderPtr
      endlist() = 0;

    /// @brief Begins building a tuple with a fixed number of fields.
    virtual const BuilderPtr
      begintuple(int64_t numfields) = 0;

    /// @brief Sets the pointer to a given tuple field index; the next
    /// command will fill that slot.
    virtual const BuilderPtr
      index(int64_t index) = 0;

    /// @brief Ends a tuple.
    virtual const BuilderPtr
      endtuple() = 0;

    /// @brief Begins building a record with an optional name.
    ///
    /// @param name If specified, this name is both used to distinguish
    /// records of different types in heterogeneous data (to build a
    /// union of record arrays, rather than a record array with union
    /// fields and optional values) and it also sets the `"__record__"`
    /// parameter to add custom behaviors in Python.
    /// @check If `true`, actually do a string comparison to see if the
    /// provided `name` matches the previous `name; if `false`, assume they
    /// are the same or different based on the pointer value.
    virtual const BuilderPtr
      beginrecord(const char* name, bool check) = 0;

    virtual const BuilderPtr
      field(const char* key, bool check) = 0;

    virtual const BuilderPtr
      endrecord() = 0;

    virtual const BuilderPtr
      append(const ContentPtr& array, int64_t at) = 0;

    void
      setthat(const BuilderPtr& that);

  protected:
    BuilderPtr that_;
  };
}

#endif // AWKWARD_FILLABLE_H_
