// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_FILLABLE_H_
#define AWKWARD_FILLABLE_H_

#include <complex>
#include <string>
#include <vector>

#include "awkward/common.h"

namespace awkward {
  class Builder;
  using BuilderPtr = std::shared_ptr<Builder>;

  /// @class BuffersContainer
  ///
  /// @brief Abstract class to represent the output of ak.to_buffers.
  /// In Python, this would be a dict of NumPy arrays.
  class BuffersContainer {
  public:
    /// @brief Copy data at `source` with `num_bytes` into the BuffersContainer
    /// with `name`.
    ///
    /// In Python, this allocates a NumPy array and copies data into it.
    virtual void
      copy_buffer(const std::string& name, const void* source, int64_t num_bytes) = 0;

    /// @brief Create an array initialized to a given fill value.
    virtual void
      full_buffer(const std::string& name, int64_t length, int64_t value, const std::string& dtype) = 0;

    virtual void*
      empty_buffer(const std::string& name, int64_t num_bytes) = 0;
  };

  /// @class Builder
  ///
  /// @brief Abstract base class for nodes within an ArrayBuilder that
  /// cumulatively discover an array's type and fill it.
  class EXPORT_SYMBOL Builder: public std::enable_shared_from_this<Builder> {
  public:
    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~Builder();

    /// @brief User-friendly name of this class.
    virtual const std::string
      classname() const = 0;

    /// @brief Copy the current snapshot into the BuffersContainer and
    /// return a Form as a std::string (JSON).
    virtual const std::string
      to_buffers(BuffersContainer& container, int64_t& form_key_id) const = 0;

    /// @brief Current length of the accumulated array.
    virtual int64_t
      length() const = 0;

    /// @brief Removes all accumulated data without resetting the type
    /// knowledge.
    virtual void
      clear() = 0;

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

    /// @brief Adds a complex value `x` to the accumulated data.
    virtual const BuilderPtr
      complex(std::complex<double> x) = 0;

    /// @brief Adds a datetime value `x` to the accumulated data.
    virtual const BuilderPtr
      datetime(int64_t x, const std::string& unit) = 0;

    /// @brief Adds a timedelta value `x` to the accumulated data.
    virtual const BuilderPtr
      timedelta(int64_t x, const std::string& unit) = 0;

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
    /// @param name If specified, this name is used to distinguish
    /// records of different types in heterogeneous data (to build a
    /// union of record arrays, rather than a record array with union
    /// fields and optional values) and it also sets the `"__record__"`
    /// parameter to later add custom behaviors in Python.
    /// @param check If `true`, actually do a string comparison to see if the
    /// provided `name` matches the previous `name`; if `false`, assume
    /// that the same pointer means the same string (safe for string
    /// literals).
    virtual const BuilderPtr
      beginrecord(const char* name, bool check) = 0;

    /// @brief Sets the pointer to a given record field `key`; the next
    /// command will fill that slot.
    ///
    /// @param key Indicates the field to fill.
    /// @param check If `true`, actually do a string comparison to see if
    /// `key` matches the previous `key`; if `false`, assume that the same
    /// pointer means the same string (safe for string literals).
    ///
    /// Record keys are checked in round-robin order. The best performance
    /// will be achieved by filling them in the same order for each record.
    /// Lookup time for random order scales with the number of fields.
    virtual void
      field(const char* key, bool check) = 0;

    /// @brief Ends a record.
    virtual const BuilderPtr
      endrecord() = 0;
  };
}

#endif // AWKWARD_FILLABLE_H_
