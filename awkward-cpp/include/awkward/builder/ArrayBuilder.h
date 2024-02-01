// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_ARRAYBUILDER_H_
#define AWKWARD_ARRAYBUILDER_H_

#include <complex>
#include <string>
#include <vector>

#include "awkward/common.h"
#include "awkward/builder/Builder.h"
#include "awkward/BuilderOptions.h"

namespace awkward {
  class Builder;
  using BuilderPtr = std::shared_ptr<Builder>;

  /// @class ArrayBuilder
  ///
  /// @brief User interface to the Builder system: the ArrayBuilder is a
  /// fixed reference while the Builder subclass instances change in
  /// response to accumulating data.
  class EXPORT_SYMBOL ArrayBuilder {
  public:
    /// @brief Creates an ArrayBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    ArrayBuilder(const BuilderOptions& options);

    /// @brief Copy the current snapshot into the BuffersContainer and
    /// return a Form as a std::string (JSON).
    const std::string
      to_buffers(BuffersContainer& container, int64_t& form_key_id) const;

    /// @brief Current length of the accumulated array.
    int64_t
      length() const;

    /// @brief Removes all accumulated data without resetting the type
    /// knowledge.
    void
      clear();

    /// @brief Adds a `null` value to the accumulated data.
    void
      null();

    /// @brief Adds a boolean value `x` to the accumulated data.
    void
      boolean(bool x);

    /// @brief Adds an integer value `x` to the accumulated data.
    void
      integer(int64_t x);

    /// @brief Adds a real value `x` to the accumulated data.
    void
      real(double x);

    /// @brief Adds a complex value `x` to the accumulated data.
    void
      complex(std::complex<double> x);

    /// @brief Adds a datetime value `x` to the accumulated data.
    void
      datetime(int64_t x, const std::string& unit);

    /// @brief Adds a timedelta value `x` to the accumulated data.
    void
      timedelta(int64_t x, const std::string& unit);

    /// @brief Adds an unencoded, null-terminated bytestring value `x` to the
    /// accumulated data.
    void
      bytestring(const char* x);

    /// @brief Adds an unencoded bytestring value `x` with a given `length`
    /// to the accumulated data.
    ///
    /// The string does not need to be null-terminated.
    void
      bytestring(const char* x, int64_t length);

    /// @brief Adds an unencoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      bytestring(const std::string& x);

    /// @brief Adds a UTF-8 encoded, null-terminated bytestring value `x` to
    /// the accumulated data.
    void
      string(const char* x);

    /// @brief Adds a UTF-8 encoded bytestring value `x` with a given `length`
    /// to the accumulated data.
    ///
    /// The string does not need to be null-terminated.
    void
      string(const char* x, int64_t length);

    /// @brief Adds a UTF-8 encoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      string(const std::string& x);

    /// @brief Begins building a nested list.
    void
      beginlist();

    /// @brief Ends a nested list.
    void
      endlist();

    /// @brief Begins building a tuple with a fixed number of fields.
    void
      begintuple(int64_t numfields);

    /// @brief Sets the pointer to a given tuple field index; the next
    /// command will fill that slot.
    void
      index(int64_t index);

    /// @brief Ends a tuple.
    void
      endtuple();

    /// @brief Begins building a record without a name.
    ///
    /// See #beginrecord_fast and #beginrecord_check.
    void
      beginrecord();

    /// @brief Begins building a record with a name.
    ///
    /// @param name This name is used to distinguish
    /// records of different types in heterogeneous data (to build a
    /// union of record arrays, rather than a record array with union
    /// fields and optional values) and it also sets the `"__record__"`
    /// parameter to later add custom behaviors in Python.
    ///
    /// In the `_fast` version of this method, a string comparison is not
    /// performed: the same pointer is assumed to have the same value each time
    /// (safe for string literals).
    ///
    /// See #beginrecord and #beginrecord_check.
    void
      beginrecord_fast(const char* name);

    /// @brief Begins building a record with a name.
    ///
    /// @param name This name is used to distinguish
    /// records of different types in heterogeneous data (to build a
    /// union of record arrays, rather than a record array with union
    /// fields and optional values) and it also sets the `"__record__"`
    /// parameter to later add custom behaviors in Python.
    ///
    /// In the `_check` version of this method, a string comparison is
    /// performed every time it is called to verify that the `name` matches
    /// a stored `name`.
    ///
    /// See #beginrecord and #beginrecord_fast.
    void
      beginrecord_check(const char* name);

    /// @brief Begins building a record with a name.
    ///
    /// @param name This name is used to distinguish
    /// records of different types in heterogeneous data (to build a
    /// union of record arrays, rather than a record array with union
    /// fields and optional values) and it also sets the `"__record__"`
    /// parameter to later add custom behaviors in Python.
    ///
    /// In the `_check` version of this method, a string comparison is
    /// performed every time it is called to verify that the `name` matches
    /// a stored `name`.
    ///
    /// See #beginrecord and #beginrecord_fast.
    void
      beginrecord_check(const std::string& name);

    /// @brief Sets the pointer to a given record field `key`; the next
    /// command will fill that slot.
    ///
    /// In the `_fast` version of this method, a string comparison is not
    /// performed: the same pointer is assumed to have the same value each time
    /// (safe for string literals). See #field_check.
    ///
    /// Record keys are checked in round-robin order. The best performance
    /// will be achieved by filling them in the same order for each record.
    /// Lookup time for random order scales with the number of fields.
    void
      field_fast(const char* key);

    /// @brief Sets the pointer to a given record field `key`; the next
    /// command will fill that slot.
    ///
    /// In the `_check` version of this method, a string comparison is
    /// performed every time it is called to verify that the `key` matches
    /// a stored `key`. See #field_fast.
    ///
    /// Record keys are checked in round-robin order. The best performance
    /// will be achieved by filling them in the same order for each record.
    /// Lookup time for random order scales with the number of fields.
    void
      field_check(const char* key);

    /// @brief Sets the pointer to a given record field `key`; the next
    /// command will fill that slot.
    ///
    /// In the `_check` version of this method, a string comparison is
    /// performed every time it is called to verify that the `key` matches
    /// a stored `key`. See #field_fast.
    ///
    /// Record keys are checked in round-robin order. The best performance
    /// will be achieved by filling them in the same order for each record.
    /// Lookup time for random order scales with the number of fields.
    void
      field_check(const std::string& key);

    /// @brief Ends a record.
    void
      endrecord();

    // @brief Root node of the Builder tree.
    const BuilderPtr builder() const { return builder_; }

    void builder_update(BuilderPtr builder) { builder_ = builder; }

    /// @brief Internal function to replace the root node of the ArrayBuilder's
    /// Builder tree with a new root.
    void
      maybeupdate(const BuilderPtr builder);

private:
    /// @brief Constant equal to `nullptr`.
    static const char* no_encoding;
    /// @brief Constant equal to `"utf-8"`.
    static const char* utf8_encoding;
    /// @brief Root node of the Builder tree.
    BuilderPtr builder_;
  };
}

extern "C" {
  /// @brief C interface to {@link awkward::ArrayBuilder#length ArrayBuilder::length}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_length(void* arraybuilder,
                                int64_t* result);
  /// @brief C interface to {@link awkward::ArrayBuilder#clear ArrayBuilder::clear}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_clear(void* arraybuilder);

  /// @brief C interface to {@link awkward::ArrayBuilder#null ArrayBuilder::null}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_null(void* arraybuilder);

  /// @brief C interface to {@link awkward::ArrayBuilder#boolean ArrayBuilder::boolean}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_boolean(void* arraybuilder,
                                 bool x);

  /// @brief C interface to {@link awkward::ArrayBuilder#integer ArrayBuilder::integer}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_integer(void* arraybuilder,
                                 int64_t x);

  /// @brief C interface to {@link awkward::ArrayBuilder#real ArrayBuilder::real}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_real(void* arraybuilder,
                              double x);

  /// @brief C interface to {@link awkward::ArrayBuilder#complex ArrayBuilder::complex}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_complex(void* arraybuilder,
                                 double real,
                                 double imag);

  /// @brief C interface to {@link awkward::ArrayBuilder#datetime ArrayBuilder::datetime}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_datetime(void* arraybuilder,
                                  int64_t x,
                                  const char* unit);

  /// @brief C interface to {@link awkward::ArrayBuilder#timedelta ArrayBuilder::timedelta}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_timedelta(void* arraybuilder,
                                   int64_t x,
                                   const char* unit);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#bytestring ArrayBuilder::bytestring}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_bytestring(void* arraybuilder,
                                    const char* x);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#bytestring ArrayBuilder::bytestring}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_bytestring_length(void* arraybuilder,
                                           const char* x,
                                           int64_t length);

  /// @brief C interface to {@link awkward::ArrayBuilder#string ArrayBuilder::string}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_string(void* arraybuilder,
                                const char* x);

  /// @brief C interface to {@link awkward::ArrayBuilder#string ArrayBuilder::string}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_string_length(void* arraybuilder,
                                       const char* x,
                                       int64_t length);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#beginlist ArrayBuilder::beginlist}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_beginlist(void* arraybuilder);

  /// @brief C interface to {@link awkward::ArrayBuilder#endlist ArrayBuilder::endlist}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_endlist(void* arraybuilder);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#begintuple ArrayBuilder::begintuple}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_begintuple(void* arraybuilder,
                                    int64_t numfields);

  /// @brief C interface to {@link awkward::ArrayBuilder#index ArrayBuilder::index}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_index(void* arraybuilder,
                               int64_t index);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#endtuple ArrayBuilder::endtuple}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_endtuple(void* arraybuilder);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#beginrecord ArrayBuilder::beginrecord}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_beginrecord(void* arraybuilder);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#beginrecord_fast ArrayBuilder::beginrecord_fast}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_beginrecord_fast(void* arraybuilder,
                                          const char* name);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#beginrecord_check ArrayBuilder::beginrecord_check}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_beginrecord_check(void* arraybuilder,
                                           const char* name);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#field_fast ArrayBuilder::field_fast}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_field_fast(void* arraybuilder,
                                    const char* key);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#field_check ArrayBuilder::field_check}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_field_check(void* arraybuilder,
                                     const char* key);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#endrecord ArrayBuilder::endrecord}.
  EXPORT_SYMBOL uint8_t
    awkward_ArrayBuilder_endrecord(void* arraybuilder);
}

#endif // AWKWARD_ARRAYBUILDER_H_
