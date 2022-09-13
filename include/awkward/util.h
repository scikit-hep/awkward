// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_UTIL_H_
#define AWKWARD_UTIL_H_

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "awkward/common.h"

#ifndef _MSC_VER
  #include "dlfcn.h"
#endif

namespace awkward {
  namespace util {
    /// @brief NumPy dtypes that can be interpreted within Awkward C++
    /// (only the primitive, fixed-width types). Non-native-endian types
    /// are considered NOT_PRIMITIVE.
    enum class dtype {
        NOT_PRIMITIVE,
        boolean,
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float16,
        float32,
        float64,
        float128,
        complex64,
        complex128,
        complex256,
        datetime64,
        timedelta64,
        size
    };

    /// @brief Returns the name associated with a given dtype.
    dtype
    name_to_dtype(const std::string& name);

    /// @brief Returns the name associated with a given dtype.
    const std::string
    dtype_to_name(dtype dt);

    /// @brief Convert a NumPy format string and itemsize into a dtype enum.
    dtype
    format_to_dtype(const std::string& format, int64_t itemsize);

    /// @brief Convert a dtype enum into a NumPy format string.
    const std::string
    dtype_to_format(dtype dt, const std::string& format = "");

    /// @brief Convert a dtype enum into an itemsize.
    int64_t
    dtype_to_itemsize(dtype dt);

    /// @brief True if the dtype is a non-boolean integer (signed or unsigned).
    bool
    is_integer(dtype dt);

    /// @brief True if the dtype is a signed integer.
    bool
    is_signed(dtype dt);

    /// @brief True if the dtype is an unsigned integer.
    bool
    is_unsigned(dtype dt);

    /// @brief True if the dtype is a non-complex floating point number.
    bool
    is_real(dtype dt);

    /// @brief True if the dtype is a complex number.
    bool
    is_complex(dtype dt);

    /// @brief If the Error struct contains an error message (from a
    /// cpu-kernel through the C interface), raise that error as a C++
    /// exception.
    ///
    /// @param err The Error struct from a cpu-kernel.
    /// @param classname The name of this class to include in the error
    /// message.
    void
    handle_error(const struct Error &err,
                 const std::string &classname = std::string(""));

    /// @brief Puts quotation marks around a string and escapes the appropriate
    /// characters.
    ///
    /// @param x The string to quote.
    /// @param doublequote If `true`, apply double-quotes (`"`); if `false`,
    /// apply single-quotes (`'`).
    ///
    /// @note The implementation does not yet escape characters: it only adds
    /// strings. See issue
    /// [scikit-hep/awkward-1.0#186](https://github.com/scikit-hep/awkward-1.0/issues/186).
    std::string
      quote(const std::string& x);

    using RecordLookup    = std::vector<std::string>;
    using RecordLookupPtr = std::shared_ptr<RecordLookup>;

    /// @brief Initializes a RecordLookup by assigning each element with
    /// a string representation of its field index position.
    ///
    /// For example, if `numfields = 3`, the return value is `["0", "1", "2"]`.
    RecordLookupPtr
      init_recordlookup(int64_t numfields);

    /// @brief Returns the field index associated with a key, given
    /// a RecordLookup and a number of fields.
    int64_t
      fieldindex(const RecordLookupPtr& recordlookup,
                 const std::string& key,
                 int64_t numfields);

    /// @brief Returns the key associated with a field index, given a
    /// RecordLookup and a number of fields.
    const std::string
      key(const RecordLookupPtr& recordlookup,
          int64_t fieldindex,
          int64_t numfields);

    /// @brief Returns `true` if a RecordLookup has a given `key`; `false`
    /// otherwise.
    bool
      haskey(const RecordLookupPtr& recordlookup,
             const std::string& key,
             int64_t numfields);

    /// @brief Returns a given RecordLookup as keys or generate anonymous ones
    /// form a number of fields.
    const std::vector<std::string>
      keys(const RecordLookupPtr& recordlookup, int64_t numfields);

    using Parameters = std::map<std::string, std::string>;

    /// @brief Returns `true` if `myvalue` is equal to `value` when interpreted
    /// as JSON.
    bool
    json_equals(const std::string &myvalue, const std::string &value);

    /// @brief Returns `true` if the value associated with a `key` in
    /// `parameters` is equal to the specified `value`.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    bool
      parameter_equals(const Parameters& parameters,
                       const std::string& key,
                       const std::string& value);

    /// @brief Returns `true` if all key-value pairs in `self` is equal to
    /// all key-value pairs in `other`.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    ///
    /// If `check_all`, every parameter is checked; otherwise, only
    /// `"__array__"` and `"__record__"` are checked.
    bool
      parameters_equal(const Parameters& self, const Parameters& other, bool check_all);

    /// @brief Merges a set of `input` parameters with `output`, keeping
    /// only those that are common to all sets.
    void
      merge_parameters(Parameters& output, const Parameters& input);

    /// @brief Returns `true` if the parameter associated with `key` is a
    /// string; `false` otherwise.
    bool
      parameter_isstring(const Parameters& parameters, const std::string& key);

    /// @brief Returns `true` if the parameter associated with `key` is a
    /// string that matches `[A-Za-z_][A-Za-z_0-9]*`; `false` otherwise.
    bool
      parameter_isname(const Parameters& parameters, const std::string& key);

    /// @brief Returns the parameter associated with `key` as a string if
    /// #parameter_isstring; raises an error otherwise.
    const std::string
      parameter_asstring(const Parameters& parameters, const std::string& key);

    using TypeStrs = std::map<std::string, std::string>;

    /// @brief Extracts a custom type string from `typestrs` if required by
    /// one of the `parameters` or an empty string if there is no match.
    std::string
      gettypestr(const Parameters& parameters,
                 const TypeStrs& typestrs);

    /// @brief Exhaustive list of runtime errors possible in the ForthMachine.
    enum class ForthError {
        // execution can continue
        none,

        // execution cannot continue
        not_ready,
        is_done,
        user_halt,
        recursion_depth_exceeded,
        stack_underflow,
        stack_overflow,
        read_beyond,
        seek_beyond,
        skip_beyond,
        rewind_beyond,
        division_by_zero,
        varint_too_big,
        text_number_missing,
        quoted_string_missing,
        enumeration_missing,

        size
    };

    /// @class array_deleter
    ///
    /// @brief Used as a `std::shared_ptr` deleter (second argument) to
    /// overload `delete ptr` with `delete[] ptr`.
    ///
    /// This is necessary for `std::shared_ptr` to contain array buffers.
    ///
    /// See also
    ///   - pyobject_deleter, which reduces the reference count of a
    ///     Python object when there are no more C++ shared pointers
    ///     referencing it.
    template <typename T>
    class LIBAWKWARD_EXPORT_SYMBOL array_deleter {
    public:
      /// @brief Called by `std::shared_ptr` when its reference count reaches
      /// zero.
      void
      operator()(T const* ptr) {
        uint8_t const* in = reinterpret_cast<uint8_t const*>(ptr);
        delete [] in;
      }
    };

  }
}

#endif // AWKWARD_UTIL_H_
