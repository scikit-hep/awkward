// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

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
    enum class EXPORT_SYMBOL dtype {
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
    EXPORT_SYMBOL const std::string
    dtype_to_name(dtype dt);

    /// @brief Convert a dtype enum into a NumPy format string.
    EXPORT_SYMBOL const std::string
    dtype_to_format(dtype dt, const std::string& format = "");

    /// @brief Puts quotation marks around a string and escapes the appropriate
    /// characters.
    ///
    /// @param x The string to quote.
    /// @param doublequote If `true`, apply double-quotes (`"`); if `false`,
    /// apply single-quotes (`'`).
    ///
    /// @note The implementation does not yet escape characters: it only adds
    /// strings. See issue
    /// [scikit-hep/awkward#186](https://github.com/scikit-hep/awkward/issues/186).
    std::string
      quote(const std::string& x);

    /// @brief Exhaustive list of runtime errors possible in the ForthMachine.
    enum class EXPORT_SYMBOL ForthError {
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
    class EXPORT_SYMBOL array_deleter {
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
