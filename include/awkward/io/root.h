// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IO_ROOT_H_
#define AWKWARD_IO_ROOT_H_

#include <cstdio>
#include <string>

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/Index.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/Content.h"
#include "awkward/array/NumpyArray.h"

namespace awkward {
  /// @brief Create a Content array from a `std::vector` of `std::vectors`
  /// in ROOT serialization.
  ///
  /// @param byteoffsets The starting byte position for each ROOT entry.
  /// @param rawdata The raw bytes containing ROOT-serialized data (not
  /// including `byteoffsets`). This buffer must be uncompressed, but otherwise
  /// in its serialized form; for instance, it must be big-endian.
  /// @param depth The number of levels of `std::vectors` deep; any
  /// non-negative integer is allowed.
  /// @param itemsize The number of bytes in each numerical value in the
  /// deepest `std::vector`.
  /// @param format The pybind11 format string for the data type.
  /// @param options Configuration options for building an ArrayBuilder array.
  LIBAWKWARD_EXPORT_SYMBOL const ContentPtr
    FromROOT_nestedvector(const Index64& byteoffsets,
                          const NumpyArray& rawdata,
                          int64_t depth,
                          int64_t itemsize,
                          std::string format,
                          const ArrayBuilderOptions& options);
}

#endif // AWKWARD_IO_ROOT_H_
