// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_argsort.cpp", line)

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "awkward/kernels.h"

template <typename T>
bool argsort_order_ascending(T l, T r)
{
  return !std::isnan(static_cast<double>(r)) && (std::isnan(static_cast<double>(l)) || l < r);
}

template <typename T>
bool argsort_order_descending(T l, T r)
{
  return !std::isnan(static_cast<double>(r)) && (std::isnan(static_cast<double>(l)) || l > r);
}

template <typename T, typename U>
ERROR awkward_argsort(
  int64_t* toptr,
  const T* fromptr,
  int64_t length,
  const U* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  std::iota(toptr, toptr + length, 0);

  for (int64_t i = 0; i < offsetslength - 1; i++) {
    int64_t start_off = static_cast<int64_t>(offsets[i]);
    int64_t stop_off = static_cast<int64_t>(offsets[i + 1]);

    int64_t* segment_start = toptr + start_off;
    int64_t* segment_stop = toptr + stop_off;

    auto comparator = [&fromptr, ascending](int64_t i1, int64_t i2) {
        if (ascending) return argsort_order_ascending<T>(fromptr[i1], fromptr[i2]);
        else return argsort_order_descending<T>(fromptr[i1], fromptr[i2]);
    };

    if (stable) {
        std::stable_sort(segment_start, segment_stop, comparator);
    } else {
        std::sort(segment_start, segment_stop, comparator);
    }

    std::transform(segment_start, segment_stop, segment_start, [start_off](int64_t j) {
        return j - start_off;
    });
  }

  return success();
}

#define EXPORT_AWKWARD_ARGSORT(TYPE, TYPE_NAME)                                            \
  ERROR awkward_argsort_##TYPE_NAME##_int32(                                               \
    int64_t* toptr,                                                                        \
    const TYPE* fromptr,                                                                   \
    int64_t length,                                                                        \
    const int32_t* offsets,                                                                \
    int64_t offsetslength,                                                                 \
    bool ascending,                                                                        \
    bool stable) {                                                                         \
    return awkward_argsort<TYPE, int32_t>(                                                 \
      toptr, fromptr, length, offsets, offsetslength, ascending, stable);                  \
  }                                                                                        \
  ERROR awkward_argsort_##TYPE_NAME##_uint32(                                               \
    int64_t* toptr,                                                                        \
    const TYPE* fromptr,                                                                   \
    int64_t length,                                                                        \
    const uint32_t* offsets,                                                               \
    int64_t offsetslength,                                                                 \
    bool ascending,                                                                        \
    bool stable) {                                                                         \
    return awkward_argsort<TYPE, uint32_t>(                                                \
      toptr, fromptr, length, offsets, offsetslength, ascending, stable);                  \
  }                                                                                        \
  ERROR awkward_argsort_##TYPE_NAME##_uint64(                                              \
    int64_t* toptr,                                                                        \
    const TYPE* fromptr,                                                                   \
    int64_t length,                                                                        \
    const uint64_t* offsets,                                                               \
    int64_t offsetslength,                                                                 \
    bool ascending,                                                                        \
    bool stable) {                                                                         \
    return awkward_argsort<TYPE, uint64_t>(                                                \
      toptr, fromptr, length, offsets, offsetslength, ascending, stable);                  \
  }                                                                                        \
  ERROR awkward_argsort_##TYPE_NAME##_int64(                                               \
    int64_t* toptr,                                                                        \
    const TYPE* fromptr,                                                                   \
    int64_t length,                                                                        \
    const int64_t* offsets,                                                                \
    int64_t offsetslength,                                                                 \
    bool ascending,                                                                        \
    bool stable) {                                                                         \
    return awkward_argsort<TYPE, int64_t>(                                                 \
      toptr, fromptr, length, offsets, offsetslength, ascending, stable);                  \
  }

EXPORT_AWKWARD_ARGSORT(bool, bool)
EXPORT_AWKWARD_ARGSORT(int8_t, int8)
EXPORT_AWKWARD_ARGSORT(uint8_t, uint8)
EXPORT_AWKWARD_ARGSORT(int16_t, int16)
EXPORT_AWKWARD_ARGSORT(uint16_t, uint16)
EXPORT_AWKWARD_ARGSORT(int32_t, int32)
EXPORT_AWKWARD_ARGSORT(uint32_t, uint32)
EXPORT_AWKWARD_ARGSORT(int64_t, int64)
EXPORT_AWKWARD_ARGSORT(uint64_t, uint64)
EXPORT_AWKWARD_ARGSORT(float, float32)
EXPORT_AWKWARD_ARGSORT(double, float64)

template <>
bool argsort_order_ascending(bool l, bool r)
{
  return l < r;
}

template <>
bool argsort_order_descending(bool l, bool r)
{
  return l > r;
}
