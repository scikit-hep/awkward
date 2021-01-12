// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_quick_argsort.cpp", line)

#include <algorithm>
#include <numeric>
#include <vector>

#include "awkward/kernels.h"

template <typename T>
bool order_ascending(T left, T right)
{
  return left <= right;
}

template <typename T>
bool order_descending(T left, T right)
{
  return left >= right;
}

template <typename T>
bool binary_op(T left, T right, bool (*f)(T, T)) {
  return (*f)(left, right);
}

template <typename T, typename P>
int quick_argsort(const T *arr,
                  int64_t* result,
                  int64_t elements,
                  int64_t* beg,
                  int64_t* end,
                  int64_t maxlevels,
                  P& predicate) {
  int64_t low = 0;
  int64_t high = 0;
  int64_t i = 0;
  beg[0] = 0;
  end[0] = elements;
  while (i >= 0) {
    low = beg[i];
    high = end[i];
    if (high - low > 1) {
      int64_t mid = low  + ((high - low) >> 1);
      int64_t ind = result[mid];
      T pivot = arr[result[mid]];
      result[mid] = result[low];

      if (i == maxlevels - 1) {
        return -1;
      }
      high--;
      while (low < high) {
        while (binary_op(pivot, arr[result[high]], predicate)  &&  low < high) {
          high--;
        }
        if (low < high) {
          result[low++] = result[high];
        }
        while (binary_op(arr[result[low]], pivot, predicate)  &&  low < high) {
          low++;
        }
        if (low < high) {
          result[high--] = result[low];
        }
      }
      result[low] = ind;
      mid = low + 1;
      while (low > beg[i]  &&  result[low - 1] == ind) {
        low--;
      }
      while (mid < end[i]  &&  result[mid] == ind) {
        mid++;
      }
      if (low - beg[i] > end[i] - mid) {
        beg[i + 1] = mid;
        end[i + 1] = end[i];
        end[i++] = low;
      } else {
        beg[i + 1] = beg[i];
        end[i + 1] = low;
        beg[i++] = mid;
      }
    } else {
      i--;
    }
  }
  return 0;
}

template <typename T>
ERROR awkward_quick_argsort(
  int64_t* toptr,
  const T* fromptr,
  int64_t length,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable,
  int64_t maxlevels) {
  for (int64_t i = 0; i < offsetslength - 1; i++) {
    for (int64_t j = 0; j < offsets[i + 1] - offsets[i]; j++) {
      toptr[offsets[i] + j] = j;
    }
  }

  if (ascending) {
    for (int64_t i = 0; i < offsetslength - 1; i++) {
      if (quick_argsort(&(fromptr[offsets[i]]),
                        &(toptr[offsets[i]]),
                        offsets[i + 1] - offsets[i],
                        tmpbeg,
                        tmpend,
                        maxlevels,
                        order_ascending<T>) < 0) {
        return failure("failed to sort an array", i, offsets[i], FILENAME(__LINE__));
      }
    }
  }
  else {
    for (int64_t i = 0; i < offsetslength - 1; i++) {
      if (quick_argsort(&(fromptr[offsets[i]]),
                        &(toptr[offsets[i]]),
                        offsets[i + 1] - offsets[i],
                        tmpbeg,
                        tmpend,
                        maxlevels,
                        order_descending<T>) < 0) {
        return failure("failed to sort an array", i, offsets[i], FILENAME(__LINE__));
      }
    }
  }

  return success();
}

ERROR awkward_quick_argsort_bool(
  int64_t* toptr,
  const bool* fromptr,
  int64_t length,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable,
  int64_t maxlevels) {
  return awkward_quick_argsort<bool>(
    toptr,
    fromptr,
    length,
    tmpbeg,
    tmpend,
    offsets,
    offsetslength,
    ascending,
    stable,
    maxlevels);
}

ERROR awkward_quick_argsort_int8(
  int64_t* toptr,
  const int8_t* fromptr,
  int64_t length,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable,
  int64_t maxlevels) {
  return awkward_quick_argsort<int8_t>(
    toptr,
    fromptr,
    length,
    tmpbeg,
    tmpend,
    offsets,
    offsetslength,
    ascending,
    stable,
    maxlevels);
}

ERROR awkward_quick_argsort_uint8(
  int64_t* toptr,
  const uint8_t* fromptr,
  int64_t length,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable,
  int64_t maxlevels) {
  return awkward_quick_argsort<uint8_t>(
    toptr,
    fromptr,
    length,
    tmpbeg,
    tmpend,
    offsets,
    offsetslength,
    ascending,
    stable,
    maxlevels);
}

ERROR awkward_quick_argsort_int16(
  int64_t* toptr,
  const int16_t* fromptr,
  int64_t length,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable,
  int64_t maxlevels) {
  return awkward_quick_argsort<int16_t>(
    toptr,
    fromptr,
    length,
    tmpbeg,
    tmpend,
    offsets,
    offsetslength,
    ascending,
    stable,
    maxlevels);
}

ERROR awkward_quick_argsort_uint16(
  int64_t* toptr,
  const uint16_t* fromptr,
  int64_t length,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable,
  int64_t maxlevels) {
  return awkward_quick_argsort<uint16_t>(
    toptr,
    fromptr,
    length,
    tmpbeg,
    tmpend,
    offsets,
    offsetslength,
    ascending,
    stable,
    maxlevels);
}

ERROR awkward_quick_argsort_int32(
  int64_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable,
  int64_t maxlevels) {
  return awkward_quick_argsort<int32_t>(
    toptr,
    fromptr,
    length,
    tmpbeg,
    tmpend,
    offsets,
    offsetslength,
    ascending,
    stable,
    maxlevels);
}

ERROR awkward_quick_argsort_uint32(
  int64_t* toptr,
  const uint32_t* fromptr,
  int64_t length,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable,
  int64_t maxlevels) {
  return awkward_quick_argsort<uint32_t>(
    toptr,
    fromptr,
    length,
    tmpbeg,
    tmpend,
    offsets,
    offsetslength,
    ascending,
    stable,
    maxlevels);
}

ERROR awkward_quick_argsort_int64(
  int64_t* toptr,
  const int64_t* fromptr,
  int64_t length,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable,
  int64_t maxlevels) {
  return awkward_quick_argsort<int64_t>(
    toptr,
    fromptr,
    length,
    tmpbeg,
    tmpend,
    offsets,
    offsetslength,
    ascending,
    stable,
    maxlevels);
}

ERROR awkward_quick_argsort_uint64(
  int64_t* toptr,
  const uint64_t* fromptr,
  int64_t length,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable,
  int64_t maxlevels) {
  return awkward_quick_argsort<uint64_t>(
    toptr,
    fromptr,
    length,
    tmpbeg,
    tmpend,
    offsets,
    offsetslength,
    ascending,
    stable,
    maxlevels);
}

ERROR awkward_quick_argsort_float32(
  int64_t* toptr,
  const float* fromptr,
  int64_t length,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable,
  int64_t maxlevels) {
  return awkward_quick_argsort<float>(
    toptr,
    fromptr,
    length,
    tmpbeg,
    tmpend,
    offsets,
    offsetslength,
    ascending,
    stable,
    maxlevels);
}

ERROR awkward_quick_argsort_float64(
  int64_t* toptr,
  const double* fromptr,
  int64_t length,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable,
  int64_t maxlevels) {
  return awkward_quick_argsort<double>(
    toptr,
    fromptr,
    length,
    tmpbeg,
    tmpend,
    offsets,
    offsetslength,
    ascending,
    stable,
    maxlevels);
}
