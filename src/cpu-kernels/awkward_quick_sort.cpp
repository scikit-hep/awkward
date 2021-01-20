// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_quick_sort.cpp", line)

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
int quick_sort(T *arr,
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
      T pivot = arr[mid];
      arr[mid] = arr[low];

      if (i == maxlevels - 1) {
        return -1;
      }
      high--;
      while (low < high) {
        while (binary_op(pivot, arr[high], predicate)  &&  low < high) {
          high--;
        }
        if (low < high) {
          arr[low++] = arr[high];
        }
        while (binary_op(arr[low], pivot, predicate)  &&  low < high) {
          low++;
        }
        if (low < high) {
          arr[high--] = arr[low];
        }
      }
      arr[low] = pivot;
      mid = low + 1;
      while (low > beg[i]  &&  arr[low - 1] == pivot) {
        low--;
      }
      while (mid < end[i]  &&  arr[mid] == pivot) {
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
ERROR awkward_quick_sort(
    T* tmpptr,
    int64_t* tmpbeg,
    int64_t* tmpend,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    bool ascending,
    int64_t length,
    int64_t maxlevels) {

  if (ascending) {
    for (int64_t i = 0; i < length; i++) {
      if (quick_sort(&(tmpptr[fromstarts[i]]),
                     fromstops[i] - fromstarts[i],
                     tmpbeg,
                     tmpend,
                     maxlevels,
                     order_ascending<T>) < 0) {
        return failure("failed to sort an array", i, fromstarts[i], FILENAME(__LINE__));
      }
    }
  }
  else {
    for (int64_t i = 0; i < length; i++) {
      if (quick_sort(&(tmpptr[fromstarts[i]]),
                     fromstops[i] - fromstarts[i],
                     tmpbeg,
                     tmpend,
                     maxlevels,
                     order_descending<T>) < 0) {
        return failure("failed to sort an array", i, fromstarts[i], FILENAME(__LINE__));
      }
    }
  }

  return success();
}

ERROR awkward_quick_sort_bool(
  bool* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  bool ascending,
  int64_t length,
  int64_t maxlevels) {
    return awkward_quick_sort<bool>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      ascending,
      length,
      maxlevels);
}
ERROR awkward_quick_sort_int8(
  int8_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  bool ascending,
  int64_t length,
  int64_t maxlevels) {
    return awkward_quick_sort<int8_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      ascending,
      length,
      maxlevels);
}
ERROR awkward_quick_sort_uint8(
  uint8_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  bool ascending,
  int64_t length,
  int64_t maxlevels) {
    return awkward_quick_sort<uint8_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      ascending,
      length,
      maxlevels);
}
ERROR awkward_quick_sort_int16(
  int16_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  bool ascending,
  int64_t length,
  int64_t maxlevels) {
    return awkward_quick_sort<int16_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      ascending,
      length,
      maxlevels);
}
ERROR awkward_quick_sort_uint16(
  uint16_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  bool ascending,
  int64_t length,
  int64_t maxlevels) {
    return awkward_quick_sort<uint16_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      ascending,
      length,
      maxlevels);
}
ERROR awkward_quick_sort_int32(
  int32_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  bool ascending,
  int64_t length,
  int64_t maxlevels) {
    return awkward_quick_sort<int32_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      ascending,
      length,
      maxlevels);
}
ERROR awkward_quick_sort_uint32(
  uint32_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  bool ascending,
  int64_t length,
  int64_t maxlevels) {
    return awkward_quick_sort<uint32_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      ascending,
      length,
      maxlevels);
}
ERROR awkward_quick_sort_int64(
  int64_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  bool ascending,
  int64_t length,
  int64_t maxlevels) {
    return awkward_quick_sort<int64_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      ascending,
      length,
      maxlevels);
}
ERROR awkward_quick_sort_uint64(
  uint64_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  bool ascending,
  int64_t length,
  int64_t maxlevels) {
    return awkward_quick_sort<uint64_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      ascending,
      length,
      maxlevels);
}
ERROR awkward_quick_sort_float32(
  float* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  bool ascending,
  int64_t length,
  int64_t maxlevels) {
    return awkward_quick_sort<float>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      ascending,
      length,
      maxlevels);
}
ERROR awkward_quick_sort_float64(
  double* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  bool ascending,
  int64_t length,
  int64_t maxlevels) {
    return awkward_quick_sort<double>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      ascending,
      length,
      maxlevels);
}
