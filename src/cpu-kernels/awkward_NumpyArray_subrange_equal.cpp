// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_subrange_equal.cpp", line)

#include "awkward/kernels.h"

template <typename T>
void swap(T* a, T* b)
{
  T t = *a;
  *a = *b;
  *b = t;
}

template <typename T>
int partition (T* arr, int64_t low, int64_t high)
{
  T pivot = arr[high];
  int64_t i = (low - 1);

  for (int64_t j = low; j <= high - 1; j++) {
    if (arr[j] < pivot) {
      i++;
      swap(&arr[i], &arr[j]);
    }
  }
  swap(&arr[i + 1], &arr[high]);
  return (i + 1);
}

template <typename T>
void quick_sort(T* arr, int64_t low, int64_t high)
{
  if (low < high)
  {
    int64_t pi = partition(arr, low, high);

    quick_sort(arr, low, pi - 1);
    quick_sort(arr, pi + 1, high);
  }
}

template <typename T>
int quick_sort(T *arr,
               int64_t elements,
               int64_t* beg,
               int64_t* end,
               int64_t maxlevels) {
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
        while (arr[high] >= pivot  &&  low < high) {
          high--;
        }
        if (low < high) {
          arr[low++] = arr[high];
        }
        while (arr[low] <= pivot  &&  low < high) {
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
ERROR awkward_NumpyArray_subrange_equal(
    T* tmpptr,
    int64_t* tmpbeg,
    int64_t* tmpend,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t length,
    int64_t maxlevels,
    bool* toequal) {

  for (int64_t i = 0; i < length; i++) {
    int result = quick_sort(&(tmpptr[fromstarts[i]]), fromstops[i] - fromstarts[i],
                            tmpbeg, tmpend, maxlevels);
    if (result < 0) {
      return failure("failed to sort an array", i, fromstarts[i], FILENAME(__LINE__));
    }
  }

  bool differ = true;
  int64_t leftlen;
  int64_t rightlen;

  for (int64_t i = 0;  i < length - 1;  i++) {
    leftlen = fromstops[i] - fromstarts[i];
    for (int64_t ii = i + 1; ii < length - 1;  ii++) {
      rightlen = fromstops[ii] - fromstarts[ii];
      if (leftlen == rightlen) {
        differ = false;
        for (int64_t j = 0; j < leftlen; j++) {
          if (tmpptr[fromstarts[i] + j] != tmpptr[fromstarts[ii] + j]) {
            differ = true;
            break;
          }
        }
      }
    }
  }

  *toequal = !differ;

  return success();
}

ERROR awkward_NumpyArray_subrange_equal_bool(
  bool* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t maxlevels,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<bool>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      length,
      maxlevels,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_int8(
  int8_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t maxlevels,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<int8_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      length,
      maxlevels,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_uint8(
  uint8_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t maxlevels,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<uint8_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      length,
      maxlevels,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_int16(
  int16_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t maxlevels,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<int16_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      length,
      maxlevels,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_uint16(
  uint16_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t maxlevels,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<uint16_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      length,
      maxlevels,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_int32(
  int32_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t maxlevels,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<int32_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      length,
      maxlevels,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_uint32(
  uint32_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t maxlevels,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<uint32_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      length,
      maxlevels,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_int64(
  int64_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t maxlevels,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<int64_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      length,
      maxlevels,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_uint64(
  uint64_t* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t maxlevels,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<uint64_t>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      length,
      maxlevels,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_float32(
  float* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t maxlevels,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<float>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      length,
      maxlevels,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_float64(
  double* tmpptr,
  int64_t* tmpbeg,
  int64_t* tmpend,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t maxlevels,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<double>(
      tmpptr,
      tmpbeg,
      tmpend,
      fromstarts,
      fromstops,
      length,
      maxlevels,
      toequal);
}
