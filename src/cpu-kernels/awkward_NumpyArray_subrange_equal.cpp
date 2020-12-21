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
  int64_t pivot = arr[high];
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
void quickSort(T* arr, int64_t low, int64_t high)
{
  if (low < high)
  {
    int64_t pi = partition(arr, low, high);

    quickSort(arr, low, pi - 1);
    quickSort(arr, pi + 1, high);
  }
}

template <typename T>
ERROR awkward_NumpyArray_subrange_equal(
    T* fromptr,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t length,
    bool* toequal) {

  for (int64_t i = 0; i < length; i++) {
    quickSort(fromptr, fromstarts[i], fromstops[i] - 1);
  }

  bool differ = true;
  int64_t leftlen;
  int64_t rightlen;
  // FIXME: sort the ranges before comparisons?
  for (int64_t i = 0;  i < length - 1;  i++) {
    leftlen = fromstops[i] - fromstarts[i];
    for (int64_t ii = i + 1; ii < length - 1;  ii++) {
      rightlen = fromstops[ii] - fromstarts[ii];
      if (leftlen == rightlen) {
        differ = false;
        for (int64_t j = 0; j < leftlen; j++) {
          if (fromptr[fromstarts[i] + j] != fromptr[fromstarts[ii] + j]) {
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
  bool* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<bool>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_int8(
  int8_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<int8_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_uint8(
  uint8_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<uint8_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_int16(
  int16_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<int16_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_uint16(
  uint16_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<uint16_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_int32(
  int32_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<int32_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_uint32(
  uint32_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<uint32_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_int64(
  int64_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<int64_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_uint64(
  uint64_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<uint64_t>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_float32(
  float* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<float>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
ERROR awkward_NumpyArray_subrange_equal_float64(
  double* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {
    return awkward_NumpyArray_subrange_equal<double>(
      fromptr,
      fromstarts,
      fromstops,
      length,
      toequal);
}
