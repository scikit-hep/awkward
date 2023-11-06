// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_combinations_length.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_combinations_length(
  int64_t* totallen,
  T* tooffsets,
  int64_t n,
  bool replacement,
  const C* starts,
  const C* stops,
  int64_t length) {
  *totallen = 0;
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    int64_t size = (int64_t)(stops[i] - starts[i]);
    if (replacement) {
      size += (n - 1);
    }
    int64_t thisn = n;
    int64_t combinationslen;
    if (thisn > size) {
      combinationslen = 0;
    }
    else if (thisn == size) {
      combinationslen = 1;
    }
    else {
      if (thisn * 2 > size) {
        thisn = size - thisn;
      }
      combinationslen = size;
      for (int64_t j = 2;  j <= thisn;  j++) {
        combinationslen *= (size - j + 1);
        combinationslen /= j;
      }
    }
    *totallen = *totallen + combinationslen;
    tooffsets[i + 1] = tooffsets[i] + combinationslen;
  }
  return success();
}
ERROR awkward_ListArray32_combinations_length_64(
  int64_t* totallen,
  int64_t* tooffsets,
  int64_t n,
  bool replacement,
  const int32_t* starts,
  const int32_t* stops,
  int64_t length) {
  return awkward_ListArray_combinations_length<int32_t, int64_t>(
    totallen,
    tooffsets,
    n,
    replacement,
    starts,
    stops,
    length);
}
ERROR awkward_ListArrayU32_combinations_length_64(
  int64_t* totallen,
  int64_t* tooffsets,
  int64_t n,
  bool replacement,
  const uint32_t* starts,
  const uint32_t* stops,
  int64_t length) {
  return awkward_ListArray_combinations_length<uint32_t, int64_t>(
    totallen,
    tooffsets,
    n,
    replacement,
    starts,
    stops,
    length);
}
ERROR awkward_ListArray64_combinations_length_64(
  int64_t* totallen,
  int64_t* tooffsets,
  int64_t n,
  bool replacement,
  const int64_t* starts,
  const int64_t* stops,
  int64_t length) {
  return awkward_ListArray_combinations_length<int64_t, int64_t>(
    totallen,
    tooffsets,
    n,
    replacement,
    starts,
    stops,
    length);
}
