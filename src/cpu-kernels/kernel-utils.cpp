// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/kernel-utils.cpp", line)

#include "awkward/kernel-utils.h"

int8_t awkward_Index8_getitem_at_nowrap(
  const int8_t* ptr,
  int64_t at) {
  return ptr[(size_t)(at)];
}
uint8_t awkward_IndexU8_getitem_at_nowrap(
  const uint8_t* ptr,
  int64_t at) {
  return ptr[(size_t)(at)];
}
int32_t awkward_Index32_getitem_at_nowrap(
  const int32_t* ptr,
  int64_t at) {
  return ptr[(size_t)(at)];
}
uint32_t awkward_IndexU32_getitem_at_nowrap(
  const uint32_t* ptr,
  int64_t at) {
  return ptr[(size_t)(at)];
}
int64_t awkward_Index64_getitem_at_nowrap(
  const int64_t* ptr,
  int64_t at) {
  return ptr[(size_t)(at)];
}

void awkward_Index8_setitem_at_nowrap(
  int8_t* ptr,
  int64_t at,
  int8_t value) {
  ptr[(size_t)(at)] = value;
}
void awkward_IndexU8_setitem_at_nowrap(
  uint8_t* ptr,
  int64_t at,
  uint8_t value) {
  ptr[(size_t)(at)] = value;
}
void awkward_Index32_setitem_at_nowrap(
  int32_t* ptr,
  int64_t at,
  int32_t value) {
  ptr[(size_t)(at)] = value;
}
void awkward_IndexU32_setitem_at_nowrap(
  uint32_t* ptr,
  int64_t at,
  uint32_t value) {
  ptr[(size_t)(at)] = value;
}
void awkward_Index64_setitem_at_nowrap(
  int64_t* ptr,
  int64_t at,
  int64_t value) {
  ptr[(size_t)(at)] = value;
}

bool awkward_NumpyArraybool_getitem_at0(
  const bool* ptr) {
  return ptr[0];
}
int8_t awkward_NumpyArray8_getitem_at0(
  const int8_t* ptr) {
  return ptr[0];
}
uint8_t awkward_NumpyArrayU8_getitem_at0(
  const uint8_t* ptr) {
  return ptr[0];
}
int16_t awkward_NumpyArray16_getitem_at0(
  const int16_t* ptr) {
  return ptr[0];
}
uint16_t awkward_NumpyArrayU16_getitem_at0(
  const uint16_t* ptr) {
  return ptr[0];
}
int32_t awkward_NumpyArray32_getitem_at0(
  const int32_t* ptr) {
  return ptr[0];
}
uint32_t awkward_NumpyArrayU32_getitem_at0(
  const uint32_t* ptr) {
  return ptr[0];
}
int64_t awkward_NumpyArray64_getitem_at0(
  const int64_t* ptr) {
  return ptr[0];
}
uint64_t awkward_NumpyArrayU64_getitem_at0(
  const uint64_t* ptr) {
  return ptr[0];
}
float awkward_NumpyArrayfloat32_getitem_at0(
  const float* ptr) {
  return ptr[0];
}
double awkward_NumpyArrayfloat64_getitem_at0(
  const double* ptr) {
  return ptr[0];
}

void awkward_regularize_rangeslice(
  int64_t* start,
  int64_t* stop,
  bool posstep,
  bool hasstart,
  bool hasstop,
  int64_t length) {
  if (posstep) {
    if (!hasstart)           *start = 0;
    else if (*start < 0)     *start += length;
    if (*start < 0)          *start = 0;
    if (*start > length)     *start = length;

    if (!hasstop)            *stop = length;
    else if (*stop < 0)      *stop += length;
    if (*stop < 0)           *stop = 0;
    if (*stop > length)      *stop = length;
    if (*stop < *start)      *stop = *start;
  }

  else {
    if (!hasstart)           *start = length - 1;
    else if (*start < 0)     *start += length;
    if (*start < -1)         *start = -1;
    if (*start > length - 1) *start = length - 1;

    if (!hasstop)            *stop = -1;
    else if (*stop < 0)      *stop += length;
    if (*stop < -1)          *stop = -1;
    if (*stop > length - 1)  *stop = length - 1;
    if (*stop > *start)      *stop = *start;
  }
}

template <typename T>
void awkward_ListArray_combinations_step(
  T** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t j,
  int64_t stop,
  int64_t n,
  bool replacement) {
  while (fromindex[j] < stop) {
    if (replacement) {
      for (int64_t k = j + 1;  k < n;  k++) {
        fromindex[k] = fromindex[j];
      }
    }
    else {
      for (int64_t k = j + 1;  k < n;  k++) {
        fromindex[k] = fromindex[j] + (k - j);
      }
    }
    if (j + 1 == n) {
      for (int64_t k = 0;  k < n;  k++) {
        tocarry[k][toindex[k]] = fromindex[k];
        toindex[k]++;
      }
    }
    else {
      awkward_ListArray_combinations_step<T>(
        tocarry,
        toindex,
        fromindex,
        j + 1,
        stop,
        n,
        replacement);
    }
    fromindex[j]++;
  }
}

void awkward_ListArray_combinations_step_64(
  int64_t** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t j,
  int64_t stop,
  int64_t n,
  bool replacement) {
    return awkward_ListArray_combinations_step<int64_t>(
      tocarry,
      toindex,
      fromindex,
      j,
      stop,
      n,
      replacement
    );
}
