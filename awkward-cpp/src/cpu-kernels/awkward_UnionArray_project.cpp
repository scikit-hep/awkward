// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_project.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename C, typename I>
ERROR awkward_UnionArray_project(
  int64_t* lenout,
  T* tocarry,
  const C* fromtags,
  const I* fromindex,
  int64_t length,
  int64_t which) {
  *lenout = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if (fromtags[i] == which) {
      tocarry[(size_t)(*lenout)] = fromindex[i];
      *lenout = *lenout + 1;
    }
  }
  return success();
}
ERROR awkward_UnionArray8_32_project_64(
  int64_t* lenout,
  int64_t* tocarry,
  const int8_t* fromtags,
  const int32_t* fromindex,
  int64_t length,
  int64_t which) {
  return awkward_UnionArray_project<int64_t, int8_t, int32_t>(
    lenout,
    tocarry,
    fromtags,
    fromindex,
    length,
    which);
}
ERROR awkward_UnionArray8_U32_project_64(
  int64_t* lenout,
  int64_t* tocarry,
  const int8_t* fromtags,
  const uint32_t* fromindex,
  int64_t length,
  int64_t which) {
  return awkward_UnionArray_project<int64_t, int8_t, uint32_t>(
    lenout,
    tocarry,
    fromtags,
    fromindex,
    length,
    which);
}
ERROR awkward_UnionArray8_64_project_64(
  int64_t* lenout,
  int64_t* tocarry,
  const int8_t* fromtags,
  const int64_t* fromindex,
  int64_t length,
  int64_t which) {
  return awkward_UnionArray_project<int64_t, int8_t, int64_t>(
    lenout,
    tocarry,
    fromtags,
    fromindex,
    length,
    which);
}
