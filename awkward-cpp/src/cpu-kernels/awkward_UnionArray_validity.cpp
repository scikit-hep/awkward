// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_validity.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename I>
ERROR awkward_UnionArray_validity(
  const T* tags,
  const I* index,
  int64_t length,
  int64_t numcontents,
  const int64_t* lencontents) {
  for (int64_t i = 0;  i < length;  i++) {
    T tag = tags[i];
    I idx = index[i];
    if (tag < 0) {
      return failure("tags[i] < 0", i, kSliceNone, FILENAME(__LINE__));
    }
    if (idx < 0) {
      return failure("index[i] < 0", i, kSliceNone, FILENAME(__LINE__));
    }
    if (tag >= numcontents) {
      return failure("tags[i] >= len(contents)", i, kSliceNone, FILENAME(__LINE__));
    }
    int64_t lencontent = lencontents[tag];
    if (idx >= lencontent) {
      return failure("index[i] >= len(content[tags[i]])", i, kSliceNone, FILENAME(__LINE__));
    }
  }
  return success();
}
ERROR awkward_UnionArray8_32_validity(
  const int8_t* tags,
  const int32_t* index,
  int64_t length,
  int64_t numcontents,
  const int64_t* lencontents) {
  return awkward_UnionArray_validity<int8_t, int32_t>(
    tags,
    index,
    length,
    numcontents,
    lencontents);
}
ERROR awkward_UnionArray8_U32_validity(
  const int8_t* tags,
  const uint32_t* index,
  int64_t length,
  int64_t numcontents,
  const int64_t* lencontents) {
  return awkward_UnionArray_validity<int8_t, uint32_t>(
    tags,
    index,
    length,
    numcontents,
    lencontents);
}
ERROR awkward_UnionArray8_64_validity(
  const int8_t* tags,
  const int64_t* index,
  int64_t length,
  int64_t numcontents,
  const int64_t* lencontents) {
  return awkward_UnionArray_validity<int8_t, int64_t>(
    tags,
    index,
    length,
    numcontents,
    lencontents);
}
