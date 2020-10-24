// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_num.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename C>
ERROR awkward_ListArray_num(
  T* tonum,
  const C* fromstarts,
  const C* fromstops,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    C start = fromstarts[i];
    C stop = fromstops[i];
    tonum[i] = (T)(stop - start);
  }
  return success();
}
ERROR awkward_ListArray32_num_64(
  int64_t* tonum,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t length) {
  return awkward_ListArray_num<int64_t, int32_t>(
    tonum,
    fromstarts,
    fromstops,
    length);
}
ERROR awkward_ListArrayU32_num_64(
  int64_t* tonum,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t length) {
  return awkward_ListArray_num<int64_t, uint32_t>(
    tonum,
    fromstarts,
    fromstops,
    length);
}
ERROR awkward_ListArray64_num_64(
  int64_t* tonum,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length) {
  return awkward_ListArray_num<int64_t, int64_t>(
    tonum,
    fromstarts,
    fromstops,
    length);
}
