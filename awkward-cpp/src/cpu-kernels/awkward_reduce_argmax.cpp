// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_argmax.cpp", line)

#include "awkward/kernels.h"

template <typename OUT, typename IN>
ERROR awkward_reduce_argmax(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* offsets,
  const int64_t* /* starts */,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    int64_t best = -1;
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      if (best == -1 || fromptr[i] > fromptr[best]) {
        best = i;
      }
    }
    toptr[bin] = static_cast<OUT>(best);
  }
  return success();
}
ERROR awkward_reduce_argmax_int8_64(
  int64_t* toptr,
  const int8_t* fromptr,
  const int64_t* offsets,
  const int64_t* starts,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, int8_t>(toptr, fromptr, offsets, starts, outlength);
}
ERROR awkward_reduce_argmax_uint8_64(
  int64_t* toptr,
  const uint8_t* fromptr,
  const int64_t* offsets,
  const int64_t* starts,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, uint8_t>(toptr, fromptr, offsets, starts, outlength);
}
ERROR awkward_reduce_argmax_int16_64(
  int64_t* toptr,
  const int16_t* fromptr,
  const int64_t* offsets,
  const int64_t* starts,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, int16_t>(toptr, fromptr, offsets, starts, outlength);
}
ERROR awkward_reduce_argmax_uint16_64(
  int64_t* toptr,
  const uint16_t* fromptr,
  const int64_t* offsets,
  const int64_t* starts,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, uint16_t>(toptr, fromptr, offsets, starts, outlength);
}
ERROR awkward_reduce_argmax_int32_64(
  int64_t* toptr,
  const int32_t* fromptr,
  const int64_t* offsets,
  const int64_t* starts,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, int32_t>(toptr, fromptr, offsets, starts, outlength);
}
ERROR awkward_reduce_argmax_uint32_64(
  int64_t* toptr,
  const uint32_t* fromptr,
  const int64_t* offsets,
  const int64_t* starts,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, uint32_t>(toptr, fromptr, offsets, starts, outlength);
}
ERROR awkward_reduce_argmax_int64_64(
  int64_t* toptr,
  const int64_t* fromptr,
  const int64_t* offsets,
  const int64_t* starts,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, int64_t>(toptr, fromptr, offsets, starts, outlength);
}
ERROR awkward_reduce_argmax_uint64_64(
  int64_t* toptr,
  const uint64_t* fromptr,
  const int64_t* offsets,
  const int64_t* starts,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, uint64_t>(toptr, fromptr, offsets, starts, outlength);
}
ERROR awkward_reduce_argmax_float32_64(
  int64_t* toptr,
  const float* fromptr,
  const int64_t* offsets,
  const int64_t* starts,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, float>(toptr, fromptr, offsets, starts, outlength);
}
ERROR awkward_reduce_argmax_float64_64(
  int64_t* toptr,
  const double* fromptr,
  const int64_t* offsets,
  const int64_t* starts,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, double>(toptr, fromptr, offsets, starts, outlength);
}
