// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_unique_strings_uint8.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_NumpyArray_unique_strings_uint8(
  uint8_t* toptr,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t* outoffsets,
  int64_t* tolength) {

  // The strings are compacted toward the front of toptr in place, so a
  // candidate string must be compared against the previously kept string at
  // its location in the (already compacted) output, not at its original
  // offset which may have been overwritten by the compaction.
  int64_t laststart = 0;
  int64_t lastlen = -1;
  int64_t index = 0;
  int64_t counter = 0;
  bool differ = false;
  outoffsets[counter++] = offsets[0];
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    int64_t slen = offsets[i + 1] - offsets[i];
    differ = false;
    if (slen != lastlen) {
      differ = true;
    }
    else {
      for (int64_t k = 0; k < slen; k++) {
        if (toptr[laststart + k] != toptr[offsets[i] + k]) {
          differ = true;
          break;
        }
      }
    }
    if (differ) {
      laststart = index;
      for (int64_t j = offsets[i]; j < offsets[i + 1]; j++) {
        toptr[index++] = toptr[j];
      }
      lastlen = slen;
      outoffsets[counter++] = index;
    }
  }
  *tolength = counter;

  return success();
}
