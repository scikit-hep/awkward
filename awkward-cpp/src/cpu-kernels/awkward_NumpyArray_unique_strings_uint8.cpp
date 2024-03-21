// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_unique_strings_uint8.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_NumpyArray_unique_strings_uint8(
  uint8_t* toptr,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t* outoffsets,
  int64_t* tolength) {

  int64_t slen = 0;
  int64_t index = 0;
  int64_t counter = 0;
  int64_t start = 0;
  int64_t k = 0;
  bool differ = false;
  outoffsets[counter++] = offsets[0];
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    differ = false;
    if (offsets[i + 1] - offsets[i] != slen) {
      differ = true;
    }
    else {
      k = 0;
      for (int64_t j = offsets[i]; j < offsets[i + 1]; j++) {
        if (toptr[start + k++] != toptr[j]) {
          differ = true;
        }
      }
    }
    if (differ) {
      for (int64_t j = offsets[i]; j < offsets[i + 1]; j++) {
        toptr[index++] = toptr[j];
        start = offsets[i];
     }
     outoffsets[counter++] = index;
   }
   slen = offsets[i + 1] - offsets[i];
  }
  *tolength = counter;

  return success();
}
