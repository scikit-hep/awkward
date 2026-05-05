// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_local_preparenext_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_IndexedArray_local_preparenext_64(
    int64_t* tocarry,
    const int64_t* /* starts */,   // used in CUDA kernels
    const int64_t* offsets,
    const int64_t* nextoffsets,
    int64_t outlength) {
  // For each outer bin, walk the elements assigned to that bin and pair them
  // up with the surviving inner elements in nextoffsets[bin..bin+1). Outer
  // elements that survived receive their inner position k; the rest get -1.
  for (int64_t bin = 0; bin < outlength; bin++) {
    int64_t outer_start = offsets[bin];
    int64_t outer_stop = offsets[bin + 1];
    int64_t inner_start = nextoffsets[bin];
    int64_t inner_stop = nextoffsets[bin + 1];

    int64_t k = inner_start;
    for (int64_t i = outer_start; i < outer_stop; i++) {
      if (k < inner_stop) {
        tocarry[i] = k;
        k++;
      }
      else {
        tocarry[i] = -1;
      }
    }
  }
  return success();
}
