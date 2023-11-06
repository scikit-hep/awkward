// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RecordArray_reduce_local_outoffsets_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_RecordArray_reduce_nonlocal_outoffsets_64(
  int64_t* outoffsets,
  int64_t* outcarry,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  int64_t i = 0;
  int64_t j_stop = 0;
  int64_t k_sublist = 0;

  // The first offset is always 0
  outoffsets[0] = 0;

  // Initialise carry to unique value, indicating "missing" parent
  for (i = 0; i < outlength; i++) {
    outcarry[i] = -1;
  }

  // Fill offsets with stop index of sublists in parents array. Ignore ordering given by parents, this is done by subsequent carry
  i = 0;
  for (j_stop = 1; j_stop < lenparents; j_stop++) {
    if (parents[i] != parents[j_stop]) {
        outoffsets[k_sublist + 1] = j_stop;
        outcarry[parents[i]] = k_sublist;
        i = j_stop;
        k_sublist++;
    }
  }
  // Close the last sublist!
  if (lenparents > 0) {
    outoffsets[k_sublist + 1] = j_stop;
    outcarry[parents[i]] = k_sublist;
    i = j_stop;
    k_sublist++;
  }

  // Append empty lists for missing parents
  for (i = k_sublist; i < outlength; i++) {
    outoffsets[i + 1] = lenparents;
  }

  // Replace unique value with index of appended empty lists
  for (i = 0; i <= outlength; i++) {
    if (outcarry[i] == -1) {
        outcarry[i] = k_sublist++;
    }
  }
  return success();
}
