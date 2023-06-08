// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RecordArray_reduce_local_outoffsets_64.cpp", line)

#include "awkward/kernels.h"

// This kernel
ERROR awkward_RecordArray_reduce_nonlocal_outoffsets_64(
  int64_t* outoffsets,
  int64_t* outcarry,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  int64_t i_start = 0;
  int64_t j_stop = 0;
  int64_t k_sublist = 0;

  outoffsets[0] = 0;

  // Initialise carry to unique value, indicating "missing" parent
  for (i_start = 0; i_start < outlength; i_start++) {
    outcarry[i_start] = -1;
  }

  // Fill offsets with lengths of sublists (in order of appearance, *NOT* parents)
  for (i_start = 0;  i_start <= outlength;  i_start++) {
    outoffsets[i_start] = 0;
  }

  // Fill offsets with lengths of sublists (in order of appearance, *NOT* parents)
  for (k_sublist = 0, i_start = 0, j_stop = 1; j_stop < lenparents; j_stop++) {
    if (parents[i_start] != parents[j_stop]) {
        outoffsets[k_sublist + 1] = j_stop;
        outcarry[parents[i_start]] = k_sublist;
        i_start = j_stop;
        k_sublist++;
    }
  }

  // Close the last sublist
  if (lenparents > 0) {
    outoffsets[k_sublist + 1] = j_stop;
    outcarry[parents[i_start]] = k_sublist;
    i_start = j_stop;
    k_sublist++;
  }

  // Append empty lists for missing parents
  for (i_start = k_sublist; i_start < outlength; i_start++) {
    outoffsets[i_start + 1] = lenparents;
  }

  // Replace unique value with index of appended empty list
  for (i_start=0; i_start <= outlength; i_start++) {
    if (outcarry[i_start] == -1) {
        outcarry[i_start] = k_sublist++;
    }
  }
  return success();
}
