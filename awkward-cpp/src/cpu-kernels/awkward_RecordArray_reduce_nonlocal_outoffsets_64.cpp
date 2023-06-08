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
  int64_t k_carry = 0;
  int64_t offset = 0;

  // Zero initialize offsets
  for (i_start = 0;  i_start <= outlength;  i_start++) {
    outoffsets[i_start] = 0;
  }

  // Compute lengths of sublists
  for (i_start = 0, j_stop = 1; j_stop < lenparents; j_stop++) {
    if (parents[i_start] != parents[j_stop]) {
        outoffsets[parents[i_start]] = j_stop - i_start;
        outcarry[parents[i_start]] = k_carry;
        i_start = j_stop;
        k_carry++;
    }
  }

  // Close final open list
  if (lenparents > 0)
  {
    outoffsets[parents[i_start]] = lenparents - i_start;
    outcarry[parents[i_start]] = k_carry;
  }

  // Convert to offsets; final `outoffsets` is always zero before entering
  // this loop, as lenoffsets = maxparent + 2
  for (j_stop=0; j_stop <= outlength; j_stop++) {
    int64_t tmp = outoffsets[j_stop];
    outoffsets[j_stop] = offset;
    offset += tmp;
  }
  return success();
}
