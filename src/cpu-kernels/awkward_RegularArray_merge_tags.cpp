// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_merge_tags.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_RegularArray_merge_tags8_index64(
  int8_t* totags,
  int64_t* toindex,
  int64_t tolength,
  int64_t length,
  int64_t size,
  int64_t otherlength,
  int64_t othersize) {
    int64_t i = 0;
    int64_t j = 0;
    int64_t left = 0;
    int64_t right = 0;
    int64_t tag = 0;
    for (int64_t m = 0; m < tolength; m++) {
      totags[m] = tag;
      if (tag == 0) {
        i++;
        left++;
      } else {
        j++;
        right++;
      }
      if (i == size) {
        i = 0;
        if (right < otherlength) {
          tag = 1;
        }
      }
      if (j == othersize) {
        j = 0;
        if (left < length) {
          tag = 0;
        }
      }
    }
    left = 0;
    right = 0;
    for (int64_t m = 0; m < tolength; m++) {
      toindex[m] = (totags[m] == 0) ? left++ : right++;
    }
    return success();
  }
