// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_Content_getitem_next_missing_jagged_getmaskstartstop.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_Content_getitem_next_missing_jagged_getmaskstartstop(
  int64_t* index_in,
  int64_t* offsets_in,
  int64_t* mask_out,
  int64_t* starts_out,
  int64_t* stops_out,
  int64_t length) {
  int64_t k = 0;
  for (int64_t i = 0; i < length; ++i) {
    starts_out[i] = offsets_in[k];
    if (index_in[i] < 0) {
      mask_out[i] = -1;
      stops_out[i] = offsets_in[k];
    }
    else {
      mask_out[i] = i;
      k++;
      stops_out[i] = offsets_in[k];
    }
  }
  return success();
}
