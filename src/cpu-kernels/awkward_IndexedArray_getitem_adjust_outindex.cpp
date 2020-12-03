// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_getitem_adjust_outindex.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_IndexedArray_getitem_adjust_outindex(
  int8_t* tomask,
  T* toindex,
  T* tononzero,
  const T* fromindex,
  int64_t fromindexlength,
  const T* nonzero,
  int64_t nonzerolength) {
  int64_t j = 0;
  int64_t k = 0;
  for (int64_t i = 0;  i < fromindexlength;  i++) {
    T fromval = fromindex[i];
    tomask[i] = (fromval < 0);
    if (fromval < 0) {
      toindex[k] = -1;
      k++;
    }
    else if (j < nonzerolength  &&  fromval == nonzero[j]) {
      tononzero[j] = fromval + (k - j);
      toindex[k] = j;
      j++;
      k++;
    }
  }
  return success();
}
ERROR awkward_IndexedArray_getitem_adjust_outindex_64(
  int8_t* tomask,
  int64_t* toindex,
  int64_t* tononzero,
  const int64_t* fromindex,
  int64_t fromindexlength,
  const int64_t* nonzero,
  int64_t nonzerolength) {
  return awkward_IndexedArray_getitem_adjust_outindex<int64_t>(
    tomask,
    toindex,
    tononzero,
    fromindex,
    fromindexlength,
    nonzero,
    nonzerolength);
}
