// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"

Error awkward_identity_new(IndexType length, IndexType* to) {
  for (IndexType i = 0;  i < length;  i++) {
    to[i] = i;
  }
  return kNoError;
}

Error awkward_identity_from_listfoffsets(IndexType length, IndexType width, IndexType* offsets, IndexType* from, IndexType tolength, IndexType* to) {
  IndexType k = 0;
  for (IndexType i = 0;  i < length;  i++) {
    for (IndexType subi = 0;  subi < offsets[i + 1] - offsets[i];  subi++) {
      for (IndexType j = 0;  j < width;  j++) {
        to[(width + 1)*k + j] = from[(width)*i + j];
      }
      to[(width + 1)*k + width] = subi;
      k++;
    }
  }

  return kNoError;
}
