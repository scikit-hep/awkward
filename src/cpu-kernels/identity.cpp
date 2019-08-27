// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"

Error awkward_identity_new(IndexType* ptr, IndexType length) {
  for (IndexType i = 0;  i < length;  i++) {
    ptr[i] = i;
  }
  return kNoError;
}
