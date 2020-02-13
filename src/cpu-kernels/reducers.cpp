// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>
#include <vector>

#include "awkward/cpu-kernels/reducers.h"

ERROR awkward_content_reduce_zeroparents_64(int64_t* toparents, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toparents[i] = 0;
  }
}
