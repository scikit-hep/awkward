// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>

#include "awkward/cpu-kernels/util.h"

Error success() {
  Error out;
  out.where1 = -1;
  out.where2 = -1;
  out.strlength = 0;
  out.str = nullptr;
  return out;
}

Error failure(int64_t where1, int64_t where2, const char* str) {
  Error out;
  out.where1 = where1;
  out.where2 = where2;
  out.strlength = (int64_t)strlen(str);
  out.str = str;
}
