// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>

#include "awkward/cpu-kernels/util.h"

Error success() {
  Error out;
  out.location = kSliceNone;
  out.attempt = kSliceNone;
  out.strlength = 0;
  out.str = nullptr;
  return out;
}

Error failure(int64_t location, int64_t attempt, const char* str) {
  Error out;
  out.location = location;
  out.attempt = attempt;
  out.strlength = (int64_t)strlen(str);
  out.str = str;
  return out;
}
