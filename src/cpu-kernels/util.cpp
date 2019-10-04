// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>

#include "awkward/cpu-kernels/util.h"

Error success() {
  Error out;
  out.str = nullptr;
  out.location = kSliceNone;
  out.attempt = kSliceNone;
  out.extra = 0;
  return out;
}

Error failure(int64_t location, int64_t attempt, const char* str) {
  Error out;
  out.str = str;
  out.location = location;
  out.attempt = attempt;
  out.extra = 0;
  return out;
}
