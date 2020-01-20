// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>

#include "awkward/cpu-kernels/util.h"

struct Error success() {
  struct Error out;
  out.str = nullptr;
  out.identity = kSliceNone;
  out.attempt = kSliceNone;
  out.extra = 0;
  return out;
}

struct Error failure(const char* str, int64_t identity, int64_t attempt) {
  struct Error out;
  out.str = str;
  out.identity = identity;
  out.attempt = attempt;
  out.extra = 0;
  return out;
}
