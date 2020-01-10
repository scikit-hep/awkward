// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>

#include "awkward/cpu-kernels/operations.h"

template <typename C, typename T>
ERROR awkward_listarray_flatten(C* tostarts, C* tostops, const C* fromstarts, const C* fromstops, const int64_t lenstarts, T* toarray, int64_t* tolen) {
  *tolen = 0;
  int64_t at = 0;
  for(int64_t i = 0, j = 0; i < lenstarts; i++) {
    int64_t start = (C)fromstarts[i];
    int64_t stop = (C)fromstops[i];
    if(start < 0 || stop < 0)
      return failure("all start and stop values must be non-negative", kSliceNone, i);
    int64_t length = stop - start;
    if(length > 0) {
      for(int64_t l = 0; l < length; l++) {
        toarray[at] = start + l;
        ++at;
      }
      tostarts[j] = start;
      tostops[j] = stop;
      ++j;
      // FIXME: return it to shrink tostarts and tostops
      //*tostartslen = j;
      *tolen += length;
    }
  }
  return success();
}
ERROR awkward_listarray32_flatten_64(int32_t* tostarts, int32_t* tostops, const int32_t* fromstarts, const int32_t* fromstops, const int64_t lenstarts, int64_t* toarray, int64_t* tolen) {
  return awkward_listarray_flatten<int32_t, int64_t>(tostarts, tostops, fromstarts, fromstops, lenstarts, toarray, tolen);
}
ERROR awkward_listarrayU32_flatten_64(uint32_t* tostarts, uint32_t* tostops, const uint32_t* fromstarts, const uint32_t* fromstops, const int64_t lenstarts, int64_t* toarray, int64_t* tolen) {
  return awkward_listarray_flatten<uint32_t, int64_t>(tostarts, tostops, fromstarts, fromstops, lenstarts, toarray, tolen);
}
ERROR awkward_listarray64_flatten_64(int64_t* tostarts, int64_t* tostops, const int64_t* fromstarts, const int64_t* fromstops, const int64_t lenstarts, int64_t* toarray, int64_t* tolen) {
  return awkward_listarray_flatten<int64_t, int64_t>(tostarts, tostops, fromstarts, fromstops, lenstarts, toarray, tolen);
}
