// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>

#include "awkward/cpu-kernels/operations.h"

template <typename C, typename T>
ERROR awkward_listarray_flatten(C* tostarts, C* tostops, const C* fromstarts, const C* fromstops, const int64_t lenstarts, T* toarray, int64_t* tolen) {
  //  The following are allowed:
  //     * out of order (4:7 before 0:1)
  //     * overlaps (0:1 and 0:4 and 1:5)
  //     * beyond content starts[i] == stops[i] (999)
  for(int64_t i = 0; i < lenstarts; i++) {
    tostarts[i] = (C)fromstarts[i];
    tostops[i] = (C)fromstops[i];
    if(tostarts[i] < 0 || tostarts[i] < 0)
      return failure("all start and stop values must be non-negative", kSliceNone, i);
    int64_t length = fromstops[i] - fromstarts[i];
    *tolen += length;
  }
  std::cout << "\nTotal length " << *tolen << "\n";
  int64_t at = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    std::cout << " #" << i << "(" << fromstarts[i] << ", " << fromstops[i] << ")\n";
    int64_t length = fromstops[i] - fromstarts[i];
    if(length > 0) {
      for(int64_t l = 0; l < length; l++) {
        toarray[at] = fromstarts[i] + l;
        ++at;
      }
    }
  }
  std::cout << "\n";
  std::cout << "new length " << at << " vs total length " << *tolen << "\n";
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
