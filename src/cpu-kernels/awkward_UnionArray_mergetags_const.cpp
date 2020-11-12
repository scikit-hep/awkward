// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_mergetags_const.cpp", line)

#include "awkward/kernels.h"

template <typename TO>
ERROR awkward_UnionArray_mergetags_const(
  TO* totags,
  int64_t* toindex,
  const int64_t* fromleft,
  const int64_t leftlen,
  const int64_t* fromright,
  const int64_t rightlen) {
  int64_t ileft = 0;
  int64_t iright = 0;
  int64_t leftsize = 0;
  int64_t rightsize = 0;
  TO tag = (TO)0;
  int64_t j = 0;
  int64_t li = 0;
  int64_t ri = 0;
  int64_t length = leftlen + rightlen;
  for (int64_t m = 0; m < length; m++) {
    if (tag == 0  &&  ileft < leftlen) {
      leftsize = fromleft[ileft + 1] - fromleft[ileft];
      for (int64_t i = 0; i < leftsize; i++) {
        toindex[j] = li++;
        totags[j++] = (TO)tag;
      }
    }
    if (tag == 1  &&  iright < rightlen) {
      rightsize = fromright[iright + 1] - fromright[iright];
      for (int64_t i = 0; i < rightsize; i++) {
        toindex[j] = ri++;
        totags[j++] = (TO)tag;
      }
    }
    if (tag == (TO)0) {
      if (iright < rightlen) {
        tag = (TO)1;
      }
      ileft++;
    } else {
      if (ileft < leftlen) {
        tag = (TO)0;
      }
      iright++;
    }
  }
  return success();
}
ERROR awkward_UnionArray_mergetags_to8_const(
  int8_t* totags,
  int64_t* toindex,
  const int64_t* fromleft,
  const int64_t leftlen,
  const int64_t* fromright,
  const int64_t rightlen) {
  return awkward_UnionArray_mergetags_const<int8_t>(
    totags,
    toindex,
    fromleft,
    leftlen,
    fromright,
    rightlen);
}
