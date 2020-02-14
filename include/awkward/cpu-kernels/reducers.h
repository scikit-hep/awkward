// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_REDUCERS_H_
#define AWKWARDCPU_REDUCERS_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  EXPORT_SYMBOL struct Error awkward_content_reduce_zeroparents_64(int64_t* toparents, int64_t length);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_global_startstop_64(int64_t* globalstart, int64_t* globalstop, const int64_t* offsets, int64_t offsetsoffset, int64_t length);

  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_nonlocal_maxcount_offsetscopy_64(int64_t* maxcount, int64_t* offsetscopy, const int64_t* offsets, int64_t offsetsoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_nonlocal_preparenext_64(int64_t* nextcarry, int64_t* nextparents, int64_t nextlen, int64_t* maxnextparents, int64_t* distincts, int64_t distinctslen, int64_t* offsetscopy, const int64_t* offsets, int64_t offsetsoffset, int64_t length, const int64_t* parents, int64_t parentsoffset, int64_t maxcount);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_nonlocal_findgaps_64(int64_t* gaps, const int64_t* parents, int64_t parentsoffset, int64_t lenparents);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_nonlocal_outstartsstops_64(int64_t* outstarts, int64_t* outstops, const int64_t* distincts, int64_t lendistincts, const int64_t* gaps);

  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_local_nextparents_64(int64_t* nextparents, const int64_t* offsets, int64_t offsetsoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_local_outoffsets_64(int64_t* outoffsets, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);

}

#endif // AWKWARDCPU_REDUCERS_H_
