// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_REDUCERS_H_
#define AWKWARDCPU_REDUCERS_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  EXPORT_SYMBOL struct Error awkward_reduce_count_64(int64_t* toptr, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);

  EXPORT_SYMBOL struct Error awkward_reduce_sum_int8_int8_64(int8_t* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_uint8_uint8_64(uint8_t* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_int16_int16_64(int16_t* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_uint16_uint16_64(uint16_t* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_int32_int32_64(int32_t* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_uint32_uint32_64(uint32_t* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_int64_int64_64(int64_t* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_uint64_uint64_64(uint64_t* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_float32_float32_64(float* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_float64_float64_64(double* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);

  EXPORT_SYMBOL struct Error awkward_reduce_sum_bool_bool_64(bool* toptr, const bool* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_bool_int8_64(bool* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_bool_uint8_64(bool* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_bool_int16_64(bool* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_bool_uint16_64(bool* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_bool_int32_64(bool* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_bool_uint32_64(bool* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_bool_int64_64(bool* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_bool_uint64_64(bool* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_bool_float32_64(bool* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_sum_bool_float64_64(bool* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);

  EXPORT_SYMBOL struct Error awkward_reduce_prod_int8_int8_64(int8_t* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_uint8_uint8_64(uint8_t* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_int16_int16_64(int16_t* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_uint16_uint16_64(uint16_t* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_int32_int32_64(int32_t* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_uint32_uint32_64(uint32_t* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_int64_int64_64(int64_t* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_uint64_uint64_64(uint64_t* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_float32_float32_64(float* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_float64_float64_64(double* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);

  EXPORT_SYMBOL struct Error awkward_reduce_prod_bool_bool_64(bool* toptr, const bool* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_bool_int8_64(bool* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_bool_uint8_64(bool* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_bool_int16_64(bool* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_bool_uint16_64(bool* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_bool_int32_64(bool* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_bool_uint32_64(bool* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_bool_int64_64(bool* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_bool_uint64_64(bool* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_bool_float32_64(bool* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);
  EXPORT_SYMBOL struct Error awkward_reduce_prod_bool_float64_64(bool* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);

  EXPORT_SYMBOL struct Error awkward_content_reduce_zeroparents_64(int64_t* toparents, int64_t length);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_global_startstop_64(int64_t* globalstart, int64_t* globalstop, const int64_t* offsets, int64_t offsetsoffset, int64_t length);

  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_nonlocal_maxcount_offsetscopy_64(int64_t* maxcount, int64_t* offsetscopy, const int64_t* offsets, int64_t offsetsoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_nonlocal_preparenext_64(int64_t* nextcarry, int64_t* nextparents, int64_t nextlen, int64_t* maxnextparents, int64_t* distincts, int64_t distinctslen, int64_t* offsetscopy, const int64_t* offsets, int64_t offsetsoffset, int64_t length, const int64_t* parents, int64_t parentsoffset, int64_t maxcount);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_nonlocal_findgaps_64(int64_t* gaps, const int64_t* parents, int64_t parentsoffset, int64_t lenparents);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_nonlocal_outstartsstops_64(int64_t* outstarts, int64_t* outstops, const int64_t* distincts, int64_t lendistincts, const int64_t* gaps);

  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_local_nextparents_64(int64_t* nextparents, const int64_t* offsets, int64_t offsetsoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray_reduce_local_outoffsets_64(int64_t* outoffsets, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength);

  EXPORT_SYMBOL struct Error awkward_indexedarray32_reduce_next_64(int64_t* nextcarry, int64_t* nextparents, const int32_t* index, int64_t indexoffset, int64_t* parents, int64_t parentsoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_reduce_next_64(int64_t* nextcarry, int64_t* nextparents, const uint32_t* index, int64_t indexoffset, int64_t* parents, int64_t parentsoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_reduce_next_64(int64_t* nextcarry, int64_t* nextparents, const int64_t* index, int64_t indexoffset, int64_t* parents, int64_t parentsoffset, int64_t length);

}

#endif // AWKWARDCPU_REDUCERS_H_
