// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_SORTING_H_
#define AWKWARDCPU_SORTING_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  EXPORT_SYMBOL struct Error
    awkward_argsort_prepare_ranges(
      int64_t* toptr,
      const int64_t* parents,
      int64_t parentsoffset,
      int64_t length,
      int64_t outlength);

  EXPORT_SYMBOL struct Error
    awkward_argsort_64(
      int64_t* toptr,
      const int64_t* fromindex,
      int64_t indexoffset,
      int64_t length);

  EXPORT_SYMBOL struct Error
    awkward_sort_bool(
      bool* toptr,
      const bool* fromptr,
      const int64_t* fromindex,
      int64_t indexoffset,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentsoffset,
      int64_t length);
  EXPORT_SYMBOL struct Error
    awkward_sort_int8(
      int8_t* toptr,
      const int8_t* fromptr,
      const int64_t* fromindex,
      int64_t indexoffset,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentsoffset,
      int64_t length);
  EXPORT_SYMBOL struct Error
    awkward_sort_uint8(
      uint8_t* toptr,
      const uint8_t* fromptr,
      const int64_t* fromindex,
      int64_t indexoffset,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentsoffset,
      int64_t length);
  EXPORT_SYMBOL struct Error
    awkward_sort_int16(
      int16_t* toptr,
      const int16_t* fromptr,
      const int64_t* fromindex,
      int64_t indexoffset,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentsoffset,
      int64_t length);
  EXPORT_SYMBOL struct Error
    awkward_sort_uint16(
      uint16_t* toptr,
      const uint16_t* fromptr,
      const int64_t* fromindex,
      int64_t indexoffset,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentsoffset,
      int64_t length);
  EXPORT_SYMBOL struct Error
    awkward_sort_int32(
      int32_t* toptr,
      const int32_t* fromptr,
      const int64_t* fromindex,
      int64_t indexoffset,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentsoffset,
      int64_t length);
  EXPORT_SYMBOL struct Error
    awkward_sort_uint32(
      uint32_t* toptr,
      const uint32_t* fromptr,
      const int64_t* fromindex,
      int64_t indexoffset,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentsoffset,
      int64_t length);
  EXPORT_SYMBOL struct Error
    awkward_sort_int64(
      int64_t* toptr,
      const int64_t* fromptr,
      const int64_t* fromindex,
      int64_t indexoffset,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentsoffset,
      int64_t length);
  EXPORT_SYMBOL struct Error
    awkward_sort_uint64(
      uint64_t* toptr,
      const uint64_t* fromptr,
      const int64_t* fromindex,
      int64_t indexoffset,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentsoffset,
      int64_t length);
  EXPORT_SYMBOL struct Error
    awkward_sort_float32(
      float* toptr,
      const float* fromptr,
      const int64_t* fromindex,
      int64_t indexoffset,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentsoffset,
      int64_t length);
  EXPORT_SYMBOL struct Error
    awkward_sort_float64(
      double* toptr,
      const double* fromptr,
      const int64_t* fromindex,
      int64_t indexoffset,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentsoffset,
      int64_t length);

  EXPORT_SYMBOL struct Error
    awkward_listoffsetarray_local_preparenext_64(
      int64_t* outcarry,
      const int64_t* result,
      int64_t nextlen);

}

#endif // AWKWARDCPU_GETITEM_H_
