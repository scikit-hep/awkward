// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNELS_REDUCERS_H_
#define AWKWARD_KERNELS_REDUCERS_H_

#include "awkward/common.h"

extern "C" {
  /// @param toptr outparam
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_count_64(
      int64_t* toptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_countnonzero_bool_64(
      int64_t* toptr,
      const bool* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_countnonzero_int8_64(
      int64_t* toptr,
      const int8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_countnonzero_uint8_64(
      int64_t* toptr,
      const uint8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_countnonzero_int16_64(
      int64_t* toptr,
      const int16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_countnonzero_uint16_64(
      int64_t* toptr,
      const uint16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_countnonzero_int32_64(
      int64_t* toptr,
      const int32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_countnonzero_uint32_64(
      int64_t* toptr,
      const uint32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_countnonzero_int64_64(
      int64_t* toptr,
      const int64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_countnonzero_uint64_64(
      int64_t* toptr,
      const uint64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_countnonzero_float32_64(
      int64_t* toptr,
      const float* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_countnonzero_float64_64(
      int64_t* toptr,
      const double* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_int64_bool_64(
      int64_t* toptr,
      const bool* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_int64_int8_64(
      int64_t* toptr,
      const int8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_uint64_uint8_64(
      uint64_t* toptr,
      const uint8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_int64_int16_64(
      int64_t* toptr,
      const int16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_uint64_uint16_64(
      uint64_t* toptr,
      const uint16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_int64_int32_64(
      int64_t* toptr,
      const int32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_uint64_uint32_64(
      uint64_t* toptr,
      const uint32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_int64_int64_64(
      int64_t* toptr,
      const int64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_uint64_uint64_64(
      uint64_t* toptr,
      const uint64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_float32_float32_64(
      float* toptr,
      const float* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_float64_float64_64(
      double* toptr,
      const double* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_int32_bool_64(
      int32_t* toptr,
      const bool* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_int32_int8_64(
      int32_t* toptr,
      const int8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_uint32_uint8_64(
      uint32_t* toptr,
      const uint8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_int32_int16_64(
      int32_t* toptr,
      const int16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_uint32_uint16_64(
      uint32_t* toptr,
      const uint16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_int32_int32_64(
      int32_t* toptr,
      const int32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_uint32_uint32_64(
      uint32_t* toptr,
      const uint32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_bool_bool_64(
      bool* toptr,
      const bool* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_bool_int8_64(
      bool* toptr,
      const int8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_bool_uint8_64(
      bool* toptr,
      const uint8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_bool_int16_64(
      bool* toptr,
      const int16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_bool_uint16_64(
      bool* toptr,
      const uint16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_bool_int32_64(
      bool* toptr,
      const int32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_bool_uint32_64(
      bool* toptr,
      const uint32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_bool_int64_64(
      bool* toptr,
      const int64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_bool_uint64_64(
      bool* toptr,
      const uint64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_bool_float32_64(
      bool* toptr,
      const float* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_sum_bool_float64_64(
      bool* toptr,
      const double* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_int64_bool_64(
      int64_t* toptr,
      const bool* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_int64_int8_64(
      int64_t* toptr,
      const int8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_uint64_uint8_64(
      uint64_t* toptr,
      const uint8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_int64_int16_64(
      int64_t* toptr,
      const int16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_uint64_uint16_64(
      uint64_t* toptr,
      const uint16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_int64_int32_64(
      int64_t* toptr,
      const int32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_uint64_uint32_64(
      uint64_t* toptr,
      const uint32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_int64_int64_64(
      int64_t* toptr,
      const int64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_uint64_uint64_64(
      uint64_t* toptr,
      const uint64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_float32_float32_64(
      float* toptr,
      const float* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_float64_float64_64(
      double* toptr,
      const double* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_int32_bool_64(
      int32_t* toptr,
      const bool* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_int32_int8_64(
      int32_t* toptr,
      const int8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_uint32_uint8_64(
      uint32_t* toptr,
      const uint8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_int32_int16_64(
      int32_t* toptr,
      const int16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_uint32_uint16_64(
      uint32_t* toptr,
      const uint16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_int32_int32_64(
      int32_t* toptr,
      const int32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_uint32_uint32_64(
      uint32_t* toptr,
      const uint32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_bool_bool_64(
      bool* toptr,
      const bool* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_bool_int8_64(
      bool* toptr,
      const int8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_bool_uint8_64(
      bool* toptr,
      const uint8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_bool_int16_64(
      bool* toptr,
      const int16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_bool_uint16_64(
      bool* toptr,
      const uint16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_bool_int32_64(
      bool* toptr,
      const int32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_bool_uint32_64(
      bool* toptr,
      const uint32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_bool_int64_64(
      bool* toptr,
      const int64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_bool_uint64_64(
      bool* toptr,
      const uint64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_bool_float32_64(
      bool* toptr,
      const float* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_prod_bool_float64_64(
      bool* toptr,
      const double* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_min_int8_int8_64(
      int8_t* toptr,
      const int8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      int8_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_min_uint8_uint8_64(
      uint8_t* toptr,
      const uint8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      uint8_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_min_int16_int16_64(
      int16_t* toptr,
      const int16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      int16_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_min_uint16_uint16_64(
      uint16_t* toptr,
      const uint16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      uint16_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_min_int32_int32_64(
      int32_t* toptr,
      const int32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      int32_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_min_uint32_uint32_64(
      uint32_t* toptr,
      const uint32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      uint32_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_min_int64_int64_64(
      int64_t* toptr,
      const int64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      int64_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_min_uint64_uint64_64(
      uint64_t* toptr,
      const uint64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      uint64_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_min_float32_float32_64(
      float* toptr,
      const float* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      float identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_min_float64_float64_64(
      double* toptr,
      const double* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      double identity);

  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_max_int8_int8_64(
      int8_t* toptr,
      const int8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      int8_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_max_uint8_uint8_64(
      uint8_t* toptr,
      const uint8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      uint8_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_max_int16_int16_64(
      int16_t* toptr,
      const int16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      int16_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_max_uint16_uint16_64(
      uint16_t* toptr,
      const uint16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      uint16_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_max_int32_int32_64(
      int32_t* toptr,
      const int32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      int32_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_max_uint32_uint32_64(
      uint32_t* toptr,
      const uint32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      uint32_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_max_int64_int64_64(
      int64_t* toptr,
      const int64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      int64_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_max_uint64_uint64_64(
      uint64_t* toptr,
      const uint64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      uint64_t identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_max_float32_float32_64(
      float* toptr,
      const float* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      float identity);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  /// @param identity inparam role: reducer-identity
  EXPORT_SYMBOL struct Error
    awkward_reduce_max_float64_float64_64(
      double* toptr,
      const double* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength,
      double identity);

  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmin_bool_64(
      int64_t* toptr,
      const bool* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmin_int8_64(
      int64_t* toptr,
      const int8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmin_uint8_64(
      int64_t* toptr,
      const uint8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmin_int16_64(
      int64_t* toptr,
      const int16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmin_uint16_64(
      int64_t* toptr,
      const uint16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmin_int32_64(
      int64_t* toptr,
      const int32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmin_uint32_64(
      int64_t* toptr,
      const uint32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmin_int64_64(
      int64_t* toptr,
      const int64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmin_uint64_64(
      int64_t* toptr,
      const uint64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmin_float32_64(
      int64_t* toptr,
      const float* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmin_float64_64(
      int64_t* toptr,
      const double* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmax_bool_64(
      int64_t* toptr,
      const bool* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmax_int8_64(
      int64_t* toptr,
      const int8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmax_uint8_64(
      int64_t* toptr,
      const uint8_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmax_int16_64(
      int64_t* toptr,
      const int16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmax_uint16_64(
      int64_t* toptr,
      const uint16_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmax_int32_64(
      int64_t* toptr,
      const int32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmax_uint32_64(
      int64_t* toptr,
      const uint32_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmax_int64_64(
      int64_t* toptr,
      const int64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmax_uint64_64(
      int64_t* toptr,
      const uint64_t* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmax_float32_64(
      int64_t* toptr,
      const float* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);
  /// @param toptr outparam
  /// @param fromptr inparam role: reducer-fromptr
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_reduce_argmax_float64_64(
      int64_t* toptr,
      const double* fromptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

  /// @param toparents outparam
  /// @param length inparam role: reducer-length
  EXPORT_SYMBOL struct Error
    awkward_content_reduce_zeroparents_64(
      int64_t* toparents,
      int64_t length);

  /// @param globalstart outparam role: pointer
  /// @param globalstop outparam role: pointer
  /// @param offsets inparam role: reducer-offsets
  /// @param length inparam role: reducer-length
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray_reduce_global_startstop_64(
      int64_t* globalstart,
      int64_t* globalstop,
      const int64_t* offsets,
      int64_t length);

  /// @param maxcount outparam role: pointer
  /// @param offsetscopy outparam
  /// @param offsets inparam role: reducer-offsets
  /// @param length inparam role: reducer-length
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64(
      int64_t* maxcount,
      int64_t* offsetscopy,
      const int64_t* offsets,
      int64_t length);

  /// @param nextcarry outparam
  /// @param nextparents outparam
  /// @param nextlen inparam role: reducer-nextlen
  /// @param maxnextparents outparam role: pointer
  /// @param distincts outparam
  /// @param distinctslen inparam role: reducer-distinctslen
  /// @param offsetscopy inparam role: reducer-offsetscopy
  /// @param offsets inparam role: reducer-offsets
  /// @param length inparam role: reducer-length
  /// @param parents inparam role: reducer-parents
  /// @param maxcount inparam role: reducer-maxcount
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray_reduce_nonlocal_preparenext_64(
      int64_t* nextcarry,
      int64_t* nextparents,
      int64_t nextlen,
      int64_t* maxnextparents,
      int64_t* distincts,
      int64_t distinctslen,
      int64_t* offsetscopy,
      const int64_t* offsets,
      int64_t length,
      const int64_t* parents,
      int64_t maxcount);

  /// @param nextstarts outparam
  /// @param nextparents inparam role: reducer-nextparents
  /// @param nextlen inparam role: reducer-nextlen
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64(
      int64_t* nextstarts,
      const int64_t* nextparents,
      int64_t nextlen);

  /// @param gaps outparam
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray_reduce_nonlocal_findgaps_64(
      int64_t* gaps,
      const int64_t* parents,
      int64_t lenparents);

  /// @param outstarts outparam
  /// @param outstops outparam
  /// @param distincts inparam role: reducer-distincts
  /// @param lendistincts inparam role: reducer-lendistincts
  /// @param gaps inparam role: reducer-gaps
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64(
      int64_t* outstarts,
      int64_t* outstops,
      const int64_t* distincts,
      int64_t lendistincts,
      const int64_t* gaps,
      int64_t outlength);

  /// @param nummissing outparam
  /// @param missing outparam
  /// @param nextshifts outparam
  /// @param offsets inparam role: reducer-offsets
  /// @param length inparam role: reducer-length
  /// @param starts inparam role: reducer-starts
  /// @param parents inparam role: reducer-parents
  /// @param maxcount inparam role: reducer-maxcount
  /// @param nextlen inparam role: reducer-nextlen
  /// @param nextcarry inparam role: reducer-nextcarry
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64(
      int64_t* nummissing,
      int64_t* missing,
      int64_t* nextshifts,
      const int64_t* offsets,
      int64_t length,
      const int64_t* starts,
      const int64_t* parents,
      int64_t maxcount,
      int64_t nextlen,
      const int64_t* nextcarry);

  /// @param nextparents outparam
  /// @param offsets inparam role: reducer-offsets
  /// @param length inparam role: reducer-length
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray_reduce_local_nextparents_64(
      int64_t* nextparents,
      const int64_t* offsets,
      int64_t length);

  /// @param outoffsets outparam
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray_reduce_local_outoffsets_64(
      int64_t* outoffsets,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

  /// @param nextcarry outparam
  /// @param nextparents outparam
  /// @param outindex outparam
  /// @param index inparam role: reducer-index
  /// @param parents inparam role: reducer-parents
  /// @param length inparam role: reducer-length
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_reduce_next_64(
      int64_t* nextcarry,
      int64_t* nextparents,
      int64_t* outindex,
      const int32_t* index,
      int64_t* parents,
      int64_t length);
  /// @param nextcarry outparam
  /// @param nextparents outparam
  /// @param outindex outparam
  /// @param index inparam role: reducer-index
  /// @param parents inparam role: reducer-parents
  /// @param length inparam role: reducer-length
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_reduce_next_64(
      int64_t* nextcarry,
      int64_t* nextparents,
      int64_t* outindex,
      const uint32_t* index,
      int64_t* parents,
      int64_t length);
  /// @param nextcarry outparam
  /// @param nextparents outparam
  /// @param outindex outparam
  /// @param index inparam role: reducer-index
  /// @param parents inparam role: reducer-parents
  /// @param length inparam role: reducer-length
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_reduce_next_64(
      int64_t* nextcarry,
      int64_t* nextparents,
      int64_t* outindex,
      const int64_t* index,
      int64_t* parents,
      int64_t length);

  /// @param nextshifts outparam
  /// @param index inparam role: reducer-nextshifts
  /// @param length inparam role: reducer-length
  /// @param shifts inparam role: reducer-shifts
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_reduce_next_nonlocal_nextshifts_64(
      int64_t* nextshifts,
      const int32_t* index,
      int64_t length);
  /// @param nextshifts outparam
  /// @param index inparam role: reducer-nextshifts
  /// @param length inparam role: reducer-length
  /// @param shifts inparam role: reducer-shifts
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_reduce_next_nonlocal_nextshifts_64(
      int64_t* nextshifts,
      const uint32_t* index,
      int64_t length);
  /// @param nextshifts outparam
  /// @param index inparam role: reducer-nextshifts
  /// @param length inparam role: reducer-length
  /// @param shifts inparam role: reducer-shifts
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_reduce_next_nonlocal_nextshifts_64(
      int64_t* nextshifts,
      const int64_t* index,
      int64_t length);

  /// @param nextshifts outparam
  /// @param index inparam role: reducer-nextshifts
  /// @param length inparam role: reducer-length
  /// @param shifts inparam role: reducer-shifts
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_reduce_next_nonlocal_nextshifts_fromshifts_64(
      int64_t* nextshifts,
      const int32_t* index,
      int64_t length,
      const int64_t* shifts);
  /// @param nextshifts outparam
  /// @param index inparam role: reducer-nextshifts
  /// @param length inparam role: reducer-length
  /// @param shifts inparam role: reducer-shifts
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_reduce_next_nonlocal_nextshifts_fromshifts_64(
      int64_t* nextshifts,
      const uint32_t* index,
      int64_t length,
      const int64_t* shifts);
  /// @param nextshifts outparam
  /// @param index inparam role: reducer-nextshifts
  /// @param length inparam role: reducer-length
  /// @param shifts inparam role: reducer-shifts
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_reduce_next_nonlocal_nextshifts_fromshifts_64(
      int64_t* nextshifts,
      const int64_t* index,
      int64_t length,
      const int64_t* shifts);

  /// @param outoffsets outparam
  /// @param starts inparam role: reducer-starts
  /// @param startslength inparam role: reducer-startslength
  /// @param outindexlength inparam role: reducer-outindexlength
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray_reduce_next_fix_offsets_64(
      int64_t* outoffsets,
      const int64_t* starts,
      int64_t startslength,
      int64_t outindexlength);

  /// @param toptr outparam
  /// @param outlength inparam role: reducer-outlength
  /// @param parents inparam role: reducer-parents
  /// @param starts inparam role: reducer-starts
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_reduce_adjust_starts_64(
      int64_t* toptr,
      int64_t outlength,
      const int64_t* parents,
      const int64_t* starts);

  /// @param toptr outparam
  /// @param outlength inparam role: reducer-outlength
  /// @param parents inparam role: reducer-parents
  /// @param starts inparam role: reducer-starts
  /// @param shifts inparam role: reducer-shifts
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_reduce_adjust_starts_shifts_64(
      int64_t* toptr,
      int64_t outlength,
      const int64_t* parents,
      const int64_t* starts,
      const int64_t* shifts);

  /// @param toptr outparam
  /// @param parents inparam role: reducer-parents
  /// @param lenparents inparam role: reducer-lenparents
  /// @param outlength inparam role: reducer-outlength
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_reduce_mask_ByteMaskedArray_64(
      int8_t* toptr,
      const int64_t* parents,
      int64_t lenparents,
      int64_t outlength);

  /// @param nextcarry outparam
  /// @param nextparents outparam
  /// @param outindex outparam
  /// @param mask inparam role: reducer-mask
  /// @param parents inparam role: reducer-parents
  /// @param length inparam role: reducer-length
  /// @param validwhen inparam role: reducer-validwhen
  EXPORT_SYMBOL struct Error
    awkward_ByteMaskedArray_reduce_next_64(
      int64_t* nextcarry,
      int64_t* nextparents,
      int64_t* outindex,
      const int8_t* mask,
      const int64_t* parents,
      int64_t length,
      bool validwhen);

  /// @param nextshifts outparam
  /// @param mask inparam role: reducer-mask
  /// @param length inparam role: reducer-length
  /// @param valid_when inparam role: reducer-valid_when
  EXPORT_SYMBOL struct Error
    awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64(
      int64_t* nextshifts,
      const int8_t* mask,
      int64_t length,
      bool valid_when);

  /// @param nextshifts outparam
  /// @param mask inparam role: reducer-mask
  /// @param length inparam role: reducer-length
  /// @param valid_when inparam role: reducer-valid_when
  /// @param shifts inparam role: reducer-shifts
  EXPORT_SYMBOL struct Error
    awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64(
      int64_t* nextshifts,
      const int8_t* mask,
      int64_t length,
      bool valid_when,
      const int64_t* shifts);

}

#endif // AWKWARD_KERNELS_REDUCERS_H_
