// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/cpu-kernels/operations.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/reducers.h"

#ifdef BUILD_CUDA_KERNELS
#include "awkward/cuda-kernels/identities.h"
#endif

#include "awkward/kernel.h"

namespace kernel {
  /////////////////////////////////// awkward/cpu-kernels/getitem.h

  void regularize_rangeslice(
    int64_t* start,
    int64_t* stop,
    bool posstep,
    bool hasstart,
    bool hasstop,
    int64_t length) {
    return awkward_regularize_rangeslice(
      start,
      stop,
      posstep,
      hasstart,
      hasstop,
      length);
  }

  ERROR regularize_arrayslice_64(
    int64_t* flatheadptr,
    int64_t lenflathead,
    int64_t length) {
    return awkward_regularize_arrayslice_64(
      flatheadptr,
      lenflathead,
      length);
  }

  template <>
  ERROR Index_to_Index64(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t length) {
    return awkward_Index8_to_Index64(
      toptr,
      fromptr,
      length);
  }
  template <>
  ERROR Index_to_Index64(
    int64_t* toptr,
    const uint8_t* fromptr,
    int64_t length) {
    return awkward_IndexU8_to_Index64(
      toptr,
      fromptr,
      length);
  }
  template <>
  ERROR Index_to_Index64(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t length) {
    return awkward_Index32_to_Index64(
      toptr,
      fromptr,
      length);
  }
  template <>
  ERROR Index_to_Index64(
    int64_t* toptr,
    const uint32_t* fromptr,
    int64_t length) {
    return awkward_IndexU32_to_Index64(
      toptr,
      fromptr,
      length);
  }

  template <>
  Error Index_carry_64<int8_t>(
    int8_t* toindex,
    const int8_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t lenfromindex,
    int64_t length) {
    return awkward_Index8_carry_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      lenfromindex,
      length);
  }
  template <>
  Error Index_carry_64<uint8_t>(
    uint8_t* toindex,
    const uint8_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t lenfromindex,
    int64_t length) {
    return awkward_IndexU8_carry_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      lenfromindex,
      length);
  }
  template <>
  Error Index_carry_64<int32_t>(
    int32_t* toindex,
    const int32_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t lenfromindex,
    int64_t length) {
    return awkward_Index32_carry_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      lenfromindex,
      length);
  }
  template <>
  Error Index_carry_64<uint32_t>(
    uint32_t* toindex,
    const uint32_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t lenfromindex,
    int64_t length) {
    return awkward_IndexU32_carry_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      lenfromindex,
      length);
  }
  template <>
  Error Index_carry_64<int64_t>(
    int64_t* toindex,
    const int64_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t lenfromindex,
    int64_t length) {
    return awkward_Index64_carry_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      lenfromindex,
      length);
  }

  template <>
  Error Index_carry_nocheck_64<int8_t>(
    int8_t* toindex,
    const int8_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t length) {
    return awkward_Index8_carry_nocheck_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      length);
  }
  template <>
  Error Index_carry_nocheck_64<uint8_t>(
    uint8_t* toindex,
    const uint8_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t length) {
    return awkward_IndexU8_carry_nocheck_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      length);
  }
  template <>
  Error Index_carry_nocheck_64<int32_t>(
    int32_t* toindex,
    const int32_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t length) {
    return awkward_Index32_carry_nocheck_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      length);
  }
  template <>
  Error Index_carry_nocheck_64<uint32_t>(
    uint32_t* toindex,
    const uint32_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t length) {
    return awkward_IndexU32_carry_nocheck_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      length);
  }
  template <>
  Error Index_carry_nocheck_64<int64_t>(
    int64_t* toindex,
    const int64_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t length) {
    return awkward_Index64_carry_nocheck_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      length);
  }

  ERROR slicearray_ravel_64(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t ndim,
    const int64_t* shape,
    const int64_t* strides) {
    return awkward_slicearray_ravel_64(
      toptr,
      fromptr,
      ndim,
      shape,
      strides);
  }

  ERROR slicemissing_check_same(
    bool* same,
    const int8_t* bytemask,
    int64_t bytemaskoffset,
    const int64_t* missingindex,
    int64_t missingindexoffset,
    int64_t length) {
    return awkward_slicemissing_check_same(
      same,
      bytemask,
      bytemaskoffset,
      missingindex,
      missingindexoffset,
      length);
  }

  template <>
  ERROR carry_arange(
    int32_t* toptr,
    int64_t length) {
    return awkward_carry_arange32(
      toptr,
      length);
  }
  template <>
  ERROR carry_arange(
    uint32_t* toptr,
    int64_t length) {
    return awkward_carry_arangeU32(
      toptr,
      length);
  }
  template <>
  ERROR carry_arange(
    int64_t* toptr,
    int64_t length) {
    return awkward_carry_arange64(
      toptr,
      length);
  }

  template <>
  ERROR Identities_getitem_carry_64(
    int32_t* newidentitiesptr,
    const int32_t* identitiesptr,
    const int64_t* carryptr,
    int64_t lencarry,
    int64_t offset,
    int64_t width,
    int64_t length) {
    return awkward_Identities32_getitem_carry_64(
      newidentitiesptr,
      identitiesptr,
      carryptr,
      lencarry,
      offset,
      width,
      length);
  }
  template <>
  ERROR Identities_getitem_carry_64(
    int64_t* newidentitiesptr,
    const int64_t* identitiesptr,
    const int64_t* carryptr,
    int64_t lencarry,
    int64_t offset,
    int64_t width,
    int64_t length) {
    return awkward_Identities64_getitem_carry_64(
      newidentitiesptr,
      identitiesptr,
      carryptr,
      lencarry,
      offset,
      width,
      length);
  }

  ERROR NumpyArray_contiguous_init_64(
    int64_t* toptr,
    int64_t skip,
    int64_t stride) {
    return awkward_NumpyArray_contiguous_init_64(
      toptr,
      skip,
      stride);
  }


  ERROR NumpyArray_contiguous_copy_64(
    uint8_t* toptr,
    const uint8_t* fromptr,
    int64_t len,
    int64_t stride,
    int64_t offset,
    const int64_t* pos) {
    return awkward_NumpyArray_contiguous_copy_64(
      toptr,
      fromptr,
      len,
      stride,
      offset,
      pos);
  }

  ERROR NumpyArray_contiguous_next_64(
    int64_t* topos,
    const int64_t* frompos,
    int64_t len,
    int64_t skip,
    int64_t stride) {
    return awkward_NumpyArray_contiguous_next_64(
      topos,
      frompos,
      len,
      skip,
      stride);
  }

  ERROR NumpyArray_getitem_next_null_64(
    uint8_t* toptr,
    const uint8_t* fromptr,
    int64_t len,
    int64_t stride,
    int64_t offset,
    const int64_t* pos) {
    return awkward_NumpyArray_getitem_next_null_64(
      toptr,
      fromptr,
      len,
      stride,
      offset,
      pos);
  }

  ERROR NumpyArray_getitem_next_at_64(
    int64_t* nextcarryptr,
    const int64_t* carryptr,
    int64_t lencarry,
    int64_t skip,
    int64_t at) {
    return awkward_NumpyArray_getitem_next_at_64(
      nextcarryptr,
      carryptr,
      lencarry,
      skip,
      at);
  }

  ERROR NumpyArray_getitem_next_range_64(
    int64_t* nextcarryptr,
    const int64_t* carryptr,
    int64_t lencarry,
    int64_t lenhead,
    int64_t skip,
    int64_t start,
    int64_t step) {
    return awkward_NumpyArray_getitem_next_range_64(
      nextcarryptr,
      carryptr,
      lencarry,
      lenhead,
      skip,
      start,
      step);
  }

  ERROR NumpyArray_getitem_next_range_advanced_64(
    int64_t* nextcarryptr,
    int64_t* nextadvancedptr,
    const int64_t* carryptr,
    const int64_t* advancedptr,
    int64_t lencarry,
    int64_t lenhead,
    int64_t skip,
    int64_t start,
    int64_t step) {
    return awkward_NumpyArray_getitem_next_range_advanced_64(
      nextcarryptr,
      nextadvancedptr,
      carryptr,
      advancedptr,
      lencarry,
      lenhead,
      skip,
      start,
      step);
  }

  ERROR NumpyArray_getitem_next_array_64(
    int64_t* nextcarryptr,
    int64_t* nextadvancedptr,
    const int64_t* carryptr,
    const int64_t* flatheadptr,
    int64_t lencarry,
    int64_t lenflathead,
    int64_t skip) {
    return awkward_NumpyArray_getitem_next_array_64(
      nextcarryptr,
      nextadvancedptr,
      carryptr,
      flatheadptr,
      lencarry,
      lenflathead,
      skip);
  }

  ERROR NumpyArray_getitem_next_array_advanced_64(
    int64_t* nextcarryptr,
    const int64_t* carryptr,
    const int64_t* advancedptr,
    const int64_t* flatheadptr,
    int64_t lencarry,
    int64_t skip) {
    return awkward_NumpyArray_getitem_next_array_advanced_64(
      nextcarryptr,
      carryptr,
      advancedptr,
      flatheadptr,
      lencarry,
      skip);
  }

  ERROR NumpyArray_getitem_boolean_numtrue(
    int64_t* numtrue,
    const int8_t* fromptr,
    int64_t byteoffset,
    int64_t length,
    int64_t stride) {
    return awkward_NumpyArray_getitem_boolean_numtrue(
      numtrue,
      fromptr,
      byteoffset,
      length,
      stride);
  }

  ERROR NumpyArray_getitem_boolean_nonzero_64(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t byteoffset,
    int64_t length,
    int64_t stride) {
    return awkward_NumpyArray_getitem_boolean_nonzero_64(
      toptr,
      fromptr,
      byteoffset,
      length,
      stride);
  }

  template <>
  Error ListArray_getitem_next_at_64<int32_t>(
    int64_t* tocarry,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t at) {
    return awkward_ListArray32_getitem_next_at_64(
      tocarry,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset,
      at);
  }
  template <>
  Error ListArray_getitem_next_at_64<uint32_t>(
    int64_t* tocarry,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t at) {
    return awkward_ListArrayU32_getitem_next_at_64(
      tocarry,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset,
      at);
  }
  template <>
  Error ListArray_getitem_next_at_64<int64_t>(
    int64_t* tocarry,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t at) {
    return awkward_ListArray64_getitem_next_at_64(
      tocarry,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset,
      at);
  }

  template <>
  Error ListArray_getitem_next_range_carrylength<int32_t>(
    int64_t* carrylength,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step) {
    return awkward_ListArray32_getitem_next_range_carrylength(
      carrylength,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset,
      start,
      stop,
      step);
  }
  template <>
  Error ListArray_getitem_next_range_carrylength<uint32_t>(
    int64_t* carrylength,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step) {
    return awkward_ListArrayU32_getitem_next_range_carrylength(
      carrylength,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset,
      start,
      stop,
      step);
  }
  template <>
  Error ListArray_getitem_next_range_carrylength<int64_t>(
    int64_t* carrylength,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step) {
    return awkward_ListArray64_getitem_next_range_carrylength(
      carrylength,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset,
      start,
      stop,
      step);
  }

  template <>
  Error ListArray_getitem_next_range_64<int32_t>(
    int32_t* tooffsets,
    int64_t* tocarry,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step) {
    return awkward_ListArray32_getitem_next_range_64(
      tooffsets,
      tocarry,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset,
      start,
      stop,
      step);
  }
  template <>
  Error ListArray_getitem_next_range_64<uint32_t>(
    uint32_t* tooffsets,
    int64_t* tocarry,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step) {
    return awkward_ListArrayU32_getitem_next_range_64(
      tooffsets,
      tocarry,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset,
      start,
      stop,
      step);
  }
  template <>
  Error ListArray_getitem_next_range_64<int64_t>(
    int64_t* tooffsets,
    int64_t* tocarry,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step) {
    return awkward_ListArray64_getitem_next_range_64(
      tooffsets,
      tocarry,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset,
      start,
      stop,
      step);
  }

  template <>
  Error ListArray_getitem_next_range_counts_64<int32_t>(
    int64_t* total,
    const int32_t* fromoffsets,
    int64_t lenstarts) {
    return awkward_ListArray32_getitem_next_range_counts_64(
      total,
      fromoffsets,
      lenstarts);
  }
  template <>
  Error ListArray_getitem_next_range_counts_64<uint32_t>(
    int64_t* total,
    const uint32_t* fromoffsets,
    int64_t lenstarts) {
    return awkward_ListArrayU32_getitem_next_range_counts_64(
      total,
      fromoffsets,
      lenstarts);
  }
  template <>
  Error ListArray_getitem_next_range_counts_64<int64_t>(
    int64_t* total,
    const int64_t* fromoffsets,
    int64_t lenstarts) {
    return awkward_ListArray64_getitem_next_range_counts_64(
      total,
      fromoffsets,
      lenstarts);
  }

  template <>
  Error ListArray_getitem_next_range_spreadadvanced_64<int32_t>(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const int32_t* fromoffsets,
    int64_t lenstarts) {
    return awkward_ListArray32_getitem_next_range_spreadadvanced_64(
      toadvanced,
      fromadvanced,
      fromoffsets,
      lenstarts);
  }
  template <>
  Error ListArray_getitem_next_range_spreadadvanced_64<uint32_t>(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const uint32_t* fromoffsets,
    int64_t lenstarts) {
    return awkward_ListArrayU32_getitem_next_range_spreadadvanced_64(
      toadvanced,
      fromadvanced,
      fromoffsets,
      lenstarts);
  }
  template <>
  Error ListArray_getitem_next_range_spreadadvanced_64<int64_t>(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const int64_t* fromoffsets,
    int64_t lenstarts) {
    return awkward_ListArray64_getitem_next_range_spreadadvanced_64(
      toadvanced,
      fromadvanced,
      fromoffsets,
      lenstarts);
  }

  template <>
  Error ListArray_getitem_next_array_64<int32_t>(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    const int64_t* fromarray,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent) {
    return awkward_ListArray32_getitem_next_array_64(
      tocarry,
      toadvanced,
      fromstarts,
      fromstops,
      fromarray,
      startsoffset,
      stopsoffset,
      lenstarts,
      lenarray,
      lencontent);
  }
  template <>
  Error ListArray_getitem_next_array_64<uint32_t>(
    int64_t* tocarry,
    int64_t* toadvanced,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    const int64_t* fromarray,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent) {
    return awkward_ListArrayU32_getitem_next_array_64(
      tocarry,
      toadvanced,
      fromstarts,
      fromstops,
      fromarray,
      startsoffset,
      stopsoffset,
      lenstarts,
      lenarray,
      lencontent);
  }
  template <>
  Error ListArray_getitem_next_array_64<int64_t>(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    const int64_t* fromarray,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent) {
    return awkward_ListArray64_getitem_next_array_64(
      tocarry,
      toadvanced,
      fromstarts,
      fromstops,
      fromarray,
      startsoffset,
      stopsoffset,
      lenstarts,
      lenarray,
      lencontent);
  }

  template <>
  Error ListArray_getitem_next_array_advanced_64<int32_t>(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    const int64_t* fromarray,
    const int64_t* fromadvanced,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent) {
    return awkward_ListArray32_getitem_next_array_advanced_64(
      tocarry,
      toadvanced,
      fromstarts,
      fromstops,
      fromarray,
      fromadvanced,
      startsoffset,
      stopsoffset,
      lenstarts,
      lenarray,
      lencontent);
  }
  template <>
  Error ListArray_getitem_next_array_advanced_64<uint32_t>(
    int64_t* tocarry,
    int64_t* toadvanced,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    const int64_t* fromarray,
    const int64_t* fromadvanced,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent) {
    return awkward_ListArrayU32_getitem_next_array_advanced_64(
      tocarry,
      toadvanced,
      fromstarts,
      fromstops,
      fromarray,
      fromadvanced,
      startsoffset,
      stopsoffset,
      lenstarts,
      lenarray,
      lencontent);
  }
  template <>
  Error ListArray_getitem_next_array_advanced_64<int64_t>(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    const int64_t* fromarray,
    const int64_t* fromadvanced,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent) {
    return awkward_ListArray64_getitem_next_array_advanced_64(
      tocarry,
      toadvanced,
      fromstarts,
      fromstops,
      fromarray,
      fromadvanced,
      startsoffset,
      stopsoffset,
      lenstarts,
      lenarray,
      lencontent);
  }

  template <>
  Error ListArray_getitem_carry_64<int32_t>(
    int32_t* tostarts,
    int32_t* tostops,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    const int64_t* fromcarry,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lencarry) {
    return awkward_ListArray32_getitem_carry_64(
      tostarts,
      tostops,
      fromstarts,
      fromstops,
      fromcarry,
      startsoffset,
      stopsoffset,
      lenstarts,
      lencarry);
  }
  template <>
  Error ListArray_getitem_carry_64<uint32_t>(
    uint32_t* tostarts,
    uint32_t* tostops,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    const int64_t* fromcarry,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lencarry) {
    return awkward_ListArrayU32_getitem_carry_64(
      tostarts,
      tostops,
      fromstarts,
      fromstops,
      fromcarry,
      startsoffset,
      stopsoffset,
      lenstarts,
      lencarry);
  }
  template <>
  Error ListArray_getitem_carry_64<int64_t>(
    int64_t* tostarts,
    int64_t* tostops,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    const int64_t* fromcarry,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lencarry) {
    return awkward_ListArray64_getitem_carry_64(
      tostarts,
      tostops,
      fromstarts,
      fromstops,
      fromcarry,
      startsoffset,
      stopsoffset,
      lenstarts,
      lencarry);
  }

  ERROR RegularArray_getitem_next_at_64(
    int64_t* tocarry,
    int64_t at,
    int64_t len,
    int64_t size) {
    return awkward_RegularArray_getitem_next_at_64(
      tocarry,
      at,
      len,
      size);
  }

  ERROR RegularArray_getitem_next_range_64(
    int64_t* tocarry,
    int64_t regular_start,
    int64_t step,
    int64_t len,
    int64_t size,
    int64_t nextsize) {
    return awkward_RegularArray_getitem_next_range_64(
      tocarry,
      regular_start,
      step,
      len,
      size,
      nextsize);
  }

  ERROR RegularArray_getitem_next_range_spreadadvanced_64(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    int64_t len,
    int64_t nextsize) {
    return awkward_RegularArray_getitem_next_range_spreadadvanced_64(
      toadvanced,
      fromadvanced,
      len,
      nextsize);
  }

  ERROR RegularArray_getitem_next_array_regularize_64(
    int64_t* toarray,
    const int64_t* fromarray,
    int64_t lenarray,
    int64_t size) {
    return awkward_RegularArray_getitem_next_array_regularize_64(
      toarray,
      fromarray,
      lenarray,
      size);
  }

  ERROR RegularArray_getitem_next_array_64(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int64_t* fromarray,
    int64_t len,
    int64_t lenarray,
    int64_t size) {
    return awkward_RegularArray_getitem_next_array_64(
      tocarry,
      toadvanced,
      fromarray,
      len,
      lenarray,
      size);
  }

  ERROR RegularArray_getitem_next_array_advanced_64(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const int64_t* fromarray,
    int64_t len,
    int64_t lenarray,
    int64_t size) {
    return awkward_RegularArray_getitem_next_array_advanced_64(
      tocarry,
      toadvanced,
      fromadvanced,
      fromarray,
      len,
      lenarray,
      size);
  }

  ERROR RegularArray_getitem_carry_64(
    int64_t* tocarry,
    const int64_t* fromcarry,
    int64_t lencarry,
    int64_t size) {
    return awkward_RegularArray_getitem_carry_64(
      tocarry,
      fromcarry,
      lencarry,
      size);
  }

  template <>
  Error IndexedArray_numnull<int32_t>(
    int64_t* numnull,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex) {
    return awkward_IndexedArray32_numnull(
      numnull,
      fromindex,
      indexoffset,
      lenindex);
  }
  template <>
  Error IndexedArray_numnull<uint32_t>(
    int64_t* numnull,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex) {
    return awkward_IndexedArrayU32_numnull(
      numnull,
      fromindex,
      indexoffset,
      lenindex);
  }
  template <>
  Error IndexedArray_numnull<int64_t>(
    int64_t* numnull,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex) {
    return awkward_IndexedArray64_numnull(
      numnull,
      fromindex,
      indexoffset,
      lenindex);
  }

  template <>
  Error IndexedArray_getitem_nextcarry_outindex_64<int32_t>(
    int64_t* tocarry,
    int32_t* toindex,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent) {
    return awkward_IndexedArray32_getitem_nextcarry_outindex_64(
      tocarry,
      toindex,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  Error IndexedArray_getitem_nextcarry_outindex_64<uint32_t>(
    int64_t* tocarry,
    uint32_t* toindex,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent) {
    return awkward_IndexedArrayU32_getitem_nextcarry_outindex_64(
      tocarry,
      toindex,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  Error IndexedArray_getitem_nextcarry_outindex_64<int64_t>(
    int64_t* tocarry,
    int64_t* toindex,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent) {
    return awkward_IndexedArray64_getitem_nextcarry_outindex_64(
      tocarry,
      toindex,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }

  template <>
  Error IndexedArray_getitem_nextcarry_outindex_mask_64<int32_t>(
    int64_t* tocarry,
    int64_t* toindex,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent) {
    return awkward_IndexedArray32_getitem_nextcarry_outindex_mask_64(
      tocarry,
      toindex,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  Error IndexedArray_getitem_nextcarry_outindex_mask_64<uint32_t>(
    int64_t* tocarry,
    int64_t* toindex,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent) {
    return awkward_IndexedArrayU32_getitem_nextcarry_outindex_mask_64(
      tocarry,
      toindex,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  Error IndexedArray_getitem_nextcarry_outindex_mask_64<int64_t>(
    int64_t* tocarry,
    int64_t* toindex,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent) {
    return awkward_IndexedArray64_getitem_nextcarry_outindex_mask_64(
      tocarry,
      toindex,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }

  ERROR ListOffsetArray_getitem_adjust_offsets_64(
    int64_t* tooffsets,
    int64_t* tononzero,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length,
    const int64_t* nonzero,
    int64_t nonzerooffset,
    int64_t nonzerolength) {
    return awkward_ListOffsetArray_getitem_adjust_offsets_64(
      tooffsets,
      tononzero,
      fromoffsets,
      offsetsoffset,
      length,
      nonzero,
      nonzerooffset,
      nonzerolength);
  }

  ERROR ListOffsetArray_getitem_adjust_offsets_index_64(
    int64_t* tooffsets,
    int64_t* tononzero,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length,
    const int64_t* index,
    int64_t indexoffset,
    int64_t indexlength,
    const int64_t* nonzero,
    int64_t nonzerooffset,
    int64_t nonzerolength,
    const int8_t* originalmask,
    int64_t maskoffset,
    int64_t masklength) {
    return awkward_ListOffsetArray_getitem_adjust_offsets_index_64(
      tooffsets,
      tononzero,
      fromoffsets,
      offsetsoffset,
      length,
      index,
      indexoffset,
      indexlength,
      nonzero,
      nonzerooffset,
      nonzerolength,
      originalmask,
      maskoffset,
      masklength);
  }

  ERROR IndexedArray_getitem_adjust_outindex_64(
    int8_t* tomask,
    int64_t* toindex,
    int64_t* tononzero,
    const int64_t* fromindex,
    int64_t fromindexoffset,
    int64_t fromindexlength,
    const int64_t* nonzero,
    int64_t nonzerooffset,
    int64_t nonzerolength) {
    return awkward_IndexedArray_getitem_adjust_outindex_64(
      tomask,
      toindex,
      tononzero,
      fromindex,
      fromindexoffset,
      fromindexlength,
      nonzero,
      nonzerooffset,
      nonzerolength);
  }

  template <>
  ERROR IndexedArray_getitem_nextcarry_64(
    int64_t* tocarry,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent) {
    return awkward_IndexedArray32_getitem_nextcarry_64(
      tocarry,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  ERROR IndexedArray_getitem_nextcarry_64(
    int64_t* tocarry,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent) {
    return awkward_IndexedArrayU32_getitem_nextcarry_64(
      tocarry,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  ERROR IndexedArray_getitem_nextcarry_64(
    int64_t* tocarry,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent) {
    return awkward_IndexedArray64_getitem_nextcarry_64(
      tocarry,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }

  template <>
  Error IndexedArray_getitem_carry_64<int32_t>(
    int32_t* toindex,
    const int32_t* fromindex,
    const int64_t* fromcarry,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencarry) {
    return awkward_IndexedArray32_getitem_carry_64(
      toindex,
      fromindex,
      fromcarry,
      indexoffset,
      lenindex,
      lencarry);
  }
  template <>
  Error IndexedArray_getitem_carry_64<uint32_t>(
    uint32_t* toindex,
    const uint32_t* fromindex,
    const int64_t* fromcarry,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencarry) {
    return awkward_IndexedArrayU32_getitem_carry_64(
      toindex,
      fromindex,
      fromcarry,
      indexoffset,
      lenindex,
      lencarry);
  }
  template <>
  Error IndexedArray_getitem_carry_64<int64_t>(
    int64_t* toindex,
    const int64_t* fromindex,
    const int64_t* fromcarry,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencarry) {
    return awkward_IndexedArray64_getitem_carry_64(
      toindex,
      fromindex,
      fromcarry,
      indexoffset,
      lenindex,
      lencarry);
  }

  template <>
  Error UnionArray_regular_index_getsize<int8_t>(
    int64_t* size,
    const int8_t* fromtags,
    int64_t tagsoffset,
    int64_t length) {
    return awkward_UnionArray8_regular_index_getsize(
      size,
      fromtags,
      tagsoffset,
      length);
  }

  template <>
  Error UnionArray_regular_index<int8_t, int32_t>(
    int32_t* toindex,
    int32_t* current,
    int64_t size,
    const int8_t* fromtags,
    int64_t tagsoffset,
    int64_t length) {
    return awkward_UnionArray8_32_regular_index(
      toindex,
      current,
      size,
      fromtags,
      tagsoffset,
      length);
  }
  template <>
  Error UnionArray_regular_index<int8_t, uint32_t>(
    uint32_t* toindex,
    uint32_t* current,
    int64_t size,
    const int8_t* fromtags,
    int64_t tagsoffset,
    int64_t length) {
    return awkward_UnionArray8_U32_regular_index(
      toindex,
      current,
      size,
      fromtags,
      tagsoffset,
      length);
  }
  template <>
  Error UnionArray_regular_index<int8_t, int64_t>(
    int64_t* toindex,
    int64_t* current,
    int64_t size,
    const int8_t* fromtags,
    int64_t tagsoffset,
    int64_t length) {
    return awkward_UnionArray8_64_regular_index(
      toindex,
      current,
      size,
      fromtags,
      tagsoffset,
      length);
  }

  template <>
  Error UnionArray_project_64<int8_t, int32_t>(
    int64_t* lenout,
    int64_t* tocarry,
    const int8_t* fromtags,
    int64_t tagsoffset,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t length,
    int64_t which) {
    return awkward_UnionArray8_32_project_64(
      lenout,
      tocarry,
      fromtags,
      tagsoffset,
      fromindex,
      indexoffset,
      length,
      which);
  }
  template <>
  Error UnionArray_project_64<int8_t, uint32_t>(
    int64_t* lenout,
    int64_t* tocarry,
    const int8_t* fromtags,
    int64_t tagsoffset,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t length,
    int64_t which) {
    return awkward_UnionArray8_U32_project_64(
      lenout,
      tocarry,
      fromtags,
      tagsoffset,
      fromindex,
      indexoffset,
      length,
      which);
  }
  template <>
  Error UnionArray_project_64<int8_t, int64_t>(
    int64_t* lenout,
    int64_t* tocarry,
    const int8_t* fromtags,
    int64_t tagsoffset,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t length,
    int64_t which) {
    return awkward_UnionArray8_64_project_64(
      lenout,
      tocarry,
      fromtags,
      tagsoffset,
      fromindex,
      indexoffset,
      length,
      which);
  }

  ERROR missing_repeat_64(
    int64_t* outindex,
    const int64_t* index,
    int64_t indexoffset,
    int64_t indexlength,
    int64_t repetitions,
    int64_t regularsize) {
    return awkward_missing_repeat_64(
      outindex,
      index,
      indexoffset,
      indexlength,
      repetitions,
      regularsize);
  }

  ERROR RegularArray_getitem_jagged_expand_64(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t regularsize,
    int64_t regularlength) {
    return awkward_RegularArray_getitem_jagged_expand_64(
      multistarts,
      multistops,
      singleoffsets,
      regularsize,
      regularlength);
  }

  template <>
  Error ListArray_getitem_jagged_expand_64<int32_t>(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t* tocarry,
    const int32_t* fromstarts,
    int64_t fromstartsoffset,
    const int32_t* fromstops,
    int64_t fromstopsoffset,
    int64_t jaggedsize,
    int64_t length) {
    return awkward_ListArray32_getitem_jagged_expand_64(
      multistarts,
      multistops,
      singleoffsets,
      tocarry,
      fromstarts,
      fromstartsoffset,
      fromstops,
      fromstopsoffset,
      jaggedsize,
      length);
  }
  template <>
  Error ListArray_getitem_jagged_expand_64(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t* tocarry,
    const uint32_t* fromstarts,
    int64_t fromstartsoffset,
    const uint32_t* fromstops,
    int64_t fromstopsoffset,
    int64_t jaggedsize,
    int64_t length) {
    return awkward_ListArrayU32_getitem_jagged_expand_64(
      multistarts,
      multistops,
      singleoffsets,
      tocarry,
      fromstarts,
      fromstartsoffset,
      fromstops,
      fromstopsoffset,
      jaggedsize,
      length);
  }
  template <>
  Error ListArray_getitem_jagged_expand_64(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t* tocarry,
    const int64_t* fromstarts,
    int64_t fromstartsoffset,
    const int64_t* fromstops,
    int64_t fromstopsoffset,
    int64_t jaggedsize,
    int64_t length) {
    return awkward_ListArray64_getitem_jagged_expand_64(
      multistarts,
      multistops,
      singleoffsets,
      tocarry,
      fromstarts,
      fromstartsoffset,
      fromstops,
      fromstopsoffset,
      jaggedsize,
      length);
  }

  ERROR ListArray_getitem_jagged_carrylen_64(
    int64_t* carrylen,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t sliceouterlen) {
    return awkward_ListArray_getitem_jagged_carrylen_64(
      carrylen,
      slicestarts,
      slicestartsoffset,
      slicestops,
      slicestopsoffset,
      sliceouterlen);
  }

  template <>
  Error ListArray_getitem_jagged_apply_64<int32_t>(
    int64_t* tooffsets,
    int64_t* tocarry,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t sliceouterlen,
    const int64_t* sliceindex,
    int64_t sliceindexoffset,
    int64_t sliceinnerlen,
    const int32_t* fromstarts,
    int64_t fromstartsoffset,
    const int32_t* fromstops,
    int64_t fromstopsoffset,
    int64_t contentlen) {
    return awkward_ListArray32_getitem_jagged_apply_64(
      tooffsets,
      tocarry,
      slicestarts,
      slicestartsoffset,
      slicestops,
      slicestopsoffset,
      sliceouterlen,
      sliceindex,
      sliceindexoffset,
      sliceinnerlen,
      fromstarts,
      fromstartsoffset,
      fromstops,
      fromstopsoffset,
      contentlen);
  }
  template <>
  Error ListArray_getitem_jagged_apply_64<uint32_t>(
    int64_t* tooffsets,
    int64_t* tocarry,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t sliceouterlen,
    const int64_t* sliceindex,
    int64_t sliceindexoffset,
    int64_t sliceinnerlen,
    const uint32_t* fromstarts,
    int64_t fromstartsoffset,
    const uint32_t* fromstops,
    int64_t fromstopsoffset,
    int64_t contentlen) {
    return awkward_ListArrayU32_getitem_jagged_apply_64(
      tooffsets,
      tocarry,
      slicestarts,
      slicestartsoffset,
      slicestops,
      slicestopsoffset,
      sliceouterlen,
      sliceindex,
      sliceindexoffset,
      sliceinnerlen,
      fromstarts,
      fromstartsoffset,
      fromstops,
      fromstopsoffset,
      contentlen);
  }
  template <>
  Error ListArray_getitem_jagged_apply_64<int64_t>(
    int64_t* tooffsets,
    int64_t* tocarry,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t sliceouterlen,
    const int64_t* sliceindex,
    int64_t sliceindexoffset,
    int64_t sliceinnerlen,
    const int64_t* fromstarts,
    int64_t fromstartsoffset,
    const int64_t* fromstops,
    int64_t fromstopsoffset,
    int64_t contentlen) {
    return awkward_ListArray64_getitem_jagged_apply_64(
      tooffsets,
      tocarry,
      slicestarts,
      slicestartsoffset,
      slicestops,
      slicestopsoffset,
      sliceouterlen,
      sliceindex,
      sliceindexoffset,
      sliceinnerlen,
      fromstarts,
      fromstartsoffset,
      fromstops,
      fromstopsoffset,
      contentlen);
  }

  ERROR ListArray_getitem_jagged_numvalid_64(
    int64_t* numvalid,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t length,
    const int64_t* missing,
    int64_t missingoffset,
    int64_t missinglength) {
    return awkward_ListArray_getitem_jagged_numvalid_64(
      numvalid,
      slicestarts,
      slicestartsoffset,
      slicestops,
      slicestopsoffset,
      length,
      missing,
      missingoffset,
      missinglength);
  }

  ERROR ListArray_getitem_jagged_shrink_64(
    int64_t* tocarry,
    int64_t* tosmalloffsets,
    int64_t* tolargeoffsets,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t length,
    const int64_t* missing,
    int64_t missingoffset) {
    return awkward_ListArray_getitem_jagged_shrink_64(
      tocarry,
      tosmalloffsets,
      tolargeoffsets,
      slicestarts,
      slicestartsoffset,
      slicestops,
      slicestopsoffset,
      length,
      missing,
      missingoffset);
  }

  template <>
  Error ListArray_getitem_jagged_descend_64<int32_t>(
    int64_t* tooffsets,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t sliceouterlen,
    const int32_t* fromstarts,
    int64_t fromstartsoffset,
    const int32_t* fromstops,
    int64_t fromstopsoffset) {
    return awkward_ListArray32_getitem_jagged_descend_64(
      tooffsets,
      slicestarts,
      slicestartsoffset,
      slicestops,
      slicestopsoffset,
      sliceouterlen,
      fromstarts,
      fromstartsoffset,
      fromstops,
      fromstopsoffset);
  }
  template <>
  Error ListArray_getitem_jagged_descend_64<uint32_t>(
    int64_t* tooffsets,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t sliceouterlen,
    const uint32_t* fromstarts,
    int64_t fromstartsoffset,
    const uint32_t* fromstops,
    int64_t fromstopsoffset) {
    return awkward_ListArrayU32_getitem_jagged_descend_64(
      tooffsets,
      slicestarts,
      slicestartsoffset,
      slicestops,
      slicestopsoffset,
      sliceouterlen,
      fromstarts,
      fromstartsoffset,
      fromstops,
      fromstopsoffset);
  }
  template <>
  Error ListArray_getitem_jagged_descend_64<int64_t>(
    int64_t* tooffsets,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t sliceouterlen,
    const int64_t* fromstarts,
    int64_t fromstartsoffset,
    const int64_t* fromstops,
    int64_t fromstopsoffset) {
    return awkward_ListArray64_getitem_jagged_descend_64(
      tooffsets,
      slicestarts,
      slicestartsoffset,
      slicestops,
      slicestopsoffset,
      sliceouterlen,
      fromstarts,
      fromstartsoffset,
      fromstops,
      fromstopsoffset);
  }

  template <>
  int8_t index_getitem_at_nowrap<int8_t>(
    const int8_t* ptr,
    int64_t offset,
    int64_t at) {
    return awkward_Index8_getitem_at_nowrap(
      ptr,
      offset,
      at);
  }
  template <>
  uint8_t index_getitem_at_nowrap<uint8_t>(
    const uint8_t* ptr,
    int64_t offset,
    int64_t at) {
    return awkward_IndexU8_getitem_at_nowrap(
      ptr,
      offset,
      at);
  }
  template <>
  int32_t index_getitem_at_nowrap<int32_t>(
    const int32_t* ptr,
    int64_t offset,
    int64_t at) {
    return awkward_Index32_getitem_at_nowrap(
      ptr,
      offset,
      at);
  }
  template <>
  uint32_t index_getitem_at_nowrap<uint32_t>(
    const uint32_t* ptr,
    int64_t offset,
    int64_t at) {
    return awkward_IndexU32_getitem_at_nowrap(
      ptr,
      offset,
      at);
  }
  template <>
  int64_t index_getitem_at_nowrap<int64_t>(
    const int64_t* ptr,
    int64_t offset,
    int64_t at) {
    return awkward_Index64_getitem_at_nowrap(
      ptr,
      offset,
      at);
  }

  template <>
  void  index_setitem_at_nowrap<int8_t>(
    int8_t* ptr,
    int64_t offset,
    int64_t at,
    int8_t value) {
    awkward_Index8_setitem_at_nowrap(
      ptr,
      offset,
      at,
      value);
  }
  template <>
  void  index_setitem_at_nowrap<uint8_t>(
    uint8_t* ptr,
    int64_t offset,
    int64_t at,
    uint8_t value) {
    awkward_IndexU8_setitem_at_nowrap(
      ptr,
      offset,
      at,
      value);
  }
  template <>
  void  index_setitem_at_nowrap<int32_t>(
    int32_t* ptr,
    int64_t offset,
    int64_t at,
    int32_t value) {
    awkward_Index32_setitem_at_nowrap(
      ptr,
      offset,
      at,
      value);
  }
  template <>
  void  index_setitem_at_nowrap<uint32_t>(
    uint32_t* ptr,
    int64_t offset,
    int64_t at,
    uint32_t value) {
    awkward_IndexU32_setitem_at_nowrap(
      ptr,
      offset,
      at,
      value);
  }
  template <>
  void  index_setitem_at_nowrap<int64_t>(
    int64_t* ptr,
    int64_t offset,
    int64_t at,
    int64_t value) {
    awkward_Index64_setitem_at_nowrap(
      ptr,
      offset,
      at,
      value);
  }

  ERROR ByteMaskedArray_getitem_carry_64(
    int8_t* tomask,
    const int8_t* frommask,
    int64_t frommaskoffset,
    int64_t lenmask,
    const int64_t* fromcarry,
    int64_t lencarry) {
    return awkward_ByteMaskedArray_getitem_carry_64(
      tomask,
      frommask,
      frommaskoffset,
      lenmask,
      fromcarry,
      lencarry);
  }

  ERROR ByteMaskedArray_numnull(
    int64_t* numnull,
    const int8_t* mask,
    int64_t maskoffset,
    int64_t length,
    bool validwhen) {
    return awkward_ByteMaskedArray_numnull(
      numnull,
      mask,
      maskoffset,
      length,
      validwhen);
  }

  ERROR ByteMaskedArray_getitem_nextcarry_64(
    int64_t* tocarry,
    const int8_t* mask,
    int64_t maskoffset,
    int64_t length,
    bool validwhen) {
    return awkward_ByteMaskedArray_getitem_nextcarry_64(
      tocarry,
      mask,
      maskoffset,
      length,
      validwhen);
  }

  ERROR ByteMaskedArray_getitem_nextcarry_outindex_64(
    int64_t* tocarry,
    int64_t* toindex,
    const int8_t* mask,
    int64_t maskoffset,
    int64_t length,
    bool validwhen) {
    return awkward_ByteMaskedArray_getitem_nextcarry_outindex_64(
      tocarry,
      toindex,
      mask,
      maskoffset,
      length,
      validwhen);
  }

  ERROR ByteMaskedArray_toIndexedOptionArray64(
    int64_t* toindex,
    const int8_t* mask,
    int64_t maskoffset,
    int64_t length,
    bool validwhen) {
    return awkward_ByteMaskedArray_toIndexedOptionArray64(
      toindex,
      mask,
      maskoffset,
      length,
      validwhen);
  }

  /////////////////////////////////// awkward/cpu-kernels/identities.h

  template <>
  ERROR new_Identities(
    int32_t* toptr,
    int64_t length) {
    return awkward_new_Identities32(
      toptr,
      length);
  }
  template <>
  ERROR new_Identities(
    int64_t* toptr,
    int64_t length) {
    return awkward_new_Identities64(
      toptr,
      length);
  }

  template <>
  ERROR Identities_to_Identities64(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t length,
    int64_t width) {
    return awkward_Identities32_to_Identities64(
      toptr,
      fromptr,
      length,
      width);
  }

  template <>
  Error Identities_from_ListOffsetArray<int32_t, int32_t>(
    int32_t* toptr,
    const int32_t* fromptr,
    const int32_t* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities32_from_ListOffsetArray32(
      toptr,
      fromptr,
      fromoffsets,
      fromptroffset,
      offsetsoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_ListOffsetArray<int32_t, uint32_t>(
    int32_t* toptr,
    const int32_t* fromptr,
    const uint32_t* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities32_from_ListOffsetArrayU32(
      toptr,
      fromptr,
      fromoffsets,
      fromptroffset,
      offsetsoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_ListOffsetArray<int32_t, int64_t>(
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities32_from_ListOffsetArray64(
      toptr,
      fromptr,
      fromoffsets,
      fromptroffset,
      offsetsoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_ListOffsetArray<int64_t, int32_t>(
    int64_t* toptr,
    const int64_t* fromptr,
    const int32_t* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities64_from_ListOffsetArray32(
      toptr,
      fromptr,
      fromoffsets,
      fromptroffset,
      offsetsoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_ListOffsetArray<int64_t, uint32_t>(
    int64_t* toptr,
    const int64_t* fromptr,
    const uint32_t* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities64_from_ListOffsetArrayU32(
      toptr,
      fromptr,
      fromoffsets,
      fromptroffset,
      offsetsoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_ListOffsetArray<int64_t, int64_t>(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities64_from_ListOffsetArray64(
      toptr,
      fromptr,
      fromoffsets,
      fromptroffset,
      offsetsoffset,
      tolength,
      fromlength,
      fromwidth);
  }

  template <>
  Error Identities_from_ListArray<int32_t, int32_t>(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t fromptroffset,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities32_from_ListArray32(
      uniquecontents,
      toptr,
      fromptr,
      fromstarts,
      fromstops,
      fromptroffset,
      startsoffset,
      stopsoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_ListArray<int32_t, uint32_t>(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t fromptroffset,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities32_from_ListArrayU32(
      uniquecontents,
      toptr,
      fromptr,
      fromstarts,
      fromstops,
      fromptroffset,
      startsoffset,
      stopsoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_ListArray<int32_t, int64_t>(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t fromptroffset,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities32_from_ListArray64(
      uniquecontents,
      toptr,
      fromptr,
      fromstarts,
      fromstops,
      fromptroffset,
      startsoffset,
      stopsoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_ListArray<int64_t, int32_t>(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t fromptroffset,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities64_from_ListArray32(
      uniquecontents,
      toptr,
      fromptr,
      fromstarts,
      fromstops,
      fromptroffset,
      startsoffset,
      stopsoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_ListArray<int64_t, uint32_t>(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t fromptroffset,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities64_from_ListArrayU32(
      uniquecontents,
      toptr,
      fromptr,
      fromstarts,
      fromstops,
      fromptroffset,
      startsoffset,
      stopsoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_ListArray<int64_t, int64_t>(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t fromptroffset,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities64_from_ListArray64(
      uniquecontents,
      toptr,
      fromptr,
      fromstarts,
      fromstops,
      fromptroffset,
      startsoffset,
      stopsoffset,
      tolength,
      fromlength,
      fromwidth);
  }

  template <>
  ERROR Identities_from_RegularArray(
    int32_t* toptr,
    const int32_t* fromptr,
    int64_t fromptroffset,
    int64_t size,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities32_from_RegularArray(
      toptr,
      fromptr,
      fromptroffset,
      size,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  ERROR Identities_from_RegularArray(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t fromptroffset,
    int64_t size,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities64_from_RegularArray(
      toptr,
      fromptr,
      fromptroffset,
      size,
      tolength,
      fromlength,
      fromwidth);
  }

  template <>
  Error Identities_from_IndexedArray<int32_t, int32_t>(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int32_t* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities32_from_IndexedArray32(
      uniquecontents,
      toptr,
      fromptr,
      fromindex,
      fromptroffset,
      indexoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_IndexedArray<int32_t, uint32_t>(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const uint32_t* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities32_from_IndexedArrayU32(
      uniquecontents,
      toptr,
      fromptr,
      fromindex,
      fromptroffset,
      indexoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_IndexedArray<int32_t, int64_t>(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities32_from_IndexedArray64(
      uniquecontents,
      toptr,
      fromptr,
      fromindex,
      fromptroffset,
      indexoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_IndexedArray<int64_t, int32_t>(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int32_t* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities64_from_IndexedArray32(
      uniquecontents,
      toptr,
      fromptr,
      fromindex,
      fromptroffset,
      indexoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_IndexedArray<int64_t, uint32_t>(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const uint32_t* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities64_from_IndexedArrayU32(
      uniquecontents,
      toptr,
      fromptr,
      fromindex,
      fromptroffset,
      indexoffset,
      tolength,
      fromlength,
      fromwidth);
  }
  template <>
  Error Identities_from_IndexedArray<int64_t, int64_t>(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth) {
    return awkward_Identities64_from_IndexedArray64(
      uniquecontents,
      toptr,
      fromptr,
      fromindex,
      fromptroffset,
      indexoffset,
      tolength,
      fromlength,
      fromwidth);
  }

  template <>
  Error Identities_from_UnionArray<int32_t, int8_t, int32_t>(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int8_t* fromtags,
    const int32_t* fromindex,
    int64_t fromptroffset,
    int64_t tagsoffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which) {
    return awkward_Identities32_from_UnionArray8_32(
      uniquecontents,
      toptr,
      fromptr,
      fromtags,
      fromindex,
      fromptroffset,
      tagsoffset,
      indexoffset,
      tolength,
      fromlength,
      fromwidth,
      which);
  }
  template <>
  Error Identities_from_UnionArray<int32_t, int8_t, uint32_t>(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int8_t* fromtags,
    const uint32_t* fromindex,
    int64_t fromptroffset,
    int64_t tagsoffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which) {
    return awkward_Identities32_from_UnionArray8_U32(
      uniquecontents,
      toptr,
      fromptr,
      fromtags,
      fromindex,
      fromptroffset,
      tagsoffset,
      indexoffset,
      tolength,
      fromlength,
      fromwidth,
      which);
  }
  template <>
  Error Identities_from_UnionArray<int32_t, int8_t, int64_t>(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int8_t* fromtags,
    const int64_t* fromindex,
    int64_t fromptroffset,
    int64_t tagsoffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which) {
    return awkward_Identities32_from_UnionArray8_64(
      uniquecontents,
      toptr,
      fromptr,
      fromtags,
      fromindex,
      fromptroffset,
      tagsoffset,
      indexoffset,
      tolength,
      fromlength,
      fromwidth,
      which);
  }
  template <>
  Error Identities_from_UnionArray<int64_t, int8_t, int32_t>(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int8_t* fromtags,
    const int32_t* fromindex,
    int64_t fromptroffset,
    int64_t tagsoffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which) {
    return awkward_Identities64_from_UnionArray8_32(
      uniquecontents,
      toptr,
      fromptr,
      fromtags,
      fromindex,
      fromptroffset,
      tagsoffset,
      indexoffset,
      tolength,
      fromlength,
      fromwidth,
      which);
  }
  template <>
  Error Identities_from_UnionArray<int64_t, int8_t, uint32_t>(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int8_t* fromtags,
    const uint32_t* fromindex,
    int64_t fromptroffset,
    int64_t tagsoffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which) {
    return awkward_Identities64_from_UnionArray8_U32(
      uniquecontents,
      toptr,
      fromptr,
      fromtags,
      fromindex,
      fromptroffset,
      tagsoffset,
      indexoffset,
      tolength,
      fromlength,
      fromwidth,
      which);
  }
  template <>
  Error Identities_from_UnionArray<int64_t, int8_t, int64_t>(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int8_t* fromtags,
    const int64_t* fromindex,
    int64_t fromptroffset,
    int64_t tagsoffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which) {
    return awkward_Identities64_from_UnionArray8_64(
      uniquecontents,
      toptr,
      fromptr,
      fromtags,
      fromindex,
      fromptroffset,
      tagsoffset,
      indexoffset,
      tolength,
      fromlength,
      fromwidth,
      which);
  }

  template <>
  ERROR Identities_extend(
    int32_t* toptr,
    const int32_t* fromptr,
    int64_t fromoffset,
    int64_t fromlength,
    int64_t tolength) {
    return awkward_Identities32_extend(
      toptr,
      fromptr,
      fromoffset,
      fromlength,
      tolength);
  }
  template <>
  ERROR Identities_extend(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t fromoffset,
    int64_t fromlength,
    int64_t tolength) {
    return awkward_Identities64_extend(
      toptr,
      fromptr,
      fromoffset,
      fromlength,
      tolength);
  }

  /////////////////////////////////// awkward/cpu-kernels/operations.h

  template <>
  Error ListArray_num_64<int32_t>(
    int64_t* tonum,
    const int32_t* fromstarts,
    int64_t startsoffset,
    const int32_t* fromstops,
    int64_t stopsoffset,
    int64_t length) {
    return awkward_ListArray32_num_64(
      tonum,
      fromstarts,
      startsoffset,
      fromstops,
      stopsoffset,
      length);
  }
  template <>
  Error ListArray_num_64<uint32_t>(
    int64_t* tonum,
    const uint32_t* fromstarts,
    int64_t startsoffset,
    const uint32_t* fromstops,
    int64_t stopsoffset,
    int64_t length) {
    return awkward_ListArrayU32_num_64(
      tonum,
      fromstarts,
      startsoffset,
      fromstops,
      stopsoffset,
      length);
  }
  template <>
  Error ListArray_num_64<int64_t>(
    int64_t* tonum,
    const int64_t* fromstarts,
    int64_t startsoffset,
    const int64_t* fromstops,
    int64_t stopsoffset,
    int64_t length) {
    return awkward_ListArray64_num_64(
      tonum,
      fromstarts,
      startsoffset,
      fromstops,
      stopsoffset,
      length);
  }

  ERROR RegularArray_num_64(
    int64_t* tonum,
    int64_t size,
    int64_t length) {
    return awkward_RegularArray_num_64(
      tonum,
      size,
      length);
  }

  template <>
  Error ListOffsetArray_flatten_offsets_64<int32_t>(
    int64_t* tooffsets,
    const int32_t* outeroffsets,
    int64_t outeroffsetsoffset,
    int64_t outeroffsetslen,
    const int64_t* inneroffsets,
    int64_t inneroffsetsoffset,
    int64_t inneroffsetslen) {
    return awkward_ListOffsetArray32_flatten_offsets_64(
      tooffsets,
      outeroffsets,
      outeroffsetsoffset,
      outeroffsetslen,
      inneroffsets,
      inneroffsetsoffset,
      inneroffsetslen);
  }
  template <>
  Error ListOffsetArray_flatten_offsets_64<uint32_t>(
    int64_t* tooffsets,
    const uint32_t* outeroffsets,
    int64_t outeroffsetsoffset,
    int64_t outeroffsetslen,
    const int64_t* inneroffsets,
    int64_t inneroffsetsoffset,
    int64_t inneroffsetslen) {
    return awkward_ListOffsetArrayU32_flatten_offsets_64(
      tooffsets,
      outeroffsets,
      outeroffsetsoffset,
      outeroffsetslen,
      inneroffsets,
      inneroffsetsoffset,
      inneroffsetslen);
  }
  template <>
  Error ListOffsetArray_flatten_offsets_64<int64_t>(
    int64_t* tooffsets,
    const int64_t* outeroffsets,
    int64_t outeroffsetsoffset,
    int64_t outeroffsetslen,
    const int64_t* inneroffsets,
    int64_t inneroffsetsoffset,
    int64_t inneroffsetslen) {
    return awkward_ListOffsetArray64_flatten_offsets_64(
      tooffsets,
      outeroffsets,
      outeroffsetsoffset,
      outeroffsetslen,
      inneroffsets,
      inneroffsetsoffset,
      inneroffsetslen);
  }

  template <>
  Error IndexedArray_flatten_none2empty_64<int32_t>(
    int64_t* outoffsets,
    const int32_t* outindex,
    int64_t outindexoffset,
    int64_t outindexlength,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t offsetslength) {
    return awkward_IndexedArray32_flatten_none2empty_64(
      outoffsets,
      outindex,
      outindexoffset,
      outindexlength,
      offsets,
      offsetsoffset,
      offsetslength);
  }
  template <>
  Error IndexedArray_flatten_none2empty_64<uint32_t>(
    int64_t* outoffsets,
    const uint32_t* outindex,
    int64_t outindexoffset,
    int64_t outindexlength,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t offsetslength) {
    return awkward_IndexedArrayU32_flatten_none2empty_64(
      outoffsets,
      outindex,
      outindexoffset,
      outindexlength,
      offsets,
      offsetsoffset,
      offsetslength);
  }
  template <>
  Error IndexedArray_flatten_none2empty_64<int64_t>(
    int64_t* outoffsets,
    const int64_t* outindex,
    int64_t outindexoffset,
    int64_t outindexlength,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t offsetslength) {
    return awkward_IndexedArray64_flatten_none2empty_64(
      outoffsets,
      outindex,
      outindexoffset,
      outindexlength,
      offsets,
      offsetsoffset,
      offsetslength);
  }

  template <>
  Error UnionArray_flatten_length_64<int8_t,
    int32_t>(
    int64_t* total_length,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    const int32_t* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t** offsetsraws,
    int64_t* offsetsoffsets) {
    return awkward_UnionArray32_flatten_length_64(
      total_length,
      fromtags,
      fromtagsoffset,
      fromindex,
      fromindexoffset,
      length,
      offsetsraws,
      offsetsoffsets);
  }
  template <>
  Error UnionArray_flatten_length_64<int8_t,
    uint32_t>(
    int64_t* total_length,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    const uint32_t* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t** offsetsraws,
    int64_t* offsetsoffsets) {
    return awkward_UnionArrayU32_flatten_length_64(
      total_length,
      fromtags,
      fromtagsoffset,
      fromindex,
      fromindexoffset,
      length,
      offsetsraws,
      offsetsoffsets);
  }
  template <>
  Error UnionArray_flatten_length_64<int8_t,
    int64_t>(
    int64_t* total_length,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    const int64_t* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t** offsetsraws,
    int64_t* offsetsoffsets) {
    return awkward_UnionArray64_flatten_length_64(
      total_length,
      fromtags,
      fromtagsoffset,
      fromindex,
      fromindexoffset,
      length,
      offsetsraws,
      offsetsoffsets);
  }

  template <>
  Error UnionArray_flatten_combine_64<int8_t,
    int32_t>(
    int8_t* totags,
    int64_t* toindex,
    int64_t* tooffsets,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    const int32_t* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t** offsetsraws,
    int64_t* offsetsoffsets) {
    return awkward_UnionArray32_flatten_combine_64(
      totags,
      toindex,
      tooffsets,
      fromtags,
      fromtagsoffset,
      fromindex,
      fromindexoffset,
      length,
      offsetsraws,
      offsetsoffsets);
  }
  template <>
  Error UnionArray_flatten_combine_64<int8_t,
    uint32_t>(
    int8_t* totags,
    int64_t* toindex,
    int64_t* tooffsets,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    const uint32_t* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t** offsetsraws,
    int64_t* offsetsoffsets) {
    return awkward_UnionArrayU32_flatten_combine_64(
      totags,
      toindex,
      tooffsets,
      fromtags,
      fromtagsoffset,
      fromindex,
      fromindexoffset,
      length,
      offsetsraws,
      offsetsoffsets);
  }
  template <>
  Error UnionArray_flatten_combine_64<int8_t,
    int64_t>(
    int8_t* totags,
    int64_t* toindex,
    int64_t* tooffsets,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    const int64_t* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t** offsetsraws,
    int64_t* offsetsoffsets) {
    return awkward_UnionArray64_flatten_combine_64(
      totags,
      toindex,
      tooffsets,
      fromtags,
      fromtagsoffset,
      fromindex,
      fromindexoffset,
      length,
      offsetsraws,
      offsetsoffsets);
  }

  template <>
  Error IndexedArray_flatten_nextcarry_64<int32_t>(
    int64_t* tocarry,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent) {
    return awkward_IndexedArray32_flatten_nextcarry_64(
      tocarry,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  Error IndexedArray_flatten_nextcarry_64<uint32_t>(
    int64_t* tocarry,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent) {
    return awkward_IndexedArrayU32_flatten_nextcarry_64(
      tocarry,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  Error IndexedArray_flatten_nextcarry_64<int64_t>(
    int64_t* tocarry,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent) {
    return awkward_IndexedArray64_flatten_nextcarry_64(
      tocarry,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }

  template <>
  Error IndexedArray_overlay_mask8_to64<int32_t>(
    int64_t* toindex,
    const int8_t* mask,
    int64_t maskoffset,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t length) {
    return awkward_IndexedArray32_overlay_mask8_to64(
      toindex,
      mask,
      maskoffset,
      fromindex,
      indexoffset,
      length);
  }
  template <>
  Error IndexedArray_overlay_mask8_to64<uint32_t>(
    int64_t* toindex,
    const int8_t* mask,
    int64_t maskoffset,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t length) {
    return awkward_IndexedArrayU32_overlay_mask8_to64(
      toindex,
      mask,
      maskoffset,
      fromindex,
      indexoffset,
      length);
  }
  template <>
  Error IndexedArray_overlay_mask8_to64<int64_t>(
    int64_t* toindex,
    const int8_t* mask,
    int64_t maskoffset,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t length) {
    return awkward_IndexedArray64_overlay_mask8_to64(
      toindex,
      mask,
      maskoffset,
      fromindex,
      indexoffset,
      length);
  }

  template <>
  Error IndexedArray_mask8<int32_t>(
    int8_t* tomask,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t length) {
    return awkward_IndexedArray32_mask8(
      tomask,
      fromindex,
      indexoffset,
      length);
  }
  template <>
  Error IndexedArray_mask8<uint32_t>(
    int8_t* tomask,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t length) {
    return awkward_IndexedArrayU32_mask8(
      tomask,
      fromindex,
      indexoffset,
      length);
  }
  template <>
  Error IndexedArray_mask8<int64_t>(
    int8_t* tomask,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t length) {
    return awkward_IndexedArray64_mask8(
      tomask,
      fromindex,
      indexoffset,
      length);
  }

  ERROR ByteMaskedArray_mask8(
    int8_t* tomask,
    const int8_t* frommask,
    int64_t maskoffset,
    int64_t length,
    bool validwhen) {
    return awkward_ByteMaskedArray_mask8(
      tomask,
      frommask,
      maskoffset,
      length,
      validwhen);
  }

  ERROR zero_mask8(
    int8_t* tomask,
    int64_t length) {
    return awkward_zero_mask8(tomask, length);
  }

  template <>
  Error IndexedArray_simplify32_to64<int32_t>(
    int64_t* toindex,
    const int32_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength) {
    return awkward_IndexedArray32_simplify32_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }
  template <>
  Error IndexedArray_simplify32_to64<uint32_t>(
    int64_t* toindex,
    const uint32_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength) {
    return awkward_IndexedArrayU32_simplify32_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }
  template <>
  Error IndexedArray_simplify32_to64<int64_t>(
    int64_t* toindex,
    const int64_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength) {
    return awkward_IndexedArray64_simplify32_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }

  template <>
  Error IndexedArray_simplifyU32_to64<int32_t>(
    int64_t* toindex,
    const int32_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const uint32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength) {
    return awkward_IndexedArray32_simplifyU32_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }
  template <>
  Error IndexedArray_simplifyU32_to64<uint32_t>(
    int64_t* toindex,
    const uint32_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const uint32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength) {
    return awkward_IndexedArrayU32_simplifyU32_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }
  template <>
  Error IndexedArray_simplifyU32_to64<int64_t>(
    int64_t* toindex,
    const int64_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const uint32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength) {
    return awkward_IndexedArray64_simplifyU32_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }

  template <>
  Error IndexedArray_simplify64_to64<int32_t>(
    int64_t* toindex,
    const int32_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int64_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength) {
    return awkward_IndexedArray32_simplify64_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }
  template <>
  Error IndexedArray_simplify64_to64<uint32_t>(
    int64_t* toindex,
    const uint32_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int64_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength) {
    return awkward_IndexedArrayU32_simplify64_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }
  template <>
  Error IndexedArray_simplify64_to64<int64_t>(
    int64_t* toindex,
    const int64_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int64_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength) {
    return awkward_IndexedArray64_simplify64_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }

  template <>
  Error ListArray_compact_offsets_64(
    int64_t* tooffsets,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t length) {
    return awkward_ListArray32_compact_offsets_64(
      tooffsets,
      fromstarts,
      fromstops,
      startsoffset,
      stopsoffset,
      length);
  }
  template <>
  Error ListArray_compact_offsets_64(
    int64_t* tooffsets,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t length) {
    return awkward_ListArrayU32_compact_offsets_64(
      tooffsets,
      fromstarts,
      fromstops,
      startsoffset,
      stopsoffset,
      length);
  }
  template <>
  Error ListArray_compact_offsets_64(
    int64_t* tooffsets,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t length) {
    return awkward_ListArray64_compact_offsets_64(
      tooffsets,
      fromstarts,
      fromstops,
      startsoffset,
      stopsoffset,
      length);
  }

  ERROR RegularArray_compact_offsets_64(
    int64_t* tooffsets,
    int64_t length,
    int64_t size) {
    return awkward_RegularArray_compact_offsets64(
      tooffsets,
      length,
      size);
  }

  template <>
  Error ListOffsetArray_compact_offsets_64(
    int64_t* tooffsets,
    const int32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length) {
    return awkward_ListOffsetArray32_compact_offsets_64(
      tooffsets,
      fromoffsets,
      offsetsoffset,
      length);
  }
  template <>
  Error ListOffsetArray_compact_offsets_64(
    int64_t* tooffsets,
    const uint32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length) {
    return awkward_ListOffsetArrayU32_compact_offsets_64(
      tooffsets,
      fromoffsets,
      offsetsoffset,
      length);
  }
  template <>
  Error ListOffsetArray_compact_offsets_64(
    int64_t* tooffsets,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length) {
    return awkward_ListOffsetArray64_compact_offsets_64(
      tooffsets,
      fromoffsets,
      offsetsoffset,
      length);
  }

  template <>
  Error ListArray_broadcast_tooffsets_64<int32_t>(
    int64_t* tocarry,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    const int32_t* fromstarts,
    int64_t startsoffset,
    const int32_t* fromstops,
    int64_t stopsoffset,
    int64_t lencontent) {
    return awkward_ListArray32_broadcast_tooffsets_64(
      tocarry,
      fromoffsets,
      offsetsoffset,
      offsetslength,
      fromstarts,
      startsoffset,
      fromstops,
      stopsoffset,
      lencontent);
  }
  template <>
  Error ListArray_broadcast_tooffsets_64<uint32_t>(
    int64_t* tocarry,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    const uint32_t* fromstarts,
    int64_t startsoffset,
    const uint32_t* fromstops,
    int64_t stopsoffset,
    int64_t lencontent) {
    return awkward_ListArrayU32_broadcast_tooffsets_64(
      tocarry,
      fromoffsets,
      offsetsoffset,
      offsetslength,
      fromstarts,
      startsoffset,
      fromstops,
      stopsoffset,
      lencontent);
  }
  template <>
  Error ListArray_broadcast_tooffsets_64<int64_t>(
    int64_t* tocarry,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    const int64_t* fromstarts,
    int64_t startsoffset,
    const int64_t* fromstops,
    int64_t stopsoffset,
    int64_t lencontent) {
    return awkward_ListArray64_broadcast_tooffsets_64(
      tocarry,
      fromoffsets,
      offsetsoffset,
      offsetslength,
      fromstarts,
      startsoffset,
      fromstops,
      stopsoffset,
      lencontent);
  }

  ERROR RegularArray_broadcast_tooffsets_64(
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    int64_t size) {
    return awkward_RegularArray_broadcast_tooffsets_64(
      fromoffsets,
      offsetsoffset,
      offsetslength,
      size);
  }

  ERROR RegularArray_broadcast_tooffsets_size1_64(
    int64_t* tocarry,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength) {
    return awkward_RegularArray_broadcast_tooffsets_size1_64(
      tocarry,
      fromoffsets,
      offsetsoffset,
      offsetslength);
  }

  template <>
  Error ListOffsetArray_toRegularArray<int32_t>(
    int64_t* size,
    const int32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength) {
    return awkward_ListOffsetArray32_toRegularArray(
      size,
      fromoffsets,
      offsetsoffset,
      offsetslength);
  }
  template <>
  Error ListOffsetArray_toRegularArray<uint32_t>(
    int64_t* size,
    const uint32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength) {
    return awkward_ListOffsetArrayU32_toRegularArray(
      size,
      fromoffsets,
      offsetsoffset,
      offsetslength);
  }
  template <>
  Error ListOffsetArray_toRegularArray(
    int64_t* size,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength) {
    return awkward_ListOffsetArray64_toRegularArray(
      size,
      fromoffsets,
      offsetsoffset,
      offsetslength);
  }

  template <>
  ERROR NumpyArray_fill(
    double* toptr,
    int64_t tooffset,
    const double* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_todouble_fromdouble(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    double* toptr,
    int64_t tooffset,
    const float* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_todouble_fromfloat(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    double* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_todouble_from64(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    double* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_todouble_fromU64(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    double* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_todouble_from32(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    double* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_todouble_fromU32(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    double* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_todouble_from16(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    double* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_todouble_fromU16(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    double* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_todouble_from8(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    double* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_todouble_fromU8(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill_frombool(
    double* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_todouble_frombool(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    uint64_t* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_toU64_fromU64(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    int64_t* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_to64_from64(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  ERROR NumpyArray_fill_to64_fromU64(
    int64_t* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_to64_fromU64(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    int64_t* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_to64_from32(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    int64_t* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_to64_fromU32(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    int64_t* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_to64_from16(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    int64_t* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_to64_fromU16(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    int64_t* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_to64_from8(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    int64_t* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_to64_fromU8(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill_frombool(
    int64_t* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_to64_frombool(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill_frombool(
    bool* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_tobool_frombool(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }
  template <>
  ERROR NumpyArray_fill(
    int8_t* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t fromoffset,
    int64_t length) {
    return awkward_NumpyArray_fill_tobyte_frombyte(
      toptr,
      tooffset,
      fromptr,
      fromoffset,
      length);
  }

  template <>
  ERROR ListArray_fill(
    int64_t* tostarts,
    int64_t tostartsoffset,
    int64_t* tostops,
    int64_t tostopsoffset,
    const int32_t* fromstarts,
    int64_t fromstartsoffset,
    const int32_t* fromstops,
    int64_t fromstopsoffset,
    int64_t length,
    int64_t base) {
    return awkward_ListArray_fill_to64_from32(
      tostarts,
      tostartsoffset,
      tostops,
      tostopsoffset,
      fromstarts,
      fromstartsoffset,
      fromstops,
      fromstopsoffset,
      length,
      base);
  }
  template <>
  ERROR ListArray_fill(
    int64_t* tostarts,
    int64_t tostartsoffset,
    int64_t* tostops,
    int64_t tostopsoffset,
    const uint32_t* fromstarts,
    int64_t fromstartsoffset,
    const uint32_t* fromstops,
    int64_t fromstopsoffset,
    int64_t length,
    int64_t base) {
    return awkward_ListArray_fill_to64_fromU32(
      tostarts,
      tostartsoffset,
      tostops,
      tostopsoffset,
      fromstarts,
      fromstartsoffset,
      fromstops,
      fromstopsoffset,
      length,
      base);
  }
  template <>
  ERROR ListArray_fill(
    int64_t* tostarts,
    int64_t tostartsoffset,
    int64_t* tostops,
    int64_t tostopsoffset,
    const int64_t* fromstarts,
    int64_t fromstartsoffset,
    const int64_t* fromstops,
    int64_t fromstopsoffset,
    int64_t length,
    int64_t base) {
    return awkward_ListArray_fill_to64_from64(
      tostarts,
      tostartsoffset,
      tostops,
      tostopsoffset,
      fromstarts,
      fromstartsoffset,
      fromstops,
      fromstopsoffset,
      length,
      base);
  }

  template <>
  ERROR IndexedArray_fill(
    int64_t* toindex,
    int64_t toindexoffset,
    const int32_t* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t base) {
    return awkward_IndexedArray_fill_to64_from32(
      toindex,
      toindexoffset,
      fromindex,
      fromindexoffset,
      length,
      base);
  }
  template <>
  ERROR IndexedArray_fill(
    int64_t* toindex,
    int64_t toindexoffset,
    const uint32_t* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t base) {
    return awkward_IndexedArray_fill_to64_fromU32(
      toindex,
      toindexoffset,
      fromindex,
      fromindexoffset,
      length,
      base);
  }
  template <>
  ERROR IndexedArray_fill(
    int64_t* toindex,
    int64_t toindexoffset,
    const int64_t* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t base) {
    return awkward_IndexedArray_fill_to64_from64(
      toindex,
      toindexoffset,
      fromindex,
      fromindexoffset,
      length,
      base);
  }

  ERROR IndexedArray_fill_to64_count(
    int64_t* toindex,
    int64_t toindexoffset,
    int64_t length,
    int64_t base) {
    return awkward_IndexedArray_fill_to64_count(
      toindex,
      toindexoffset,
      length,
      base);
  }

  ERROR UnionArray_filltags_to8_from8(
    int8_t* totags,
    int64_t totagsoffset,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray_filltags_to8_from8(
      totags,
      totagsoffset,
      fromtags,
      fromtagsoffset,
      length,
      base);
  }

  template <>
  ERROR UnionArray_fillindex(
    int64_t* toindex,
    int64_t toindexoffset,
    const int32_t* fromindex,
    int64_t fromindexoffset,
    int64_t length) {
    return awkward_UnionArray_fillindex_to64_from32(
      toindex,
      toindexoffset,
      fromindex,
      fromindexoffset,
      length);
  }
  template <>
  ERROR UnionArray_fillindex(
    int64_t* toindex,
    int64_t toindexoffset,
    const uint32_t* fromindex,
    int64_t fromindexoffset,
    int64_t length) {
    return awkward_UnionArray_fillindex_to64_fromU32(
      toindex,
      toindexoffset,
      fromindex,
      fromindexoffset,
      length);
  }
  template <>
  ERROR UnionArray_fillindex(
    int64_t* toindex,
    int64_t toindexoffset,
    const int64_t* fromindex,
    int64_t fromindexoffset,
    int64_t length) {
    return awkward_UnionArray_fillindex_to64_from64(
      toindex,
      toindexoffset,
      fromindex,
      fromindexoffset,
      length);
  }

  ERROR UnionArray_filltags_to8_const(
    int8_t* totags,
    int64_t totagsoffset,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray_filltags_to8_const(
      totags,
      totagsoffset,
      length,
      base);
  }

  ERROR UnionArray_fillindex_count_64(
    int64_t* toindex,
    int64_t toindexoffset,
    int64_t length) {
    return awkward_UnionArray_fillindex_to64_count(
      toindex,
      toindexoffset,
      length);
  }

  template <>
  Error UnionArray_simplify8_32_to8_64<int8_t,
    int32_t>(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* outertags,
    int64_t outertagsoffset,
    const int32_t* outerindex,
    int64_t outerindexoffset,
    const int8_t* innertags,
    int64_t innertagsoffset,
    const int32_t* innerindex,
    int64_t innerindexoffset,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray8_32_simplify8_32_to8_64(
      totags,
      toindex,
      outertags,
      outertagsoffset,
      outerindex,
      outerindexoffset,
      innertags,
      innertagsoffset,
      innerindex,
      innerindexoffset,
      towhich,
      innerwhich,
      outerwhich,
      length,
      base);
  }
  template <>
  Error UnionArray_simplify8_32_to8_64<int8_t,
    uint32_t>(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* outertags,
    int64_t outertagsoffset,
    const uint32_t* outerindex,
    int64_t outerindexoffset,
    const int8_t* innertags,
    int64_t innertagsoffset,
    const int32_t* innerindex,
    int64_t innerindexoffset,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray8_U32_simplify8_32_to8_64(
      totags,
      toindex,
      outertags,
      outertagsoffset,
      outerindex,
      outerindexoffset,
      innertags,
      innertagsoffset,
      innerindex,
      innerindexoffset,
      towhich,
      innerwhich,
      outerwhich,
      length,
      base);
  }
  template <>
  Error UnionArray_simplify8_32_to8_64<int8_t,
    int64_t>(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* outertags,
    int64_t outertagsoffset,
    const int64_t* outerindex,
    int64_t outerindexoffset,
    const int8_t* innertags,
    int64_t innertagsoffset,
    const int32_t* innerindex,
    int64_t innerindexoffset,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray8_64_simplify8_32_to8_64(
      totags,
      toindex,
      outertags,
      outertagsoffset,
      outerindex,
      outerindexoffset,
      innertags,
      innertagsoffset,
      innerindex,
      innerindexoffset,
      towhich,
      innerwhich,
      outerwhich,
      length,
      base);
  }

  template <>
  Error UnionArray_simplify8_U32_to8_64<int8_t,
    int32_t>(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* outertags,
    int64_t outertagsoffset,
    const int32_t* outerindex,
    int64_t outerindexoffset,
    const int8_t* innertags,
    int64_t innertagsoffset,
    const uint32_t* innerindex,
    int64_t innerindexoffset,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray8_32_simplify8_U32_to8_64(
      totags,
      toindex,
      outertags,
      outertagsoffset,
      outerindex,
      outerindexoffset,
      innertags,
      innertagsoffset,
      innerindex,
      innerindexoffset,
      towhich,
      innerwhich,
      outerwhich,
      length,
      base);
  }
  template <>
  Error UnionArray_simplify8_U32_to8_64<int8_t,
    uint32_t>(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* outertags,
    int64_t outertagsoffset,
    const uint32_t* outerindex,
    int64_t outerindexoffset,
    const int8_t* innertags,
    int64_t innertagsoffset,
    const uint32_t* innerindex,
    int64_t innerindexoffset,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray8_U32_simplify8_U32_to8_64(
      totags,
      toindex,
      outertags,
      outertagsoffset,
      outerindex,
      outerindexoffset,
      innertags,
      innertagsoffset,
      innerindex,
      innerindexoffset,
      towhich,
      innerwhich,
      outerwhich,
      length,
      base);
  }
  template <>
  Error UnionArray_simplify8_U32_to8_64<int8_t,
    int64_t>(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* outertags,
    int64_t outertagsoffset,
    const int64_t* outerindex,
    int64_t outerindexoffset,
    const int8_t* innertags,
    int64_t innertagsoffset,
    const uint32_t* innerindex,
    int64_t innerindexoffset,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray8_64_simplify8_U32_to8_64(
      totags,
      toindex,
      outertags,
      outertagsoffset,
      outerindex,
      outerindexoffset,
      innertags,
      innertagsoffset,
      innerindex,
      innerindexoffset,
      towhich,
      innerwhich,
      outerwhich,
      length,
      base);
  }

  template <>
  Error UnionArray_simplify8_64_to8_64<int8_t,
    int32_t>(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* outertags,
    int64_t outertagsoffset,
    const int32_t* outerindex,
    int64_t outerindexoffset,
    const int8_t* innertags,
    int64_t innertagsoffset,
    const int64_t* innerindex,
    int64_t innerindexoffset,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray8_32_simplify8_64_to8_64(
      totags,
      toindex,
      outertags,
      outertagsoffset,
      outerindex,
      outerindexoffset,
      innertags,
      innertagsoffset,
      innerindex,
      innerindexoffset,
      towhich,
      innerwhich,
      outerwhich,
      length,
      base);
  }
  template <>
  Error UnionArray_simplify8_64_to8_64<int8_t,
    uint32_t>(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* outertags,
    int64_t outertagsoffset,
    const uint32_t* outerindex,
    int64_t outerindexoffset,
    const int8_t* innertags,
    int64_t innertagsoffset,
    const int64_t* innerindex,
    int64_t innerindexoffset,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray8_U32_simplify8_64_to8_64(
      totags,
      toindex,
      outertags,
      outertagsoffset,
      outerindex,
      outerindexoffset,
      innertags,
      innertagsoffset,
      innerindex,
      innerindexoffset,
      towhich,
      innerwhich,
      outerwhich,
      length,
      base);
  }
  template <>
  Error UnionArray_simplify8_64_to8_64<int8_t,
    int64_t>(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* outertags,
    int64_t outertagsoffset,
    const int64_t* outerindex,
    int64_t outerindexoffset,
    const int8_t* innertags,
    int64_t innertagsoffset,
    const int64_t* innerindex,
    int64_t innerindexoffset,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray8_64_simplify8_64_to8_64(
      totags,
      toindex,
      outertags,
      outertagsoffset,
      outerindex,
      outerindexoffset,
      innertags,
      innertagsoffset,
      innerindex,
      innerindexoffset,
      towhich,
      innerwhich,
      outerwhich,
      length,
      base);
  }

  template <>
  Error UnionArray_simplify_one_to8_64<int8_t,
    int32_t>(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    const int32_t* fromindex,
    int64_t fromindexoffset,
    int64_t towhich,
    int64_t fromwhich,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray8_32_simplify_one_to8_64(
      totags,
      toindex,
      fromtags,
      fromtagsoffset,
      fromindex,
      fromindexoffset,
      towhich,
      fromwhich,
      length,
      base);
  }
  template <>
  Error UnionArray_simplify_one_to8_64<int8_t,
    uint32_t>(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    const uint32_t* fromindex,
    int64_t fromindexoffset,
    int64_t towhich,
    int64_t fromwhich,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray8_U32_simplify_one_to8_64(
      totags,
      toindex,
      fromtags,
      fromtagsoffset,
      fromindex,
      fromindexoffset,
      towhich,
      fromwhich,
      length,
      base);
  }
  template <>
  Error UnionArray_simplify_one_to8_64<int8_t,
    int64_t>(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    const int64_t* fromindex,
    int64_t fromindexoffset,
    int64_t towhich,
    int64_t fromwhich,
    int64_t length,
    int64_t base) {
    return awkward_UnionArray8_64_simplify_one_to8_64(
      totags,
      toindex,
      fromtags,
      fromtagsoffset,
      fromindex,
      fromindexoffset,
      towhich,
      fromwhich,
      length,
      base);
  }

  template <>
  Error ListArray_validity<int32_t>(
    const int32_t* starts,
    int64_t startsoffset,
    const int32_t* stops,
    int64_t stopsoffset,
    int64_t length,
    int64_t lencontent) {
    return awkward_ListArray32_validity(
      starts,
      startsoffset,
      stops,
      stopsoffset,
      length,
      lencontent);
  }
  template <>
  Error ListArray_validity<uint32_t>(
    const uint32_t* starts,
    int64_t startsoffset,
    const uint32_t* stops,
    int64_t stopsoffset,
    int64_t length,
    int64_t lencontent) {
    return awkward_ListArrayU32_validity(
      starts,
      startsoffset,
      stops,
      stopsoffset,
      length,
      lencontent);
  }
  template <>
  Error ListArray_validity<int64_t>(
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* stops,
    int64_t stopsoffset,
    int64_t length,
    int64_t lencontent) {
    return awkward_ListArray64_validity(
      starts,
      startsoffset,
      stops,
      stopsoffset,
      length,
      lencontent);
  }

  template <>
  Error IndexedArray_validity<int32_t>(
    const int32_t* index,
    int64_t indexoffset,
    int64_t length,
    int64_t lencontent,
    bool isoption) {
    return awkward_IndexedArray32_validity(
      index,
      indexoffset,
      length,
      lencontent,
      isoption);
  }
  template <>
  Error IndexedArray_validity<uint32_t>(
    const uint32_t* index,
    int64_t indexoffset,
    int64_t length,
    int64_t lencontent,
    bool isoption) {
    return awkward_IndexedArrayU32_validity(
      index,
      indexoffset,
      length,
      lencontent,
      isoption);
  }
  template <>
  Error IndexedArray_validity<int64_t>(
    const int64_t* index,
    int64_t indexoffset,
    int64_t length,
    int64_t lencontent,
    bool isoption) {
    return awkward_IndexedArray64_validity(
      index,
      indexoffset,
      length,
      lencontent,
      isoption);
  }

  template <>
  Error UnionArray_validity<int8_t, int32_t>(
    const int8_t* tags,
    int64_t tagsoffset,
    const int32_t* index,
    int64_t indexoffset,
    int64_t length,
    int64_t numcontents,
    const int64_t* lencontents) {
    return awkward_UnionArray8_32_validity(
      tags,
      tagsoffset,
      index,
      indexoffset,
      length,
      numcontents,
      lencontents);
  }
  template <>
  Error UnionArray_validity<int8_t, uint32_t>(
    const int8_t* tags,
    int64_t tagsoffset,
    const uint32_t* index,
    int64_t indexoffset,
    int64_t length,
    int64_t numcontents,
    const int64_t* lencontents) {
    return awkward_UnionArray8_U32_validity(
      tags,
      tagsoffset,
      index,
      indexoffset,
      length,
      numcontents,
      lencontents);
  }
  template <>
  Error UnionArray_validity<int8_t, int64_t>(
    const int8_t* tags,
    int64_t tagsoffset,
    const int64_t* index,
    int64_t indexoffset,
    int64_t length,
    int64_t numcontents,
    const int64_t* lencontents) {
    return awkward_UnionArray8_64_validity(
      tags,
      tagsoffset,
      index,
      indexoffset,
      length,
      numcontents,
      lencontents);
  }

  template <>
  Error UnionArray_fillna_64<int32_t>(
    int64_t* toindex,
    const int32_t* fromindex,
    int64_t offset,
    int64_t length) {
    return awkward_UnionArray_fillna_from32_to64(
      toindex,
      fromindex,
      offset,
      length);
  }
  template <>
  Error UnionArray_fillna_64<uint32_t>(
    int64_t* toindex,
    const uint32_t* fromindex,
    int64_t offset,
    int64_t length) {
    return awkward_UnionArray_fillna_fromU32_to64(
      toindex,
      fromindex,
      offset,
      length);
  }
  template <>
  Error UnionArray_fillna_64<int64_t>(
    int64_t* toindex,
    const int64_t* fromindex,
    int64_t offset,
    int64_t length) {
    return awkward_UnionArray_fillna_from64_to64(
      toindex,
      fromindex,
      offset,
      length);
  }

  ERROR IndexedOptionArray_rpad_and_clip_mask_axis1_64(
    int64_t* toindex,
    const int8_t* frommask,
    int64_t length) {
    return awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64(
      toindex,
      frommask,
      length);
  }

  ERROR index_rpad_and_clip_axis0_64(
    int64_t* toindex,
    int64_t target,
    int64_t length) {
    return awkward_index_rpad_and_clip_axis0_64(
      toindex,
      target,
      length);
  }

  ERROR index_rpad_and_clip_axis1_64(
    int64_t* tostarts,
    int64_t* tostops,
    int64_t target,
    int64_t length) {
    return awkward_index_rpad_and_clip_axis1_64(
      tostarts,
      tostops,
      target,
      length);
  }

  ERROR RegularArray_rpad_and_clip_axis1_64(
    int64_t* toindex,
    int64_t target,
    int64_t size,
    int64_t length) {
    return awkward_RegularArray_rpad_and_clip_axis1_64(
      toindex,
      target,
      size,
      length);
  }

  template <>
  Error ListArray_min_range<int32_t>(
    int64_t* tomin,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset) {
    return awkward_ListArray32_min_range(
      tomin,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset);
  }
  template <>
  Error ListArray_min_range<uint32_t>(
    int64_t* tomin,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset) {
    return awkward_ListArrayU32_min_range(
      tomin,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset);
  }
  template <>
  Error ListArray_min_range<int64_t>(
    int64_t* tomin,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset) {
    return awkward_ListArray64_min_range(
      tomin,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset);
  }

  template <>
  Error ListArray_rpad_and_clip_length_axis1<int32_t>(
    int64_t* tolength,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t target,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset) {
    return awkward_ListArray32_rpad_and_clip_length_axis1(
      tolength,
      fromstarts,
      fromstops,
      target,
      lenstarts,
      startsoffset,
      stopsoffset);
  }
  template <>
  Error ListArray_rpad_and_clip_length_axis1<uint32_t>(
    int64_t* tolength,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t target,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset) {
    return awkward_ListArrayU32_rpad_and_clip_length_axis1(
      tolength,
      fromstarts,
      fromstops,
      target,
      lenstarts,
      startsoffset,
      stopsoffset);
  }
  template <>
  Error ListArray_rpad_and_clip_length_axis1<int64_t>(
    int64_t* tolength,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t target,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset) {
    return awkward_ListArray64_rpad_and_clip_length_axis1(
      tolength,
      fromstarts,
      fromstops,
      target,
      lenstarts,
      startsoffset,
      stopsoffset);
  }

  template <>
  Error ListArray_rpad_axis1_64<int32_t>(
    int64_t* toindex,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int32_t* tostarts,
    int32_t* tostops,
    int64_t target,
    int64_t length,
    int64_t startsoffset,
    int64_t stopsoffset) {
    return awkward_ListArray32_rpad_axis1_64(
      toindex,
      fromstarts,
      fromstops,
      tostarts,
      tostops,
      target,
      length,
      startsoffset,
      stopsoffset);
  }
  template <>
  Error ListArray_rpad_axis1_64<uint32_t>(
    int64_t* toindex,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    uint32_t* tostarts,
    uint32_t* tostops,
    int64_t target,
    int64_t length,
    int64_t startsoffset,
    int64_t stopsoffset) {
    return awkward_ListArrayU32_rpad_axis1_64(
      toindex,
      fromstarts,
      fromstops,
      tostarts,
      tostops,
      target,
      length,
      startsoffset,
      stopsoffset);
  }
  template <>
  Error ListArray_rpad_axis1_64<int64_t>(
    int64_t* toindex,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t* tostarts,
    int64_t* tostops,
    int64_t target,
    int64_t length,
    int64_t startsoffset,
    int64_t stopsoffset) {
    return awkward_ListArray64_rpad_axis1_64(
      toindex,
      fromstarts,
      fromstops,
      tostarts,
      tostops,
      target,
      length,
      startsoffset,
      stopsoffset);
  }

  template <>
  Error ListOffsetArray_rpad_and_clip_axis1_64<int32_t>(
    int64_t* toindex,
    const int32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length,
    int64_t target) {
    return awkward_ListOffsetArray32_rpad_and_clip_axis1_64(
      toindex,
      fromoffsets,
      offsetsoffset,
      length,
      target);
  }
  template <>
  Error ListOffsetArray_rpad_and_clip_axis1_64<uint32_t>(
    int64_t* toindex,
    const uint32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length,
    int64_t target) {
    return awkward_ListOffsetArrayU32_rpad_and_clip_axis1_64(
      toindex,
      fromoffsets,
      offsetsoffset,
      length,
      target);
  }
  template <>
  Error ListOffsetArray_rpad_and_clip_axis1_64<int64_t>(
    int64_t* toindex,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length,
    int64_t target) {
    return awkward_ListOffsetArray64_rpad_and_clip_axis1_64(
      toindex,
      fromoffsets,
      offsetsoffset,
      length,
      target);
  }

  template <>
  Error ListOffsetArray_rpad_length_axis1<int32_t>(
    int32_t* tooffsets,
    const int32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t fromlength,
    int64_t length,
    int64_t* tocount) {
    return awkward_ListOffsetArray32_rpad_length_axis1(
      tooffsets,
      fromoffsets,
      offsetsoffset,
      fromlength,
      length,
      tocount);
  }
  template <>
  Error ListOffsetArray_rpad_length_axis1<uint32_t>(
    uint32_t* tooffsets,
    const uint32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t fromlength,
    int64_t length,
    int64_t* tocount) {
    return awkward_ListOffsetArrayU32_rpad_length_axis1(
      tooffsets,
      fromoffsets,
      offsetsoffset,
      fromlength,
      length,
      tocount);
  }
  template <>
  Error ListOffsetArray_rpad_length_axis1<int64_t>(
    int64_t* tooffsets,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t fromlength,
    int64_t length,
    int64_t* tocount) {
    return awkward_ListOffsetArray64_rpad_length_axis1(
      tooffsets,
      fromoffsets,
      offsetsoffset,
      fromlength,
      length,
      tocount);
  }

  template <>
  Error ListOffsetArray_rpad_axis1_64<int32_t>(
    int64_t* toindex,
    const int32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t fromlength,
    int64_t target) {
    return awkward_ListOffsetArray32_rpad_axis1_64(
      toindex,
      fromoffsets,
      offsetsoffset,
      fromlength,
      target);
  }
  template <>
  Error ListOffsetArray_rpad_axis1_64<uint32_t>(
    int64_t* toindex,
    const uint32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t fromlength,
    int64_t target) {
    return awkward_ListOffsetArrayU32_rpad_axis1_64(
      toindex,
      fromoffsets,
      offsetsoffset,
      fromlength,
      target);
  }
  template <>
  Error ListOffsetArray_rpad_axis1_64<int64_t>(
    int64_t* toindex,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t fromlength,
    int64_t target) {
    return awkward_ListOffsetArray64_rpad_axis1_64(
      toindex,
      fromoffsets,
      offsetsoffset,
      fromlength,
      target);
  }

  ERROR localindex_64(
    int64_t* toindex,
    int64_t length) {
    return awkward_localindex_64(
      toindex,
      length);
  }

  template <>
  Error ListArray_localindex_64<int32_t>(
    int64_t* toindex,
    const int32_t* offsets,
    int64_t offsetsoffset,
    int64_t length) {
    return awkward_ListArray32_localindex_64(
      toindex,
      offsets,
      offsetsoffset,
      length);
  }
  template <>
  Error ListArray_localindex_64<uint32_t>(
    int64_t* toindex,
    const uint32_t* offsets,
    int64_t offsetsoffset,
    int64_t length) {
    return awkward_ListArrayU32_localindex_64(
      toindex,
      offsets,
      offsetsoffset,
      length);
  }
  template <>
  Error ListArray_localindex_64<int64_t>(
    int64_t* toindex,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t length) {
    return awkward_ListArray64_localindex_64(
      toindex,
      offsets,
      offsetsoffset,
      length);
  }

  ERROR RegularArray_localindex_64(
    int64_t* toindex,
    int64_t size,
    int64_t length) {
    return awkward_RegularArray_localindex_64(
      toindex,
      size,
      length);
  }

  template <>
  ERROR combinations(
    int64_t* toindex,
    int64_t n,
    bool replacement,
    int64_t singlelen) {
    return awkward_combinations_64(
      toindex,
      n,
      replacement,
      singlelen);
  }

  template <>
  Error ListArray_combinations_length_64<int32_t>(
    int64_t* totallen,
    int64_t* tooffsets,
    int64_t n,
    bool replacement,
    const int32_t* starts,
    int64_t startsoffset,
    const int32_t* stops,
    int64_t stopsoffset,
    int64_t length) {
    return awkward_ListArray32_combinations_length_64(
      totallen,
      tooffsets,
      n,
      replacement,
      starts,
      startsoffset,
      stops,
      stopsoffset,
      length);
  }
  template <>
  Error ListArray_combinations_length_64<uint32_t>(
    int64_t* totallen,
    int64_t* tooffsets,
    int64_t n,
    bool replacement,
    const uint32_t* starts,
    int64_t startsoffset,
    const uint32_t* stops,
    int64_t stopsoffset,
    int64_t length) {
    return awkward_ListArrayU32_combinations_length_64(
      totallen,
      tooffsets,
      n,
      replacement,
      starts,
      startsoffset,
      stops,
      stopsoffset,
      length);
  }
  template <>
  Error ListArray_combinations_length_64<int64_t>(
    int64_t* totallen,
    int64_t* tooffsets,
    int64_t n,
    bool replacement,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* stops,
    int64_t stopsoffset,
    int64_t length) {
    return awkward_ListArray64_combinations_length_64(
      totallen,
      tooffsets,
      n,
      replacement,
      starts,
      startsoffset,
      stops,
      stopsoffset,
      length);
  }

  template <>
  Error ListArray_combinations_64<int32_t>(
    int64_t** tocarry,
    int64_t* toindex,
    int64_t* fromindex,
    int64_t n,
    bool replacement,
    const int32_t* starts,
    int64_t startsoffset,
    const int32_t* stops,
    int64_t stopsoffset,
    int64_t length) {
    return awkward_ListArray32_combinations_64(
      tocarry,
      toindex,
      fromindex,
      n,
      replacement,
      starts,
      startsoffset,
      stops,
      stopsoffset,
      length);
  }
  template <>
  Error ListArray_combinations_64<uint32_t>(
    int64_t** tocarry,
    int64_t* toindex,
    int64_t* fromindex,
    int64_t n,
    bool replacement,
    const uint32_t* starts,
    int64_t startsoffset,
    const uint32_t* stops,
    int64_t stopsoffset,
    int64_t length) {
    return awkward_ListArrayU32_combinations_64(
      tocarry,
      toindex,
      fromindex,
      n,
      replacement,
      starts,
      startsoffset,
      stops,
      stopsoffset,
      length);
  }
  template <>
  Error ListArray_combinations_64<int64_t>(
    int64_t** tocarry,
    int64_t* toindex,
    int64_t* fromindex,
    int64_t n,
    bool replacement,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* stops,
    int64_t stopsoffset,
    int64_t length) {
    return awkward_ListArray64_combinations_64(
      tocarry,
      toindex,
      fromindex,
      n,
      replacement,
      starts,
      startsoffset,
      stops,
      stopsoffset,
      length);
  }

  ERROR RegularArray_combinations_64(
    int64_t** tocarry,
    int64_t* toindex,
    int64_t* fromindex,
    int64_t n,
    bool replacement,
    int64_t size,
    int64_t length) {
    return awkward_RegularArray_combinations_64(
      tocarry,
      toindex,
      fromindex,
      n,
      replacement,
      size,
      length);
  }

  ERROR ByteMaskedArray_overlay_mask8(
    int8_t* tomask,
    const int8_t* theirmask,
    int64_t theirmaskoffset,
    const int8_t* mymask,
    int64_t mymaskoffset,
    int64_t length,
    bool validwhen) {
    return awkward_ByteMaskedArray_overlay_mask8(
      tomask,
      theirmask,
      theirmaskoffset,
      mymask,
      mymaskoffset,
      length,
      validwhen);
  }

  ERROR BitMaskedArray_to_ByteMaskedArray(
    int8_t* tobytemask,
    const uint8_t* frombitmask,
    int64_t bitmaskoffset,
    int64_t bitmasklength,
    bool validwhen,
    bool lsb_order) {
    return awkward_BitMaskedArray_to_ByteMaskedArray(
      tobytemask,
      frombitmask,
      bitmaskoffset,
      bitmasklength,
      validwhen,
      lsb_order);
  }

  ERROR BitMaskedArray_to_IndexedOptionArray64(
    int64_t* toindex,
    const uint8_t* frombitmask,
    int64_t bitmaskoffset,
    int64_t bitmasklength,
    bool validwhen,
    bool lsb_order) {
    return awkward_BitMaskedArray_to_IndexedOptionArray64(
      toindex,
      frombitmask,
      bitmaskoffset,
      bitmasklength,
      validwhen,
      lsb_order);
  }

  /////////////////////////////////// awkward/cpu-kernels/reducers.h

  ERROR reduce_count_64(
    int64_t* toptr,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_count_64(
      toptr,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }

  template <>
  ERROR reduce_countnonzero_64(
    int64_t* toptr,
    const bool* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_countnonzero_bool_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_countnonzero_64(
    int64_t* toptr,
    const uint8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_countnonzero_uint8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_countnonzero_64(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_countnonzero_int8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_countnonzero_64(
    int64_t* toptr,
    const int16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_countnonzero_int16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_countnonzero_64(
    int64_t* toptr,
    const uint16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_countnonzero_uint16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_countnonzero_64(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_countnonzero_int32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_countnonzero_64(
    int64_t* toptr,
    const uint32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_countnonzero_uint32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_countnonzero_64(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_countnonzero_int64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_countnonzero_64(
    int64_t* toptr,
    const uint64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_countnonzero_uint64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_countnonzero_64(
    int64_t* toptr,
    const float* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_countnonzero_float32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_countnonzero_64(
    int64_t* toptr,
    const double* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_countnonzero_float64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }

  template <>
  ERROR reduce_sum_64(
    int64_t* toptr,
    const bool* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_int64_bool_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_int64_int8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    uint64_t* toptr,
    const uint8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_uint64_uint8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    int64_t* toptr,
    const int16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_int64_int16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    uint64_t* toptr,
    const uint16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_uint64_uint16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_int64_int32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    uint64_t* toptr,
    const uint32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_uint64_uint32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_int64_int64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    uint64_t* toptr,
    const uint64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_uint64_uint64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    float* toptr,
    const float* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_float32_float32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    double* toptr,
    const double* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_float64_float64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    int32_t* toptr,
    const bool* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_int32_bool_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    int32_t* toptr,
    const int8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_int32_int8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    uint32_t* toptr,
    const uint8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_uint32_uint8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    int32_t* toptr,
    const int16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_int32_int16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    uint32_t* toptr,
    const uint16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_uint32_uint16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    int32_t* toptr,
    const int32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_int32_int32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_64(
    uint32_t* toptr,
    const uint32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_uint32_uint32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }

  template <>
  ERROR reduce_sum_bool_64(
    bool* toptr,
    const bool* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_bool_bool_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_bool_64(
    bool* toptr,
    const int8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_bool_int8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_bool_64(
    bool* toptr,
    const uint8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_bool_uint8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_bool_64(
    bool* toptr,
    const int16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_bool_int16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_bool_64(
    bool* toptr,
    const uint16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_bool_uint16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_bool_64(
    bool* toptr,
    const int32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_bool_int32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_bool_64(
    bool* toptr,
    const uint32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_bool_uint32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_bool_64(
    bool* toptr,
    const int64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_bool_int64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_bool_64(
    bool* toptr,
    const uint64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_bool_uint64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_bool_64(
    bool* toptr,
    const float* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_bool_float32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_sum_bool_64(
    bool* toptr,
    const double* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_sum_bool_float64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }

  template <>
  ERROR reduce_prod_64(
    int64_t* toptr,
    const bool* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_int64_bool_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_int64_int8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    uint64_t* toptr,
    const uint8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_uint64_uint8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    int64_t* toptr,
    const int16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_int64_int16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    uint64_t* toptr,
    const uint16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_uint64_uint16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_int64_int32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    uint64_t* toptr,
    const uint32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_uint64_uint32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_int64_int64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    uint64_t* toptr,
    const uint64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_uint64_uint64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    float* toptr,
    const float* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_float32_float32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    double* toptr,
    const double* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_float64_float64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    int32_t* toptr,
    const bool* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_int32_bool_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    int32_t* toptr,
    const int8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_int32_int8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    uint32_t* toptr,
    const uint8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_uint32_uint8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    int32_t* toptr,
    const int16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_int32_int16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    uint32_t* toptr,
    const uint16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_uint32_uint16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    int32_t* toptr,
    const int32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_int32_int32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_64(
    uint32_t* toptr,
    const uint32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_uint32_uint32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }

  template <>
  ERROR reduce_prod_bool_64(
    bool* toptr,
    const bool* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_bool_bool_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_bool_64(
    bool* toptr,
    const int8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_bool_int8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_bool_64(
    bool* toptr,
    const uint8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_bool_uint8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_bool_64(
    bool* toptr,
    const int16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_bool_int16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_bool_64(
    bool* toptr,
    const uint16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_bool_uint16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_bool_64(
    bool* toptr,
    const int32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_bool_int32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_bool_64(
    bool* toptr,
    const uint32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_bool_uint32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_bool_64(
    bool* toptr,
    const int64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_bool_int64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_bool_64(
    bool* toptr,
    const uint64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_bool_uint64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_bool_64(
    bool* toptr,
    const float* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_bool_float32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_prod_bool_64(
    bool* toptr,
    const double* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_prod_bool_float64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }

  template <>
  ERROR reduce_min_64(
    int8_t* toptr,
    const int8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    int8_t identity) {
    return awkward_reduce_min_int8_int8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_min_64(
    uint8_t* toptr,
    const uint8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    uint8_t identity) {
    return awkward_reduce_min_uint8_uint8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_min_64(
    int16_t* toptr,
    const int16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    int16_t identity) {
    return awkward_reduce_min_int16_int16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_min_64(
    uint16_t* toptr,
    const uint16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    uint16_t identity) {
    return awkward_reduce_min_uint16_uint16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_min_64(
    int32_t* toptr,
    const int32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    int32_t identity) {
    return awkward_reduce_min_int32_int32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_min_64(
    uint32_t* toptr,
    const uint32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    uint32_t identity) {
    return awkward_reduce_min_uint32_uint32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_min_64(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    int64_t identity) {
    return awkward_reduce_min_int64_int64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_min_64(
    uint64_t* toptr,
    const uint64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    uint64_t identity) {
    return awkward_reduce_min_uint64_uint64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_min_64(
    float* toptr,
    const float* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    float identity) {
    return awkward_reduce_min_float32_float32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_min_64(
    double* toptr,
    const double* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    double identity) {
    return awkward_reduce_min_float64_float64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }

  template <>
  ERROR reduce_max_64(
    int8_t* toptr,
    const int8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    int8_t identity) {
    return awkward_reduce_max_int8_int8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_max_64(
    uint8_t* toptr,
    const uint8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    uint8_t identity) {
    return awkward_reduce_max_uint8_uint8_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_max_64(
    int16_t* toptr,
    const int16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    int16_t identity) {
    return awkward_reduce_max_int16_int16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_max_64(
    uint16_t* toptr,
    const uint16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    uint16_t identity) {
    return awkward_reduce_max_uint16_uint16_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_max_64(
    int32_t* toptr,
    const int32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    int32_t identity) {
    return awkward_reduce_max_int32_int32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_max_64(
    uint32_t* toptr,
    const uint32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    uint32_t identity) {
    return awkward_reduce_max_uint32_uint32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_max_64(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    int64_t identity) {
    return awkward_reduce_max_int64_int64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_max_64(
    uint64_t* toptr,
    const uint64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    uint64_t identity) {
    return awkward_reduce_max_uint64_uint64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_max_64(
    float* toptr,
    const float* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    float identity) {
    return awkward_reduce_max_float32_float32_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }
  template <>
  ERROR reduce_max_64(
    double* toptr,
    const double* fromptr,
    int64_t fromptroffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength,
    double identity) {
    return awkward_reduce_max_float64_float64_64(
      toptr,
      fromptr,
      fromptroffset,
      parents,
      parentsoffset,
      lenparents,
      outlength,
      identity);
  }

  template <>
  ERROR reduce_argmin_64(
    int64_t* toptr,
    const bool* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmin_bool_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmin_64(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmin_int8_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmin_64(
    int64_t* toptr,
    const uint8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmin_uint8_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmin_64(
    int64_t* toptr,
    const int16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmin_int16_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmin_64(
    int64_t* toptr,
    const uint16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmin_uint16_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmin_64(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmin_int32_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmin_64(
    int64_t* toptr,
    const uint32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmin_uint32_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmin_64(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmin_int64_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmin_64(
    int64_t* toptr,
    const uint64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmin_uint64_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmin_64(
    int64_t* toptr,
    const float* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmin_float32_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmin_64(
    int64_t* toptr,
    const double* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmin_float64_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }


  template <>
  ERROR reduce_argmax_64(
    int64_t* toptr,
    const bool* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmax_bool_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmax_64(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmax_int8_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmax_64(
    int64_t* toptr,
    const uint8_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmax_uint8_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmax_64(
    int64_t* toptr,
    const int16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmax_int16_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmax_64(
    int64_t* toptr,
    const uint16_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmax_uint16_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmax_64(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmax_int32_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmax_64(
    int64_t* toptr,
    const uint32_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmax_uint32_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmax_64(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmax_int64_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmax_64(
    int64_t* toptr,
    const uint64_t* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmax_uint64_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmax_64(
    int64_t* toptr,
    const float* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmax_float32_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }
  template <>
  ERROR reduce_argmax_64(
    int64_t* toptr,
    const double* fromptr,
    int64_t fromptroffset,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_reduce_argmax_float64_64(
      toptr,
      fromptr,
      fromptroffset,
      starts,
      startsoffset,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }

  ERROR content_reduce_zeroparents_64(
    int64_t* toparents,
    int64_t length) {
    return awkward_content_reduce_zeroparents_64(
      toparents,
      length);
  }

  ERROR ListOffsetArray_reduce_global_startstop_64(
    int64_t* globalstart,
    int64_t* globalstop,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t length) {
    return awkward_ListOffsetArray_reduce_global_startstop_64(
      globalstart,
      globalstop,
      offsets,
      offsetsoffset,
      length);
  }

  ERROR ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64(
    int64_t* maxcount,
    int64_t* offsetscopy,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t length) {
    return awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64(
      maxcount,
      offsetscopy,
      offsets,
      offsetsoffset,
      length);
  }

  ERROR ListOffsetArray_reduce_nonlocal_preparenext_64(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t nextlen,
    int64_t* maxnextparents,
    int64_t* distincts,
    int64_t distinctslen,
    int64_t* offsetscopy,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t length,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t maxcount) {
    return awkward_ListOffsetArray_reduce_nonlocal_preparenext_64(
      nextcarry,
      nextparents,
      nextlen,
      maxnextparents,
      distincts,
      distinctslen,
      offsetscopy,
      offsets,
      offsetsoffset,
      length,
      parents,
      parentsoffset,
      maxcount);
  }

  ERROR ListOffsetArray_reduce_nonlocal_nextstarts_64(
    int64_t* nextstarts,
    const int64_t* nextparents,
    int64_t nextlen) {
    return awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64(
      nextstarts,
      nextparents,
      nextlen);
  }

  ERROR ListOffsetArray_reduce_nonlocal_findgaps_64(
    int64_t* gaps,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents) {
    return awkward_ListOffsetArray_reduce_nonlocal_findgaps_64(
      gaps,
      parents,
      parentsoffset,
      lenparents);
  }

  ERROR ListOffsetArray_reduce_nonlocal_outstartsstops_64(
    int64_t* outstarts,
    int64_t* outstops,
    const int64_t* distincts,
    int64_t lendistincts,
    const int64_t* gaps,
    int64_t outlength) {
    return awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64(
      outstarts,
      outstops,
      distincts,
      lendistincts,
      gaps,
      outlength);
  }

  ERROR ListOffsetArray_reduce_local_nextparents_64(
    int64_t* nextparents,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t length) {
    return awkward_ListOffsetArray_reduce_local_nextparents_64(
      nextparents,
      offsets,
      offsetsoffset,
      length);
  }

  ERROR ListOffsetArray_reduce_local_outoffsets_64(
    int64_t* outoffsets,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_ListOffsetArray_reduce_local_outoffsets_64(
      outoffsets,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }

  template <>
  Error IndexedArray_reduce_next_64<int32_t>(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t* outindex,
    const int32_t* index,
    int64_t indexoffset,
    int64_t* parents,
    int64_t parentsoffset,
    int64_t length) {
    return awkward_IndexedArray32_reduce_next_64(
      nextcarry,
      nextparents,
      outindex,
      index,
      indexoffset,
      parents,
      parentsoffset,
      length);
  }

  template <>
  Error IndexedArray_reduce_next_64<uint32_t>(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t* outindex,
    const uint32_t* index,
    int64_t indexoffset,
    int64_t* parents,
    int64_t parentsoffset,
    int64_t length) {
    return awkward_IndexedArrayU32_reduce_next_64(
      nextcarry,
      nextparents,
      outindex,
      index,
      indexoffset,
      parents,
      parentsoffset,
      length);
  }

  template <>
  Error IndexedArray_reduce_next_64<int64_t>(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t* outindex,
    const int64_t* index,
    int64_t indexoffset,
    int64_t* parents,
    int64_t parentsoffset,
    int64_t length) {
    return awkward_IndexedArray64_reduce_next_64(
      nextcarry,
      nextparents,
      outindex,
      index,
      indexoffset,
      parents,
      parentsoffset,
      length);
  }

  ERROR IndexedArray_reduce_next_fix_offsets_64(
    int64_t* outoffsets,
    const int64_t* starts,
    int64_t startsoffset,
    int64_t startslength,
    int64_t outindexlength) {
    return awkward_IndexedArray_reduce_next_fix_offsets_64(
      outoffsets,
      starts,
      startsoffset,
      startslength,
      outindexlength);
  }

  ERROR NumpyArray_reduce_mask_ByteMaskedArray_64(
    int8_t* toptr,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t lenparents,
    int64_t outlength) {
    return awkward_NumpyArray_reduce_mask_ByteMaskedArray_64(
      toptr,
      parents,
      parentsoffset,
      lenparents,
      outlength);
  }

  ERROR ByteMaskedArray_reduce_next_64(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t* outindex,
    const int8_t* mask,
    int64_t maskoffset,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t length,
    bool validwhen) {
    return awkward_ByteMaskedArray_reduce_next_64(
      nextcarry,
      nextparents,
      outindex,
      mask,
      maskoffset,
      parents,
      parentsoffset,
      length,
      validwhen);
  }

  /////////////////////////////////// awkward/cpu-kernels/sorting.h

  ERROR sorting_ranges(
    int64_t* toindex,
    int64_t tolength,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t parentslength,
    int64_t outlength) {
    return awkward_sorting_ranges(
      toindex,
      tolength,
      parents,
      parentsoffset,
      parentslength,
      outlength);
  }
  ERROR sorting_ranges_length(
    int64_t* tolength,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t parentslength,
    int64_t outlength) {
    return awkward_sorting_ranges_length(
      tolength,
      parents,
      parentsoffset,
      parentslength,
      outlength);
  }
  template <>
  Error NumpyArray_argsort<bool>(
    int64_t* toptr,
    const bool* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable) {
    return awkward_argsort_bool(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_argsort<int8_t>(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable) {
    return awkward_argsort_int8(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_argsort<uint8_t>(
    int64_t* toptr,
    const uint8_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable) {
    return awkward_argsort_uint8(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_argsort<int16_t>(
    int64_t* toptr,
    const int16_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable) {
    return awkward_argsort_int16(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_argsort<uint16_t>(
    int64_t* toptr,
    const uint16_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable) {
    return awkward_argsort_uint16(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_argsort<int32_t>(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable) {
    return awkward_argsort_int32(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_argsort<uint32_t>(
    int64_t* toptr,
    const uint32_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable) {
    return awkward_argsort_uint32(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_argsort<int64_t>(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable) {
    return awkward_argsort_int64(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_argsort<uint64_t>(
    int64_t* toptr,
    const uint64_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable) {
    return awkward_argsort_uint64(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_argsort<float>(
    int64_t* toptr,
    const float* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable) {
    return awkward_argsort_float32(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      ascending,
      stable);
  }

  template <>
  Error NumpyArray_argsort<double>(
    int64_t* toptr,
    const double* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable) {
    return awkward_argsort_float64(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      ascending,
      stable);
  }

  template <>
  Error NumpyArray_sort<bool>(
    bool* toptr,
    const bool* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable) {
    return awkward_sort_bool(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      parentslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_sort<uint8_t>(
    uint8_t* toptr,
    const uint8_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable) {
    return awkward_sort_uint8(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      parentslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_sort<int8_t>(
    int8_t* toptr,
    const int8_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable) {
    return awkward_sort_int8(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      parentslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_sort<uint16_t>(
    uint16_t* toptr,
    const uint16_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable) {
    return awkward_sort_uint16(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      parentslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_sort<int16_t>(
    int16_t* toptr,
    const int16_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable) {
    return awkward_sort_int16(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      parentslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_sort<uint32_t>(
    uint32_t* toptr,
    const uint32_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable) {
    return awkward_sort_uint32(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      parentslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_sort<int32_t>(
    int32_t* toptr,
    const int32_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable) {
    return awkward_sort_int32(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      parentslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_sort<uint64_t>(
    uint64_t* toptr,
    const uint64_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable) {
    return awkward_sort_uint64(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      parentslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_sort<int64_t>(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable) {
    return awkward_sort_int64(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      parentslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_sort<float>(
    float* toptr,
    const float* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable) {
    return awkward_sort_float32(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      parentslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_sort<double>(
    double* toptr,
    const double* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable) {
    return awkward_sort_float64(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      parentslength,
      ascending,
      stable);
  }
  template <>
  Error NumpyArray_sort_asstrings<uint8_t>(
    uint8_t* toptr,
    const uint8_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    bool ascending,
    bool stable) {
    return awkward_NumpyArray_sort_asstrings_uint8(
      toptr,
      fromptr,
      length,
      offsets,
      offsetslength,
      outoffsets,
      ascending,
      stable);
  }

  ERROR ListOffsetArray_local_preparenext_64(
    int64_t* tocarry,
    const int64_t* fromindex,
    int64_t length) {
    return awkward_ListOffsetArray_local_preparenext_64(
      tocarry,
      fromindex,
      length);
  }

  ERROR IndexedArray_local_preparenext_64(
    int64_t* tocarry,
    const int64_t* starts,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t parentslength,
    const int64_t* nextparents,
    int64_t nextparentsoffset) {
    return awkward_IndexedArray_local_preparenext_64(
      tocarry,
      starts,
      parents,
      parentsoffset,
      parentslength,
      nextparents,
      nextparentsoffset);
  }
}
