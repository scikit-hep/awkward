// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef KERNEL_H
#define KERNEL_H

#include "awkward/common.h"

namespace kernel {
  class Identities;
  template <typename T>
  class IndexOf;

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  identities32_from_listoffsetarray(
    int32_t* toptr,
    const int32_t* fromptr,
    const T* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  identities64_from_listoffsetarray(
    int64_t* toptr,
    const int64_t* fromptr,
    const T* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  identities32_from_listarray(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const T* fromstarts,
    const T* fromstops,
    int64_t fromptroffset,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  identities64_from_listarray(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const T* fromstarts,
    const T* fromstops,
    int64_t fromptroffset,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  identities32_from_indexedarray(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const T* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  identities64_from_indexedarray(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const T* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T, typename I>
  ERROR
  identities32_from_unionarray(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const T* fromtags,
    const I* fromindex,
    int64_t fromptroffset,
    int64_t tagsoffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T, typename I>
  ERROR
  identities64_from_unionarray(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const T* fromtags,
    const I* fromindex,
    int64_t fromptroffset,
    int64_t tagsoffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  index_carry_64(
    T* toindex,
    const T* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t lenfromindex,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  index_carry_nocheck_64(
    T* toindex,
    const T* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_getitem_next_at_64(
    int64_t* tocarry,
    const T* fromstarts,
    const T* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t at  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_getitem_next_range_carrylength(
    int64_t* carrylength,
    const T* fromstarts,
    const T* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_getitem_next_range_64(
    T* tooffsets,
    int64_t* tocarry,
    const T* fromstarts,
    const T* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_getitem_next_range_counts_64(
    int64_t* total,
    const T* fromoffsets,
    int64_t lenstarts  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_getitem_next_range_spreadadvanced_64(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const T* fromoffsets,
    int64_t lenstarts  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_getitem_next_array_64(
    int64_t* tocarry,
    int64_t* toadvanced,
    const T* fromstarts,
    const T* fromstops,
    const int64_t* fromarray,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_getitem_next_array_advanced_64(
    int64_t* tocarry,
    int64_t* toadvanced,
    const T* fromstarts,
    const T* fromstops,
    const int64_t* fromarray,
    const int64_t* fromadvanced,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_getitem_carry_64(
    T* tostarts,
    T* tostops,
    const T* fromstarts,
    const T* fromstops,
    const int64_t* fromcarry,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lencarry  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_num_64(
    int64_t* tonum,
    const T* fromstarts,
    int64_t startsoffset,
    const T* fromstops,
    int64_t stopsoffset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listoffsetarray_flatten_offsets_64(
    int64_t* tooffsets,
    const T* outeroffsets,
    int64_t outeroffsetsoffset,
    int64_t outeroffsetslen,
    const int64_t* inneroffsets,
    int64_t inneroffsetsoffset,
    int64_t inneroffsetslen  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_flatten_none2empty_64(
    int64_t* outoffsets,
    const T* outindex,
    int64_t outindexoffset,
    int64_t outindexlength,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t offsetslength  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T, typename I>
  ERROR
  unionarray_flatten_length_64(
    int64_t* total_length,
    const T* fromtags,
    int64_t fromtagsoffset,
    const I* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t** offsetsraws,
    int64_t* offsetsoffsets  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T, typename I>
  ERROR
  unionarray_flatten_combine_64(
    int8_t* totags,
    int64_t* toindex,
    int64_t* tooffsets,
    const T* fromtags,
    int64_t fromtagsoffset,
    const I* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t** offsetsraws,
    int64_t* offsetsoffsets  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_flatten_nextcarry_64(
    int64_t* tocarry,
    const T* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_numnull(
    int64_t* numnull,
    const T* fromindex,
    int64_t indexoffset,
    int64_t lenindex  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_getitem_nextcarry_outindex_64(
    int64_t* tocarry,
    T* toindex,
    const T* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_getitem_nextcarry_outindex_mask_64(
    int64_t* tocarry,
    int64_t* toindex,
    const T* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_getitem_nextcarry_64(
    int64_t* tocarry,
    const T* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_getitem_carry_64(
    T* toindex,
    const T* fromindex,
    const int64_t* fromcarry,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencarry  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_overlay_mask8_to64(
    int64_t* toindex,
    const int8_t* mask,
    int64_t maskoffset,
    const T* fromindex,
    int64_t indexoffset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_mask8(
    int8_t* tomask,
    const T* fromindex,
    int64_t indexoffset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_simplify32_to64(
    int64_t* toindex,
    const T* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_simplifyU32_to64(
    int64_t* toindex,
    const T* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const uint32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_simplify64_to64(
    int64_t* toindex,
    const T* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int64_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  unionarray_regular_index_getsize(
    int64_t* size,
    const T* fromtags,
    int64_t tagsoffset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T, typename I>
  ERROR
  unionarray_regular_index(
    I* toindex,
    I* current,
    int64_t size,
    const T* fromtags,
    int64_t tagsoffset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T, typename I>
  ERROR
  unionarray_project_64(
    int64_t* lenout,
    int64_t* tocarry,
    const T* fromtags,
    int64_t tagsoffset,
    const I* fromindex,
    int64_t indexoffset,
    int64_t length,
    int64_t which  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_compact_offsets64(
    int64_t* tooffsets,
    const T* fromstarts,
    const T* fromstops,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listoffsetarray_compact_offsets64(
    int64_t* tooffsets,
    const T* fromoffsets,
    int64_t offsetsoffset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_broadcast_tooffsets64(
    int64_t* tocarry,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    const T* fromstarts,
    int64_t startsoffset,
    const T* fromstops,
    int64_t stopsoffset,
    int64_t lencontent  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listoffsetarray_toRegularArray(
    int64_t* size,
    const T* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T, typename I>
  ERROR
  unionarray_simplify8_32_to8_64(
    int8_t* totags,
    int64_t* toindex,
    const T* outertags,
    int64_t outertagsoffset,
    const I* outerindex,
    int64_t outerindexoffset,
    const int8_t* innertags,
    int64_t innertagsoffset,
    const int32_t* innerindex,
    int64_t innerindexoffset,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T, typename I>
  ERROR
  unionarray_simplify8_U32_to8_64(
    int8_t* totags,
    int64_t* toindex,
    const T* outertags,
    int64_t outertagsoffset,
    const I* outerindex,
    int64_t outerindexoffset,
    const int8_t* innertags,
    int64_t innertagsoffset,
    const uint32_t* innerindex,
    int64_t innerindexoffset,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T, typename I>
  ERROR
  unionarray_simplify8_64_to8_64(
    int8_t* totags,
    int64_t* toindex,
    const T* outertags,
    int64_t outertagsoffset,
    const I* outerindex,
    int64_t outerindexoffset,
    const int8_t* innertags,
    int64_t innertagsoffset,
    const int64_t* innerindex,
    int64_t innerindexoffset,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T, typename I>
  ERROR
  unionarray_simplify_one_to8_64(
    int8_t* totags,
    int64_t* toindex,
    const T* fromtags,
    int64_t fromtagsoffset,
    const I* fromindex,
    int64_t fromindexoffset,
    int64_t towhich,
    int64_t fromwhich,
    int64_t length,
    int64_t base  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_getitem_jagged_expand_64(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t* tocarry,
    const T* fromstarts,
    int64_t fromstartsoffset,
    const T* fromstops,
    int64_t fromstopsoffset,
    int64_t jaggedsize,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_getitem_jagged_apply_64(
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
    const T* fromstarts,
    int64_t fromstartsoffset,
    const T* fromstops,
    int64_t fromstopsoffset,
    int64_t contentlen  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_getitem_jagged_descend_64(
    int64_t* tooffsets,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t sliceouterlen,
    const T* fromstarts,
    int64_t fromstartsoffset,
    const T* fromstops,
    int64_t fromstopsoffset  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_reduce_next_64(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t* outindex,
    const T* index,
    int64_t indexoffset,
    int64_t* parents,
    int64_t parentsoffset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  UnionArray_fillna_64(
    int64_t* toindex,
    const T* fromindex,
    int64_t offset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  ListArray_min_range(
    int64_t* tomin,
    const T* fromstarts,
    const T* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  ListArray_rpad_axis1_64(
    int64_t* toindex,
    const T* fromstarts,
    const T* fromstops,
    T* tostarts,
    T* tostops,
    int64_t target,
    int64_t length,
    int64_t startsoffset,
    int64_t stopsoffset  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  ListArray_rpad_and_clip_length_axis1(
    int64_t* tolength,
    const T* fromstarts,
    const T* fromstops,
    int64_t target,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  ListOffsetArray_rpad_length_axis1(
    T* tooffsets,
    const T* fromoffsets,
    int64_t offsetsoffset,
    int64_t fromlength,
    int64_t target,
    int64_t* tolength  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  ListOffsetArray_rpad_axis1_64(
    int64_t* toindex,
    const T* fromoffsets,
    int64_t offsetsoffset,
    int64_t fromlength,
    int64_t target  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  ListOffsetArray_rpad_and_clip_axis1_64(
    int64_t* toindex,
    const T* fromoffsets,
    int64_t offsetsoffset,
    int64_t length,
    int64_t target  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_validity(
    const T* starts,
    int64_t startsoffset,
    const T* stops,
    int64_t stopsoffset,
    int64_t length,
    int64_t lencontent  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  indexedarray_validity(
    const T* index,
    int64_t indexoffset,
    int64_t length,
    int64_t lencontent,
    bool isoption  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T, typename I>
  ERROR
  unionarray_validity(
    const T* tags,
    int64_t tagsoffset,
    const I* index,
    int64_t indexoffset,
    int64_t length,
    int64_t numcontents,
    const int64_t* lencontents  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_localindex_64(
    int64_t* toindex,
    const T* offsets,
    int64_t offsetsoffset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_combinations_length_64(
    int64_t* totallen,
    int64_t* tooffsets,
    int64_t n,
    bool replacement,
    const T* starts,
    int64_t startsoffset,
    const T* stops,
    int64_t stopsoffset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template <typename T>
  ERROR
  listarray_combinations_64(
    int64_t** tocarry,
    int64_t* toindex,
    int64_t* fromindex,
    int64_t n,
    bool replacement,
    const T* starts,
    int64_t startsoffset,
    const T* stops,
    int64_t stopsoffset,
    int64_t length  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template<typename T>
  T
  index_getitem_at_nowrap(
    const T* ptr,
    int64_t offset,
    int64_t at  ,
    KernelsLib ptr_lib = cpu_kernels);

  /// @brief Wraps several cpu-kernels from the C interface with a template
  /// to make it easier and more type-safe to call.
  template<typename T>
  void
  index_setitem_at_nowrap(
    T* ptr,
    int64_t offset,
    int64_t at,
    T value  ,
    KernelsLib ptr_lib = cpu_kernels);

};

#endif //KERNEL_H
