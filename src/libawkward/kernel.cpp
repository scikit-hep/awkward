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
  template <>
  Error identities32_from_listoffsetarray<int32_t>(
    int32_t* toptr,
    const int32_t* fromptr,
    const int32_t* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities32_from_listoffsetarray32(
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
  Error identities32_from_listoffsetarray<uint32_t>(
    int32_t* toptr,
    const int32_t* fromptr,
    const uint32_t* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities32_from_listoffsetarrayU32(
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
  Error identities32_from_listoffsetarray<int64_t>(
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities32_from_listoffsetarray64(
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
  Error identities64_from_listoffsetarray<int32_t>(
    int64_t* toptr,
    const int64_t* fromptr,
    const int32_t* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities64_from_listoffsetarray32(
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
  Error identities64_from_listoffsetarray<uint32_t>(
    int64_t* toptr,
    const int64_t* fromptr,
    const uint32_t* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities64_from_listoffsetarrayU32(
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
  Error identities64_from_listoffsetarray<int64_t>(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* fromoffsets,
    int64_t fromptroffset,
    int64_t offsetsoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities64_from_listoffsetarray64(
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
  Error identities32_from_listarray<int32_t>(
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
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities32_from_listarray32(
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
  Error identities32_from_listarray<uint32_t>(
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
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities32_from_listarrayU32(
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
  Error identities32_from_listarray<int64_t>(
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
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities32_from_listarray64(
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
  Error identities64_from_listarray<int32_t>(
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
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities64_from_listarray32(
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
  Error identities64_from_listarray<uint32_t>(
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
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities64_from_listarrayU32(
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
  Error identities64_from_listarray<int64_t>(
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
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities64_from_listarray64(
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
  Error identities32_from_indexedarray<int32_t>(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int32_t* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities32_from_indexedarray32(
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
  Error identities32_from_indexedarray<uint32_t>(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const uint32_t* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities32_from_indexedarrayU32(
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
  Error identities32_from_indexedarray<int64_t>(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities32_from_indexedarray64(
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
  Error identities64_from_indexedarray<int32_t>(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int32_t* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities64_from_indexedarray32(
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
  Error identities64_from_indexedarray<uint32_t>(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const uint32_t* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities64_from_indexedarrayU32(
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
  Error identities64_from_indexedarray<int64_t>(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* fromindex,
    int64_t fromptroffset,
    int64_t indexoffset,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    KernelsLib ptr_lib) {
    return awkward_identities64_from_indexedarray64(
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
  Error identities32_from_unionarray<int8_t,
    int32_t>(
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
    int64_t which,
    KernelsLib ptr_lib) {
    return awkward_identities32_from_unionarray8_32(
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
  Error identities32_from_unionarray<int8_t,
    uint32_t>(
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
    int64_t which,
    KernelsLib ptr_lib) {
    return awkward_identities32_from_unionarray8_U32(
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
  Error identities32_from_unionarray<int8_t,
    int64_t>(
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
    int64_t which,
    KernelsLib ptr_lib) {
    return awkward_identities32_from_unionarray8_64(
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
  Error identities64_from_unionarray<int8_t,
    int32_t>(
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
    int64_t which,
    KernelsLib ptr_lib) {
    return awkward_identities64_from_unionarray8_32(
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
  Error identities64_from_unionarray<int8_t,
    uint32_t>(
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
    int64_t which,
    KernelsLib ptr_lib) {
    return awkward_identities64_from_unionarray8_U32(
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
  Error identities64_from_unionarray<int8_t,
    int64_t>(
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
    int64_t which,
    KernelsLib ptr_lib) {
    return awkward_identities64_from_unionarray8_64(
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
  Error listoffsetarray_flatten_offsets_64<int32_t>(
    int64_t* tooffsets,
    const int32_t* outeroffsets,
    int64_t outeroffsetsoffset,
    int64_t outeroffsetslen,
    const int64_t* inneroffsets,
    int64_t inneroffsetsoffset,
    int64_t inneroffsetslen,
    KernelsLib ptr_lib) {
    return awkward_listoffsetarray32_flatten_offsets_64(
      tooffsets,
      outeroffsets,
      outeroffsetsoffset,
      outeroffsetslen,
      inneroffsets,
      inneroffsetsoffset,
      inneroffsetslen);
  }
  template <>
  Error listoffsetarray_flatten_offsets_64<uint32_t>(
    int64_t* tooffsets,
    const uint32_t* outeroffsets,
    int64_t outeroffsetsoffset,
    int64_t outeroffsetslen,
    const int64_t* inneroffsets,
    int64_t inneroffsetsoffset,
    int64_t inneroffsetslen,
    KernelsLib ptr_lib) {
    return awkward_listoffsetarrayU32_flatten_offsets_64(
      tooffsets,
      outeroffsets,
      outeroffsetsoffset,
      outeroffsetslen,
      inneroffsets,
      inneroffsetsoffset,
      inneroffsetslen);
  }
  template <>
  Error listoffsetarray_flatten_offsets_64<int64_t>(
    int64_t* tooffsets,
    const int64_t* outeroffsets,
    int64_t outeroffsetsoffset,
    int64_t outeroffsetslen,
    const int64_t* inneroffsets,
    int64_t inneroffsetsoffset,
    int64_t inneroffsetslen,
    KernelsLib ptr_lib) {
    return awkward_listoffsetarray64_flatten_offsets_64(
      tooffsets,
      outeroffsets,
      outeroffsetsoffset,
      outeroffsetslen,
      inneroffsets,
      inneroffsetsoffset,
      inneroffsetslen);
  }

  template <>
  Error indexedarray_flatten_none2empty_64<int32_t>(
    int64_t* outoffsets,
    const int32_t* outindex,
    int64_t outindexoffset,
    int64_t outindexlength,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_flatten_none2empty_64(
      outoffsets,
      outindex,
      outindexoffset,
      outindexlength,
      offsets,
      offsetsoffset,
      offsetslength);
  }
  template <>
  Error indexedarray_flatten_none2empty_64<uint32_t>(
    int64_t* outoffsets,
    const uint32_t* outindex,
    int64_t outindexoffset,
    int64_t outindexlength,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_flatten_none2empty_64(
      outoffsets,
      outindex,
      outindexoffset,
      outindexlength,
      offsets,
      offsetsoffset,
      offsetslength);
  }
  template <>
  Error indexedarray_flatten_none2empty_64<int64_t>(
    int64_t* outoffsets,
    const int64_t* outindex,
    int64_t outindexoffset,
    int64_t outindexlength,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_flatten_none2empty_64(
      outoffsets,
      outindex,
      outindexoffset,
      outindexlength,
      offsets,
      offsetsoffset,
      offsetslength);
  }

  template <>
  Error unionarray_flatten_length_64<int8_t,
    int32_t>(
    int64_t* total_length,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    const int32_t* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t** offsetsraws,
    int64_t* offsetsoffsets,
    KernelsLib ptr_lib) {
    return awkward_unionarray32_flatten_length_64(
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
  Error unionarray_flatten_length_64<int8_t,
    uint32_t>(
    int64_t* total_length,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    const uint32_t* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t** offsetsraws,
    int64_t* offsetsoffsets,
    KernelsLib ptr_lib) {
    return awkward_unionarrayU32_flatten_length_64(
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
  Error unionarray_flatten_length_64<int8_t,
    int64_t>(
    int64_t* total_length,
    const int8_t* fromtags,
    int64_t fromtagsoffset,
    const int64_t* fromindex,
    int64_t fromindexoffset,
    int64_t length,
    int64_t** offsetsraws,
    int64_t* offsetsoffsets,
    KernelsLib ptr_lib) {
    return awkward_unionarray64_flatten_length_64(
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
  Error unionarray_flatten_combine_64<int8_t,
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
    int64_t* offsetsoffsets,
    KernelsLib ptr_lib) {
    return awkward_unionarray32_flatten_combine_64(
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
  Error unionarray_flatten_combine_64<int8_t,
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
    int64_t* offsetsoffsets,
    KernelsLib ptr_lib) {
    return awkward_unionarrayU32_flatten_combine_64(
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
  Error unionarray_flatten_combine_64<int8_t,
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
    int64_t* offsetsoffsets,
    KernelsLib ptr_lib) {
    return awkward_unionarray64_flatten_combine_64(
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
  Error indexedarray_flatten_nextcarry_64<int32_t>(
    int64_t* tocarry,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_flatten_nextcarry_64(
      tocarry,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  Error indexedarray_flatten_nextcarry_64<uint32_t>(
    int64_t* tocarry,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_flatten_nextcarry_64(
      tocarry,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  Error indexedarray_flatten_nextcarry_64<int64_t>(
    int64_t* tocarry,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_flatten_nextcarry_64(
      tocarry,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }

  template <>
  Error indexedarray_overlay_mask8_to64<int32_t>(
    int64_t* toindex,
    const int8_t* mask,
    int64_t maskoffset,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_overlay_mask8_to64(
      toindex,
      mask,
      maskoffset,
      fromindex,
      indexoffset,
      length);
  }
  template <>
  Error indexedarray_overlay_mask8_to64<uint32_t>(
    int64_t* toindex,
    const int8_t* mask,
    int64_t maskoffset,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_overlay_mask8_to64(
      toindex,
      mask,
      maskoffset,
      fromindex,
      indexoffset,
      length);
  }
  template <>
  Error indexedarray_overlay_mask8_to64<int64_t>(
    int64_t* toindex,
    const int8_t* mask,
    int64_t maskoffset,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_overlay_mask8_to64(
      toindex,
      mask,
      maskoffset,
      fromindex,
      indexoffset,
      length);
  }

  template <>
  Error indexedarray_mask8<int32_t>(
    int8_t* tomask,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_mask8(
      tomask,
      fromindex,
      indexoffset,
      length);
  }
  template <>
  Error indexedarray_mask8<uint32_t>(
    int8_t* tomask,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_mask8(
      tomask,
      fromindex,
      indexoffset,
      length);
  }
  template <>
  Error indexedarray_mask8<int64_t>(
    int8_t* tomask,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_mask8(
      tomask,
      fromindex,
      indexoffset,
      length);
  }

  template <>
  Error indexedarray_simplify32_to64<int32_t>(
    int64_t* toindex,
    const int32_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_simplify32_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }
  template <>
  Error indexedarray_simplify32_to64<uint32_t>(
    int64_t* toindex,
    const uint32_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_simplify32_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }
  template <>
  Error indexedarray_simplify32_to64<int64_t>(
    int64_t* toindex,
    const int64_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_simplify32_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }

  template <>
  Error indexedarray_simplifyU32_to64<int32_t>(
    int64_t* toindex,
    const int32_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const uint32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_simplifyU32_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }
  template <>
  Error indexedarray_simplifyU32_to64<uint32_t>(
    int64_t* toindex,
    const uint32_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const uint32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_simplifyU32_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }
  template <>
  Error indexedarray_simplifyU32_to64<int64_t>(
    int64_t* toindex,
    const int64_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const uint32_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_simplifyU32_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }

  template <>
  Error indexedarray_simplify64_to64<int32_t>(
    int64_t* toindex,
    const int32_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int64_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_simplify64_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }
  template <>
  Error indexedarray_simplify64_to64<uint32_t>(
    int64_t* toindex,
    const uint32_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int64_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_simplify64_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }
  template <>
  Error indexedarray_simplify64_to64<int64_t>(
    int64_t* toindex,
    const int64_t* outerindex,
    int64_t outeroffset,
    int64_t outerlength,
    const int64_t* innerindex,
    int64_t inneroffset,
    int64_t innerlength,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_simplify64_to64(
      toindex,
      outerindex,
      outeroffset,
      outerlength,
      innerindex,
      inneroffset,
      innerlength);
  }

  template <>
  Error listarray_compact_offsets64(
    int64_t* tooffsets,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarray32_compact_offsets64(
      tooffsets,
      fromstarts,
      fromstops,
      startsoffset,
      stopsoffset,
      length);
  }
  template <>
  Error listarray_compact_offsets64(
    int64_t* tooffsets,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_compact_offsets64(
      tooffsets,
      fromstarts,
      fromstops,
      startsoffset,
      stopsoffset,
      length);
  }
  template <>
  Error listarray_compact_offsets64(
    int64_t* tooffsets,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarray64_compact_offsets64(
      tooffsets,
      fromstarts,
      fromstops,
      startsoffset,
      stopsoffset,
      length);
  }

  template <>
  Error listoffsetarray_compact_offsets64(
    int64_t* tooffsets,
    const int32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listoffsetarray32_compact_offsets64(
      tooffsets,
      fromoffsets,
      offsetsoffset,
      length);
  }
  template <>
  Error listoffsetarray_compact_offsets64(
    int64_t* tooffsets,
    const uint32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listoffsetarrayU32_compact_offsets64(
      tooffsets,
      fromoffsets,
      offsetsoffset,
      length);
  }
  template <>
  Error listoffsetarray_compact_offsets64(
    int64_t* tooffsets,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listoffsetarray64_compact_offsets64(
      tooffsets,
      fromoffsets,
      offsetsoffset,
      length);
  }

  template <>
  Error listarray_broadcast_tooffsets64<int32_t>(
    int64_t* tocarry,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    const int32_t* fromstarts,
    int64_t startsoffset,
    const int32_t* fromstops,
    int64_t stopsoffset,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_listarray32_broadcast_tooffsets64(
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
  Error listarray_broadcast_tooffsets64<uint32_t>(
    int64_t* tocarry,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    const uint32_t* fromstarts,
    int64_t startsoffset,
    const uint32_t* fromstops,
    int64_t stopsoffset,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_broadcast_tooffsets64(
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
  Error listarray_broadcast_tooffsets64<int64_t>(
    int64_t* tocarry,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    const int64_t* fromstarts,
    int64_t startsoffset,
    const int64_t* fromstops,
    int64_t stopsoffset,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_listarray64_broadcast_tooffsets64(
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
  Error listoffsetarray_toRegularArray<int32_t>(
    int64_t* size,
    const int32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    KernelsLib ptr_lib) {
    return awkward_listoffsetarray32_toRegularArray(
      size,
      fromoffsets,
      offsetsoffset,
      offsetslength);
  }
  template <>
  Error listoffsetarray_toRegularArray<uint32_t>(
    int64_t* size,
    const uint32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    KernelsLib ptr_lib) {
    return awkward_listoffsetarrayU32_toRegularArray(
      size,
      fromoffsets,
      offsetsoffset,
      offsetslength);
  }
  template <>
  Error listoffsetarray_toRegularArray(
    int64_t* size,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t offsetslength,
    KernelsLib ptr_lib) {
    return awkward_listoffsetarray64_toRegularArray(
      size,
      fromoffsets,
      offsetsoffset,
      offsetslength);
  }

  template <>
  Error unionarray_simplify8_32_to8_64<int8_t,
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
    int64_t base,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_32_simplify8_32_to8_64(
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
  Error unionarray_simplify8_32_to8_64<int8_t,
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
    int64_t base,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_U32_simplify8_32_to8_64(
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
  Error unionarray_simplify8_32_to8_64<int8_t,
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
    int64_t base,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_64_simplify8_32_to8_64(
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
  Error unionarray_simplify8_U32_to8_64<int8_t,
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
    int64_t base,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_32_simplify8_U32_to8_64(
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
  Error unionarray_simplify8_U32_to8_64<int8_t,
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
    int64_t base,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_U32_simplify8_U32_to8_64(
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
  Error unionarray_simplify8_U32_to8_64<int8_t,
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
    int64_t base,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_64_simplify8_U32_to8_64(
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
  Error unionarray_simplify8_64_to8_64<int8_t,
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
    int64_t base,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_32_simplify8_64_to8_64(
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
  Error unionarray_simplify8_64_to8_64<int8_t,
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
    int64_t base,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_U32_simplify8_64_to8_64(
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
  Error unionarray_simplify8_64_to8_64<int8_t,
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
    int64_t base,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_64_simplify8_64_to8_64(
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
  Error unionarray_simplify_one_to8_64<int8_t,
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
    int64_t base,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_32_simplify_one_to8_64(
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
  Error unionarray_simplify_one_to8_64<int8_t,
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
    int64_t base,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_U32_simplify_one_to8_64(
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
  Error unionarray_simplify_one_to8_64<int8_t,
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
    int64_t base,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_64_simplify_one_to8_64(
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
  Error indexedarray_reduce_next_64<int32_t>(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t* outindex,
    const int32_t* index,
    int64_t indexoffset,
    int64_t* parents,
    int64_t parentsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_reduce_next_64(
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
  Error indexedarray_reduce_next_64<uint32_t>(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t* outindex,
    const uint32_t* index,
    int64_t indexoffset,
    int64_t* parents,
    int64_t parentsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_reduce_next_64(
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
  Error indexedarray_reduce_next_64<int64_t>(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t* outindex,
    const int64_t* index,
    int64_t indexoffset,
    int64_t* parents,
    int64_t parentsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_reduce_next_64(
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
  Error UnionArray_fillna_64<int32_t>(
    int64_t* toindex,
    const int32_t* fromindex,
    int64_t offset,
    int64_t length,
    KernelsLib ptr_lib) {
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
    int64_t length,
    KernelsLib ptr_lib) {
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
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_UnionArray_fillna_from64_to64(
      toindex,
      fromindex,
      offset,
      length);
  }

  template <>
  Error ListArray_min_range<int32_t>(
    int64_t* tomin,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    KernelsLib ptr_lib) {
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
    int64_t stopsoffset,
    KernelsLib ptr_lib) {
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
    int64_t stopsoffset,
    KernelsLib ptr_lib) {
    return awkward_ListArray64_min_range(
      tomin,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset);
  }
  template <>
  Error ListOffsetArray_rpad_length_axis1<int32_t>(
    int32_t* tooffsets,
    const int32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t fromlength,
    int64_t length,
    int64_t* tocount,
    KernelsLib ptr_lib) {
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
    int64_t* tocount,
    KernelsLib ptr_lib) {
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
    int64_t* tocount,
    KernelsLib ptr_lib) {
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
    int64_t target,
    KernelsLib ptr_lib) {
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
    int64_t target,
    KernelsLib ptr_lib) {
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
    int64_t target,
    KernelsLib ptr_lib) {
    return awkward_ListOffsetArray64_rpad_axis1_64(
      toindex,
      fromoffsets,
      offsetsoffset,
      fromlength,
      target);
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
    int64_t stopsoffset,
    KernelsLib ptr_lib) {
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
    int64_t stopsoffset,
    KernelsLib ptr_lib) {
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
    int64_t stopsoffset,
    KernelsLib ptr_lib) {
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
  Error ListArray_rpad_and_clip_length_axis1<int32_t>(
    int64_t* tolength,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t target,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    KernelsLib ptr_lib) {
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
    int64_t stopsoffset,
    KernelsLib ptr_lib) {
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
    int64_t stopsoffset,
    KernelsLib ptr_lib) {
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
  Error ListOffsetArray_rpad_and_clip_axis1_64<int32_t>(
    int64_t* toindex,
    const int32_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length,
    int64_t target,
    KernelsLib ptr_lib) {
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
    int64_t target,
    KernelsLib ptr_lib) {
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
    int64_t target,
    KernelsLib ptr_lib) {
    return awkward_ListOffsetArray64_rpad_and_clip_axis1_64(
      toindex,
      fromoffsets,
      offsetsoffset,
      length,
      target);
  }

  template <>
  Error listarray_validity<int32_t>(
    const int32_t* starts,
    int64_t startsoffset,
    const int32_t* stops,
    int64_t stopsoffset,
    int64_t length,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_listarray32_validity(
      starts,
      startsoffset,
      stops,
      stopsoffset,
      length,
      lencontent);
  }
  template <>
  Error listarray_validity<uint32_t>(
    const uint32_t* starts,
    int64_t startsoffset,
    const uint32_t* stops,
    int64_t stopsoffset,
    int64_t length,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_validity(
      starts,
      startsoffset,
      stops,
      stopsoffset,
      length,
      lencontent);
  }
  template <>
  Error listarray_validity<int64_t>(
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* stops,
    int64_t stopsoffset,
    int64_t length,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_listarray64_validity(
      starts,
      startsoffset,
      stops,
      stopsoffset,
      length,
      lencontent);
  }

  template <>
  Error indexedarray_validity<int32_t>(
    const int32_t* index,
    int64_t indexoffset,
    int64_t length,
    int64_t lencontent,
    bool isoption,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_validity(
      index,
      indexoffset,
      length,
      lencontent,
      isoption);
  }
  template <>
  Error indexedarray_validity<uint32_t>(
    const uint32_t* index,
    int64_t indexoffset,
    int64_t length,
    int64_t lencontent,
    bool isoption,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_validity(
      index,
      indexoffset,
      length,
      lencontent,
      isoption);
  }
  template <>
  Error indexedarray_validity<int64_t>(
    const int64_t* index,
    int64_t indexoffset,
    int64_t length,
    int64_t lencontent,
    bool isoption,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_validity(
      index,
      indexoffset,
      length,
      lencontent,
      isoption);
  }

  template <>
  Error unionarray_validity<int8_t,
    int32_t>(
    const int8_t* tags,
    int64_t tagsoffset,
    const int32_t* index,
    int64_t indexoffset,
    int64_t length,
    int64_t numcontents,
    const int64_t* lencontents,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_32_validity(
      tags,
      tagsoffset,
      index,
      indexoffset,
      length,
      numcontents,
      lencontents);
  }
  template <>
  Error unionarray_validity<int8_t,
    uint32_t>(
    const int8_t* tags,
    int64_t tagsoffset,
    const uint32_t* index,
    int64_t indexoffset,
    int64_t length,
    int64_t numcontents,
    const int64_t* lencontents,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_U32_validity(
      tags,
      tagsoffset,
      index,
      indexoffset,
      length,
      numcontents,
      lencontents);
  }
  template <>
  Error unionarray_validity<int8_t,
    int64_t>(
    const int8_t* tags,
    int64_t tagsoffset,
    const int64_t* index,
    int64_t indexoffset,
    int64_t length,
    int64_t numcontents,
    const int64_t* lencontents,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_64_validity(
      tags,
      tagsoffset,
      index,
      indexoffset,
      length,
      numcontents,
      lencontents);
  }

  template <>
  Error listarray_localindex_64<int32_t>(
    int64_t* toindex,
    const int32_t* offsets,
    int64_t offsetsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarray32_localindex_64(
      toindex,
      offsets,
      offsetsoffset,
      length);
  }
  template <>
  Error listarray_localindex_64<uint32_t>(
    int64_t* toindex,
    const uint32_t* offsets,
    int64_t offsetsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_localindex_64(
      toindex,
      offsets,
      offsetsoffset,
      length);
  }
  template <>
  Error listarray_localindex_64<int64_t>(
    int64_t* toindex,
    const int64_t* offsets,
    int64_t offsetsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarray64_localindex_64(
      toindex,
      offsets,
      offsetsoffset,
      length);
  }

  template <>
  Error listarray_combinations_length_64<int32_t>(
    int64_t* totallen,
    int64_t* tooffsets,
    int64_t n,
    bool replacement,
    const int32_t* starts,
    int64_t startsoffset,
    const int32_t* stops,
    int64_t stopsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarray32_combinations_length_64(
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
  Error listarray_combinations_length_64<uint32_t>(
    int64_t* totallen,
    int64_t* tooffsets,
    int64_t n,
    bool replacement,
    const uint32_t* starts,
    int64_t startsoffset,
    const uint32_t* stops,
    int64_t stopsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_combinations_length_64(
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
  Error listarray_combinations_length_64<int64_t>(
    int64_t* totallen,
    int64_t* tooffsets,
    int64_t n,
    bool replacement,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* stops,
    int64_t stopsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarray64_combinations_length_64(
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
  Error listarray_combinations_64<int32_t>(
    int64_t** tocarry,
    int64_t* toindex,
    int64_t* fromindex,
    int64_t n,
    bool replacement,
    const int32_t* starts,
    int64_t startsoffset,
    const int32_t* stops,
    int64_t stopsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarray32_combinations_64(
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
  Error listarray_combinations_64<uint32_t>(
    int64_t** tocarry,
    int64_t* toindex,
    int64_t* fromindex,
    int64_t n,
    bool replacement,
    const uint32_t* starts,
    int64_t startsoffset,
    const uint32_t* stops,
    int64_t stopsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_combinations_64(
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
  Error listarray_combinations_64<int64_t>(
    int64_t** tocarry,
    int64_t* toindex,
    int64_t* fromindex,
    int64_t n,
    bool replacement,
    const int64_t* starts,
    int64_t startsoffset,
    const int64_t* stops,
    int64_t stopsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarray64_combinations_64(
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
  Error listarray_num_64<int32_t>(
    int64_t* tonum,
    const int32_t* fromstarts,
    int64_t startsoffset,
    const int32_t* fromstops,
    int64_t stopsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarray32_num_64(
      tonum,
      fromstarts,
      startsoffset,
      fromstops,
      stopsoffset,
      length);
  }
  template <>
  Error listarray_num_64<uint32_t>(
    int64_t* tonum,
    const uint32_t* fromstarts,
    int64_t startsoffset,
    const uint32_t* fromstops,
    int64_t stopsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_num_64(
      tonum,
      fromstarts,
      startsoffset,
      fromstops,
      stopsoffset,
      length);
  }
  template <>
  Error listarray_num_64<int64_t>(
    int64_t* tonum,
    const int64_t* fromstarts,
    int64_t startsoffset,
    const int64_t* fromstops,
    int64_t stopsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarray64_num_64(
      tonum,
      fromstarts,
      startsoffset,
      fromstops,
      stopsoffset,
      length);
  }

  /// @brief getitem kernels
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

  template <>
  ERROR regularize_arrayslice(
    int64_t* flatheadptr,
    int64_t lenflathead,
    int64_t length) {
    return awkward_regularize_arrayslice_64(
      flatheadptr,
      lenflathead,
      length);
  }

  template <>
  ERROR index_to_index64(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t length) {
    return awkward_index8_to_index64(
      toptr,
      fromptr,
      length);
  }
  template <>
  ERROR index_to_index64(
    int64_t* toptr,
    const uint8_t* fromptr,
    int64_t length) {
    return awkward_indexU8_to_index64(
      toptr,
      fromptr,
      length);
  }
  template <>
  ERROR index_to_index64(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t length) {
    return awkward_index32_to_index64(
      toptr,
      fromptr,
      length);
  }
  template <>
  ERROR index_to_index64(
    int64_t* toptr,
    const uint32_t* fromptr,
    int64_t length) {
    return awkward_indexU32_to_index64(
      toptr,
      fromptr,
      length);
  }

  template <>
  Error index_carry_64<int8_t>(
    int8_t* toindex,
    const int8_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t lenfromindex,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_index8_carry_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      lenfromindex,
      length);
  }
  template <>
  Error index_carry_64<uint8_t>(
    uint8_t* toindex,
    const uint8_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t lenfromindex,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexU8_carry_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      lenfromindex,
      length);
  }
  template <>
  Error index_carry_64<int32_t>(
    int32_t* toindex,
    const int32_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t lenfromindex,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_index32_carry_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      lenfromindex,
      length);
  }
  template <>
  Error index_carry_64<uint32_t>(
    uint32_t* toindex,
    const uint32_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t lenfromindex,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexU32_carry_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      lenfromindex,
      length);
  }
  template <>
  Error index_carry_64<int64_t>(
    int64_t* toindex,
    const int64_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t lenfromindex,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_index64_carry_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      lenfromindex,
      length);
  }

  template <>
  Error index_carry_nocheck_64<int8_t>(
    int8_t* toindex,
    const int8_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_index8_carry_nocheck_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      length);
  }
  template <>
  Error index_carry_nocheck_64<uint8_t>(
    uint8_t* toindex,
    const uint8_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexU8_carry_nocheck_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      length);
  }
  template <>
  Error index_carry_nocheck_64<int32_t>(
    int32_t* toindex,
    const int32_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_index32_carry_nocheck_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      length);
  }
  template <>
  Error index_carry_nocheck_64<uint32_t>(
    uint32_t* toindex,
    const uint32_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_indexU32_carry_nocheck_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      length);
  }
  template <>
  Error index_carry_nocheck_64<int64_t>(
    int64_t* toindex,
    const int64_t* fromindex,
    const int64_t* carry,
    int64_t fromindexoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_index64_carry_nocheck_64(
      toindex,
      fromindex,
      carry,
      fromindexoffset,
      length);
  }

  template <>
  ERROR slicearray_ravel(
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
    return awkward_carry_arange_32(
      toptr,
      length);
  }
  template <>
  ERROR carry_arange(
    uint32_t* toptr,
    int64_t length) {
    return awkward_carry_arange_U32(
      toptr,
      length);
  }
  template <>
  ERROR carry_arange(
    int64_t* toptr,
    int64_t length) {
    return awkward_carry_arange_64(
      toptr,
      length);
  }

  template <>
  ERROR identities_getitem_carry(
    int32_t* newidentitiesptr,
    const int32_t* identitiesptr,
    const int64_t* carryptr,
    int64_t lencarry,
    int64_t offset,
    int64_t width,
    int64_t length) {
    return awkward_identities32_getitem_carry_64(
      newidentitiesptr,
      identitiesptr,
      carryptr,
      lencarry,
      offset,
      width,
      length);
  }
  template <>
  ERROR identities_getitem_carry(
    int64_t* newidentitiesptr,
    const int64_t* identitiesptr,
    const int64_t* carryptr,
    int64_t lencarry,
    int64_t offset,
    int64_t width,
    int64_t length) {
    return awkward_identities64_getitem_carry_64(
      newidentitiesptr,
      identitiesptr,
      carryptr,
      lencarry,
      offset,
      width,
      length);
  }

  template <>
  ERROR numpyarray_contiguous_copy(
    uint8_t* toptr,
    const uint8_t* fromptr,
    int64_t len,
    int64_t stride,
    int64_t offset,
    const int64_t* pos) {
    return awkward_numpyarray_contiguous_copy_64(
      toptr,
      fromptr,
      len,
      stride,
      offset,
      pos);
  }

  template <>
  ERROR numpyarray_contiguous_next(
    int64_t* topos,
    const int64_t* frompos,
    int64_t len,
    int64_t skip,
    int64_t stride) {
    return awkward_numpyarray_contiguous_next_64(
      topos,
      frompos,
      len,
      skip,
      stride);
  }

  template <>
  ERROR numpyarray_getitem_next_null(
    uint8_t* toptr,
    const uint8_t* fromptr,
    int64_t len,
    int64_t stride,
    int64_t offset,
    const int64_t* pos) {
    return awkward_numpyarray_getitem_next_null_64(
      toptr,
      fromptr,
      len,
      stride,
      offset,
      pos);
  }

  template <>
  ERROR numpyarray_getitem_next_at(
    int64_t* nextcarryptr,
    const int64_t* carryptr,
    int64_t lencarry,
    int64_t skip,
    int64_t at) {
    return awkward_numpyarray_getitem_next_at_64(
      nextcarryptr,
      carryptr,
      lencarry,
      skip,
      at);
  }

  template <>
  ERROR numpyarray_getitem_next_range(
    int64_t* nextcarryptr,
    const int64_t* carryptr,
    int64_t lencarry,
    int64_t lenhead,
    int64_t skip,
    int64_t start,
    int64_t step) {
    return awkward_numpyarray_getitem_next_range_64(
      nextcarryptr,
      carryptr,
      lencarry,
      lenhead,
      skip,
      start,
      step);
  }

  template <>
  ERROR numpyarray_getitem_next_range_advanced(
    int64_t* nextcarryptr,
    int64_t* nextadvancedptr,
    const int64_t* carryptr,
    const int64_t* advancedptr,
    int64_t lencarry,
    int64_t lenhead,
    int64_t skip,
    int64_t start,
    int64_t step) {
    return awkward_numpyarray_getitem_next_range_advanced_64(
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

  template <>
  ERROR numpyarray_getitem_next_array(
    int64_t* nextcarryptr,
    int64_t* nextadvancedptr,
    const int64_t* carryptr,
    const int64_t* flatheadptr,
    int64_t lencarry,
    int64_t lenflathead,
    int64_t skip) {
    return awkward_numpyarray_getitem_next_array_64(
      nextcarryptr,
      nextadvancedptr,
      carryptr,
      flatheadptr,
      lencarry,
      lenflathead,
      skip);
  }

  template <>
  ERROR numpyarray_getitem_next_array_advanced(
    int64_t* nextcarryptr,
    const int64_t* carryptr,
    const int64_t* advancedptr,
    const int64_t* flatheadptr,
    int64_t lencarry,
    int64_t skip) {
    return awkward_numpyarray_getitem_next_array_advanced_64(
      nextcarryptr,
      carryptr,
      advancedptr,
      flatheadptr,
      lencarry,
      skip);
  }

  ERROR numpyarray_getitem_boolean_numtrue(
    int64_t* numtrue,
    const int8_t* fromptr,
    int64_t byteoffset,
    int64_t length,
    int64_t stride) {
    return awkward_numpyarray_getitem_boolean_numtrue(
      numtrue,
      fromptr,
      byteoffset,
      length,
      stride);
  }

  template <>
  ERROR numpyarray_getitem_boolean_nonzero(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t byteoffset,
    int64_t length,
    int64_t stride) {
    return awkward_numpyarray_getitem_boolean_nonzero_64(
      toptr,
      fromptr,
      byteoffset,
      length,
      stride);
  }

  template <>
  Error listarray_getitem_next_at_64<int32_t>(
    int64_t* tocarry,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t at,
    KernelsLib ptr_lib) {
    return awkward_listarray32_getitem_next_at_64(
      tocarry,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset,
      at);
  }
  template <>
  Error listarray_getitem_next_at_64<uint32_t>(
    int64_t* tocarry,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t at,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_getitem_next_at_64(
      tocarry,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset,
      at);
  }
  template <>
  Error listarray_getitem_next_at_64<int64_t>(
    int64_t* tocarry,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t at,
    KernelsLib ptr_lib) {
    return awkward_listarray64_getitem_next_at_64(
      tocarry,
      fromstarts,
      fromstops,
      lenstarts,
      startsoffset,
      stopsoffset,
      at);
  }

  template <>
  Error listarray_getitem_next_range_carrylength<int32_t>(
    int64_t* carrylength,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step,
    KernelsLib ptr_lib) {
    return awkward_listarray32_getitem_next_range_carrylength(
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
  Error listarray_getitem_next_range_carrylength<uint32_t>(
    int64_t* carrylength,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_getitem_next_range_carrylength(
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
  Error listarray_getitem_next_range_carrylength<int64_t>(
    int64_t* carrylength,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step,
    KernelsLib ptr_lib) {
    return awkward_listarray64_getitem_next_range_carrylength(
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
  Error listarray_getitem_next_range_64<int32_t>(
    int32_t* tooffsets,
    int64_t* tocarry,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step,
    KernelsLib ptr_lib) {
    return awkward_listarray32_getitem_next_range_64(
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
  Error listarray_getitem_next_range_64<uint32_t>(
    uint32_t* tooffsets,
    int64_t* tocarry,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_getitem_next_range_64(
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
  Error listarray_getitem_next_range_64<int64_t>(
    int64_t* tooffsets,
    int64_t* tocarry,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t lenstarts,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t start,
    int64_t stop,
    int64_t step,
    KernelsLib ptr_lib) {
    return awkward_listarray64_getitem_next_range_64(
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
  Error listarray_getitem_next_range_counts_64<int32_t>(
    int64_t* total,
    const int32_t* fromoffsets,
    int64_t lenstarts,
    KernelsLib ptr_lib) {
    return awkward_listarray32_getitem_next_range_counts_64(
      total,
      fromoffsets,
      lenstarts);
  }
  template <>
  Error listarray_getitem_next_range_counts_64<uint32_t>(
    int64_t* total,
    const uint32_t* fromoffsets,
    int64_t lenstarts,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_getitem_next_range_counts_64(
      total,
      fromoffsets,
      lenstarts);
  }
  template <>
  Error listarray_getitem_next_range_counts_64<int64_t>(
    int64_t* total,
    const int64_t* fromoffsets,
    int64_t lenstarts,
    KernelsLib ptr_lib) {
    return awkward_listarray64_getitem_next_range_counts_64(
      total,
      fromoffsets,
      lenstarts);
  }

  template <>
  Error listarray_getitem_next_range_spreadadvanced_64<int32_t>(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const int32_t* fromoffsets,
    int64_t lenstarts,
    KernelsLib ptr_lib) {
    return awkward_listarray32_getitem_next_range_spreadadvanced_64(
      toadvanced,
      fromadvanced,
      fromoffsets,
      lenstarts);
  }
  template <>
  Error listarray_getitem_next_range_spreadadvanced_64<uint32_t>(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const uint32_t* fromoffsets,
    int64_t lenstarts,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_getitem_next_range_spreadadvanced_64(
      toadvanced,
      fromadvanced,
      fromoffsets,
      lenstarts);
  }
  template <>
  Error listarray_getitem_next_range_spreadadvanced_64<int64_t>(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const int64_t* fromoffsets,
    int64_t lenstarts,
    KernelsLib ptr_lib) {
    return awkward_listarray64_getitem_next_range_spreadadvanced_64(
      toadvanced,
      fromadvanced,
      fromoffsets,
      lenstarts);
  }

  template <>
  Error listarray_getitem_next_array_64<int32_t>(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    const int64_t* fromarray,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_listarray32_getitem_next_array_64(
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
  Error listarray_getitem_next_array_64<uint32_t>(
    int64_t* tocarry,
    int64_t* toadvanced,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    const int64_t* fromarray,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_getitem_next_array_64(
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
  Error listarray_getitem_next_array_64<int64_t>(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    const int64_t* fromarray,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_listarray64_getitem_next_array_64(
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
  Error listarray_getitem_next_array_advanced_64<int32_t>(
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
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_listarray32_getitem_next_array_advanced_64(
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
  Error listarray_getitem_next_array_advanced_64<uint32_t>(
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
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_getitem_next_array_advanced_64(
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
  Error listarray_getitem_next_array_advanced_64<int64_t>(
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
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_listarray64_getitem_next_array_advanced_64(
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
  Error listarray_getitem_carry_64<int32_t>(
    int32_t* tostarts,
    int32_t* tostops,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    const int64_t* fromcarry,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lencarry,
    KernelsLib ptr_lib) {
    return awkward_listarray32_getitem_carry_64(
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
  Error listarray_getitem_carry_64<uint32_t>(
    uint32_t* tostarts,
    uint32_t* tostops,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    const int64_t* fromcarry,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lencarry,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_getitem_carry_64(
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
  Error listarray_getitem_carry_64<int64_t>(
    int64_t* tostarts,
    int64_t* tostops,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    const int64_t* fromcarry,
    int64_t startsoffset,
    int64_t stopsoffset,
    int64_t lenstarts,
    int64_t lencarry,
    KernelsLib ptr_lib) {
    return awkward_listarray64_getitem_carry_64(
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
  ERROR regulararray_getitem_next_at(
    int64_t* tocarry,
    int64_t at,
    int64_t len,
    int64_t size) {
    return awkward_regulararray_getitem_next_at_64(
      tocarry,
      at,
      len,
      size);
  }

  template <>
  ERROR regulararray_getitem_next_range(
    int64_t* tocarry,
    int64_t regular_start,
    int64_t step,
    int64_t len,
    int64_t size,
    int64_t nextsize) {
    return awkward_regulararray_getitem_next_range_64(
      tocarry,
      regular_start,
      step,
      len,
      size,
      nextsize);
  }

  template <>
  ERROR regulararray_getitem_next_range_spreadadvanced(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    int64_t len,
    int64_t nextsize) {
    return awkward_regulararray_getitem_next_range_spreadadvanced_64(
      toadvanced,
      fromadvanced,
      len,
      nextsize);
  }

  template <>
  ERROR regulararray_getitem_next_array_regularize(
    int64_t* toarray,
    const int64_t* fromarray,
    int64_t lenarray,
    int64_t size) {
    return awkward_regulararray_getitem_next_array_regularize_64(
      toarray,
      fromarray,
      lenarray,
      size);
  }

  template <>
  ERROR regulararray_getitem_next_array(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int64_t* fromarray,
    int64_t len,
    int64_t lenarray,
    int64_t size) {
    return awkward_regulararray_getitem_next_array_64(
      tocarry,
      toadvanced,
      fromarray,
      len,
      lenarray,
      size);
  }

  template <>
  ERROR regulararray_getitem_next_array_advanced(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const int64_t* fromarray,
    int64_t len,
    int64_t lenarray,
    int64_t size) {
    return awkward_regulararray_getitem_next_array_advanced_64(
      tocarry,
      toadvanced,
      fromadvanced,
      fromarray,
      len,
      lenarray,
      size);
  }

  template <>
  ERROR regulararray_getitem_carry(
    int64_t* tocarry,
    const int64_t* fromcarry,
    int64_t lencarry,
    int64_t size) {
    return awkward_regulararray_getitem_carry_64(
      tocarry,
      fromcarry,
      lencarry,
      size);
  }

  template <>
  Error indexedarray_numnull<int32_t>(
    int64_t* numnull,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_numnull(
      numnull,
      fromindex,
      indexoffset,
      lenindex);
  }
  template <>
  Error indexedarray_numnull<uint32_t>(
    int64_t* numnull,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_numnull(
      numnull,
      fromindex,
      indexoffset,
      lenindex);
  }
  template <>
  Error indexedarray_numnull<int64_t>(
    int64_t* numnull,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_numnull(
      numnull,
      fromindex,
      indexoffset,
      lenindex);
  }

  template <>
  Error indexedarray_getitem_nextcarry_outindex_64<int32_t>(
    int64_t* tocarry,
    int32_t* toindex,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_getitem_nextcarry_outindex_64(
      tocarry,
      toindex,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  Error indexedarray_getitem_nextcarry_outindex_64<uint32_t>(
    int64_t* tocarry,
    uint32_t* toindex,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_getitem_nextcarry_outindex_64(
      tocarry,
      toindex,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  Error indexedarray_getitem_nextcarry_outindex_64<int64_t>(
    int64_t* tocarry,
    int64_t* toindex,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_getitem_nextcarry_outindex_64(
      tocarry,
      toindex,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }

  template <>
  Error indexedarray_getitem_nextcarry_outindex_mask_64<int32_t>(
    int64_t* tocarry,
    int64_t* toindex,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_getitem_nextcarry_outindex_mask_64(
      tocarry,
      toindex,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  Error indexedarray_getitem_nextcarry_outindex_mask_64<uint32_t>(
    int64_t* tocarry,
    int64_t* toindex,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_getitem_nextcarry_outindex_mask_64(
      tocarry,
      toindex,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  Error indexedarray_getitem_nextcarry_outindex_mask_64<int64_t>(
    int64_t* tocarry,
    int64_t* toindex,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_getitem_nextcarry_outindex_mask_64(
      tocarry,
      toindex,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }

  template <>
  ERROR listoffsetarray_getitem_adjust_offsets(
    int64_t* tooffsets,
    int64_t* tononzero,
    const int64_t* fromoffsets,
    int64_t offsetsoffset,
    int64_t length,
    const int64_t* nonzero,
    int64_t nonzerooffset,
    int64_t nonzerolength) {
    return awkward_listoffsetarray_getitem_adjust_offsets_64(
      tooffsets,
      tononzero,
      fromoffsets,
      offsetsoffset,
      length,
      nonzero,
      nonzerooffset,
      nonzerolength);
  }

  template <>
  ERROR listoffsetarray_getitem_adjust_offsets_index(
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
    return awkward_listoffsetarray_getitem_adjust_offsets_index_64(
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

  template <>
  ERROR indexedarray_getitem_adjust_outindex(
    int8_t* tomask,
    int64_t* toindex,
    int64_t* tononzero,
    const int64_t* fromindex,
    int64_t fromindexoffset,
    int64_t fromindexlength,
    const int64_t* nonzero,
    int64_t nonzerooffset,
    int64_t nonzerolength) {
    return awkward_indexedarray_getitem_adjust_outindex_64(
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
  ERROR indexedarray_getitem_nextcarry_64(
    int64_t* tocarry,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_getitem_nextcarry_64(
      tocarry,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  ERROR indexedarray_getitem_nextcarry_64(
    int64_t* tocarry,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_getitem_nextcarry_64(
      tocarry,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }
  template <>
  ERROR indexedarray_getitem_nextcarry_64(
    int64_t* tocarry,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencontent,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_getitem_nextcarry_64(
      tocarry,
      fromindex,
      indexoffset,
      lenindex,
      lencontent);
  }

  template <>
  Error indexedarray_getitem_carry_64<int32_t>(
    int32_t* toindex,
    const int32_t* fromindex,
    const int64_t* fromcarry,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencarry,
    KernelsLib ptr_lib) {
    return awkward_indexedarray32_getitem_carry_64(
      toindex,
      fromindex,
      fromcarry,
      indexoffset,
      lenindex,
      lencarry);
  }
  template <>
  Error indexedarray_getitem_carry_64<uint32_t>(
    uint32_t* toindex,
    const uint32_t* fromindex,
    const int64_t* fromcarry,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencarry,
    KernelsLib ptr_lib) {
    return awkward_indexedarrayU32_getitem_carry_64(
      toindex,
      fromindex,
      fromcarry,
      indexoffset,
      lenindex,
      lencarry);
  }
  template <>
  Error indexedarray_getitem_carry_64<int64_t>(
    int64_t* toindex,
    const int64_t* fromindex,
    const int64_t* fromcarry,
    int64_t indexoffset,
    int64_t lenindex,
    int64_t lencarry,
    KernelsLib ptr_lib) {
    return awkward_indexedarray64_getitem_carry_64(
      toindex,
      fromindex,
      fromcarry,
      indexoffset,
      lenindex,
      lencarry);
  }

  template <>
  Error unionarray_regular_index_getsize<int8_t>(
    int64_t* size,
    const int8_t* fromtags,
    int64_t tagsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_regular_index_getsize(
      size,
      fromtags,
      tagsoffset,
      length);
  }

  template <>
  Error unionarray_regular_index<int8_t,
    int32_t>(
    int32_t* toindex,
    int32_t* current,
    int64_t size,
    const int8_t* fromtags,
    int64_t tagsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_32_regular_index(
      toindex,
      current,
      size,
      fromtags,
      tagsoffset,
      length);
  }
  template <>
  Error unionarray_regular_index<int8_t,
    uint32_t>(
    uint32_t* toindex,
    uint32_t* current,
    int64_t size,
    const int8_t* fromtags,
    int64_t tagsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_U32_regular_index(
      toindex,
      current,
      size,
      fromtags,
      tagsoffset,
      length);
  }
  template <>
  Error unionarray_regular_index<int8_t,
    int64_t>(
    int64_t* toindex,
    int64_t* current,
    int64_t size,
    const int8_t* fromtags,
    int64_t tagsoffset,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_64_regular_index(
      toindex,
      current,
      size,
      fromtags,
      tagsoffset,
      length);
  }

  template <>
  Error unionarray_project_64<int8_t,
    int32_t>(
    int64_t* lenout,
    int64_t* tocarry,
    const int8_t* fromtags,
    int64_t tagsoffset,
    const int32_t* fromindex,
    int64_t indexoffset,
    int64_t length,
    int64_t which,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_32_project_64(
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
  Error unionarray_project_64<int8_t,
    uint32_t>(
    int64_t* lenout,
    int64_t* tocarry,
    const int8_t* fromtags,
    int64_t tagsoffset,
    const uint32_t* fromindex,
    int64_t indexoffset,
    int64_t length,
    int64_t which,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_U32_project_64(
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
  Error unionarray_project_64<int8_t,
    int64_t>(
    int64_t* lenout,
    int64_t* tocarry,
    const int8_t* fromtags,
    int64_t tagsoffset,
    const int64_t* fromindex,
    int64_t indexoffset,
    int64_t length,
    int64_t which,
    KernelsLib ptr_lib) {
    return awkward_unionarray8_64_project_64(
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
  ERROR missing_repeat(
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

  template <>
  ERROR regulararray_getitem_jagged_expand(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t regularsize,
    int64_t regularlength) {
    return awkward_regulararray_getitem_jagged_expand_64(
      multistarts,
      multistops,
      singleoffsets,
      regularsize,
      regularlength);
  }

  template <>
  Error listarray_getitem_jagged_expand_64<int32_t>(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t* tocarry,
    const int32_t* fromstarts,
    int64_t fromstartsoffset,
    const int32_t* fromstops,
    int64_t fromstopsoffset,
    int64_t jaggedsize,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarray32_getitem_jagged_expand_64(
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
  Error listarray_getitem_jagged_expand_64(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t* tocarry,
    const uint32_t* fromstarts,
    int64_t fromstartsoffset,
    const uint32_t* fromstops,
    int64_t fromstopsoffset,
    int64_t jaggedsize,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_getitem_jagged_expand_64(
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
  Error listarray_getitem_jagged_expand_64(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t* tocarry,
    const int64_t* fromstarts,
    int64_t fromstartsoffset,
    const int64_t* fromstops,
    int64_t fromstopsoffset,
    int64_t jaggedsize,
    int64_t length,
    KernelsLib ptr_lib) {
    return awkward_listarray64_getitem_jagged_expand_64(
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
  ERROR listarray_getitem_jagged_carrylen(
    int64_t* carrylen,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t sliceouterlen) {
    return awkward_listarray_getitem_jagged_carrylen_64(
      carrylen,
      slicestarts,
      slicestartsoffset,
      slicestops,
      slicestopsoffset,
      sliceouterlen);
  }

  template <>
  Error listarray_getitem_jagged_apply_64<int32_t>(
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
    int64_t contentlen,
    KernelsLib ptr_lib) {
    return awkward_listarray32_getitem_jagged_apply_64(
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
  Error listarray_getitem_jagged_apply_64<uint32_t>(
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
    int64_t contentlen,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_getitem_jagged_apply_64(
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
  Error listarray_getitem_jagged_apply_64<int64_t>(
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
    int64_t contentlen,
    KernelsLib ptr_lib) {
    return awkward_listarray64_getitem_jagged_apply_64(
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
  ERROR listarray_getitem_jagged_numvalid(
    int64_t* numvalid,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t length,
    const int64_t* missing,
    int64_t missingoffset,
    int64_t missinglength) {
    return awkward_listarray_getitem_jagged_numvalid_64(
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

  template <>
  ERROR listarray_getitem_jagged_shrink(
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
    return awkward_listarray_getitem_jagged_shrink_64(
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
  Error listarray_getitem_jagged_descend_64<int32_t>(
    int64_t* tooffsets,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t sliceouterlen,
    const int32_t* fromstarts,
    int64_t fromstartsoffset,
    const int32_t* fromstops,
    int64_t fromstopsoffset,
    KernelsLib ptr_lib) {
    return awkward_listarray32_getitem_jagged_descend_64(
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
  Error listarray_getitem_jagged_descend_64<uint32_t>(
    int64_t* tooffsets,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t sliceouterlen,
    const uint32_t* fromstarts,
    int64_t fromstartsoffset,
    const uint32_t* fromstops,
    int64_t fromstopsoffset,
    KernelsLib ptr_lib) {
    return awkward_listarrayU32_getitem_jagged_descend_64(
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
  Error listarray_getitem_jagged_descend_64<int64_t>(
    int64_t* tooffsets,
    const int64_t* slicestarts,
    int64_t slicestartsoffset,
    const int64_t* slicestops,
    int64_t slicestopsoffset,
    int64_t sliceouterlen,
    const int64_t* fromstarts,
    int64_t fromstartsoffset,
    const int64_t* fromstops,
    int64_t fromstopsoffset,
    KernelsLib ptr_lib) {
    return awkward_listarray64_getitem_jagged_descend_64(
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
    int64_t at,
    KernelsLib ptr_lib) {
    return awkward_index8_getitem_at_nowrap(
      ptr,
      offset,
      at);
  }
  template <>
  uint8_t index_getitem_at_nowrap<uint8_t>(
    const uint8_t* ptr,
    int64_t offset,
    int64_t at,
    KernelsLib ptr_lib) {
    return awkward_indexU8_getitem_at_nowrap(
      ptr,
      offset,
      at);
  }
  template <>
  int32_t index_getitem_at_nowrap<int32_t>(
    const int32_t* ptr,
    int64_t offset,
    int64_t at,
    KernelsLib ptr_lib) {
    return awkward_index32_getitem_at_nowrap(
      ptr,
      offset,
      at);
  }
  template <>
  uint32_t index_getitem_at_nowrap<uint32_t>(
    const uint32_t* ptr,
    int64_t offset,
    int64_t at,
    KernelsLib ptr_lib) {
    return awkward_indexU32_getitem_at_nowrap(
      ptr,
      offset,
      at);
  }
  template <>
  int64_t index_getitem_at_nowrap<int64_t>(
    const int64_t* ptr,
    int64_t offset,
    int64_t at,
    KernelsLib ptr_lib) {
    return awkward_index64_getitem_at_nowrap(
      ptr,
      offset,
      at);
  }

  template <>
  void  index_setitem_at_nowrap<int8_t>(
    int8_t* ptr,
    int64_t offset,
    int64_t at,
    int8_t value,
    KernelsLib ptr_lib) {
    awkward_index8_setitem_at_nowrap(
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
    uint8_t value,
    KernelsLib ptr_lib) {
    awkward_indexU8_setitem_at_nowrap(
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
    int32_t value,
    KernelsLib ptr_lib) {
    awkward_index32_setitem_at_nowrap(
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
    uint32_t value,
    KernelsLib ptr_lib) {
    awkward_indexU32_setitem_at_nowrap(
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
    int64_t value,
    KernelsLib ptr_lib) {
    awkward_index64_setitem_at_nowrap(
      ptr,
      offset,
      at,
      value);
  }

  template <>
  ERROR bytemaskedarray_getitem_carry(
    int8_t* tomask,
    const int8_t* frommask,
    int64_t frommaskoffset,
    int64_t lenmask,
    const int64_t* fromcarry,
    int64_t lencarry) {
    return awkward_bytemaskedarray_getitem_carry_64(
      tomask,
      frommask,
      frommaskoffset,
      lenmask,
      fromcarry,
      lencarry);
  }

  ERROR bytemaskedarray_numnull(
    int64_t* numnull,
    const int8_t* mask,
    int64_t maskoffset,
    int64_t length,
    bool validwhen) {
    return awkward_bytemaskedarray_numnull(
      numnull,
      mask,
      maskoffset,
      length,
      validwhen);
  }

  template <>
  ERROR bytemaskedarray_getitem_nextcarry(
    int64_t* tocarry,
    const int8_t* mask,
    int64_t maskoffset,
    int64_t length,
    bool validwhen) {
    return awkward_bytemaskedarray_getitem_nextcarry_64(
      tocarry,
      mask,
      maskoffset,
      length,
      validwhen);
  }

  template <>
  ERROR bytemaskedarray_getitem_nextcarry_outindex(
    int64_t* tocarry,
    int64_t* toindex,
    const int8_t* mask,
    int64_t maskoffset,
    int64_t length,
    bool validwhen) {
    return awkward_bytemaskedarray_getitem_nextcarry_outindex_64(
      tocarry,
      toindex,
      mask,
      maskoffset,
      length,
      validwhen);
  }

  template <>
  ERROR bytemaskedarray_toindexedarray(
    int64_t* toindex,
    const int8_t* mask,
    int64_t maskoffset,
    int64_t length,
    bool validwhen) {
    return awkward_bytemaskedarray_toindexedarray_64(
      toindex,
      mask,
      maskoffset,
      length,
      validwhen);
  }
}
