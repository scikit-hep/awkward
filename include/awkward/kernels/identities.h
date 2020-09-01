// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNELS_IDENTITIES_H_
#define AWKWARD_KERNELS_IDENTITIES_H_

#include "awkward/common.h"

extern "C" {
/// @param toptr outparam
/// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_new_Identities32(
      int32_t* toptr,
      int64_t length);
/// @param toptr outparam
/// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_new_Identities64(
      int64_t* toptr,
      int64_t length);

/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param length inparam
/// @param width inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_to_Identities64(
      int64_t* toptr,
      const int32_t* fromptr,
      int64_t length,
      int64_t width);

/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromoffsets inparam role: ListOffsetArray-offsets
/// @param tolength inparam role: ListOffsetArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_ListOffsetArray32(
      int32_t* toptr,
      const int32_t* fromptr,
      const int32_t* fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromoffsets inparam role: ListOffsetArray-offsets
/// @param tolength inparam role: ListOffsetArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_ListOffsetArrayU32(
      int32_t* toptr,
      const int32_t* fromptr,
      const uint32_t* fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromoffsets inparam role: ListOffsetArray-offsets
/// @param tolength inparam role: ListOffsetArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_ListOffsetArray64(
      int32_t* toptr,
      const int32_t* fromptr,
      const int64_t* fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromoffsets inparam role: ListOffsetArray-offsets
/// @param tolength inparam role: ListOffsetArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_ListOffsetArray32(
      int64_t* toptr,
      const int64_t* fromptr,
      const int32_t* fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromoffsets inparam role: ListOffsetArray-offsets
/// @param tolength inparam role: ListOffsetArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_ListOffsetArrayU32(
      int64_t* toptr,
      const int64_t* fromptr,
      const uint32_t* fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromoffsets inparam role: ListOffsetArray-offsets
/// @param tolength inparam role: ListOffsetArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_ListOffsetArray64(
      int64_t* toptr,
      const int64_t* fromptr,
      const int64_t* fromoffsets,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);

/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromstarts inparam role: ListArray-starts
/// @param fromstops inparam role: ListArray-stops
/// @param tolength inparam role: ListArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_ListArray32(
      bool* uniquecontents,
      int32_t* toptr,
      const int32_t* fromptr,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromstarts inparam role: ListArray-starts
/// @param fromstops inparam role: ListArray-stops
/// @param tolength inparam role: ListArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_ListArrayU32(
      bool* uniquecontents,
      int32_t* toptr,
      const int32_t* fromptr,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromstarts inparam role: ListArray-starts
/// @param fromstops inparam role: ListArray-stops
/// @param tolength inparam role: ListArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_ListArray64(
      bool* uniquecontents,
      int32_t* toptr,
      const int32_t* fromptr,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromstarts inparam role: ListArray-starts
/// @param fromstops inparam role: ListArray-stops
/// @param tolength inparam role: ListArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_ListArray32(
      bool* uniquecontents,
      int64_t* toptr,
      const int64_t* fromptr,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromstarts inparam role: ListArray-starts
/// @param fromstops inparam role: ListArray-stops
/// @param tolength inparam role: ListArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_ListArrayU32(
      bool* uniquecontents,
      int64_t* toptr,
      const int64_t* fromptr,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromstarts inparam role: ListArray-starts
/// @param fromstops inparam role: ListArray-stops
/// @param tolength inparam role: ListArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_ListArray64(
      bool* uniquecontents,
      int64_t* toptr,
      const int64_t* fromptr,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);

/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param size inparam role: RegularArray-size
/// @param tolength inparam
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_RegularArray(
      int32_t* toptr,
      const int32_t* fromptr,
      int64_t size,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param size inparam role: RegularArray-size
/// @param tolength inparam
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_RegularArray(
      int64_t* toptr,
      const int64_t* fromptr,
      int64_t size,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);

/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromindex inparam role: IndexedArray-index
/// @param tolength inparam role: IndexedArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_IndexedArray32(
      bool* uniquecontents,
      int32_t* toptr,
      const int32_t* fromptr,
      const int32_t* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromindex inparam role: IndexedArray-index
/// @param tolength inparam role: IndexedArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_IndexedArrayU32(
      bool* uniquecontents,
      int32_t* toptr,
      const int32_t* fromptr,
      const uint32_t* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromindex inparam role: IndexedArray-index
/// @param tolength inparam role: IndexedArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_IndexedArray64(
      bool* uniquecontents,
      int32_t* toptr,
      const int32_t* fromptr,
      const int64_t* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromindex inparam role: IndexedArray-index
/// @param tolength inparam role: IndexedArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_IndexedArray32(
      bool* uniquecontents,
      int64_t* toptr,
      const int64_t* fromptr,
      const int32_t* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromindex inparam role: IndexedArray-index
/// @param tolength inparam role: IndexedArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_IndexedArrayU32(
      bool* uniquecontents,
      int64_t* toptr,
      const int64_t* fromptr,
      const uint32_t* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromindex inparam role: IndexedArray-index
/// @param tolength inparam role: IndexedArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_IndexedArray64(
      bool* uniquecontents,
      int64_t* toptr,
      const int64_t* fromptr,
      const int64_t* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth);

/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromtags inparam role: UnionArray-tags
/// @param fromindex inparam role: IndexedArray-index
/// @param tolength inparam role: IndexedArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
/// @param which inparam role: UnionArray-which
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_UnionArray8_32(
      bool* uniquecontents,
      int32_t* toptr,
      const int32_t* fromptr,
      const int8_t* fromtags,
      const int32_t* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromtags inparam role: UnionArray-tags
/// @param fromindex inparam role: IndexedArray-index
/// @param tolength inparam role: IndexedArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
/// @param which inparam role: UnionArray-which
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_UnionArray8_U32(
      bool* uniquecontents,
      int32_t* toptr,
      const int32_t* fromptr,
      const int8_t* fromtags,
      const uint32_t* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromtags inparam role: UnionArray-tags
/// @param fromindex inparam role: IndexedArray-index
/// @param tolength inparam role: IndexedArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
/// @param which inparam role: UnionArray-which
  EXPORT_SYMBOL struct Error
    awkward_Identities32_from_UnionArray8_64(
      bool* uniquecontents,
      int32_t* toptr,
      const int32_t* fromptr,
      const int8_t* fromtags,
      const int64_t* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromtags inparam role: UnionArray-tags
/// @param fromindex inparam role: IndexedArray-index
/// @param tolength inparam role: IndexedArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
/// @param which inparam role: UnionArray-which
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_UnionArray8_32(
      bool* uniquecontents,
      int64_t* toptr,
      const int64_t* fromptr,
      const int8_t* fromtags,
      const int32_t* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromtags inparam role: UnionArray-tags
/// @param fromindex inparam role: IndexedArray-index
/// @param tolength inparam role: IndexedArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
/// @param which inparam role: UnionArray-which
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_UnionArray8_U32(
      bool* uniquecontents,
      int64_t* toptr,
      const int64_t* fromptr,
      const int8_t* fromtags,
      const uint32_t* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which);
/// @param uniquecontents outparam role: pointer
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromtags inparam role: UnionArray-tags
/// @param fromindex inparam role: IndexedArray-index
/// @param tolength inparam role: IndexedArray-length
/// @param fromlength inparam
/// @param fromwidth inparam
/// @param which inparam role: UnionArray-which
  EXPORT_SYMBOL struct Error
    awkward_Identities64_from_UnionArray8_64(
      bool* uniquecontents,
      int64_t* toptr,
      const int64_t* fromptr,
      const int8_t* fromtags,
      const int64_t* fromindex,
      int64_t tolength,
      int64_t fromlength,
      int64_t fromwidth,
      int64_t which);

/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromlength inparam
/// @param tolength inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_extend(
      int32_t* toptr,
      const int32_t* fromptr,
      int64_t fromlength,
      int64_t tolength);
/// @param toptr outparam
/// @param fromptr inparam role: Identities-array
/// @param fromlength inparam
/// @param tolength inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities64_extend(
      int64_t* toptr,
      const int64_t* fromptr,
      int64_t fromlength,
      int64_t tolength);

}

#endif // AWKWARD_KERNELS_IDENTITIES_H_
