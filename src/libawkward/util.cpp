// BSD 3-Clause License; see
// https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cassert>
#include <sstream>

#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/identity.h"

#include "awkward/Identity.h"
#include "awkward/util.h"

namespace awkward {
namespace util {
void handle_error(const struct Error &err, const std::string classname,
                  const Identity *id) {
  if (err.str != nullptr) {
    std::stringstream out;
    out << "in " << classname;
    if (err.location != kSliceNone && id != nullptr) {
      assert(err.location > 0);
      if (0 <= err.location && err.location < id->length()) {
        out << " at id[" << id->location_at(err.location) << "]";
      } else {
        out << " at id[???]";
      }
    }
    if (err.attempt != kSliceNone) {
      out << " attempting to get " << err.attempt;
    }
    out << ", " << err.str;
    throw std::invalid_argument(out.str());
  }
}

std::string quote(std::string x, bool doublequote) {
  // TODO: escape characters, possibly using RapidJSON.
  if (doublequote) {
    return std::string("\"") + x + std::string("\"");
  } else {
    return std::string("'") + x + std::string("'");
  }
}

bool subset(const std::vector<std::string> &super,
            const std::vector<std::string> &sub) {
  if (super.size() < sub.size()) {
    return false;
  }
  for (auto x : sub) {
    bool found = false;
    for (auto y : super) {
      if (x == y) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

template <>
Error awkward_identity64_from_listoffsetarray<uint32_t>(
    int64_t *toptr, const int64_t *fromptr, const uint32_t *fromoffsets,
    int64_t fromptroffset, int64_t offsetsoffset, int64_t tolength,
    int64_t fromlength, int64_t fromwidth) {
  return awkward_identity64_from_listoffsetarrayU32(
      toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength,
      fromlength, fromwidth);
}
template <>
Error awkward_identity64_from_listoffsetarray<int64_t>(
    int64_t *toptr, const int64_t *fromptr, const int64_t *fromoffsets,
    int64_t fromptroffset, int64_t offsetsoffset, int64_t tolength,
    int64_t fromlength, int64_t fromwidth) {
  return awkward_identity64_from_listoffsetarray64(
      toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength,
      fromlength, fromwidth);
}

template <>
Error awkward_identity64_from_listarray<uint32_t>(
    int64_t *toptr, const int64_t *fromptr, const uint32_t *fromstarts,
    const uint32_t *fromstops, int64_t fromptroffset, int64_t startsoffset,
    int64_t stopsoffset, int64_t tolength, int64_t fromlength,
    int64_t fromwidth) {
  return awkward_identity64_from_listarrayU32(
      toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset,
      stopsoffset, tolength, fromlength, fromwidth);
}
template <>
Error awkward_identity64_from_listarray<int64_t>(
    int64_t *toptr, const int64_t *fromptr, const int64_t *fromstarts,
    const int64_t *fromstops, int64_t fromptroffset, int64_t startsoffset,
    int64_t stopsoffset, int64_t tolength, int64_t fromlength,
    int64_t fromwidth) {
  return awkward_identity64_from_listarray64(
      toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset,
      stopsoffset, tolength, fromlength, fromwidth);
}

template <>
Error awkward_listarray_getitem_next_at_64<int32_t>(
    int64_t *tocarry, const int32_t *fromstarts, const int32_t *fromstops,
    int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at) {
  return awkward_listarray32_getitem_next_at_64(
      tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at);
}
template <>
Error awkward_listarray_getitem_next_at_64<uint32_t>(
    int64_t *tocarry, const uint32_t *fromstarts, const uint32_t *fromstops,
    int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at) {
  return awkward_listarrayU32_getitem_next_at_64(
      tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at);
}
template <>
Error awkward_listarray_getitem_next_at_64<int64_t>(
    int64_t *tocarry, const int64_t *fromstarts, const int64_t *fromstops,
    int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at) {
  return awkward_listarray64_getitem_next_at_64(
      tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at);
}

template <>
Error awkward_listarray_getitem_next_range_carrylength<int32_t>(
    int64_t *carrylength, const int32_t *fromstarts, const int32_t *fromstops,
    int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start,
    int64_t stop, int64_t step) {
  return awkward_listarray32_getitem_next_range_carrylength(
      carrylength, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset,
      start, stop, step);
}
template <>
Error awkward_listarray_getitem_next_range_carrylength<uint32_t>(
    int64_t *carrylength, const uint32_t *fromstarts, const uint32_t *fromstops,
    int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start,
    int64_t stop, int64_t step) {
  return awkward_listarrayU32_getitem_next_range_carrylength(
      carrylength, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset,
      start, stop, step);
}
template <>
Error awkward_listarray_getitem_next_range_carrylength<int64_t>(
    int64_t *carrylength, const int64_t *fromstarts, const int64_t *fromstops,
    int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start,
    int64_t stop, int64_t step) {
  return awkward_listarray64_getitem_next_range_carrylength(
      carrylength, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset,
      start, stop, step);
}

template <>
Error awkward_listarray_getitem_next_range_64<int32_t>(
    int32_t *tooffsets, int64_t *tocarry, const int32_t *fromstarts,
    const int32_t *fromstops, int64_t lenstarts, int64_t startsoffset,
    int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  return awkward_listarray32_getitem_next_range_64(
      tooffsets, tocarry, fromstarts, fromstops, lenstarts, startsoffset,
      stopsoffset, start, stop, step);
}
template <>
Error awkward_listarray_getitem_next_range_64<uint32_t>(
    uint32_t *tooffsets, int64_t *tocarry, const uint32_t *fromstarts,
    const uint32_t *fromstops, int64_t lenstarts, int64_t startsoffset,
    int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  return awkward_listarrayU32_getitem_next_range_64(
      tooffsets, tocarry, fromstarts, fromstops, lenstarts, startsoffset,
      stopsoffset, start, stop, step);
}
template <>
Error awkward_listarray_getitem_next_range_64<int64_t>(
    int64_t *tooffsets, int64_t *tocarry, const int64_t *fromstarts,
    const int64_t *fromstops, int64_t lenstarts, int64_t startsoffset,
    int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  return awkward_listarray64_getitem_next_range_64(
      tooffsets, tocarry, fromstarts, fromstops, lenstarts, startsoffset,
      stopsoffset, start, stop, step);
}

template <>
Error awkward_listarray_getitem_next_range_counts_64<int32_t>(
    int64_t *total, const int32_t *fromoffsets, int64_t lenstarts) {
  return awkward_listarray32_getitem_next_range_counts_64(total, fromoffsets,
                                                          lenstarts);
}
template <>
Error awkward_listarray_getitem_next_range_counts_64<uint32_t>(
    int64_t *total, const uint32_t *fromoffsets, int64_t lenstarts) {
  return awkward_listarrayU32_getitem_next_range_counts_64(total, fromoffsets,
                                                           lenstarts);
}
template <>
Error awkward_listarray_getitem_next_range_counts_64<int64_t>(
    int64_t *total, const int64_t *fromoffsets, int64_t lenstarts) {
  return awkward_listarray64_getitem_next_range_counts_64(total, fromoffsets,
                                                          lenstarts);
}

template <>
Error awkward_listarray_getitem_next_range_spreadadvanced_64<int32_t>(
    int64_t *toadvanced, const int64_t *fromadvanced,
    const int32_t *fromoffsets, int64_t lenstarts) {
  return awkward_listarray32_getitem_next_range_spreadadvanced_64(
      toadvanced, fromadvanced, fromoffsets, lenstarts);
}
template <>
Error awkward_listarray_getitem_next_range_spreadadvanced_64<uint32_t>(
    int64_t *toadvanced, const int64_t *fromadvanced,
    const uint32_t *fromoffsets, int64_t lenstarts) {
  return awkward_listarrayU32_getitem_next_range_spreadadvanced_64(
      toadvanced, fromadvanced, fromoffsets, lenstarts);
}
template <>
Error awkward_listarray_getitem_next_range_spreadadvanced_64<int64_t>(
    int64_t *toadvanced, const int64_t *fromadvanced,
    const int64_t *fromoffsets, int64_t lenstarts) {
  return awkward_listarray64_getitem_next_range_spreadadvanced_64(
      toadvanced, fromadvanced, fromoffsets, lenstarts);
}

template <>
Error awkward_listarray_getitem_next_array_64<int32_t>(
    int64_t *tocarry, int64_t *toadvanced, const int32_t *fromstarts,
    const int32_t *fromstops, const int64_t *fromarray, int64_t startsoffset,
    int64_t stopsoffset, int64_t lenstarts, int64_t lenarray,
    int64_t lencontent) {
  return awkward_listarray32_getitem_next_array_64(
      tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset,
      stopsoffset, lenstarts, lenarray, lencontent);
}
template <>
Error awkward_listarray_getitem_next_array_64<uint32_t>(
    int64_t *tocarry, int64_t *toadvanced, const uint32_t *fromstarts,
    const uint32_t *fromstops, const int64_t *fromarray, int64_t startsoffset,
    int64_t stopsoffset, int64_t lenstarts, int64_t lenarray,
    int64_t lencontent) {
  return awkward_listarrayU32_getitem_next_array_64(
      tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset,
      stopsoffset, lenstarts, lenarray, lencontent);
}
template <>
Error awkward_listarray_getitem_next_array_64<int64_t>(
    int64_t *tocarry, int64_t *toadvanced, const int64_t *fromstarts,
    const int64_t *fromstops, const int64_t *fromarray, int64_t startsoffset,
    int64_t stopsoffset, int64_t lenstarts, int64_t lenarray,
    int64_t lencontent) {
  return awkward_listarray64_getitem_next_array_64(
      tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset,
      stopsoffset, lenstarts, lenarray, lencontent);
}

template <>
Error awkward_listarray_getitem_next_array_advanced_64<int32_t>(
    int64_t *tocarry, int64_t *toadvanced, const int32_t *fromstarts,
    const int32_t *fromstops, const int64_t *fromarray,
    const int64_t *fromadvanced, int64_t startsoffset, int64_t stopsoffset,
    int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray32_getitem_next_array_advanced_64(
      tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced,
      startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}
template <>
Error awkward_listarray_getitem_next_array_advanced_64<uint32_t>(
    int64_t *tocarry, int64_t *toadvanced, const uint32_t *fromstarts,
    const uint32_t *fromstops, const int64_t *fromarray,
    const int64_t *fromadvanced, int64_t startsoffset, int64_t stopsoffset,
    int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarrayU32_getitem_next_array_advanced_64(
      tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced,
      startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}
template <>
Error awkward_listarray_getitem_next_array_advanced_64<int64_t>(
    int64_t *tocarry, int64_t *toadvanced, const int64_t *fromstarts,
    const int64_t *fromstops, const int64_t *fromarray,
    const int64_t *fromadvanced, int64_t startsoffset, int64_t stopsoffset,
    int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray64_getitem_next_array_advanced_64(
      tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced,
      startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}

template <>
Error awkward_listarray_getitem_carry_64<int32_t>(
    int32_t *tostarts, int32_t *tostops, const int32_t *fromstarts,
    const int32_t *fromstops, const int64_t *fromcarry, int64_t startsoffset,
    int64_t stopsoffset, int64_t lenstarts, int64_t lencarry) {
  return awkward_listarray32_getitem_carry_64(
      tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset,
      stopsoffset, lenstarts, lencarry);
}
template <>
Error awkward_listarray_getitem_carry_64<uint32_t>(
    uint32_t *tostarts, uint32_t *tostops, const uint32_t *fromstarts,
    const uint32_t *fromstops, const int64_t *fromcarry, int64_t startsoffset,
    int64_t stopsoffset, int64_t lenstarts, int64_t lencarry) {
  return awkward_listarrayU32_getitem_carry_64(
      tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset,
      stopsoffset, lenstarts, lencarry);
}
template <>
Error awkward_listarray_getitem_carry_64<int64_t>(
    int64_t *tostarts, int64_t *tostops, const int64_t *fromstarts,
    const int64_t *fromstops, const int64_t *fromcarry, int64_t startsoffset,
    int64_t stopsoffset, int64_t lenstarts, int64_t lencarry) {
  return awkward_listarray64_getitem_carry_64(
      tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset,
      stopsoffset, lenstarts, lencarry);
}

} // namespace util
} // namespace awkward
