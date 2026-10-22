// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_argsort_strings.cpp", line)

#include <algorithm>
#include <cstring>
#include <numeric>

#include "awkward/kernels.h"

template <bool is_stable, bool is_ascending, bool is_local>
ERROR awkward_ListOffsetArray_argsort_strings_impl(
  int64_t* __restrict__ tocarry,
  const int64_t* __restrict__ offsets,
  int64_t outlength,
  const char* __restrict__ stringdata,
  const int64_t* __restrict__ stringstarts,
  const int64_t* __restrict__ stringstops) {

  auto sorter =
        [&stringdata, &stringstarts, &stringstops](int64_t left, int64_t right) -> bool {
          size_t left_n = stringstops[left] - stringstarts[left];
          size_t right_n = stringstops[right] - stringstarts[right];
          const char* left_str = &stringdata[stringstarts[left]];
          const char* right_str = &stringdata[stringstarts[right]];
          int cmp = memcmp(left_str, right_str, std::min(left_n, right_n));
          if (cmp == 0) {
            return is_ascending ? left_n < right_n : left_n > right_n;
          }
          else {
            return is_ascending ? cmp < 0 : cmp > 0;
          }
        };

  for (int64_t i = 0; i < outlength; i++) {
    int64_t start = offsets[i];
    int64_t stop = offsets[i + 1];
    if (start == stop) {
      continue;
    }
    int64_t* segment_start = tocarry + start;
    int64_t* segment_stop = tocarry + stop;
    std::iota(segment_start, segment_stop, start);
    if (is_stable) {
      std::stable_sort(segment_start, segment_stop, sorter);
    }
    else {
      std::sort(segment_start, segment_stop, sorter);
    }
    if (is_local) {
      for (int64_t j = start; j < stop; j++) {
        tocarry[j] -= start;
      }
    }
  }

  return success();
}

ERROR awkward_ListOffsetArray_argsort_strings(
  int64_t* __restrict__ tocarry,
  const int64_t* __restrict__ offsets,
  int64_t outlength,
  const uint8_t* __restrict__ stringdata,
  const int64_t* __restrict__ stringstarts,
  const int64_t* __restrict__ stringstops,
  bool is_stable,
  bool is_ascending,
  bool is_local) {
  if (is_stable) {
    if (is_ascending) {
      if (is_local) {
        return awkward_ListOffsetArray_argsort_strings_impl<true, true, true>(
          tocarry, offsets, outlength, (char*)stringdata, stringstarts, stringstops);
      }
      else {
        return awkward_ListOffsetArray_argsort_strings_impl<true, true, false>(
          tocarry, offsets, outlength, (char*)stringdata, stringstarts, stringstops);
      }
    }
    else {
      if (is_local) {
        return awkward_ListOffsetArray_argsort_strings_impl<true, false, true>(
          tocarry, offsets, outlength, (char*)stringdata, stringstarts, stringstops);
      }
      else {
        return awkward_ListOffsetArray_argsort_strings_impl<true, false, false>(
          tocarry, offsets, outlength, (char*)stringdata, stringstarts, stringstops);
      }
    }
  }
  else {
    if (is_ascending) {
      if (is_local) {
        return awkward_ListOffsetArray_argsort_strings_impl<false, true, true>(
          tocarry, offsets, outlength, (char*)stringdata, stringstarts, stringstops);
      }
      else {
        return awkward_ListOffsetArray_argsort_strings_impl<false, true, false>(
          tocarry, offsets, outlength, (char*)stringdata, stringstarts, stringstops);
      }
    }
    else {
      if (is_local) {
        return awkward_ListOffsetArray_argsort_strings_impl<false, false, true>(
          tocarry, offsets, outlength, (char*)stringdata, stringstarts, stringstops);
      }
      else {
        return awkward_ListOffsetArray_argsort_strings_impl<false, false, false>(
          tocarry, offsets, outlength, (char*)stringdata, stringstarts, stringstops);
      }
    }
  }
}
