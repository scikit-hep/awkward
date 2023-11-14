// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_argsort_strings.cpp", line)

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "awkward/kernels.h"

template <bool is_stable, bool is_ascending, bool is_local>
ERROR awkward_ListOffsetArray_argsort_strings_impl(
  int64_t* tocarry,
  const int64_t* fromparents,
  int64_t length,
  const char* stringdata,
  const int64_t* stringstarts,
  const int64_t* stringstops) {

  auto sorter =
        [&stringdata, &stringstarts, &stringstops](int64_t left, int64_t right) -> bool {
          size_t left_n = stringstops[left] - stringstarts[left];
          size_t right_n = stringstops[right] - stringstarts[right];
          const char* left_str = &stringdata[stringstarts[left]];
          const char* right_str = &stringdata[stringstarts[right]];
          int cmp = strncmp(left_str, right_str, std::min(left_n, right_n));
          bool out;
          if (cmp == 0) {
            out = left_n < right_n;
          }
          else {
            out = cmp < 0;
          }
          if (is_ascending) {
            return out;
          }
          else {
            return !out;
          }
        };

  int64_t firstindex = 0;
  int64_t lastparent = -1;
  std::vector<int64_t> index;
  for (int64_t i = 0;  i < length + 1;  i++) {
    if (i == length  ||  fromparents[i] != lastparent) {
      if (is_stable) {
        std::stable_sort(index.begin(), index.end(), sorter);
      }
      else {
        std::sort(index.begin(), index.end(), sorter);
      }
      for (int64_t j = 0;  j < (int64_t)index.size();  j++) {
        if (is_local) {
          tocarry[firstindex + j] = index[j] - firstindex;
        }
        else {
          tocarry[firstindex + j] = index[j];
        }
      }
      index.clear();
    }

    if (i != length) {
      if (index.empty()) {
        firstindex = i;
      }
      index.push_back(i);
      lastparent = fromparents[i];
    }
  }

  return success();
}

ERROR awkward_ListOffsetArray_argsort_strings(
  int64_t* tocarry,
  const int64_t* fromparents,
  int64_t length,
  const uint8_t* stringdata,
  const int64_t* stringstarts,
  const int64_t* stringstops,
  bool is_stable,
  bool is_ascending,
  bool is_local) {
  if (is_stable) {
    if (is_ascending) {
      if (is_local) {
        return awkward_ListOffsetArray_argsort_strings_impl<true, true, true>(
          tocarry, fromparents, length, (char*)stringdata, stringstarts, stringstops);
      }
      else {
        return awkward_ListOffsetArray_argsort_strings_impl<true, true, false>(
          tocarry, fromparents, length, (char*)stringdata, stringstarts, stringstops);
      }
    }
    else {
      if (is_local) {
        return awkward_ListOffsetArray_argsort_strings_impl<true, false, true>(
          tocarry, fromparents, length, (char*)stringdata, stringstarts, stringstops);
      }
      else {
        return awkward_ListOffsetArray_argsort_strings_impl<true, false, false>(
          tocarry, fromparents, length, (char*)stringdata, stringstarts, stringstops);
      }
    }
  }
  else {
    if (is_ascending) {
      if (is_local) {
        return awkward_ListOffsetArray_argsort_strings_impl<false, true, true>(
          tocarry, fromparents, length, (char*)stringdata, stringstarts, stringstops);
      }
      else {
        return awkward_ListOffsetArray_argsort_strings_impl<false, true, false>(
          tocarry, fromparents, length, (char*)stringdata, stringstarts, stringstops);
      }
    }
    else {
      if (is_local) {
        return awkward_ListOffsetArray_argsort_strings_impl<false, false, true>(
          tocarry, fromparents, length, (char*)stringdata, stringstarts, stringstops);
      }
      else {
        return awkward_ListOffsetArray_argsort_strings_impl<false, false, false>(
          tocarry, fromparents, length, (char*)stringdata, stringstarts, stringstops);
      }
    }
  }
}
