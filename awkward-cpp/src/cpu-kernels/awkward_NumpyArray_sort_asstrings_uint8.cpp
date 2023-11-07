// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_sort_asstrings_uint8.cpp", line)

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "awkward/kernels.h"

// This function relies on std::sort to do the right
// thing with std::strings
ERROR awkward_NumpyArray_sort_asstrings_uint8(
    uint8_t* toptr,
    const uint8_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    bool ascending,
    bool stable) {

  // convert array of characters to
  // an std container of strings
  std::vector<std::string> words;

  for (int64_t k = 0;  k < offsetslength - 1;  k++) {
    int64_t start = offsets[k];
    int64_t stop = offsets[k + 1];
    int64_t slen = start;
    std::string strvar;
    for (uint8_t i = (uint8_t)start;  slen < stop;  i++) {
      slen++;
      strvar += (char)fromptr[i];
    }
    words.emplace_back(strvar);
  }

  // sort the container
  if (ascending  &&  !stable) {
    std::sort(words.begin(), words.end(), std::less<std::string>());
  }
  else if (!ascending  &&  !stable) {
    std::sort(words.begin(), words.end(), std::greater<std::string>());
  }
  else if (ascending  &&  stable) {
    std::stable_sort(words.begin(), words.end(), std::less<std::string>());
  }
  else if (!ascending  &&  stable) {
    std::stable_sort(words.begin(), words.end(), std::greater<std::string>());
  }

  // convert the strings to an array of characters
  // and fill the outer memory via a pointer
  int64_t k = 0;
  for (const auto& strvar : words) {
    std::vector<char> cstr(strvar.c_str(), strvar.c_str() + strvar.size());
    for (const auto& c : cstr) {
      toptr[k] = (uint8_t)c;
      k++;
    }
  }

  // collect sorted string lengths
  // that are the new offsets for a ListOffsetArray
  int64_t o = 0;
  outoffsets[o] = (int64_t)0;
  o++;
  for (const auto& r : words) {
    outoffsets[o] = outoffsets[o - 1] + (int64_t)r.size();
    o++;
  }

  return success();
}
