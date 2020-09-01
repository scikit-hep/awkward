// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/sorting.cpp", line)

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "awkward/kernels/sorting.h"

ERROR awkward_sorting_ranges(
  int64_t* toindex,
  int64_t tolength,
  const int64_t* parents,
  int64_t parentslength) {
  int64_t j = 0;
  int64_t k = 0;
  toindex[0] = k;
  k++; j++;
  for (int64_t i = 1;  i < parentslength;  i++) {
    if (parents[i - 1] != parents[i]) {
      toindex[j] = k;
      j++;
    }
    k++;
  }
  toindex[tolength - 1] = parentslength;
  return success();
}

ERROR awkward_sorting_ranges_length(
  int64_t* tolength,
  const int64_t* parents,
  int64_t parentslength) {
  int64_t length = 2;
  for (int64_t i = 1;  i < parentslength;  i++) {
    if (parents[i - 1] != parents[i]) {
      length++;
    }
  }
  *tolength = length;
  return success();
}

template <typename T>
ERROR awkward_argsort(
  int64_t* toptr,
  const T* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  std::vector<int64_t> result(length);
  std::iota(result.begin(), result.end(), 0);

  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    auto start = std::next(result.begin(), offsets[i]);
    auto stop = std::next(result.begin(), offsets[i + 1]);

    if (ascending  &&  stable) {
      std::stable_sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return fromptr[i1] < fromptr[i2];
      });
    }
    else if (!ascending  &&  stable) {
      std::stable_sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return fromptr[i1] > fromptr[i2];
      });
    }
    else if (ascending  &&  !stable) {
      std::sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return fromptr[i1] < fromptr[i2];
      });
    }
    else {
      std::sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return fromptr[i1] > fromptr[i2];
      });
    }

    std::transform(start, stop, start, [&](int64_t j) -> int64_t {
      return j - offsets[i];
    });
  }

  for (int64_t i = 0;  i < length;  i++) {
    toptr[i] = result[i];
  }
  return success();
}

ERROR awkward_argsort_bool(
  int64_t* toptr,
  const bool* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<bool>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_int8(
  int64_t* toptr,
  const int8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<int8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_uint8(
  int64_t* toptr,
  const uint8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<uint8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_int16(
  int64_t* toptr,
  const int16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<int16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_uint16(
  int64_t* toptr,
  const uint16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<uint16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_int32(
  int64_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<int32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_uint32(
  int64_t* toptr,
  const uint32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<uint32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_int64(
  int64_t* toptr,
  const int64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<int64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_uint64(
  int64_t* toptr,
  const uint64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<uint64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_float32(
  int64_t* toptr,
  const float* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<float>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_float64(
  int64_t* toptr,
  const double* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<double>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

template <typename T>
ERROR awkward_sort(
  T* toptr,
  const T* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  std::vector<int64_t> index(length);
  std::iota(index.begin(), index.end(), 0);

  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    auto start = std::next(index.begin(), offsets[i]);
    auto stop = std::next(index.begin(), offsets[i + 1]);

    if (ascending  &&  stable) {
      std::stable_sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return fromptr[i1] < fromptr[i2];
      });
    }
    else if (!ascending  &&  stable) {
      std::stable_sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return fromptr[i1] > fromptr[i2];
      });
    }
    else if (ascending  &&  !stable) {
      std::sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return fromptr[i1] < fromptr[i2];
      });
    }
    else {
      std::sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return fromptr[i1] > fromptr[i2];
      });
    }
  }

  for (int64_t i = 0;  i < parentslength;  i++) {
    toptr[i] = fromptr[index[i]];
  }
  return success();
}

ERROR awkward_sort_bool(
  bool* toptr,
  const bool* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<bool>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}

ERROR awkward_sort_int8(
  int8_t* toptr,
  const int8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}

ERROR awkward_sort_uint8(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}

ERROR awkward_sort_int16(
  int16_t* toptr,
  const int16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}

ERROR awkward_sort_uint16(
  uint16_t* toptr,
  const uint16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}

ERROR awkward_sort_int32(
  int32_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}

ERROR awkward_sort_uint32(
  uint32_t* toptr,
  const uint32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}

ERROR awkward_sort_int64(
  int64_t* toptr,
  const int64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}

ERROR awkward_sort_uint64(
  uint64_t* toptr,
  const uint64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}

ERROR awkward_sort_float32(
  float* toptr,
  const float* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<float>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}

ERROR awkward_sort_float64(
  double* toptr,
  const double* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<double>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}

ERROR awkward_ListOffsetArray_local_preparenext_64(
  int64_t* tocarry,
  const int64_t* fromindex,
  int64_t length) {
  std::vector<int64_t> result(length);
  std::iota(result.begin(), result.end(), 0);
  std::sort(result.begin(), result.end(),
    [&fromindex](int64_t i1, int64_t i2) {
      return fromindex[i1] < fromindex[i2];
    });

  for(int64_t i = 0; i < length; i++) {
    tocarry[i] = result[i];
  }
  return success();
}

ERROR awkward_IndexedArray_local_preparenext_64(
    int64_t* tocarry,
    const int64_t* starts,
    const int64_t* parents,
    int64_t parentslength,
    const int64_t* nextparents) {
  int64_t j = 0;
  for (int64_t i = 0;  i < parentslength;  i++) {
    int64_t parent = parents[i];
    int64_t start = starts[parent];
    int64_t nextparent = nextparents[j];
    if (parent == nextparent) {
      tocarry[i] = j;
      ++j;
    }
    else {
      tocarry[i] = -1;
    }
  }
  return success();
}

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
