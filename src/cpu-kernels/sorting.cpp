// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "awkward/cpu-kernels/sorting.h"

ERROR awkward_sorting_ranges(
  int64_t* toindex,
  int64_t tolength,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  int64_t outlength) {
  std::vector<int64_t> ranges(parentslength + 1);
  for (int64_t i = 0; i < parentslength; i++) {
    ranges[i] = parents[i];
  }
  ranges[parentslength] = outlength;

  std::vector<int64_t> result;
  for (auto const& it : ranges) {
    auto res = std::find(std::begin(ranges), std::end(ranges), it);
    if (res != std::end(ranges)) {
      if (result.empty() || result.back() != std::distance(std::begin(ranges), res)) {
        result.emplace_back(std::distance(std::begin(ranges), res));
      }
    }
  }
  for (int64_t i = 0; i < tolength; i++) {
    toindex[i] = result[i];
  }
  return success();
}

ERROR awkward_sorting_ranges_length(
  int64_t* tolength,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  int64_t outlength) {
  std::vector<int64_t> ranges(parentslength + 1);
  for (int64_t i = 0; i < parentslength; i++) {
    ranges[i] = parents[i];
  }
  ranges[parentslength] = outlength;

  int64_t length = 0;
  std::vector<int64_t> result;
  for (auto const& it : ranges) {
    auto res = std::find(std::begin(ranges), std::end(ranges), it);
    if (res != std::end(ranges)) {
      if (result.empty() || result.back() != std::distance(std::begin(ranges), res)) {
        result.emplace_back(std::distance(std::begin(ranges), res));
        length++;
      }
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

  for (int64_t i = 0; i < offsetslength - 1; i++) {
    auto start = std::next(result.begin(), offsets[i]);
    auto stop = std::next(result.begin(), offsets[i + 1]);

    if (ascending  &&  !stable) {
      std::sort(start, stop,
        [&fromptr](int64_t i1, int64_t i2) {return fromptr[i1] < fromptr[i2];});
    }
    else if (!ascending  &&  !stable) {
      std::sort(start, stop,
        [&fromptr](int64_t i1, int64_t i2) {return fromptr[i1] > fromptr[i2];});
    }
    else if (ascending  &&  stable) {
      std::stable_sort(start, stop,
        [&fromptr](int64_t i1, int64_t i2) {return fromptr[i1] < fromptr[i2];});
    }
    else if (!ascending  &&  stable) {
      std::stable_sort(start, stop,
        [&fromptr](int64_t i1, int64_t i2) {return fromptr[i1] > fromptr[i2];});
    }
    std::transform(start, stop, start,
               [&](int64_t j) -> int64_t { return j - offsets[i]; });
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
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  std::vector<int64_t> index(length);
  std::iota(index.begin(), index.end(), 0);
  for (int64_t i = 0; i < offsetslength - 1; i++) {
    auto start = std::next(index.begin(), offsets[i]);
    auto stop = std::next(index.begin(), offsets[i + 1]);

    if (ascending  &&  !stable) {
      std::sort(start, stop,
        [&fromptr](int64_t i1, int64_t i2) {return fromptr[i1] < fromptr[i2];});
    }
    else if (!ascending  &&  !stable) {
      std::sort(start, stop,
        [&fromptr](int64_t i1, int64_t i2) {return fromptr[i1] > fromptr[i2];});
    }
    else if (ascending  &&  stable) {
      std::stable_sort(start, stop,
        [&fromptr](int64_t i1, int64_t i2) {return fromptr[i1] < fromptr[i2];});
    }
    else if (!ascending  &&  stable) {
      std::stable_sort(start, stop,
        [&fromptr](int64_t i1, int64_t i2) {return fromptr[i1] > fromptr[i2];});
    }
  }

  for (int64_t i = 0;  i < parentslength;  i++) {
    int64_t parent = parents[parentsoffset + i];
    int64_t start = starts[parent];

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
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<bool>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    starts,
    parents,
    parentsoffset,
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
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    starts,
    parents,
    parentsoffset,
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
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    starts,
    parents,
    parentsoffset,
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
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    starts,
    parents,
    parentsoffset,
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
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    starts,
    parents,
    parentsoffset,
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
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    starts,
    parents,
    parentsoffset,
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
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    starts,
    parents,
    parentsoffset,
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
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    starts,
    parents,
    parentsoffset,
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
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    starts,
    parents,
    parentsoffset,
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
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<float>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    starts,
    parents,
    parentsoffset,
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
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<double>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    starts,
    parents,
    parentsoffset,
    parentslength,
    ascending,
    stable);
}

ERROR awkward_listoffsetarray_local_preparenext_64(
  int64_t* outcarry,
  const int64_t* incarry,
  int64_t nextlen) {
  std::vector<int64_t> result(nextlen);
  std::iota(result.begin(), result.end(), 0);
  std::sort(result.begin(), result.end(),
    [&incarry](int64_t i1, int64_t i2) {return incarry[i1] < incarry[i2];});

  for(int64_t i = 0; i < nextlen; i++) {
    outcarry[i] = result[i];
  }
  return success();
}

ERROR awkward_indexedarray_local_preparenext_64(
    int64_t* nextoutindex,
    const int64_t* starts,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t parentslength,
    const int64_t* nextparents,
    int64_t nextparentsoffset) {
  int64_t j = 0;
  for (int64_t i = 0; i < parentslength; i++) {
    int64_t parent = parents[i] + parentsoffset;
    int64_t start = starts[parent];
    int64_t nextparent = nextparents[j] + nextparentsoffset;
    if (parent == nextparent) {
      nextoutindex[i] = j;
      ++j;
    }
    else {
      nextoutindex[i] = -1;
    }
  }
  return success();
}
