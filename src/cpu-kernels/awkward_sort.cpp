// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_sort.cpp", line)

#include "awkward/kernels.h"

template <typename T>
int quick_sort_ascending(T *arr,
                         int64_t elements,
                         int64_t* beg,
                         int64_t* end,
                         int64_t maxlevels) {
  int64_t low = 0;
  int64_t high = 0;
  int64_t i = 0;
  beg[0] = 0;
  end[0] = elements;
  while (i >= 0) {
    low = beg[i];
    high = end[i];
    if (high - low > 1) {
      int64_t mid = low  + ((high - low) >> 1);
      T pivot = arr[mid];
      arr[mid] = arr[low];

      if (i == maxlevels - 1) {
        return -1;
      }
      high--;
      while (low < high) {
        while (arr[high] >= pivot  &&  low < high) {
          high--;
        }
        if (low < high) {
          arr[low++] = arr[high];
        }
        while (arr[low] <= pivot  &&  low < high) {
          low++;
        }
        if (low < high) {
          arr[high--] = arr[low];
        }
      }
      arr[low] = pivot;
      mid = low + 1;
      while (low > beg[i]  &&  arr[low - 1] == pivot) {
        low--;
      }
      while (mid < end[i]  &&  arr[mid] == pivot) {
        mid++;
      }
      if (low - beg[i] > end[i] - mid) {
        beg[i + 1] = mid;
        end[i + 1] = end[i];
        end[i++] = low;
      } else {
        beg[i + 1] = beg[i];
        end[i + 1] = low;
        beg[i++] = mid;
      }
    } else {
      i--;
    }
  }
  return 0;
}

template <typename T>
int quick_sort_descending(T *arr,
                          int64_t elements,
                          int64_t* beg,
                          int64_t* end,
                          int64_t maxlevels) {
  int64_t low = 0;
  int64_t high = 0;
  int64_t i = 0;
  beg[0] = 0;
  end[0] = elements;
  while (i >= 0) {
    low = beg[i];
    high = end[i];
    if (high - low > 1) {
      int64_t mid = low  + ((high - low) >> 1);
      T pivot = arr[mid];
      arr[mid] = arr[low];

      if (i == maxlevels - 1) {
        return -1;
      }
      high--;
      while (low < high) {
        while (arr[high] <= pivot  &&  low < high) {
          high--;
        }
        if (low < high) {
          arr[low++] = arr[high];
        }
        while (arr[low] >= pivot  &&  low < high) {
          low++;
        }
        if (low < high) {
          arr[high--] = arr[low];
        }
      }
      arr[low] = pivot;
      mid = low + 1;
      while (low > beg[i]  &&  arr[low - 1] == pivot) {
        low--;
      }
      while (mid < end[i]  &&  arr[mid] == pivot) {
        mid++;
      }
      if (low - beg[i] > end[i] - mid) {
        beg[i + 1] = mid;
        end[i + 1] = end[i];
        end[i++] = low;
      } else {
        beg[i + 1] = beg[i];
        end[i + 1] = low;
        beg[i++] = mid;
      }
    } else {
      i--;
    }
  }
  return 0;
}

template <typename T>
ERROR awkward_sort(
  T* toptr,
  const T* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  // FIXME: sort a copy or do it in place without copying?
  for (int64_t i = 0; i < length; i++) {
    toptr[i] = fromptr[i];
  }
  // FIXME: these buffers will go out of the kernel
  int64_t tmpbeg[kMaxLevels], tmpend[kMaxLevels];

  int result = 0;
  if (ascending) {
    for (int64_t i = 0; i < offsetslength - 1; i++) {
      int result = quick_sort_ascending(&(toptr[offsets[i]]), offsets[i + 1] - offsets[i],
                                        tmpbeg, tmpend, kMaxLevels);
      if (result < 0) {
        return failure("failed to sort an array", i, offsets[i], FILENAME(__LINE__));
      }
    }
  }
  else {
    for (int64_t i = 0; i < offsetslength - 1; i++) {
      int result = quick_sort_descending(&(toptr[offsets[i]]), offsets[i + 1] - offsets[i],
                                         tmpbeg, tmpend, kMaxLevels);
      if (result < 0) {
        return failure("failed to sort an array", i, offsets[i], FILENAME(__LINE__));
      }
    }
  }

  return success();
}
ERROR awkward_sort_bool(
  bool* toptr,
  const bool* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_sort<bool>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}
ERROR awkward_sort_int8(
  int8_t* toptr,
  const int8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}
ERROR awkward_sort_uint8(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}
ERROR awkward_sort_int16(
  int16_t* toptr,
  const int16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}
ERROR awkward_sort_uint16(
  uint16_t* toptr,
  const uint16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}
ERROR awkward_sort_int32(
  int32_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}
ERROR awkward_sort_uint32(
  uint32_t* toptr,
  const uint32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}
ERROR awkward_sort_int64(
  int64_t* toptr,
  const int64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}
ERROR awkward_sort_uint64(
  uint64_t* toptr,
  const uint64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}
ERROR awkward_sort_float32(
  float* toptr,
  const float* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_sort<float>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}
ERROR awkward_sort_float64(
  double* toptr,
  const double* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_sort<double>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}
