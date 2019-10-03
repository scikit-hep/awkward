// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>

#include "awkward/cpu-kernels/getitem.h"

void awkward_regularize_rangeslice(int64_t* start, int64_t* stop, bool posstep, bool hasstart, bool hasstop, int64_t length) {
  if (posstep) {
    if (!hasstart)           *start = 0;
    else if (*start < 0)     *start += length;
    if (*start < 0)          *start = 0;
    if (*start > length)     *start = length;

    if (!hasstop)            *stop = length;
    else if (*stop < 0)      *stop += length;
    if (*stop < 0)           *stop = 0;
    if (*stop > length)      *stop = length;
    if (*stop < *start)      *stop = *start;
  }

  else {
    if (!hasstart)           *start = length - 1;
    else if (*start < 0)     *start += length;
    if (*start < -1)         *start = -1;
    if (*start > length - 1) *start = length - 1;

    if (!hasstop)            *stop = -1;
    else if (*stop < 0)      *stop += length;
    if (*stop < -1)          *stop = -1;
    if (*stop > length - 1)  *stop = length - 1;
    if (*stop > *start)      *stop = *start;
  }
}

template <typename T>
Error awkward_regularize_arrayslice(T* flatheadptr, int64_t lenflathead, int64_t length) {
  for (int64_t i = 0;  i < lenflathead;  i++) {
    T original = flatheadptr[i];
    if (flatheadptr[i] < 0) {
      flatheadptr[i] += length;
    }
    if (flatheadptr[i] < 0  ||  flatheadptr[i] >= length) {
      return failure(kSliceNone, original, "index out of range");
    }
  }
  return success();
}
Error awkward_regularize_arrayslice_64(int64_t* flatheadptr, int64_t lenflathead, int64_t length) {
  return awkward_regularize_arrayslice<int64_t>(flatheadptr, lenflathead, length);
}

template <typename T>
void awkward_slicearray_ravel(T* toptr, const T* fromptr, int64_t ndim, const int64_t* shape, const int64_t* strides) {
  if (ndim == 1) {
    for (T i = 0;  i < shape[0];  i++) {
      toptr[i] = fromptr[i*strides[0]];
    }
  }
  else {
    for (T i = 0;  i < shape[0];  i++) {
      awkward_slicearray_ravel<T>(&toptr[i*shape[1]], &fromptr[i*strides[0]], ndim - 1, &shape[1], &strides[1]);
    }
  }
}
void awkward_slicearray_ravel_64(int64_t* toptr, const int64_t* fromptr, int64_t ndim, const int64_t* shape, const int64_t* strides) {
  awkward_slicearray_ravel<int64_t>(toptr, fromptr, ndim, shape, strides);
}

template <typename T>
void awkward_carry_arange(T* toptr, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i] = i;
  }
}
void awkward_carry_arange_64(int64_t* toptr, int64_t length) {
  awkward_carry_arange<int64_t>(toptr, length);
}

template <typename ID, typename T>
Error awkward_identity_getitem_carry(ID* newidentityptr, const ID* identityptr, const T* carryptr, int64_t lencarry, int64_t offset, int64_t width, int64_t length) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (carryptr[i] >= length) {
      return failure(kSliceNone, carryptr[i], "index out of range");
    }
    for (int64_t j = 0;  j < width;  j++) {
      newidentityptr[width*i + j] = identityptr[offset + width*carryptr[i] + j];
    }
  }
  return success();
}
Error awkward_identity32_getitem_carry_64(int32_t* newidentityptr, const int32_t* identityptr, const int64_t* carryptr, int64_t lencarry, int64_t offset, int64_t width, int64_t length) {
  return awkward_identity_getitem_carry<int32_t, int64_t>(newidentityptr, identityptr, carryptr, lencarry, offset, width, length);
}
Error awkward_identity64_getitem_carry_64(int64_t* newidentityptr, const int64_t* identityptr, const int64_t* carryptr, int64_t lencarry, int64_t offset, int64_t width, int64_t length) {
  return awkward_identity_getitem_carry<int64_t, int64_t>(newidentityptr, identityptr, carryptr, lencarry, offset, width, length);
}

template <typename T>
void awkward_numpyarray_contiguous_init(T* toptr, int64_t skip, int64_t stride) {
  for (int64_t i = 0;  i < skip;  i++) {
    toptr[i] = i*stride;
  }
}
void awkward_numpyarray_contiguous_init_64(int64_t* toptr, int64_t skip, int64_t stride) {
  awkward_numpyarray_contiguous_init<int64_t>(toptr, skip, stride);
}

template <typename T>
void awkward_numpyarray_contiguous_copy(uint8_t* toptr, const uint8_t* fromptr, int64_t len, int64_t stride, int64_t offset, const T* pos) {
  for (int64_t i = 0;  i < len;  i++) {
    memcpy(&toptr[i*stride], &fromptr[offset + (int64_t)pos[i]], (size_t)stride);
  }
}
void awkward_numpyarray_contiguous_copy_64(uint8_t* toptr, const uint8_t* fromptr, int64_t len, int64_t stride, int64_t offset, const int64_t* pos) {
  awkward_numpyarray_contiguous_copy<int64_t>(toptr, fromptr, len, stride, offset, pos);
}

template <typename T>
void awkward_numpyarray_contiguous_next(T* topos, const T* frompos, int64_t len, int64_t skip, int64_t stride) {
  for (int64_t i = 0;  i < len;  i++) {
    for (int64_t j = 0;  j < skip;  j++) {
      topos[i*skip + j] = frompos[i] + j*stride;
    }
  }
}
void awkward_numpyarray_contiguous_next_64(int64_t* topos, const int64_t* frompos, int64_t len, int64_t skip, int64_t stride) {
  awkward_numpyarray_contiguous_next<int64_t>(topos, frompos, len, skip, stride);
}

template <typename T>
void awkward_numpyarray_getitem_next_null(uint8_t* toptr, const uint8_t* fromptr, int64_t len, int64_t stride, int64_t offset, const T* pos) {
  for (int64_t i = 0;  i < len;  i++) {
    std::memcpy(&toptr[i*stride], &fromptr[offset + pos[i]*stride], (size_t)stride);
  }
}
void awkward_numpyarray_getitem_next_null_64(uint8_t* toptr, const uint8_t* fromptr, int64_t len, int64_t stride, int64_t offset, const int64_t* pos) {
  awkward_numpyarray_getitem_next_null(toptr, fromptr, len, stride, offset, pos);
}

template <typename T>
void awkward_numpyarray_getitem_next_at(T* nextcarryptr, const T* carryptr, int64_t lencarry, int64_t skip, int64_t at) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    nextcarryptr[i] = skip*carryptr[i] + at;
  }
}
void awkward_numpyarray_getitem_next_at_64(int64_t* nextcarryptr, const int64_t* carryptr, int64_t lencarry, int64_t skip, int64_t at) {
  awkward_numpyarray_getitem_next_at(nextcarryptr, carryptr, lencarry, skip, at);
}

template <typename T>
void awkward_numpyarray_getitem_next_range(T* nextcarryptr, const T* carryptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenhead;  j++) {
      nextcarryptr[i*lenhead + j] = skip*carryptr[i] + start + j*step;
    }
  }
}
void awkward_numpyarray_getitem_next_range_64(int64_t* nextcarryptr, const int64_t* carryptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step) {
  awkward_numpyarray_getitem_next_range(nextcarryptr, carryptr, lencarry, lenhead, skip, start, step);
}

template <typename T>
void awkward_numpyarray_getitem_next_range_advanced(T* nextcarryptr, T* nextadvancedptr, const T* carryptr, const T* advancedptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenhead;  j++) {
      nextcarryptr[i*lenhead + j] = skip*carryptr[i] + start + j*step;
      nextadvancedptr[i*lenhead + j] = advancedptr[i];
    }
  }
}
void awkward_numpyarray_getitem_next_range_advanced_64(int64_t* nextcarryptr, int64_t* nextadvancedptr, const int64_t* carryptr, const int64_t* advancedptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step) {
  awkward_numpyarray_getitem_next_range_advanced(nextcarryptr, nextadvancedptr, carryptr, advancedptr, lencarry, lenhead, skip, start, step);
}

template <typename T>
void awkward_numpyarray_getitem_next_array(T* nextcarryptr, T* nextadvancedptr, const T* carryptr, const T* flatheadptr, int64_t lencarry, int64_t lenflathead, int64_t skip) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenflathead;  j++) {
      nextcarryptr[i*lenflathead + j] = skip*carryptr[i] + flatheadptr[j];
      nextadvancedptr[i*lenflathead + j] = j;
    }
  }
}
void awkward_numpyarray_getitem_next_array_64(int64_t* nextcarryptr, int64_t* nextadvancedptr, const int64_t* carryptr, const int64_t* flatheadptr, int64_t lencarry, int64_t lenflathead, int64_t skip) {
  awkward_numpyarray_getitem_next_array(nextcarryptr, nextadvancedptr, carryptr, flatheadptr, lencarry, lenflathead, skip);
}

template <typename T>
void awkward_numpyarray_getitem_next_array_advanced(T* nextcarryptr, const T* carryptr, const T* advancedptr, const T* flatheadptr, int64_t lencarry, int64_t skip) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    nextcarryptr[i] = skip*carryptr[i] + flatheadptr[advancedptr[i]];
  }
}
void awkward_numpyarray_getitem_next_array_advanced_64(int64_t* nextcarryptr, const int64_t* carryptr, const int64_t* advancedptr, const int64_t* flatheadptr, int64_t lencarry, int64_t skip) {
  awkward_numpyarray_getitem_next_array_advanced(nextcarryptr, carryptr, advancedptr, flatheadptr, lencarry, skip);
}

template <typename C, typename T>
Error awkward_listarray_getitem_next_at(T* tocarry, const C* fromstarts, const C* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length;
    }
    if (!(0 <= regular_at  &&  regular_at < length)) {
      return failure(i, at, "index out of range");
    }
    tocarry[i] = fromstarts[startsoffset + i] + regular_at;
  }
  return success();
}
Error awkward_listarray32_getitem_next_at_64(int64_t* tocarry, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at) {
  return awkward_listarray_getitem_next_at<int32_t, int64_t>(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at);
}
Error awkward_listarray64_getitem_next_at_64(int64_t* tocarry, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at) {
  return awkward_listarray_getitem_next_at<int64_t, int64_t>(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at);
}

template <typename C>
void awkward_listarray_getitem_next_range_carrylength(int64_t* carrylength, const C* fromstarts, const C* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  *carrylength = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, step > 0, start != kSliceNone, stop != kSliceNone, length);
    if (step > 0) {
      for (int64_t j = regular_start;  j < regular_stop;  j += step) {
        *carrylength = *carrylength + 1;
      }
    }
    else {
      for (int64_t j = regular_start;  j > regular_stop;  j += step) {
        *carrylength = *carrylength + 1;
      }
    }
  }
}
void awkward_listarray32_getitem_next_range_carrylength(int64_t* carrylength, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  awkward_listarray_getitem_next_range_carrylength<int32_t>(carrylength, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, start, stop, step);
}
void awkward_listarray64_getitem_next_range_carrylength(int64_t* carrylength, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  awkward_listarray_getitem_next_range_carrylength<int64_t>(carrylength, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, start, stop, step);
}


template <typename C, typename T>
void awkward_listarray_getitem_next_range(C* tooffsets, T* tocarry, const C* fromstarts, const C* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  int64_t k = 0;
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, step > 0, start != kSliceNone, stop != kSliceNone, length);
    if (step > 0) {
      for (int64_t j = regular_start;  j < regular_stop;  j += step) {
        tocarry[k] = fromstarts[startsoffset + i] + j;
        k++;
      }
    }
    else {
      for (int64_t j = regular_start;  j > regular_stop;  j += step) {
        tocarry[k] = fromstarts[startsoffset + i] + j;
        k++;
      }
    }
    tooffsets[i + 1] = (C)k;
  }
}
void awkward_listarray32_getitem_next_range_64(int32_t* tooffsets, int64_t* tocarry, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  awkward_listarray_getitem_next_range<int32_t, int64_t>(tooffsets, tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, start, stop, step);
}
void awkward_listarray64_getitem_next_range_64(int64_t* tooffsets, int64_t* tocarry, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  awkward_listarray_getitem_next_range<int64_t, int64_t>(tooffsets, tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, start, stop, step);
}

template <typename C, typename T>
void awkward_listarray_getitem_next_range_counts(int64_t* total, const C* fromoffsets, int64_t lenstarts) {
  *total = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    *total = *total + fromoffsets[i + 1] - fromoffsets[i];
  }
}
void awkward_listarray32_getitem_next_range_counts_64(int64_t* total, const int32_t* fromoffsets, int64_t lenstarts) {
  awkward_listarray_getitem_next_range_counts<int32_t, int64_t>(total, fromoffsets, lenstarts);
}
void awkward_listarray64_getitem_next_range_counts_64(int64_t* total, const int64_t* fromoffsets, int64_t lenstarts) {
  awkward_listarray_getitem_next_range_counts<int64_t, int64_t>(total, fromoffsets, lenstarts);
}

template <typename C, typename T>
void awkward_listarray_getitem_next_range_spreadadvanced(T* toadvanced, const T* fromadvanced, const C* fromoffsets, int64_t lenstarts) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    C count = fromoffsets[i + 1] - fromoffsets[i];
    for (int64_t j = 0;  j < count;  j++) {
      toadvanced[fromoffsets[i] + j] = fromadvanced[i];
    }
  }
}
void awkward_listarray32_getitem_next_range_spreadadvanced_64(int64_t* toadvanced, const int64_t* fromadvanced, const int32_t* fromoffsets, int64_t lenstarts) {
  awkward_listarray_getitem_next_range_spreadadvanced<int32_t, int64_t>(toadvanced, fromadvanced, fromoffsets, lenstarts);
}
void awkward_listarray64_getitem_next_range_spreadadvanced_64(int64_t* toadvanced, const int64_t* fromadvanced, const int64_t* fromoffsets, int64_t lenstarts) {
  awkward_listarray_getitem_next_range_spreadadvanced<int64_t, int64_t>(toadvanced, fromadvanced, fromoffsets, lenstarts);
}

template <typename C, typename T>
Error awkward_listarray_getitem_next_array(C* tooffsets, T* tocarry, T* toadvanced, const C* fromstarts, const C* fromstops, const T* fromarray, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    if (fromstops[stopsoffset + i] < fromstarts[startsoffset + i]) {
      return failure(i, kSliceNone, "stops[i] < starts[i]");
    }
    if (fromstarts[startsoffset + i] != fromstops[stopsoffset + i]  &&  fromstops[stopsoffset + i] > lencontent) {
      return failure(i, kSliceNone, "stops[i] > len(content)");
    }
    int64_t length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
    for (int64_t j = 0;  j < lenarray;  j++) {
      int64_t regular_at = fromarray[j];
      if (regular_at < 0) {
        regular_at += length;
      }
      if (!(0 <= regular_at  &&  regular_at < length)) {
        return failure(i, kSliceNone, "array[i] is out of range for at least one sublist");
      }
      tocarry[i*lenarray + j] = fromstarts[startsoffset + i] + regular_at;
      toadvanced[i*lenarray + j] = j;
    }
    tooffsets[i + 1] = (C)((i + 1)*lenarray);
  }
  return success();
}
Error awkward_listarray32_getitem_next_array_64(int32_t* tooffsets, int64_t* tocarry, int64_t* toadvanced, const int32_t* fromstarts, const int32_t* fromstops, const int64_t* fromarray, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray_getitem_next_array<int32_t, int64_t>(tooffsets, tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}
Error awkward_listarray64_getitem_next_array_64(int64_t* tooffsets, int64_t* tocarry, int64_t* toadvanced, const int64_t* fromstarts, const int64_t* fromstops, const int64_t* fromarray, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray_getitem_next_array<int64_t, int64_t>(tooffsets, tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}

template <typename C, typename T>
Error awkward_listarray_getitem_next_array_advanced(T* tocarry, T* toadvanced, const C* fromstarts, const C* fromstops, const T* fromarray, const T* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    if (fromstops[stopsoffset + i] < fromstarts[startsoffset + i]) {
      return failure(i, kSliceNone, "stops[i] < starts[i]");
    }
    if (fromstarts[startsoffset + i] != fromstops[stopsoffset + i]  &&  fromstops[stopsoffset + i] > lencontent) {
      return failure(i, kSliceNone, "stops[i] > len(content)");
    }
    int64_t length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
    if (fromadvanced[i] >= lenarray) {
      return failure(i, kSliceNone, "lengths of advanced indexes must match");
    }
    int64_t regular_at = fromarray[fromadvanced[i]];
    if (regular_at < 0) {
      regular_at += length;
    }
    if (!(0 <= regular_at  &&  regular_at < length)) {
      return failure(i, kSliceNone, "array[i] is out of range for at least one sublist");
    }
    tocarry[i] = fromstarts[startsoffset + i] + regular_at;
    toadvanced[i] = i;
  }
  return success();
}
Error awkward_listarray32_getitem_next_array_advanced_64(int64_t* tocarry, int64_t* toadvanced, const int32_t* fromstarts, const int32_t* fromstops, const int64_t* fromarray, const int64_t* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray_getitem_next_array_advanced<int32_t, int64_t>(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}
Error awkward_listarray64_getitem_next_array_advanced_64(int64_t* tocarry, int64_t* toadvanced, const int64_t* fromstarts, const int64_t* fromstops, const int64_t* fromarray, const int64_t* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray_getitem_next_array_advanced<int64_t, int64_t>(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}

template <typename C, typename T>
Error awkward_listarray_getitem_carry(C* tostarts, C* tostops, const C* fromstarts, const C* fromstops, const T* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lencarry) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (fromcarry[i] >= lenstarts) {
      return failure(kSliceNone, fromcarry[i], "index out of range");
    }
    tostarts[i] = (C)(fromstarts[startsoffset + fromcarry[i]]);
    tostops[i] = (C)(fromstops[stopsoffset + fromcarry[i]]);
  }
  return success();
}
Error awkward_listarray32_getitem_carry_64(int32_t* tostarts, int32_t* tostops, const int32_t* fromstarts, const int32_t* fromstops, const int64_t* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lencarry) {
  return awkward_listarray_getitem_carry<int32_t, int64_t>(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lenstarts, lencarry);
}
Error awkward_listarray64_getitem_carry_64(int64_t* tostarts, int64_t* tostops, const int64_t* fromstarts, const int64_t* fromstops, const int64_t* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lencarry) {
  return awkward_listarray_getitem_carry<int64_t, int64_t>(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lenstarts, lencarry);
}
