// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/getitem.cpp", line)

#include "awkward/kernels/getitem.h"

void awkward_regularize_rangeslice(
  int64_t* start,
  int64_t* stop,
  bool posstep,
  bool hasstart,
  bool hasstop,
  int64_t length) {
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
ERROR awkward_regularize_arrayslice(
  T* flatheadptr,
  int64_t lenflathead,
  int64_t length) {
  for (int64_t i = 0;  i < lenflathead;  i++) {
    T original = flatheadptr[i];
    if (flatheadptr[i] < 0) {
      flatheadptr[i] += length;
    }
    if (flatheadptr[i] < 0  ||  flatheadptr[i] >= length) {
      return failure("index out of range", kSliceNone, original, FILENAME(__LINE__));
    }
  }
  return success();
}
ERROR awkward_regularize_arrayslice_64(
  int64_t* flatheadptr,
  int64_t lenflathead,
  int64_t length) {
  return awkward_regularize_arrayslice<int64_t>(
    flatheadptr,
    lenflathead,
    length);
}

ERROR awkward_Index8_to_Index64(
  int64_t* toptr,
  const int8_t* fromptr,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}
ERROR awkward_IndexU8_to_Index64(
  int64_t* toptr,
  const uint8_t* fromptr,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}
ERROR awkward_Index32_to_Index64(
  int64_t* toptr,
  const int32_t* fromptr,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}
ERROR awkward_IndexU32_to_Index64(
  int64_t* toptr,
  const uint32_t* fromptr,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}

template <typename C, typename T>
ERROR awkward_index_carry(
  C* toindex,
  const C* fromindex,
  const T* carry,
  int64_t lenfromindex,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    T j = carry[i];
    if (j > lenfromindex) {
      return failure("index out of range", kSliceNone, j, FILENAME(__LINE__));
    }
    toindex[i] = fromindex[(size_t)(j)];
  }
  return success();
}
ERROR awkward_Index8_carry_64(
  int8_t* toindex,
  const int8_t* fromindex,
  const int64_t* carry,
  int64_t lenfromindex,
  int64_t length) {
  return awkward_index_carry<int8_t, int64_t>(
    toindex,
    fromindex,
    carry,
    lenfromindex,
    length);
}
ERROR awkward_IndexU8_carry_64(
  uint8_t* toindex,
  const uint8_t* fromindex,
  const int64_t* carry,
  int64_t lenfromindex,
  int64_t length) {
  return awkward_index_carry<uint8_t, int64_t>(
    toindex,
    fromindex,
    carry,
    lenfromindex,
    length);
}
ERROR awkward_Index32_carry_64(
  int32_t* toindex,
  const int32_t* fromindex,
  const int64_t* carry,
  int64_t lenfromindex,
  int64_t length) {
  return awkward_index_carry<int32_t, int64_t>(
    toindex,
    fromindex,
    carry,
    lenfromindex,
    length);
}
ERROR awkward_IndexU32_carry_64(
  uint32_t* toindex,
  const uint32_t* fromindex,
  const int64_t* carry,
  int64_t lenfromindex,
  int64_t length) {
  return awkward_index_carry<uint32_t, int64_t>(
    toindex,
    fromindex,
    carry,
    lenfromindex,
    length);
}
ERROR awkward_Index64_carry_64(
  int64_t* toindex,
  const int64_t* fromindex,
  const int64_t* carry,
  int64_t lenfromindex,
  int64_t length) {
  return awkward_index_carry<int64_t, int64_t>(
    toindex,
    fromindex,
    carry,
    lenfromindex,
    length);
}

template <typename C, typename T>
ERROR awkward_index_carry_nocheck(
  C* toindex,
  const C* fromindex,
  const T* carry,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[i] = fromindex[(size_t)(carry[i])];
  }
  return success();
}
ERROR awkward_Index8_carry_nocheck_64(
  int8_t* toindex,
  const int8_t* fromindex,
  const int64_t* carry,
  int64_t length) {
  return awkward_index_carry_nocheck<int8_t, int64_t>(
    toindex,
    fromindex,
    carry,
    length);
}
ERROR awkward_IndexU8_carry_nocheck_64(
  uint8_t* toindex,
  const uint8_t* fromindex,
  const int64_t* carry,
  int64_t length) {
  return awkward_index_carry_nocheck<uint8_t, int64_t>(
    toindex,
    fromindex,
    carry,
    length);
}
ERROR awkward_Index32_carry_nocheck_64(
  int32_t* toindex,
  const int32_t* fromindex,
  const int64_t* carry,
  int64_t length) {
  return awkward_index_carry_nocheck<int32_t, int64_t>(
    toindex,
    fromindex,
    carry,
    length);
}
ERROR awkward_IndexU32_carry_nocheck_64(
  uint32_t* toindex,
  const uint32_t* fromindex,
  const int64_t* carry,
  int64_t length) {
  return awkward_index_carry_nocheck<uint32_t, int64_t>(
    toindex,
    fromindex,
    carry,
    length);
}
ERROR awkward_Index64_carry_nocheck_64(
  int64_t* toindex,
  const int64_t* fromindex,
  const int64_t* carry,
  int64_t length) {
  return awkward_index_carry_nocheck<int64_t, int64_t>(
    toindex,
    fromindex,
    carry,
    length);
}

template <typename T>
ERROR awkward_slicearray_ravel(
  T* toptr,
  const T* fromptr,
  int64_t ndim,
  const int64_t* shape,
  const int64_t* strides) {
  if (ndim == 1) {
    for (T i = 0;  i < shape[0];  i++) {
      toptr[i] = fromptr[i*strides[0]];
    }
  }
  else {
    for (T i = 0;  i < shape[0];  i++) {
      ERROR err =
        awkward_slicearray_ravel<T>(
          &toptr[i*shape[1]],
          &fromptr[i*strides[0]],
          ndim - 1,
          &shape[1],
          &strides[1]);
      if (err.str != nullptr) {
        return err;
      }
    }
  }
  return success();
}
ERROR awkward_slicearray_ravel_64(
  int64_t* toptr,
  const int64_t* fromptr,
  int64_t ndim,
  const int64_t* shape,
  const int64_t* strides) {
  return awkward_slicearray_ravel<int64_t>(
    toptr,
    fromptr,
    ndim,
    shape,
    strides);
}

ERROR awkward_slicemissing_check_same(
  bool* same,
  const int8_t* bytemask,
  const int64_t* missingindex,
  int64_t length) {
  *same = true;
  for (int64_t i = 0;  i < length;  i++) {
    bool left = (bytemask[i] != 0);
    bool right = (missingindex[i] < 0);
    if (left != right) {
      *same = false;
      return success();
    }
  }
  return success();
}

template <typename T>
ERROR awkward_carry_arange(
  T* toptr,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i] = i;
  }
  return success();
}
ERROR awkward_carry_arange32(
  int32_t* toptr,
  int64_t length) {
  return awkward_carry_arange<int32_t>(
    toptr,
    length);
}
ERROR awkward_carry_arangeU32(
  uint32_t* toptr,
  int64_t length) {
  return awkward_carry_arange<uint32_t>(
    toptr,
    length);
}
ERROR awkward_carry_arange64(
  int64_t* toptr,
  int64_t length) {
  return awkward_carry_arange<int64_t>(
    toptr,
    length);
}

template <typename ID, typename T>
ERROR awkward_Identities_getitem_carry(
  ID* newidentitiesptr,
  const ID* identitiesptr,
  const T* carryptr,
  int64_t lencarry,
  int64_t width,
  int64_t length) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (carryptr[i] >= length) {
      return failure("index out of range", kSliceNone, carryptr[i], FILENAME(__LINE__));
    }
    for (int64_t j = 0;  j < width;  j++) {
      newidentitiesptr[width*i + j] =
        identitiesptr[width*carryptr[i] + j];
    }
  }
  return success();
}
ERROR awkward_Identities32_getitem_carry_64(
  int32_t* newidentitiesptr,
  const int32_t* identitiesptr,
  const int64_t* carryptr,
  int64_t lencarry,
  int64_t width,
  int64_t length) {
  return awkward_Identities_getitem_carry<int32_t, int64_t>(
    newidentitiesptr,
    identitiesptr,
    carryptr,
    lencarry,
    width,
    length);
}
ERROR awkward_Identities64_getitem_carry_64(
  int64_t* newidentitiesptr,
  const int64_t* identitiesptr,
  const int64_t* carryptr,
  int64_t lencarry,
  int64_t width,
  int64_t length) {
  return awkward_Identities_getitem_carry<int64_t, int64_t>(
    newidentitiesptr,
    identitiesptr,
    carryptr,
    lencarry,
    width,
    length);
}

template <typename T>
ERROR awkward_NumpyArray_contiguous_init(
  T* toptr,
  int64_t skip,
  int64_t stride) {
  for (int64_t i = 0;  i < skip;  i++) {
    toptr[i] = i*stride;
  }
  return success();
}
ERROR awkward_NumpyArray_contiguous_init_64(
  int64_t* toptr,
  int64_t skip,
  int64_t stride) {
  return awkward_NumpyArray_contiguous_init<int64_t>(
    toptr,
    skip,
    stride);
}

template <typename T>
ERROR awkward_NumpyArray_contiguous_copy(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t len,
  int64_t stride,
  const T* pos) {
  for (int64_t i = 0;  i < len;  i++) {
    memcpy(&toptr[i*stride], &fromptr[pos[i]], (size_t)stride);
  }
  return success();
}
ERROR awkward_NumpyArray_contiguous_copy_64(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t len,
  int64_t stride,
  const int64_t* pos) {
  return awkward_NumpyArray_contiguous_copy<int64_t>(
    toptr,
    fromptr,
    len,
    stride,
    pos);
}

template <typename T>
ERROR awkward_NumpyArray_contiguous_next(
  T* topos,
  const T* frompos,
  int64_t length,
  int64_t skip,
  int64_t stride) {
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = 0;  j < skip;  j++) {
      topos[i*skip + j] = frompos[i] + j*stride;
    }
  }
  return success();
}
ERROR awkward_NumpyArray_contiguous_next_64(
  int64_t* topos,
  const int64_t* frompos,
  int64_t length,
  int64_t skip,
  int64_t stride) {
  return awkward_NumpyArray_contiguous_next<int64_t>(
    topos,
    frompos,
    length,
    skip,
    stride);
}

template <typename T>
ERROR awkward_NumpyArray_getitem_next_null(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t len,
  int64_t stride,
  const T* pos) {
  for (int64_t i = 0;  i < len;  i++) {
    std::memcpy(&toptr[i*stride], &fromptr[pos[i]*stride], (size_t)stride);
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_next_null_64(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t len,
  int64_t stride,
  const int64_t* pos) {
  return awkward_NumpyArray_getitem_next_null(
    toptr,
    fromptr,
    len,
    stride,
    pos);
}

template <typename T>
ERROR awkward_NumpyArray_getitem_next_at(
  T* nextcarryptr,
  const T* carryptr,
  int64_t lencarry,
  int64_t skip,
  int64_t at) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    nextcarryptr[i] = skip*carryptr[i] + at;
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_next_at_64(
  int64_t* nextcarryptr,
  const int64_t* carryptr,
  int64_t lencarry,
  int64_t skip,
  int64_t at) {
  return awkward_NumpyArray_getitem_next_at(
    nextcarryptr,
    carryptr,
    lencarry,
    skip,
    at);
}

template <typename T>
ERROR awkward_NumpyArray_getitem_next_range(
  T* nextcarryptr,
  const T* carryptr,
  int64_t lencarry,
  int64_t lenhead,
  int64_t skip,
  int64_t start,
  int64_t step) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenhead;  j++) {
      nextcarryptr[i*lenhead + j] = skip*carryptr[i] + start + j*step;
    }
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_next_range_64(
  int64_t* nextcarryptr,
  const int64_t* carryptr,
  int64_t lencarry,
  int64_t lenhead,
  int64_t skip,
  int64_t start,
  int64_t step) {
  return awkward_NumpyArray_getitem_next_range(
    nextcarryptr,
    carryptr,
    lencarry,
    lenhead,
    skip,
    start,
    step);
}

template <typename T>
ERROR awkward_NumpyArray_getitem_next_range_advanced(
  T* nextcarryptr,
  T* nextadvancedptr,
  const T* carryptr,
  const T* advancedptr,
  int64_t lencarry,
  int64_t lenhead,
  int64_t skip,
  int64_t start,
  int64_t step) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenhead;  j++) {
      nextcarryptr[i*lenhead + j] = skip*carryptr[i] + start + j*step;
      nextadvancedptr[i*lenhead + j] = advancedptr[i];
    }
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_next_range_advanced_64(
  int64_t* nextcarryptr,
  int64_t* nextadvancedptr,
  const int64_t* carryptr,
  const int64_t* advancedptr,
  int64_t lencarry,
  int64_t lenhead,
  int64_t skip,
  int64_t start,
  int64_t step) {
  return awkward_NumpyArray_getitem_next_range_advanced(
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

template <typename T>
ERROR awkward_NumpyArray_getitem_next_array(
  T* nextcarryptr,
  T* nextadvancedptr,
  const T* carryptr,
  const T* flatheadptr,
  int64_t lencarry,
  int64_t lenflathead,
  int64_t skip) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenflathead;  j++) {
      nextcarryptr[i*lenflathead + j] = skip*carryptr[i] + flatheadptr[j];
      nextadvancedptr[i*lenflathead + j] = j;
    }
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_next_array_64(
  int64_t* nextcarryptr,
  int64_t* nextadvancedptr,
  const int64_t* carryptr,
  const int64_t* flatheadptr,
  int64_t lencarry,
  int64_t lenflathead,
  int64_t skip) {
  return awkward_NumpyArray_getitem_next_array(
    nextcarryptr,
    nextadvancedptr,
    carryptr,
    flatheadptr,
    lencarry,
    lenflathead,
    skip);
}

template <typename T>
ERROR awkward_NumpyArray_getitem_next_array_advanced(
  T* nextcarryptr,
  const T* carryptr,
  const T* advancedptr,
  const T* flatheadptr,
  int64_t lencarry,
  int64_t skip) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    nextcarryptr[i] = skip*carryptr[i] + flatheadptr[advancedptr[i]];
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_next_array_advanced_64(
  int64_t* nextcarryptr,
  const int64_t* carryptr,
  const int64_t* advancedptr,
  const int64_t* flatheadptr,
  int64_t lencarry,
  int64_t skip) {
  return awkward_NumpyArray_getitem_next_array_advanced(
    nextcarryptr,
    carryptr,
    advancedptr,
    flatheadptr,
    lencarry,
    skip);
}

ERROR awkward_NumpyArray_getitem_boolean_numtrue(
  int64_t* numtrue,
  const int8_t* fromptr,
  int64_t length,
  int64_t stride) {
  *numtrue = 0;
  for (int64_t i = 0;  i < length;  i += stride) {
    *numtrue = *numtrue + (fromptr[i] != 0);
  }
  return success();
}

template <typename T>
ERROR awkward_NumpyArray_getitem_boolean_nonzero(
  T* toptr,
  const int8_t* fromptr,
  int64_t length,
  int64_t stride) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i += stride) {
    if (fromptr[i] != 0) {
      toptr[k] = i;
      k++;
    }
  }
  return success();
}

ERROR awkward_NumpyArray_getitem_boolean_nonzero_64(
  int64_t* toptr,
  const int8_t* fromptr,
  int64_t length,
  int64_t stride) {
  return awkward_NumpyArray_getitem_boolean_nonzero<int64_t>(
    toptr,
    fromptr,
    length,
    stride);
}

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_at(
  T* tocarry,
  const C* fromstarts,
  const C* fromstops,
  int64_t lenstarts,
  int64_t at) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t length = fromstops[i] - fromstarts[i];
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length;
    }
    if (!(0 <= regular_at  &&  regular_at < length)) {
      return failure("index out of range", i, at, FILENAME(__LINE__));
    }
    tocarry[i] = fromstarts[i] + regular_at;
  }
  return success();
}
ERROR awkward_ListArray32_getitem_next_at_64(
  int64_t* tocarry,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t lenstarts,
  int64_t at) {
  return awkward_ListArray_getitem_next_at<int32_t, int64_t>(
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    at);
}
ERROR awkward_ListArrayU32_getitem_next_at_64(
  int64_t* tocarry,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t lenstarts,
  int64_t at) {
  return awkward_ListArray_getitem_next_at<uint32_t, int64_t>(
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    at);
}
ERROR awkward_ListArray64_getitem_next_at_64(
  int64_t* tocarry,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t lenstarts,
  int64_t at) {
  return awkward_ListArray_getitem_next_at<int64_t, int64_t>(
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    at);
}

template <typename C>
ERROR awkward_ListArray_getitem_next_range_carrylength(
  int64_t* carrylength,
  const C* fromstarts,
  const C* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  *carrylength = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t length = fromstops[i] - fromstarts[i];
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, step > 0,
                                  start != kSliceNone, stop != kSliceNone,
                                  length);
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
  return success();
}
ERROR awkward_ListArray32_getitem_next_range_carrylength(
  int64_t* carrylength,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  return awkward_ListArray_getitem_next_range_carrylength<int32_t>(
    carrylength,
    fromstarts,
    fromstops,
    lenstarts,
    start,
    stop,
    step);
}
ERROR awkward_ListArrayU32_getitem_next_range_carrylength(
  int64_t* carrylength,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  return awkward_ListArray_getitem_next_range_carrylength<uint32_t>(
    carrylength,
    fromstarts,
    fromstops,
    lenstarts,
    start,
    stop,
    step);
}
ERROR awkward_ListArray64_getitem_next_range_carrylength(
  int64_t* carrylength,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  return awkward_ListArray_getitem_next_range_carrylength<int64_t>(
    carrylength,
    fromstarts,
    fromstops,
    lenstarts,
    start,
    stop,
    step);
}

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_range(
  C* tooffsets,
  T* tocarry,
  const C* fromstarts,
  const C* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  int64_t k = 0;
  tooffsets[0] = 0;
  if (step > 0) {
    for (int64_t i = 0;  i < lenstarts;  i++) {
      int64_t length = fromstops[i] - fromstarts[i];
      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop, step > 0,
                                    start != kSliceNone, stop != kSliceNone,
                                    length);
      for (int64_t j = regular_start;  j < regular_stop;  j += step) {
        tocarry[k] = fromstarts[i] + j;
        k++;
      }
      tooffsets[i + 1] = (C)k;
    }
  }
  else {
    for (int64_t i = 0;  i < lenstarts;  i++) {
      int64_t length = fromstops[i] - fromstarts[i];
      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop, step > 0,
                                    start != kSliceNone, stop != kSliceNone,
                                    length);
      for (int64_t j = regular_start;  j > regular_stop;  j += step) {
        tocarry[k] = fromstarts[i] + j;
        k++;
      }
      tooffsets[i + 1] = (C)k;
    }
  }
  return success();
}
ERROR awkward_ListArray32_getitem_next_range_64(
  int32_t* tooffsets,
  int64_t* tocarry,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  return awkward_ListArray_getitem_next_range<int32_t, int64_t>(
    tooffsets,
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    start,
    stop,
    step);
}
ERROR awkward_ListArrayU32_getitem_next_range_64(
  uint32_t* tooffsets,
  int64_t* tocarry,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  return awkward_ListArray_getitem_next_range<uint32_t, int64_t>(
    tooffsets,
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    start,
    stop,
    step);
}
ERROR awkward_ListArray64_getitem_next_range_64(
  int64_t* tooffsets,
  int64_t* tocarry,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  return awkward_ListArray_getitem_next_range<int64_t, int64_t>(
    tooffsets,
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    start,
    stop,
    step);
}

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_range_counts(
  int64_t* total,
  const C* fromoffsets,
  int64_t lenstarts) {
  *total = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    *total = *total + fromoffsets[i + 1] - fromoffsets[i];
  }
  return success();
}
ERROR awkward_ListArray32_getitem_next_range_counts_64(
  int64_t* total,
  const int32_t* fromoffsets,
  int64_t lenstarts) {
  return awkward_ListArray_getitem_next_range_counts<int32_t, int64_t>(
    total,
    fromoffsets,
    lenstarts);
}
ERROR awkward_ListArrayU32_getitem_next_range_counts_64(
  int64_t* total,
  const uint32_t* fromoffsets,
  int64_t lenstarts) {
  return awkward_ListArray_getitem_next_range_counts<uint32_t, int64_t>(
    total,
    fromoffsets,
    lenstarts);
}
ERROR awkward_ListArray64_getitem_next_range_counts_64(
  int64_t* total,
  const int64_t* fromoffsets,
  int64_t lenstarts) {
  return awkward_ListArray_getitem_next_range_counts<int64_t, int64_t>(
    total,
    fromoffsets,
    lenstarts);
}

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_range_spreadadvanced(
  T* toadvanced,
  const T* fromadvanced,
  const C* fromoffsets,
  int64_t lenstarts) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    C count = fromoffsets[i + 1] - fromoffsets[i];
    for (int64_t j = 0;  j < count;  j++) {
      toadvanced[fromoffsets[i] + j] = fromadvanced[i];
    }
  }
  return success();
}
ERROR awkward_ListArray32_getitem_next_range_spreadadvanced_64(
  int64_t* toadvanced,
  const int64_t* fromadvanced,
  const int32_t* fromoffsets,
  int64_t lenstarts) {
  return awkward_ListArray_getitem_next_range_spreadadvanced<int32_t, int64_t>(
    toadvanced,
    fromadvanced,
    fromoffsets,
    lenstarts);
}
ERROR awkward_ListArrayU32_getitem_next_range_spreadadvanced_64(
  int64_t* toadvanced,
  const int64_t* fromadvanced,
  const uint32_t* fromoffsets,
  int64_t lenstarts) {
  return awkward_ListArray_getitem_next_range_spreadadvanced<uint32_t, int64_t>(
    toadvanced,
    fromadvanced,
    fromoffsets,
    lenstarts);
}
ERROR awkward_ListArray64_getitem_next_range_spreadadvanced_64(
  int64_t* toadvanced,
  const int64_t* fromadvanced,
  const int64_t* fromoffsets,
  int64_t lenstarts) {
  return awkward_ListArray_getitem_next_range_spreadadvanced<int64_t, int64_t>(
    toadvanced,
    fromadvanced,
    fromoffsets,
    lenstarts);
}

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_array(
  T* tocarry,
  T* toadvanced,
  const C* fromstarts,
  const C* fromstops,
  const T* fromarray,
  int64_t lenstarts,
  int64_t lenarray,
  int64_t lencontent) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    if (fromstops[i] < fromstarts[i]) {
      return failure("stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
    }
    if ((fromstarts[i] != fromstops[i])  &&
        (fromstops[i] > lencontent)) {
      return failure("stops[i] > len(content)", i, kSliceNone, FILENAME(__LINE__));
    }
    int64_t length = fromstops[i] - fromstarts[i];
    for (int64_t j = 0;  j < lenarray;  j++) {
      int64_t regular_at = fromarray[j];
      if (regular_at < 0) {
        regular_at += length;
      }
      if (!(0 <= regular_at  &&  regular_at < length)) {
        return failure("index out of range", i, fromarray[j], FILENAME(__LINE__));
      }
      tocarry[i*lenarray + j] = fromstarts[i] + regular_at;
      toadvanced[i*lenarray + j] = j;
    }
  }
  return success();
}
ERROR awkward_ListArray32_getitem_next_array_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  const int64_t* fromarray,
  int64_t lenstarts,
  int64_t lenarray,
  int64_t lencontent) {
  return awkward_ListArray_getitem_next_array<int32_t, int64_t>(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    lenstarts,
    lenarray,
    lencontent);
}
ERROR awkward_ListArrayU32_getitem_next_array_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  const int64_t* fromarray,
  int64_t lenstarts,
  int64_t lenarray,
  int64_t lencontent) {
  return awkward_ListArray_getitem_next_array<uint32_t, int64_t>(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    lenstarts,
    lenarray,
    lencontent);
}
ERROR awkward_ListArray64_getitem_next_array_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  const int64_t* fromarray,
  int64_t lenstarts,
  int64_t lenarray,
  int64_t lencontent) {
  return awkward_ListArray_getitem_next_array<int64_t, int64_t>(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    lenstarts,
    lenarray,
    lencontent);
}

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_array_advanced(
  T* tocarry,
  T* toadvanced,
  const C* fromstarts,
  const C* fromstops,
  const T* fromarray,
  const T* fromadvanced,
  int64_t lenstarts,
  int64_t lenarray,
  int64_t lencontent) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    if (fromstops[i] < fromstarts[i]) {
      return failure("stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
    }
    if ((fromstarts[i] != fromstops[i])  &&
        (fromstops[i] > lencontent)) {
      return failure("stops[i] > len(content)", i, kSliceNone, FILENAME(__LINE__));
    }
    int64_t length = fromstops[i] - fromstarts[i];
    int64_t regular_at = fromarray[fromadvanced[i]];
    if (regular_at < 0) {
      regular_at += length;
    }
    if (!(0 <= regular_at  &&  regular_at < length)) {
      return failure("index out of range", i, fromarray[fromadvanced[i]], FILENAME(__LINE__));
    }
    tocarry[i] = fromstarts[i] + regular_at;
    toadvanced[i] = i;
  }
  return success();
}
ERROR awkward_ListArray32_getitem_next_array_advanced_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  const int64_t* fromarray,
  const int64_t* fromadvanced,
  int64_t lenstarts,
  int64_t lenarray,
  int64_t lencontent) {
  return awkward_ListArray_getitem_next_array_advanced<int32_t, int64_t>(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    fromadvanced,
    lenstarts,
    lenarray,
    lencontent);
}
ERROR awkward_ListArrayU32_getitem_next_array_advanced_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  const int64_t* fromarray,
  const int64_t* fromadvanced,
  int64_t lenstarts,
  int64_t lenarray,
  int64_t lencontent) {
  return awkward_ListArray_getitem_next_array_advanced<uint32_t, int64_t>(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    fromadvanced,
    lenstarts,
    lenarray,
    lencontent);
}
ERROR awkward_ListArray64_getitem_next_array_advanced_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  const int64_t* fromarray,
  const int64_t* fromadvanced,
  int64_t lenstarts,
  int64_t lenarray,
  int64_t lencontent) {
  return awkward_ListArray_getitem_next_array_advanced<int64_t, int64_t>(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    fromadvanced,
    lenstarts,
    lenarray,
    lencontent);
}

template <typename C, typename T>
ERROR awkward_ListArray_getitem_carry(
  C* tostarts,
  C* tostops,
  const C* fromstarts,
  const C* fromstops,
  const T* fromcarry,
  int64_t lenstarts,
  int64_t lencarry) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (fromcarry[i] >= lenstarts) {
      return failure("index out of range", i, fromcarry[i], FILENAME(__LINE__));
    }
    tostarts[i] = (C)(fromstarts[fromcarry[i]]);
    tostops[i] = (C)(fromstops[fromcarry[i]]);
  }
  return success();
}
ERROR awkward_ListArray32_getitem_carry_64(
  int32_t* tostarts,
  int32_t* tostops,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  const int64_t* fromcarry,
  int64_t lenstarts,
  int64_t lencarry) {
  return awkward_ListArray_getitem_carry<int32_t, int64_t>(
    tostarts,
    tostops,
    fromstarts,
    fromstops,
    fromcarry,
    lenstarts,
    lencarry);
}
ERROR awkward_ListArrayU32_getitem_carry_64(
  uint32_t* tostarts,
  uint32_t* tostops,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  const int64_t* fromcarry,
  int64_t lenstarts,
  int64_t lencarry) {
  return awkward_ListArray_getitem_carry<uint32_t, int64_t>(
    tostarts,
    tostops,
    fromstarts,
    fromstops,
    fromcarry,
    lenstarts,
    lencarry);
}
ERROR awkward_ListArray64_getitem_carry_64(
  int64_t* tostarts,
  int64_t* tostops,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  const int64_t* fromcarry,
  int64_t lenstarts,
  int64_t lencarry) {
  return awkward_ListArray_getitem_carry<int64_t, int64_t>(
    tostarts,
    tostops,
    fromstarts,
    fromstops,
    fromcarry,
    lenstarts,
    lencarry);
}

template <typename T>
ERROR awkward_RegularArray_getitem_next_at(
  T* tocarry,
  int64_t at,
  int64_t length,
  int64_t size) {
  int64_t regular_at = at;
  if (regular_at < 0) {
    regular_at += size;
  }
  if (!(0 <= regular_at  &&  regular_at < size)) {
    return failure("index out of range", kSliceNone, at, FILENAME(__LINE__));
  }
  for (int64_t i = 0;  i < length;  i++) {
    tocarry[i] = i*size + regular_at;
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_at_64(
  int64_t* tocarry,
  int64_t at,
  int64_t length,
  int64_t size) {
  return awkward_RegularArray_getitem_next_at<int64_t>(
    tocarry,
    at,
    length,
    size);
}

template <typename T>
ERROR awkward_RegularArray_getitem_next_range(
  T* tocarry,
  int64_t regular_start,
  int64_t step,
  int64_t length,
  int64_t size,
  int64_t nextsize) {
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = 0;  j < nextsize;  j++) {
      tocarry[i*nextsize + j] = i*size + regular_start + j*step;
    }
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_range_64(
  int64_t* tocarry,
  int64_t regular_start,
  int64_t step,
  int64_t length,
  int64_t size,
  int64_t nextsize) {
  return awkward_RegularArray_getitem_next_range<int64_t>(
    tocarry,
    regular_start,
    step,
    length,
    size,
    nextsize);
}

template <typename T>
ERROR awkward_RegularArray_getitem_next_range_spreadadvanced(
  T* toadvanced,
  const T* fromadvanced,
  int64_t length,
  int64_t nextsize) {
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = 0;  j < nextsize;  j++) {
      toadvanced[i*nextsize + j] = fromadvanced[i];
    }
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_range_spreadadvanced_64(
  int64_t* toadvanced,
  const int64_t* fromadvanced,
  int64_t length,
  int64_t nextsize) {
  return awkward_RegularArray_getitem_next_range_spreadadvanced<int64_t>(
    toadvanced,
    fromadvanced,
    length,
    nextsize);
}

template <typename T>
ERROR awkward_RegularArray_getitem_next_array_regularize(
  T* toarray,
  const T* fromarray,
  int64_t lenarray,
  int64_t size) {
  for (int64_t j = 0;  j < lenarray;  j++) {
    toarray[j] = fromarray[j];
    if (toarray[j] < 0) {
      toarray[j] += size;
    }
    if (!(0 <= toarray[j]  &&  toarray[j] < size)) {
      return failure("index out of range", kSliceNone, fromarray[j], FILENAME(__LINE__));
    }
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_array_regularize_64(
  int64_t* toarray,
  const int64_t* fromarray,
  int64_t lenarray,
  int64_t size) {
  return awkward_RegularArray_getitem_next_array_regularize<int64_t>(
    toarray,
    fromarray,
    lenarray,
    size);
}

template <typename T>
ERROR awkward_RegularArray_getitem_next_array(
  T* tocarry,
  T* toadvanced,
  const T* fromarray,
  int64_t length,
  int64_t lenarray,
  int64_t size) {
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = 0;  j < lenarray;  j++) {
      tocarry[i*lenarray + j] = i*size + fromarray[j];
      toadvanced[i*lenarray + j] = j;
    }
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_array_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const int64_t* fromarray,
  int64_t length,
  int64_t lenarray,
  int64_t size) {
  return awkward_RegularArray_getitem_next_array<int64_t>(
    tocarry,
    toadvanced,
    fromarray,
    length,
    lenarray,
    size);
}

template <typename T>
ERROR awkward_RegularArray_getitem_next_array_advanced(
  T* tocarry,
  T* toadvanced,
  const T* fromadvanced,
  const T* fromarray,
  int64_t length,
  int64_t lenarray,
  int64_t size) {
  for (int64_t i = 0;  i < length;  i++) {
    tocarry[i] = i*size + fromarray[fromadvanced[i]];
    toadvanced[i] = i;
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_array_advanced_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const int64_t* fromadvanced,
  const int64_t* fromarray,
  int64_t length,
  int64_t lenarray,
  int64_t size) {
  return awkward_RegularArray_getitem_next_array_advanced<int64_t>(
    tocarry,
    toadvanced,
    fromadvanced,
    fromarray,
    length,
    lenarray,
    size);
}

template <typename T>
ERROR awkward_RegularArray_getitem_carry(
  T* tocarry,
  const T* fromcarry,
  int64_t lencarry,
  int64_t size) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < size;  j++) {
      tocarry[i*size + j] = fromcarry[i]*size + j;
    }
  }
  return success();
}
ERROR awkward_RegularArray_getitem_carry_64(
  int64_t* tocarry,
  const int64_t* fromcarry,
  int64_t lencarry,
  int64_t size) {
  return awkward_RegularArray_getitem_carry<int64_t>(
    tocarry,
    fromcarry,
    lencarry,
    size);
}

template <typename C>
ERROR awkward_IndexedArray_numnull(
  int64_t* numnull,
  const C* fromindex,
  int64_t lenindex) {
  *numnull = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    if (fromindex[i] < 0) {
      *numnull = *numnull + 1;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_numnull(
  int64_t* numnull,
  const int32_t* fromindex,
  int64_t lenindex) {
  return awkward_IndexedArray_numnull<int32_t>(
    numnull,
    fromindex,
    lenindex);
}
ERROR awkward_IndexedArrayU32_numnull(
  int64_t* numnull,
  const uint32_t* fromindex,
  int64_t lenindex) {
  return awkward_IndexedArray_numnull<uint32_t>(
    numnull,
    fromindex,
    lenindex);
}
ERROR awkward_IndexedArray64_numnull(
  int64_t* numnull,
  const int64_t* fromindex,
  int64_t lenindex) {
  return awkward_IndexedArray_numnull<int64_t>(
    numnull,
    fromindex,
    lenindex);
}

template <typename C, typename T>
ERROR awkward_IndexedArray_getitem_nextcarry_outindex(
  T* tocarry,
  C* toindex,
  const C* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  int64_t k = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    C j = fromindex[i];
    if (j >= lencontent) {
      return failure("index out of range", i, j, FILENAME(__LINE__));
    }
    else if (j < 0) {
      toindex[i] = -1;
    }
    else {
      tocarry[k] = j;
      toindex[i] = (C)k;
      k++;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_getitem_nextcarry_outindex_64(
  int64_t* tocarry,
  int32_t* toindex,
  const int32_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_getitem_nextcarry_outindex<int32_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    lenindex,
    lencontent);
}
ERROR awkward_IndexedArrayU32_getitem_nextcarry_outindex_64(
  int64_t* tocarry,
  uint32_t* toindex,
  const uint32_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_getitem_nextcarry_outindex<uint32_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    lenindex,
    lencontent);
}
ERROR awkward_IndexedArray64_getitem_nextcarry_outindex_64(
  int64_t* tocarry,
  int64_t* toindex,
  const int64_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_getitem_nextcarry_outindex<int64_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    lenindex,
    lencontent);
}

template <typename C, typename T>
ERROR awkward_IndexedArray_getitem_nextcarry_outindex_mask(
  T* tocarry,
  T* toindex,
  const C* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  int64_t k = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    C j = fromindex[i];
    if (j >= lencontent) {
      return failure("index out of range", i, j, FILENAME(__LINE__));
    }
    else if (j < 0) {
      toindex[i] = -1;
    }
    else {
      tocarry[k] = j;
      toindex[i] = (T)k;
      k++;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_getitem_nextcarry_outindex_mask_64(
  int64_t* tocarry,
  int64_t* toindex,
  const int32_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_getitem_nextcarry_outindex_mask<int32_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    lenindex,
    lencontent);
}
ERROR awkward_IndexedArrayU32_getitem_nextcarry_outindex_mask_64(
  int64_t* tocarry,
  int64_t* toindex,
  const uint32_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_getitem_nextcarry_outindex_mask<uint32_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    lenindex,
    lencontent);
}
ERROR awkward_IndexedArray64_getitem_nextcarry_outindex_mask_64(
  int64_t* tocarry,
  int64_t* toindex,
  const int64_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_getitem_nextcarry_outindex_mask<int64_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    lenindex,
    lencontent);
}

template <typename T>
ERROR awkward_ListOffsetArray_getitem_adjust_offsets(
  T* tooffsets,
  T* tononzero,
  const T* fromoffsets,
  int64_t length,
  const T* nonzero,
  int64_t nonzerolength) {
  int64_t j = 0;
  tooffsets[0] = fromoffsets[0];
  for (int64_t i = 0;  i < length;  i++) {
    T slicestart = fromoffsets[i];
    T slicestop = fromoffsets[i + 1];
    int64_t count = 0;
    while (j < nonzerolength  &&  nonzero[j] < slicestop) {
      tononzero[j] = nonzero[j] - slicestart;
      j++;
      count++;
    }
    tooffsets[i + 1] = tooffsets[i] + count;
  }
  return success();
}
ERROR awkward_ListOffsetArray_getitem_adjust_offsets_64(
  int64_t* tooffsets,
  int64_t* tononzero,
  const int64_t* fromoffsets,
  int64_t length,
  const int64_t* nonzero,
  int64_t nonzerolength) {
  return awkward_ListOffsetArray_getitem_adjust_offsets<int64_t>(
    tooffsets,
    tononzero,
    fromoffsets,
    length,
    nonzero,
    nonzerolength);
}

template <typename T>
ERROR awkward_ListOffsetArray_getitem_adjust_offsets_index(
  T* tooffsets,
  T* tononzero,
  const T* fromoffsets,
  int64_t length,
  const T* index,
  int64_t indexlength,
  const T* nonzero,
  int64_t nonzerolength,
  const int8_t* originalmask,
  int64_t masklength) {
  int64_t k = 0;
  tooffsets[0] = fromoffsets[0];
  for (int64_t i = 0;  i < length;  i++) {
    T slicestart = fromoffsets[i];
    T slicestop = fromoffsets[i + 1];
    int64_t numnull = 0;
    for (int64_t j = slicestart;  j < slicestop;  j++) {
      numnull += (originalmask[j] != 0 ? 1 : 0);
    }
    int64_t nullcount = 0;
    int64_t count = 0;
    while (k < indexlength  &&
           ((index[k] < 0  && nullcount < numnull)  ||
            (index[k] >= 0  &&
             index[k] < nonzerolength  &&
             nonzero[index[k]] < slicestop))) {
      if (index[k] < 0) {
        nullcount++;
      }
      else {
        int64_t j = index[k];
        tononzero[j] = nonzero[j] - slicestart;
      }
      k++;
      count++;
    }
    tooffsets[i + 1] = tooffsets[i] + count;
  }
  return success();
}
ERROR awkward_ListOffsetArray_getitem_adjust_offsets_index_64(
  int64_t* tooffsets,
  int64_t* tononzero,
  const int64_t* fromoffsets,
  int64_t length,
  const int64_t* index,
  int64_t indexlength,
  const int64_t* nonzero,
  int64_t nonzerolength,
  const int8_t* originalmask,
  int64_t masklength) {
  return awkward_ListOffsetArray_getitem_adjust_offsets_index<int64_t>(
    tooffsets,
    tononzero,
    fromoffsets,
    length,
    index,
    indexlength,
    nonzero,
    nonzerolength,
    originalmask,
    masklength);
}

template <typename T>
ERROR awkward_IndexedArray_getitem_adjust_outindex(
  int8_t* tomask,
  T* toindex,
  T* tononzero,
  const T* fromindex,
  int64_t fromindexlength,
  const T* nonzero,
  int64_t nonzerolength) {
  int64_t j = 0;
  int64_t k = 0;
  for (int64_t i = 0;  i < fromindexlength;  i++) {
    T fromval = fromindex[i];
    tomask[i] = (fromval < 0);
    if (fromval < 0) {
      toindex[k] = -1;
      k++;
    }
    else if (j < nonzerolength  &&  fromval == nonzero[j]) {
      tononzero[j] = fromval + (k - j);
      toindex[k] = j;
      j++;
      k++;
    }
  }
  return success();
}
ERROR awkward_IndexedArray_getitem_adjust_outindex_64(
  int8_t* tomask,
  int64_t* toindex,
  int64_t* tononzero,
  const int64_t* fromindex,
  int64_t fromindexlength,
  const int64_t* nonzero,
  int64_t nonzerolength) {
  return awkward_IndexedArray_getitem_adjust_outindex<int64_t>(
    tomask,
    toindex,
    tononzero,
    fromindex,
    fromindexlength,
    nonzero,
    nonzerolength);
}

template <typename C, typename T>
ERROR awkward_IndexedArray_getitem_nextcarry(
  T* tocarry,
  const C* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  int64_t k = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    C j = fromindex[i];
    if (j < 0  ||  j >= lencontent) {
      return failure("index out of range", i, j, FILENAME(__LINE__));
    }
    else {
      tocarry[k] = j;
      k++;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_getitem_nextcarry_64(
  int64_t* tocarry,
  const int32_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_getitem_nextcarry<int32_t, int64_t>(
    tocarry,
    fromindex,
    lenindex,
    lencontent);
}
ERROR awkward_IndexedArrayU32_getitem_nextcarry_64(
  int64_t* tocarry,
  const uint32_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_getitem_nextcarry<uint32_t, int64_t>(
    tocarry,
    fromindex,
    lenindex,
    lencontent);
}
ERROR awkward_IndexedArray64_getitem_nextcarry_64(
  int64_t* tocarry,
  const int64_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_getitem_nextcarry<int64_t, int64_t>(
    tocarry,
    fromindex,
    lenindex,
    lencontent);
}

template <typename C, typename T>
ERROR awkward_IndexedArray_getitem_carry(
  C* toindex,
  const C* fromindex,
  const T* fromcarry,
  int64_t lenindex,
  int64_t lencarry) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (fromcarry[i] >= lenindex) {
      return failure("index out of range", i, fromcarry[i], FILENAME(__LINE__));
    }
    toindex[i] = (C)(fromindex[fromcarry[i]]);
  }
  return success();
}
ERROR awkward_IndexedArray32_getitem_carry_64(
  int32_t* toindex,
  const int32_t* fromindex,
  const int64_t* fromcarry,
  int64_t lenindex,
  int64_t lencarry) {
  return awkward_IndexedArray_getitem_carry<int32_t, int64_t>(
    toindex,
    fromindex,
    fromcarry,
    lenindex,
    lencarry);
}
ERROR awkward_IndexedArrayU32_getitem_carry_64(
  uint32_t* toindex,
  const uint32_t* fromindex,
  const int64_t* fromcarry,
  int64_t lenindex,
  int64_t lencarry) {
  return awkward_IndexedArray_getitem_carry<uint32_t, int64_t>(
    toindex,
    fromindex,
    fromcarry,
    lenindex,
    lencarry);
}
ERROR awkward_IndexedArray64_getitem_carry_64(
  int64_t* toindex,
  const int64_t* fromindex,
  const int64_t* fromcarry,
  int64_t lenindex,
  int64_t lencarry) {
  return awkward_IndexedArray_getitem_carry<int64_t, int64_t>(
    toindex,
    fromindex,
    fromcarry,
    lenindex,
    lencarry);
}

template <typename C>
ERROR awkward_UnionArray_regular_index_getsize(
  int64_t* size,
  const C* fromtags,
  int64_t length) {
  *size = 0;
  for (int64_t i = 0;  i < length;  i++) {
    int64_t tag = (int64_t)fromtags[i];
    if (*size < tag) {
      *size = tag;
    }
  }
  *size = *size + 1;
  return success();
}

ERROR awkward_UnionArray8_regular_index_getsize(
  int64_t* size,
  const int8_t* fromtags,
  int64_t length) {
  return awkward_UnionArray_regular_index_getsize<int8_t>(
    size,
    fromtags,
    length);
}

template <typename C, typename I>
ERROR awkward_UnionArray_regular_index(
  I* toindex,
  I* current,
  int64_t size,
  const C* fromtags,
  int64_t length) {
  int64_t count = 0;
  for (int64_t k = 0;  k < size;  k++) {
    current[k] = 0;
  }
  for (int64_t i = 0;  i < length;  i++) {
    C tag = fromtags[i];
    toindex[(size_t)i] = current[(size_t)tag];
    current[(size_t)tag]++;
  }
  return success();
}
ERROR awkward_UnionArray8_32_regular_index(
  int32_t* toindex,
  int32_t* current,
  int64_t size,
  const int8_t* fromtags,
  int64_t length) {
  return awkward_UnionArray_regular_index<int8_t, int32_t>(
    toindex,
    current,
    size,
    fromtags,
    length);
}
ERROR awkward_UnionArray8_U32_regular_index(
  uint32_t* toindex,
  uint32_t* current,
  int64_t size,
  const int8_t* fromtags,
  int64_t length) {
  return awkward_UnionArray_regular_index<int8_t, uint32_t>(
    toindex,
    current,
    size,
    fromtags,
    length);
}
ERROR awkward_UnionArray8_64_regular_index(
  int64_t* toindex,
  int64_t* current,
  int64_t size,
  const int8_t* fromtags,
  int64_t length) {
  return awkward_UnionArray_regular_index<int8_t, int64_t>(
    toindex,
    current,
    size,
    fromtags,
    length);
}

template <typename T, typename C, typename I>
ERROR awkward_UnionArray_project(
  int64_t* lenout,
  T* tocarry,
  const C* fromtags,
  const I* fromindex,
  int64_t length,
  int64_t which) {
  *lenout = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if (fromtags[i] == which) {
      tocarry[(size_t)(*lenout)] = fromindex[i];
      *lenout = *lenout + 1;
    }
  }
  return success();
}
ERROR awkward_UnionArray8_32_project_64(
  int64_t* lenout,
  int64_t* tocarry,
  const int8_t* fromtags,
  const int32_t* fromindex,
  int64_t length,
  int64_t which) {
  return awkward_UnionArray_project<int64_t, int8_t, int32_t>(
    lenout,
    tocarry,
    fromtags,
    fromindex,
    length,
    which);
}
ERROR awkward_UnionArray8_U32_project_64(
  int64_t* lenout,
  int64_t* tocarry,
  const int8_t* fromtags,
  const uint32_t* fromindex,
  int64_t length,
  int64_t which) {
  return awkward_UnionArray_project<int64_t, int8_t, uint32_t>(
    lenout,
    tocarry,
    fromtags,
    fromindex,
    length,
    which);
}
ERROR awkward_UnionArray8_64_project_64(
  int64_t* lenout,
  int64_t* tocarry,
  const int8_t* fromtags,
  const int64_t* fromindex,
  int64_t length,
  int64_t which) {
  return awkward_UnionArray_project<int64_t, int8_t, int64_t>(
    lenout,
    tocarry,
    fromtags,
    fromindex,
    length,
    which);
}

template <typename T>
ERROR awkward_missing_repeat(
  T* outindex,
  const T* index,
  int64_t indexlength,
  int64_t repetitions,
  int64_t regularsize) {
  for (int64_t i = 0;  i < repetitions;  i++) {
    for (int64_t j = 0;  j < indexlength;  j++) {
      T base = index[j];
      outindex[i*indexlength + j] = base + (base >= 0 ? i*regularsize : 0);
    }
  }

  return success();
}
ERROR awkward_missing_repeat_64(
  int64_t* outindex,
  const int64_t* index,
  int64_t indexlength,
  int64_t repetitions,
  int64_t regularsize) {
  return awkward_missing_repeat<int64_t>(
    outindex,
    index,
    indexlength,
    repetitions,
    regularsize);
}

template <typename T>
ERROR awkward_RegularArray_getitem_jagged_expand(
  T* multistarts,
  T* multistops,
  const T* singleoffsets,
  int64_t regularsize,
  int64_t regularlength) {
  for (int64_t i = 0;  i < regularlength;  i++) {
    for (int64_t j = 0;  j < regularsize;  j++) {
      multistarts[i*regularsize + j] = singleoffsets[j];
      multistops[i*regularsize + j] = singleoffsets[j + 1];
    }
  }
  return success();
}
ERROR awkward_RegularArray_getitem_jagged_expand_64(
  int64_t* multistarts,
  int64_t* multistops,
  const int64_t* singleoffsets,
  int64_t regularsize,
  int64_t regularlength) {
  return awkward_RegularArray_getitem_jagged_expand<int64_t>(
    multistarts,
    multistops,
    singleoffsets,
    regularsize,
    regularlength);
}

template <typename C, typename T>
ERROR awkward_ListArray_getitem_jagged_expand(
  T* multistarts,
  T* multistops,
  const T* singleoffsets,
  T* tocarry,
  const C* fromstarts,
  const C* fromstops,
  int64_t jaggedsize,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    C start = fromstarts[i];
    C stop = fromstops[i];
    if (stop < start) {
      return failure("stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
    }
    if (stop - start != jaggedsize) {
      return failure("cannot fit jagged slice into nested list", i, kSliceNone, FILENAME(__LINE__));
    }
    for (int64_t j = 0;  j < jaggedsize;  j++) {
      multistarts[i*jaggedsize + j] = singleoffsets[j];
      multistops[i*jaggedsize + j] = singleoffsets[j + 1];
      tocarry[i*jaggedsize + j] = start + j;
    }
  }
  return success();
}
ERROR awkward_ListArray32_getitem_jagged_expand_64(
  int64_t* multistarts,
  int64_t* multistops,
  const int64_t* singleoffsets,
  int64_t* tocarry,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t jaggedsize,
  int64_t length) {
  return awkward_ListArray_getitem_jagged_expand<int32_t, int64_t>(
    multistarts,
    multistops,
    singleoffsets,
    tocarry,
    fromstarts,
    fromstops,
    jaggedsize,
    length);
}
ERROR awkward_ListArrayU32_getitem_jagged_expand_64(
  int64_t* multistarts,
  int64_t* multistops,
  const int64_t* singleoffsets,
  int64_t* tocarry,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t jaggedsize,
  int64_t length) {
  return awkward_ListArray_getitem_jagged_expand<uint32_t, int64_t>(
    multistarts,
    multistops,
    singleoffsets,
    tocarry,
    fromstarts,
    fromstops,
    jaggedsize,
    length);
}
ERROR awkward_ListArray64_getitem_jagged_expand_64(
  int64_t* multistarts,
  int64_t* multistops,
  const int64_t* singleoffsets,
  int64_t* tocarry,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t jaggedsize,
  int64_t length) {
  return awkward_ListArray_getitem_jagged_expand<int64_t, int64_t>(
    multistarts,
    multistops,
    singleoffsets,
    tocarry,
    fromstarts,
    fromstops,
    jaggedsize,
    length);
}

template <typename T>
ERROR awkward_ListArray_getitem_jagged_carrylen(
  int64_t* carrylen,
  const T* slicestarts,
  const T* slicestops,
  int64_t sliceouterlen) {
  *carrylen = 0;
  for (int64_t i = 0;  i < sliceouterlen;  i++) {
    *carrylen = *carrylen + (int64_t)(slicestops[i] - slicestarts[i]);
  }
  return success();
}
ERROR awkward_ListArray_getitem_jagged_carrylen_64(
  int64_t* carrylen,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen) {
  return awkward_ListArray_getitem_jagged_carrylen<int64_t>(
    carrylen,
    slicestarts,
    slicestops,
    sliceouterlen);
}

template <typename C, typename T>
ERROR awkward_ListArray_getitem_jagged_apply(
  T* tooffsets,
  T* tocarry,
  const T* slicestarts,
  const T* slicestops,
  int64_t sliceouterlen,
  const T* sliceindex,
  int64_t sliceinnerlen,
  const C* fromstarts,
  const C* fromstops,
  int64_t contentlen) {
  int64_t k = 0;
  for (int64_t i = 0;  i < sliceouterlen;  i++) {
    T slicestart = slicestarts[i];
    T slicestop = slicestops[i];
    tooffsets[i] = (T)k;
    if (slicestart != slicestop) {
      if (slicestop < slicestart) {
        return failure("jagged slice's stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
      }
      if (slicestop > sliceinnerlen) {
        return failure("jagged slice's offsets extend beyond its content", i, slicestop, FILENAME(__LINE__));
      }
      int64_t start = (int64_t)fromstarts[i];
      int64_t stop = (int64_t)fromstops[i];
      if (stop < start) {
        return failure("stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
      }
      if (start != stop  &&  stop > contentlen) {
        return failure("stops[i] > len(content)", i, kSliceNone, FILENAME(__LINE__));
      }
      int64_t count = stop - start;
      for (int64_t j = slicestart;  j < slicestop;  j++) {
        int64_t index = (int64_t)sliceindex[j];
        if (index < 0) {
          index += count;
        }
        if (!(0 <= index  &&  index < count)) {
          return failure("index out of range", i, (int64_t)sliceindex[j], FILENAME(__LINE__));
        }
        tocarry[k] = start + index;
        k++;
      }
    }
    tooffsets[i + 1] = (T)k;
  }
  return success();
}
ERROR awkward_ListArray32_getitem_jagged_apply_64(
  int64_t* tooffsets,
  int64_t* tocarry,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen,
  const int64_t* sliceindex,
  int64_t sliceinnerlen,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t contentlen) {
  return awkward_ListArray_getitem_jagged_apply<int32_t, int64_t>(
    tooffsets,
    tocarry,
    slicestarts,
    slicestops,
    sliceouterlen,
    sliceindex,
    sliceinnerlen,
    fromstarts,
    fromstops,
    contentlen);
}
ERROR awkward_ListArrayU32_getitem_jagged_apply_64(
  int64_t* tooffsets,
  int64_t* tocarry,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen,
  const int64_t* sliceindex,
  int64_t sliceinnerlen,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t contentlen) {
  return awkward_ListArray_getitem_jagged_apply<uint32_t, int64_t>(
    tooffsets,
    tocarry,
    slicestarts,
    slicestops,
    sliceouterlen,
    sliceindex,
    sliceinnerlen,
    fromstarts,
    fromstops,
    contentlen);
}
ERROR awkward_ListArray64_getitem_jagged_apply_64(
  int64_t* tooffsets,
  int64_t* tocarry,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen,
  const int64_t* sliceindex,
  int64_t sliceinnerlen,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t contentlen) {
  return awkward_ListArray_getitem_jagged_apply<int64_t, int64_t>(
    tooffsets,
    tocarry,
    slicestarts,
    slicestops,
    sliceouterlen,
    sliceindex,
    sliceinnerlen,
    fromstarts,
    fromstops,
    contentlen);
}

template <typename T>
ERROR awkward_ListArray_getitem_jagged_numvalid(
  int64_t* numvalid,
  const T* slicestarts,
  const T* slicestops,
  int64_t length,
  const T* missing,
  int64_t missinglength) {
  *numvalid = 0;
  for (int64_t i = 0;  i < length;  i++) {
    T slicestart = slicestarts[i];
    T slicestop = slicestops[i];
    if (slicestart != slicestop) {
      if (slicestop < slicestart) {
        return failure("jagged slice's stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
      }
      if (slicestop > missinglength) {
        return failure("jagged slice's offsets extend beyond its content", i, slicestop, FILENAME(__LINE__));
      }
      for (int64_t j = slicestart;  j < slicestop;  j++) {
        *numvalid = *numvalid + (missing[j] >= 0 ? 1 : 0);
      }
    }
  }
  return success();
}
ERROR awkward_ListArray_getitem_jagged_numvalid_64(
  int64_t* numvalid,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t length,
  const int64_t* missing,
  int64_t missinglength) {
  return awkward_ListArray_getitem_jagged_numvalid<int64_t>(
    numvalid,
    slicestarts,
    slicestops,
    length,
    missing,
    missinglength);
}

template <typename T>
ERROR awkward_ListArray_getitem_jagged_shrink(
  T* tocarry,
  T* tosmalloffsets,
  T* tolargeoffsets,
  const T* slicestarts,
  const T* slicestops,
  int64_t length,
  const T* missing) {
  int64_t k = 0;
  if (length == 0) {
    tosmalloffsets[0] = 0;
    tolargeoffsets[0] = 0;
  }
  else {
    tosmalloffsets[0] = slicestarts[0];
    tolargeoffsets[0] = slicestarts[0];
  }
  for (int64_t i = 0;  i < length;  i++) {
    T slicestart = slicestarts[i];
    T slicestop = slicestops[i];
    if (slicestart != slicestop) {
      T smallcount = 0;
      for (int64_t j = slicestart;  j < slicestop;  j++) {
        if (missing[j] >= 0) {
          tocarry[k] = j;
          k++;
          smallcount++;
        }
      }
      tosmalloffsets[i + 1] = tosmalloffsets[i] + smallcount;
    }
    else {
      tosmalloffsets[i + 1] = tosmalloffsets[i];
    }
    tolargeoffsets[i + 1] = tolargeoffsets[i] + (slicestop - slicestart);
  }
  return success();
}
ERROR awkward_ListArray_getitem_jagged_shrink_64(
  int64_t* tocarry,
  int64_t* tosmalloffsets,
  int64_t* tolargeoffsets,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t length,
  const int64_t* missing) {
  return awkward_ListArray_getitem_jagged_shrink<int64_t>(
    tocarry,
    tosmalloffsets,
    tolargeoffsets,
    slicestarts,
    slicestops,
    length,
    missing);
}

template <typename C, typename T>
ERROR awkward_ListArray_getitem_jagged_descend(
  T* tooffsets,
  const T* slicestarts,
  const T* slicestops,
  int64_t sliceouterlen,
  const C* fromstarts,
  const C* fromstops) {
  if (sliceouterlen == 0) {
    tooffsets[0] = 0;
  }
  else {
    tooffsets[0] = slicestarts[0];
  }
  for (int64_t i = 0;  i < sliceouterlen;  i++) {
    int64_t slicecount = (int64_t)(slicestops[i] -
                                   slicestarts[i]);
    int64_t count = (int64_t)(fromstops[i] -
                              fromstarts[i]);
    if (slicecount != count) {
      return failure("jagged slice inner length differs from array inner length", i, kSliceNone, FILENAME(__LINE__));
    }
    tooffsets[i + 1] = tooffsets[i] + (T)count;
  }
  return success();
}
ERROR awkward_ListArray32_getitem_jagged_descend_64(
  int64_t* tooffsets,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen,
  const int32_t* fromstarts,
  const int32_t* fromstops) {
  return awkward_ListArray_getitem_jagged_descend<int32_t, int64_t>(
    tooffsets,
    slicestarts,
    slicestops,
    sliceouterlen,
    fromstarts,
    fromstops);
}
ERROR awkward_ListArrayU32_getitem_jagged_descend_64(
  int64_t* tooffsets,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen,
  const uint32_t* fromstarts,
  const uint32_t* fromstops) {
  return awkward_ListArray_getitem_jagged_descend<uint32_t, int64_t>(
    tooffsets,
    slicestarts,
    slicestops,
    sliceouterlen,
    fromstarts,
    fromstops);
}
ERROR awkward_ListArray64_getitem_jagged_descend_64(
  int64_t* tooffsets,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen,
  const int64_t* fromstarts,
  const int64_t* fromstops) {
  return awkward_ListArray_getitem_jagged_descend<int64_t, int64_t>(
    tooffsets,
    slicestarts,
    slicestops,
    sliceouterlen,
    fromstarts,
    fromstops);
}

int8_t awkward_Index8_getitem_at_nowrap(
  const int8_t* ptr,
  int64_t at) {
  return ptr[(size_t)(at)];
}
uint8_t awkward_IndexU8_getitem_at_nowrap(
  const uint8_t* ptr,
  int64_t at) {
  return ptr[(size_t)(at)];
}
int32_t awkward_Index32_getitem_at_nowrap(
  const int32_t* ptr,
  int64_t at) {
  return ptr[(size_t)(at)];
}
uint32_t awkward_IndexU32_getitem_at_nowrap(
  const uint32_t* ptr,
  int64_t at) {
  return ptr[(size_t)(at)];
}
int64_t awkward_Index64_getitem_at_nowrap(
  const int64_t* ptr,
  int64_t at) {
  return ptr[(size_t)(at)];
}

bool awkward_NumpyArraybool_getitem_at0(
  const bool* ptr) {
  return ptr[0];
}
int8_t awkward_NumpyArray8_getitem_at0(
  const int8_t* ptr) {
  return ptr[0];
}
uint8_t awkward_NumpyArrayU8_getitem_at0(
  const uint8_t* ptr) {
  return ptr[0];
}
int16_t awkward_NumpyArray16_getitem_at0(
  const int16_t* ptr) {
  return ptr[0];
}
uint16_t awkward_NumpyArrayU16_getitem_at0(
  const uint16_t* ptr) {
  return ptr[0];
}
int32_t awkward_NumpyArray32_getitem_at0(
  const int32_t* ptr) {
  return ptr[0];
}
uint32_t awkward_NumpyArrayU32_getitem_at0(
  const uint32_t* ptr) {
  return ptr[0];
}
int64_t awkward_NumpyArray64_getitem_at0(
  const int64_t* ptr) {
  return ptr[0];
}
uint64_t awkward_NumpyArrayU64_getitem_at0(
  const uint64_t* ptr) {
  return ptr[0];
}
float awkward_NumpyArrayfloat32_getitem_at0(
  const float* ptr) {
  return ptr[0];
}
double awkward_NumpyArrayfloat64_getitem_at0(
  const double* ptr) {
  return ptr[0];
}

void awkward_Index8_setitem_at_nowrap(
  int8_t* ptr,
  int64_t at,
  int8_t value) {
  ptr[(size_t)(at)] = value;
}
void awkward_IndexU8_setitem_at_nowrap(
  uint8_t* ptr,
  int64_t at,
  uint8_t value) {
  ptr[(size_t)(at)] = value;
}
void awkward_Index32_setitem_at_nowrap(
  int32_t* ptr,
  int64_t at,
  int32_t value) {
  ptr[(size_t)(at)] = value;
}
void awkward_IndexU32_setitem_at_nowrap(
  uint32_t* ptr,
  int64_t at,
  uint32_t value) {
  ptr[(size_t)(at)] = value;
}
void awkward_Index64_setitem_at_nowrap(
  int64_t* ptr,
  int64_t at,
  int64_t value) {
  ptr[(size_t)(at)] = value;
}

template <typename T>
ERROR awkward_ByteMaskedArray_getitem_carry(
  int8_t* tomask,
  const int8_t* frommask,
  int64_t lenmask,
  const T* fromcarry,
  int64_t lencarry) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (fromcarry[i] >= lenmask) {
      return failure("index out of range", i, fromcarry[i], FILENAME(__LINE__));
    }
    tomask[i] = frommask[fromcarry[i]];
  }
  return success();
}
ERROR awkward_ByteMaskedArray_getitem_carry_64(
  int8_t* tomask,
  const int8_t* frommask,
  int64_t lenmask,
  const int64_t* fromcarry,
  int64_t lencarry) {
  return awkward_ByteMaskedArray_getitem_carry(
    tomask,
    frommask,
    lenmask,
    fromcarry,
    lencarry);
}

ERROR awkward_ByteMaskedArray_numnull(
  int64_t* numnull,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  *numnull = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if ((mask[i] != 0) != validwhen) {
      *numnull = *numnull + 1;
    }
  }
  return success();
}

template <typename T>
ERROR awkward_ByteMaskedArray_getitem_nextcarry(
  T* tocarry,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if ((mask[i] != 0) == validwhen) {
      tocarry[k] = i;
      k++;
    }
  }
  return success();
}
ERROR awkward_ByteMaskedArray_getitem_nextcarry_64(
  int64_t* tocarry,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  return awkward_ByteMaskedArray_getitem_nextcarry<int64_t>(
    tocarry,
    mask,
    length,
    validwhen);
}

template <typename T>
ERROR awkward_ByteMaskedArray_getitem_nextcarry_outindex(
  T* tocarry,
  T* outindex,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if ((mask[i] != 0) == validwhen) {
      tocarry[k] = i;
      outindex[i] = (T)k;
      k++;
    }
    else {
      outindex[i] = -1;
    }
  }
  return success();
}
ERROR awkward_ByteMaskedArray_getitem_nextcarry_outindex_64(
  int64_t* tocarry,
  int64_t* outindex,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  return awkward_ByteMaskedArray_getitem_nextcarry_outindex<int64_t>(
    tocarry,
    outindex,
    mask,
    length,
    validwhen);
}

template <typename T>
ERROR awkward_ByteMaskedArray_toIndexedOptionArray(
  T* toindex,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[i] = ((mask[i] != 0) == validwhen ? i : -1);
  }
  return success();
}
ERROR awkward_ByteMaskedArray_toIndexedOptionArray64(
  int64_t* toindex,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  return awkward_ByteMaskedArray_toIndexedOptionArray<int64_t>(
    toindex,
    mask,
    length,
    validwhen);
}

ERROR awkward_Content_getitem_next_missing_jagged_getmaskstartstop(
  int64_t* index_in,
  int64_t* offsets_in,
  int64_t* mask_out,
  int64_t* starts_out,
  int64_t* stops_out,
  int64_t length) {
  int64_t k = 0;
  for (int64_t i = 0; i < length; ++i) {
    starts_out[i] = offsets_in[k];
    if (index_in[i] < 0) {
      mask_out[i] = -1;
      stops_out[i] = offsets_in[k];
    }
    else {
      mask_out[i] = i;
      k++;
      stops_out[i] = offsets_in[k];
    }
  }
  return success();
}

template <typename T>
ERROR awkward_MaskedArray_getitem_next_jagged_project(
  T* index,
  int64_t* starts_in,
  int64_t* stops_in,
  int64_t* starts_out,
  int64_t* stops_out,
  int64_t length) {
  int64_t k = 0;
  for (int64_t i = 0; i < length; ++i) {
    if (index[i] >= 0) {
      starts_out[k] = starts_in[i];
      stops_out[k] = stops_in[i];
      k++;
    }
  }
  return success();
}

ERROR awkward_MaskedArray32_getitem_next_jagged_project(
  int32_t* index,
  int64_t* starts_in,
  int64_t* stops_in,
  int64_t* starts_out,
  int64_t* stops_out,
  int64_t length) {
  return awkward_MaskedArray_getitem_next_jagged_project<int32_t>(
    index,
    starts_in,
    stops_in,
    starts_out,
    stops_out,
    length);
}
ERROR awkward_MaskedArrayU32_getitem_next_jagged_project(
  uint32_t* index,
  int64_t* starts_in,
  int64_t* stops_in,
  int64_t* starts_out,
  int64_t* stops_out,
  int64_t length) {
  return awkward_MaskedArray_getitem_next_jagged_project<uint32_t>(
    index,
    starts_in,
    stops_in,
    starts_out,
    stops_out,
    length);
}
ERROR awkward_MaskedArray64_getitem_next_jagged_project(
  int64_t* index,
  int64_t* starts_in,
  int64_t* stops_in,
  int64_t* starts_out,
  int64_t* stops_out,
  int64_t length) {
  return awkward_MaskedArray_getitem_next_jagged_project<int64_t>(
    index,
    starts_in,
    stops_in,
    starts_out,
    stops_out,
    length);
}
