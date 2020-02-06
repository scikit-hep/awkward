// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>
#include <vector>

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
ERROR awkward_regularize_arrayslice(T* flatheadptr, int64_t lenflathead, int64_t length) {
  for (int64_t i = 0;  i < lenflathead;  i++) {
    T original = flatheadptr[i];
    if (flatheadptr[i] < 0) {
      flatheadptr[i] += length;
    }
    if (flatheadptr[i] < 0  ||  flatheadptr[i] >= length) {
      return failure("index out of range", kSliceNone, original);
    }
  }
  return success();
}
ERROR awkward_regularize_arrayslice_64(int64_t* flatheadptr, int64_t lenflathead, int64_t length) {
  return awkward_regularize_arrayslice<int64_t>(flatheadptr, lenflathead, length);
}

ERROR awkward_index8_to_index64(int64_t* toptr, const int8_t* fromptr, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}
ERROR awkward_indexU8_to_index64(int64_t* toptr, const uint8_t* fromptr, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}
ERROR awkward_index32_to_index64(int64_t* toptr, const int32_t* fromptr, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}
ERROR awkward_indexU32_to_index64(int64_t* toptr, const uint32_t* fromptr, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}

template <typename C, typename T>
ERROR awkward_index_carry(C* toindex, const C* fromindex, const T* carry, int64_t fromindexoffset, int64_t lenfromindex, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    T j = carry[i];
    if (j > lenfromindex) {
      return failure("index out of range", kSliceNone, j);
    }
    toindex[i] = fromindex[(size_t)(fromindexoffset + j)];
  }
  return success();
}
ERROR awkward_index8_carry_64(int8_t* toindex, const int8_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t lenfromindex, int64_t length) {
  return awkward_index_carry<int8_t, int64_t>(toindex, fromindex, carry, fromindexoffset, lenfromindex, length);
}
ERROR awkward_indexU8_carry_64(uint8_t* toindex, const uint8_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t lenfromindex, int64_t length) {
  return awkward_index_carry<uint8_t, int64_t>(toindex, fromindex, carry, fromindexoffset, lenfromindex, length);
}
ERROR awkward_index32_carry_64(int32_t* toindex, const int32_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t lenfromindex, int64_t length) {
  return awkward_index_carry<int32_t, int64_t>(toindex, fromindex, carry, fromindexoffset, lenfromindex, length);
}
ERROR awkward_indexU32_carry_64(uint32_t* toindex, const uint32_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t lenfromindex, int64_t length) {
  return awkward_index_carry<uint32_t, int64_t>(toindex, fromindex, carry, fromindexoffset, lenfromindex, length);
}
ERROR awkward_index64_carry_64(int64_t* toindex, const int64_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t lenfromindex, int64_t length) {
  return awkward_index_carry<int64_t, int64_t>(toindex, fromindex, carry, fromindexoffset, lenfromindex, length);
}

template <typename C, typename T>
ERROR awkward_index_carry_nocheck(C* toindex, const C* fromindex, const T* carry, int64_t fromindexoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[i] = fromindex[(size_t)(fromindexoffset + carry[i])];
  }
  return success();
}
ERROR awkward_index8_carry_nocheck_64(int8_t* toindex, const int8_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t length) {
  return awkward_index_carry_nocheck<int8_t, int64_t>(toindex, fromindex, carry, fromindexoffset, length);
}
ERROR awkward_indexU8_carry_nocheck_64(uint8_t* toindex, const uint8_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t length) {
  return awkward_index_carry_nocheck<uint8_t, int64_t>(toindex, fromindex, carry, fromindexoffset, length);
}
ERROR awkward_index32_carry_nocheck_64(int32_t* toindex, const int32_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t length) {
  return awkward_index_carry_nocheck<int32_t, int64_t>(toindex, fromindex, carry, fromindexoffset, length);
}
ERROR awkward_indexU32_carry_nocheck_64(uint32_t* toindex, const uint32_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t length) {
  return awkward_index_carry_nocheck<uint32_t, int64_t>(toindex, fromindex, carry, fromindexoffset, length);
}
ERROR awkward_index64_carry_nocheck_64(int64_t* toindex, const int64_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t length) {
  return awkward_index_carry_nocheck<int64_t, int64_t>(toindex, fromindex, carry, fromindexoffset, length);
}

template <typename T>
ERROR awkward_slicearray_ravel(T* toptr, const T* fromptr, int64_t ndim, const int64_t* shape, const int64_t* strides) {
  if (ndim == 1) {
    for (T i = 0;  i < shape[0];  i++) {
      toptr[i] = fromptr[i*strides[0]];
    }
  }
  else {
    for (T i = 0;  i < shape[0];  i++) {
      ERROR err = awkward_slicearray_ravel<T>(&toptr[i*shape[1]], &fromptr[i*strides[0]], ndim - 1, &shape[1], &strides[1]);
      if (err.str != nullptr) {
        return err;
      }
    }
  }
  return success();
}
ERROR awkward_slicearray_ravel_64(int64_t* toptr, const int64_t* fromptr, int64_t ndim, const int64_t* shape, const int64_t* strides) {
  return awkward_slicearray_ravel<int64_t>(toptr, fromptr, ndim, shape, strides);
}

template <typename T>
ERROR awkward_carry_arange(T* toptr, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i] = i;
  }
  return success();
}
ERROR awkward_carry_arange_64(int64_t* toptr, int64_t length) {
  return awkward_carry_arange<int64_t>(toptr, length);
}

template <typename ID, typename T>
ERROR awkward_identities_getitem_carry(ID* newidentitiesptr, const ID* identitiesptr, const T* carryptr, int64_t lencarry, int64_t offset, int64_t width, int64_t length) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (carryptr[i] >= length) {
      return failure("index out of range", kSliceNone, carryptr[i]);
    }
    for (int64_t j = 0;  j < width;  j++) {
      newidentitiesptr[width*i + j] = identitiesptr[offset + width*carryptr[i] + j];
    }
  }
  return success();
}
ERROR awkward_identities32_getitem_carry_64(int32_t* newidentitiesptr, const int32_t* identitiesptr, const int64_t* carryptr, int64_t lencarry, int64_t offset, int64_t width, int64_t length) {
  return awkward_identities_getitem_carry<int32_t, int64_t>(newidentitiesptr, identitiesptr, carryptr, lencarry, offset, width, length);
}
ERROR awkward_identities64_getitem_carry_64(int64_t* newidentitiesptr, const int64_t* identitiesptr, const int64_t* carryptr, int64_t lencarry, int64_t offset, int64_t width, int64_t length) {
  return awkward_identities_getitem_carry<int64_t, int64_t>(newidentitiesptr, identitiesptr, carryptr, lencarry, offset, width, length);
}

template <typename T>
ERROR awkward_numpyarray_contiguous_init(T* toptr, int64_t skip, int64_t stride) {
  for (int64_t i = 0;  i < skip;  i++) {
    toptr[i] = i*stride;
  }
  return success();
}
ERROR awkward_numpyarray_contiguous_init_64(int64_t* toptr, int64_t skip, int64_t stride) {
  return awkward_numpyarray_contiguous_init<int64_t>(toptr, skip, stride);
}

template <typename T>
ERROR awkward_numpyarray_contiguous_copy(uint8_t* toptr, const uint8_t* fromptr, int64_t len, int64_t stride, int64_t offset, const T* pos) {
  for (int64_t i = 0;  i < len;  i++) {
    memcpy(&toptr[i*stride], &fromptr[offset + (int64_t)pos[i]], (size_t)stride);
  }
  return success();
}
ERROR awkward_numpyarray_contiguous_copy_64(uint8_t* toptr, const uint8_t* fromptr, int64_t len, int64_t stride, int64_t offset, const int64_t* pos) {
  return awkward_numpyarray_contiguous_copy<int64_t>(toptr, fromptr, len, stride, offset, pos);
}

template <typename T>
ERROR awkward_numpyarray_contiguous_next(T* topos, const T* frompos, int64_t len, int64_t skip, int64_t stride) {
  for (int64_t i = 0;  i < len;  i++) {
    for (int64_t j = 0;  j < skip;  j++) {
      topos[i*skip + j] = frompos[i] + j*stride;
    }
  }
  return success();
}
ERROR awkward_numpyarray_contiguous_next_64(int64_t* topos, const int64_t* frompos, int64_t len, int64_t skip, int64_t stride) {
  return awkward_numpyarray_contiguous_next<int64_t>(topos, frompos, len, skip, stride);
}

template <typename T>
ERROR awkward_numpyarray_getitem_next_null(uint8_t* toptr, const uint8_t* fromptr, int64_t len, int64_t stride, int64_t offset, const T* pos) {
  for (int64_t i = 0;  i < len;  i++) {
    std::memcpy(&toptr[i*stride], &fromptr[offset + pos[i]*stride], (size_t)stride);
  }
  return success();
}
ERROR awkward_numpyarray_getitem_next_null_64(uint8_t* toptr, const uint8_t* fromptr, int64_t len, int64_t stride, int64_t offset, const int64_t* pos) {
  return awkward_numpyarray_getitem_next_null(toptr, fromptr, len, stride, offset, pos);
}

template <typename T>
ERROR awkward_numpyarray_getitem_next_at(T* nextcarryptr, const T* carryptr, int64_t lencarry, int64_t skip, int64_t at) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    nextcarryptr[i] = skip*carryptr[i] + at;
  }
  return success();
}
ERROR awkward_numpyarray_getitem_next_at_64(int64_t* nextcarryptr, const int64_t* carryptr, int64_t lencarry, int64_t skip, int64_t at) {
  return awkward_numpyarray_getitem_next_at(nextcarryptr, carryptr, lencarry, skip, at);
}

template <typename T>
ERROR awkward_numpyarray_getitem_next_range(T* nextcarryptr, const T* carryptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenhead;  j++) {
      nextcarryptr[i*lenhead + j] = skip*carryptr[i] + start + j*step;
    }
  }
  return success();
}
ERROR awkward_numpyarray_getitem_next_range_64(int64_t* nextcarryptr, const int64_t* carryptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step) {
  return awkward_numpyarray_getitem_next_range(nextcarryptr, carryptr, lencarry, lenhead, skip, start, step);
}

template <typename T>
ERROR awkward_numpyarray_getitem_next_range_advanced(T* nextcarryptr, T* nextadvancedptr, const T* carryptr, const T* advancedptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenhead;  j++) {
      nextcarryptr[i*lenhead + j] = skip*carryptr[i] + start + j*step;
      nextadvancedptr[i*lenhead + j] = advancedptr[i];
    }
  }
  return success();
}
ERROR awkward_numpyarray_getitem_next_range_advanced_64(int64_t* nextcarryptr, int64_t* nextadvancedptr, const int64_t* carryptr, const int64_t* advancedptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step) {
  return awkward_numpyarray_getitem_next_range_advanced(nextcarryptr, nextadvancedptr, carryptr, advancedptr, lencarry, lenhead, skip, start, step);
}

template <typename T>
ERROR awkward_numpyarray_getitem_next_array(T* nextcarryptr, T* nextadvancedptr, const T* carryptr, const T* flatheadptr, int64_t lencarry, int64_t lenflathead, int64_t skip) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenflathead;  j++) {
      nextcarryptr[i*lenflathead + j] = skip*carryptr[i] + flatheadptr[j];
      nextadvancedptr[i*lenflathead + j] = j;
    }
  }
  return success();
}
ERROR awkward_numpyarray_getitem_next_array_64(int64_t* nextcarryptr, int64_t* nextadvancedptr, const int64_t* carryptr, const int64_t* flatheadptr, int64_t lencarry, int64_t lenflathead, int64_t skip) {
  return awkward_numpyarray_getitem_next_array(nextcarryptr, nextadvancedptr, carryptr, flatheadptr, lencarry, lenflathead, skip);
}

template <typename T>
ERROR awkward_numpyarray_getitem_next_array_advanced(T* nextcarryptr, const T* carryptr, const T* advancedptr, const T* flatheadptr, int64_t lencarry, int64_t skip) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    nextcarryptr[i] = skip*carryptr[i] + flatheadptr[advancedptr[i]];
  }
  return success();
}
ERROR awkward_numpyarray_getitem_next_array_advanced_64(int64_t* nextcarryptr, const int64_t* carryptr, const int64_t* advancedptr, const int64_t* flatheadptr, int64_t lencarry, int64_t skip) {
  return awkward_numpyarray_getitem_next_array_advanced(nextcarryptr, carryptr, advancedptr, flatheadptr, lencarry, skip);
}

ERROR awkward_numpyarray_getitem_boolean_numtrue(int64_t* numtrue, const int8_t* fromptr, int64_t byteoffset, int64_t length, int64_t stride) {
  *numtrue = 0;
  for (int64_t i = 0;  i < length;  i += stride) {
    *numtrue = *numtrue + (fromptr[byteoffset + i] != 0);
  }
  return success();
}

template <typename T>
ERROR awkward_numpyarray_getitem_boolean_nonzero(T* toptr, const int8_t* fromptr, int64_t byteoffset, int64_t length, int64_t stride) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i += stride) {
    if (fromptr[byteoffset + i] != 0) {
      toptr[k] = i;
      k++;
    }
  }
  return success();
}

ERROR awkward_numpyarray_getitem_boolean_nonzero_64(int64_t* toptr, const int8_t* fromptr, int64_t byteoffset, int64_t length, int64_t stride) {
  return awkward_numpyarray_getitem_boolean_nonzero<int64_t>(toptr, fromptr, byteoffset, length, stride);
}

template <typename C, typename T>
ERROR awkward_listarray_getitem_next_at(T* tocarry, const C* fromstarts, const C* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length;
    }
    if (!(0 <= regular_at  &&  regular_at < length)) {
      return failure("index out of range", i, at);
    }
    tocarry[i] = fromstarts[startsoffset + i] + regular_at;
  }
  return success();
}
ERROR awkward_listarray32_getitem_next_at_64(int64_t* tocarry, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at) {
  return awkward_listarray_getitem_next_at<int32_t, int64_t>(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at);
}
ERROR awkward_listarrayU32_getitem_next_at_64(int64_t* tocarry, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at) {
  return awkward_listarray_getitem_next_at<uint32_t, int64_t>(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at);
}
ERROR awkward_listarray64_getitem_next_at_64(int64_t* tocarry, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at) {
  return awkward_listarray_getitem_next_at<int64_t, int64_t>(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at);
}

template <typename C>
ERROR awkward_listarray_getitem_next_range_carrylength(int64_t* carrylength, const C* fromstarts, const C* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
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
  return success();
}
ERROR awkward_listarray32_getitem_next_range_carrylength(int64_t* carrylength, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  return awkward_listarray_getitem_next_range_carrylength<int32_t>(carrylength, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, start, stop, step);
}
ERROR awkward_listarrayU32_getitem_next_range_carrylength(int64_t* carrylength, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  return awkward_listarray_getitem_next_range_carrylength<uint32_t>(carrylength, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, start, stop, step);
}
ERROR awkward_listarray64_getitem_next_range_carrylength(int64_t* carrylength, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  return awkward_listarray_getitem_next_range_carrylength<int64_t>(carrylength, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, start, stop, step);
}

template <typename C, typename T>
ERROR awkward_listarray_getitem_next_range(C* tooffsets, T* tocarry, const C* fromstarts, const C* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  int64_t k = 0;
  tooffsets[0] = 0;
  if (step > 0) {
    for (int64_t i = 0;  i < lenstarts;  i++) {
      int64_t length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop, step > 0, start != kSliceNone, stop != kSliceNone, length);
      for (int64_t j = regular_start;  j < regular_stop;  j += step) {
        tocarry[k] = fromstarts[startsoffset + i] + j;
        k++;
      }
      tooffsets[i + 1] = (C)k;
    }
  }
  else {
    for (int64_t i = 0;  i < lenstarts;  i++) {
      int64_t length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop, step > 0, start != kSliceNone, stop != kSliceNone, length);
      for (int64_t j = regular_start;  j > regular_stop;  j += step) {
        tocarry[k] = fromstarts[startsoffset + i] + j;
        k++;
      }
      tooffsets[i + 1] = (C)k;
    }
  }
  return success();
}
ERROR awkward_listarray32_getitem_next_range_64(int32_t* tooffsets, int64_t* tocarry, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  return awkward_listarray_getitem_next_range<int32_t, int64_t>(tooffsets, tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, start, stop, step);
}
ERROR awkward_listarrayU32_getitem_next_range_64(uint32_t* tooffsets, int64_t* tocarry, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  return awkward_listarray_getitem_next_range<uint32_t, int64_t>(tooffsets, tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, start, stop, step);
}
ERROR awkward_listarray64_getitem_next_range_64(int64_t* tooffsets, int64_t* tocarry, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step) {
  return awkward_listarray_getitem_next_range<int64_t, int64_t>(tooffsets, tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, start, stop, step);
}

template <typename C, typename T>
ERROR awkward_listarray_getitem_next_range_counts(int64_t* total, const C* fromoffsets, int64_t lenstarts) {
  *total = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    *total = *total + fromoffsets[i + 1] - fromoffsets[i];
  }
  return success();
}
ERROR awkward_listarray32_getitem_next_range_counts_64(int64_t* total, const int32_t* fromoffsets, int64_t lenstarts) {
  return awkward_listarray_getitem_next_range_counts<int32_t, int64_t>(total, fromoffsets, lenstarts);
}
ERROR awkward_listarrayU32_getitem_next_range_counts_64(int64_t* total, const uint32_t* fromoffsets, int64_t lenstarts) {
  return awkward_listarray_getitem_next_range_counts<uint32_t, int64_t>(total, fromoffsets, lenstarts);
}
ERROR awkward_listarray64_getitem_next_range_counts_64(int64_t* total, const int64_t* fromoffsets, int64_t lenstarts) {
  return awkward_listarray_getitem_next_range_counts<int64_t, int64_t>(total, fromoffsets, lenstarts);
}

template <typename C, typename T>
ERROR awkward_listarray_getitem_next_range_spreadadvanced(T* toadvanced, const T* fromadvanced, const C* fromoffsets, int64_t lenstarts) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    C count = fromoffsets[i + 1] - fromoffsets[i];
    for (int64_t j = 0;  j < count;  j++) {
      toadvanced[fromoffsets[i] + j] = fromadvanced[i];
    }
  }
  return success();
}
ERROR awkward_listarray32_getitem_next_range_spreadadvanced_64(int64_t* toadvanced, const int64_t* fromadvanced, const int32_t* fromoffsets, int64_t lenstarts) {
  return awkward_listarray_getitem_next_range_spreadadvanced<int32_t, int64_t>(toadvanced, fromadvanced, fromoffsets, lenstarts);
}
ERROR awkward_listarrayU32_getitem_next_range_spreadadvanced_64(int64_t* toadvanced, const int64_t* fromadvanced, const uint32_t* fromoffsets, int64_t lenstarts) {
  return awkward_listarray_getitem_next_range_spreadadvanced<uint32_t, int64_t>(toadvanced, fromadvanced, fromoffsets, lenstarts);
}
ERROR awkward_listarray64_getitem_next_range_spreadadvanced_64(int64_t* toadvanced, const int64_t* fromadvanced, const int64_t* fromoffsets, int64_t lenstarts) {
  return awkward_listarray_getitem_next_range_spreadadvanced<int64_t, int64_t>(toadvanced, fromadvanced, fromoffsets, lenstarts);
}

template <typename C, typename T>
ERROR awkward_listarray_getitem_next_array(T* tocarry, T* toadvanced, const C* fromstarts, const C* fromstops, const T* fromarray, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    if (fromstops[stopsoffset + i] < fromstarts[startsoffset + i]) {
      return failure("stops[i] < starts[i]", i, kSliceNone);
    }
    if (fromstarts[startsoffset + i] != fromstops[stopsoffset + i]  &&  fromstops[stopsoffset + i] > lencontent) {
      return failure("stops[i] > len(content)", i, kSliceNone);
    }
    int64_t length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
    for (int64_t j = 0;  j < lenarray;  j++) {
      int64_t regular_at = fromarray[j];
      if (regular_at < 0) {
        regular_at += length;
      }
      if (!(0 <= regular_at  &&  regular_at < length)) {
        return failure("index out of range", i, fromarray[j]);
      }
      tocarry[i*lenarray + j] = fromstarts[startsoffset + i] + regular_at;
      toadvanced[i*lenarray + j] = j;
    }
  }
  return success();
}
ERROR awkward_listarray32_getitem_next_array_64(int64_t* tocarry, int64_t* toadvanced, const int32_t* fromstarts, const int32_t* fromstops, const int64_t* fromarray, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray_getitem_next_array<int32_t, int64_t>(tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}
ERROR awkward_listarrayU32_getitem_next_array_64(int64_t* tocarry, int64_t* toadvanced, const uint32_t* fromstarts, const uint32_t* fromstops, const int64_t* fromarray, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray_getitem_next_array<uint32_t, int64_t>(tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}
ERROR awkward_listarray64_getitem_next_array_64(int64_t* tocarry, int64_t* toadvanced, const int64_t* fromstarts, const int64_t* fromstops, const int64_t* fromarray, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray_getitem_next_array<int64_t, int64_t>(tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}

template <typename C, typename T>
ERROR awkward_listarray_getitem_next_array_advanced(T* tocarry, T* toadvanced, const C* fromstarts, const C* fromstops, const T* fromarray, const T* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    if (fromstops[stopsoffset + i] < fromstarts[startsoffset + i]) {
      return failure("stops[i] < starts[i]", i, kSliceNone);
    }
    if (fromstarts[startsoffset + i] != fromstops[stopsoffset + i]  &&  fromstops[stopsoffset + i] > lencontent) {
      return failure("stops[i] > len(content)", i, kSliceNone);
    }
    int64_t length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
    int64_t regular_at = fromarray[fromadvanced[i]];
    if (regular_at < 0) {
      regular_at += length;
    }
    if (!(0 <= regular_at  &&  regular_at < length)) {
      return failure("index out of range", i, fromarray[fromadvanced[i]]);
    }
    tocarry[i] = fromstarts[startsoffset + i] + regular_at;
    toadvanced[i] = i;
  }
  return success();
}
ERROR awkward_listarray32_getitem_next_array_advanced_64(int64_t* tocarry, int64_t* toadvanced, const int32_t* fromstarts, const int32_t* fromstops, const int64_t* fromarray, const int64_t* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray_getitem_next_array_advanced<int32_t, int64_t>(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}
ERROR awkward_listarrayU32_getitem_next_array_advanced_64(int64_t* tocarry, int64_t* toadvanced, const uint32_t* fromstarts, const uint32_t* fromstops, const int64_t* fromarray, const int64_t* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray_getitem_next_array_advanced<uint32_t, int64_t>(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}
ERROR awkward_listarray64_getitem_next_array_advanced_64(int64_t* tocarry, int64_t* toadvanced, const int64_t* fromstarts, const int64_t* fromstops, const int64_t* fromarray, const int64_t* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray_getitem_next_array_advanced<int64_t, int64_t>(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}

template <typename C, typename T>
ERROR awkward_listarray_getitem_carry(C* tostarts, C* tostops, const C* fromstarts, const C* fromstops, const T* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lencarry) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (fromcarry[i] >= lenstarts) {
      return failure("index out of range", i, fromcarry[i]);
    }
    tostarts[i] = (C)(fromstarts[startsoffset + fromcarry[i]]);
    tostops[i] = (C)(fromstops[stopsoffset + fromcarry[i]]);
  }
  return success();
}
ERROR awkward_listarray32_getitem_carry_64(int32_t* tostarts, int32_t* tostops, const int32_t* fromstarts, const int32_t* fromstops, const int64_t* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lencarry) {
  return awkward_listarray_getitem_carry<int32_t, int64_t>(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lenstarts, lencarry);
}
ERROR awkward_listarrayU32_getitem_carry_64(uint32_t* tostarts, uint32_t* tostops, const uint32_t* fromstarts, const uint32_t* fromstops, const int64_t* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lencarry) {
  return awkward_listarray_getitem_carry<uint32_t, int64_t>(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lenstarts, lencarry);
}
ERROR awkward_listarray64_getitem_carry_64(int64_t* tostarts, int64_t* tostops, const int64_t* fromstarts, const int64_t* fromstops, const int64_t* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lencarry) {
  return awkward_listarray_getitem_carry<int64_t, int64_t>(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lenstarts, lencarry);
}

template <typename T>
ERROR awkward_regulararray_getitem_next_at(T* tocarry, int64_t at, int64_t len, int64_t size) {
  int64_t regular_at = at;
  if (regular_at < 0) {
    regular_at += size;
  }
  if (!(0 <= regular_at  &&  regular_at < size)) {
    return failure("index out of range", kSliceNone, at);
  }
  for (int64_t i = 0;  i < len;  i++) {
    tocarry[i] = i*size + regular_at;
  }
  return success();
}
ERROR awkward_regulararray_getitem_next_at_64(int64_t* tocarry, int64_t at, int64_t len, int64_t size) {
  return awkward_regulararray_getitem_next_at<int64_t>(tocarry, at, len, size);
}

template <typename T>
ERROR awkward_regulararray_getitem_next_range(T* tocarry, int64_t regular_start, int64_t step, int64_t len, int64_t size, int64_t nextsize) {
  for (int64_t i = 0;  i < len;  i++) {
    for (int64_t j = 0;  j < nextsize;  j++) {
      tocarry[i*nextsize + j] = i*size + regular_start + j*step;
    }
  }
  return success();
}
ERROR awkward_regulararray_getitem_next_range_64(int64_t* tocarry, int64_t regular_start, int64_t step, int64_t len, int64_t size, int64_t nextsize) {
  return awkward_regulararray_getitem_next_range<int64_t>(tocarry, regular_start, step, len, size, nextsize);
}

template <typename T>
ERROR awkward_regulararray_getitem_next_range_spreadadvanced(T* toadvanced, const T* fromadvanced, int64_t len, int64_t nextsize) {
  for (int64_t i = 0;  i < len;  i++) {
    for (int64_t j = 0;  j < nextsize;  j++) {
      toadvanced[i*nextsize + j] = fromadvanced[i];
    }
  }
  return success();
}
ERROR awkward_regulararray_getitem_next_range_spreadadvanced_64(int64_t* toadvanced, const int64_t* fromadvanced, int64_t len, int64_t nextsize) {
  return awkward_regulararray_getitem_next_range_spreadadvanced<int64_t>(toadvanced, fromadvanced, len, nextsize);
}

template <typename T>
ERROR awkward_regulararray_getitem_next_array_regularize(T* toarray, const T* fromarray, int64_t lenarray, int64_t size) {
  for (int64_t j = 0;  j < lenarray;  j++) {
    toarray[j] = fromarray[j];
    if (toarray[j] < 0) {
      toarray[j] += size;
    }
    if (!(0 <= toarray[j]  &&  toarray[j] < size)) {
      return failure("index out of range", kSliceNone, fromarray[j]);
    }
  }
  return success();
}
ERROR awkward_regulararray_getitem_next_array_regularize_64(int64_t* toarray, const int64_t* fromarray, int64_t lenarray, int64_t size) {
  return awkward_regulararray_getitem_next_array_regularize<int64_t>(toarray, fromarray, lenarray, size);
}

template <typename T>
ERROR awkward_regulararray_getitem_next_array(T* tocarry, T* toadvanced, const T* fromarray, int64_t len, int64_t lenarray, int64_t size) {
  for (int64_t i = 0;  i < len;  i++) {
    for (int64_t j = 0;  j < lenarray;  j++) {
      tocarry[i*lenarray + j] = i*size + fromarray[j];
      toadvanced[i*lenarray + j] = j;
    }
  }
  return success();
}
ERROR awkward_regulararray_getitem_next_array_64(int64_t* tocarry, int64_t* toadvanced, const int64_t* fromarray, int64_t len, int64_t lenarray, int64_t size) {
  return awkward_regulararray_getitem_next_array<int64_t>(tocarry, toadvanced, fromarray, len, lenarray, size);
}

template <typename T>
ERROR awkward_regulararray_getitem_next_array_advanced(T* tocarry, T* toadvanced, const T* fromadvanced, const T* fromarray, int64_t len, int64_t lenarray, int64_t size) {
  for (int64_t i = 0;  i < len;  i++) {
    tocarry[i] = i*size + fromarray[fromadvanced[i]];
    toadvanced[i] = i;
  }
  return success();
}
ERROR awkward_regulararray_getitem_next_array_advanced_64(int64_t* tocarry, int64_t* toadvanced, const int64_t* fromadvanced, const int64_t* fromarray, int64_t len, int64_t lenarray, int64_t size) {
  return awkward_regulararray_getitem_next_array_advanced<int64_t>(tocarry, toadvanced, fromadvanced, fromarray, len, lenarray, size);
}

template <typename T>
ERROR awkward_regulararray_getitem_carry(T* tocarry, const T* fromcarry, int64_t lencarry, int64_t size) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < size;  j++) {
      tocarry[i*size + j] = fromcarry[i]*size + j;
    }
  }
  return success();
}
ERROR awkward_regulararray_getitem_carry_64(int64_t* tocarry, const int64_t* fromcarry, int64_t lencarry, int64_t size) {
  return awkward_regulararray_getitem_carry<int64_t>(tocarry, fromcarry, lencarry, size);
}

template <typename C>
ERROR awkward_indexedarray_numnull(int64_t* numnull, const C* fromindex, int64_t indexoffset, int64_t lenindex) {
  *numnull = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    if (fromindex[indexoffset + i] < 0) {
      *numnull = *numnull + 1;
    }
  }
  return success();
}
ERROR awkward_indexedarray32_numnull(int64_t* numnull, const int32_t* fromindex, int64_t indexoffset, int64_t lenindex) {
  return awkward_indexedarray_numnull<int32_t>(numnull, fromindex, indexoffset, lenindex);
}
ERROR awkward_indexedarrayU32_numnull(int64_t* numnull, const uint32_t* fromindex, int64_t indexoffset, int64_t lenindex) {
  return awkward_indexedarray_numnull<uint32_t>(numnull, fromindex, indexoffset, lenindex);
}
ERROR awkward_indexedarray64_numnull(int64_t* numnull, const int64_t* fromindex, int64_t indexoffset, int64_t lenindex) {
  return awkward_indexedarray_numnull<int64_t>(numnull, fromindex, indexoffset, lenindex);
}

template <typename CIN, typename COUT, typename T>
ERROR awkward_indexedarray_getitem_nextcarry_outindex(T* tocarry, COUT* toindex, const CIN* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  int64_t k = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    CIN j = fromindex[indexoffset + i];
    if (j >= lencontent) {
      return failure("index out of range", i, j);
    }
    else if (j < 0) {
      toindex[i] = -1;
    }
    else {
      tocarry[k] = j;
      toindex[i] = (COUT)k;
      k++;
    }
  }
  return success();
}
ERROR awkward_indexedarray32_getitem_nextcarry_outindex_64(int64_t* tocarry, int32_t* toindex, const int32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  return awkward_indexedarray_getitem_nextcarry_outindex<int32_t, int32_t, int64_t>(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent);
}
ERROR awkward_indexedarrayU32_getitem_nextcarry_outindex_64(int64_t* tocarry, uint32_t* toindex, const uint32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  return awkward_indexedarray_getitem_nextcarry_outindex<uint32_t, uint32_t, int64_t>(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent);
}
ERROR awkward_indexedarray64_getitem_nextcarry_outindex_64(int64_t* tocarry, int64_t* toindex, const int64_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  return awkward_indexedarray_getitem_nextcarry_outindex<int64_t, int64_t, int64_t>(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent);
}
ERROR awkward_indexedarray32_getitem_nextcarry_outindex64_64(int64_t* tocarry, int64_t* toindex, const int32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  return awkward_indexedarray_getitem_nextcarry_outindex<int32_t, int64_t, int64_t>(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent);
}
ERROR awkward_indexedarrayU32_getitem_nextcarry_outindex64_64(int64_t* tocarry, int64_t* toindex, const uint32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  return awkward_indexedarray_getitem_nextcarry_outindex<uint32_t, int64_t, int64_t>(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent);
}
ERROR awkward_indexedarray64_getitem_nextcarry_outindex64_64(int64_t* tocarry, int64_t* toindex, const int64_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  return awkward_indexedarray_getitem_nextcarry_outindex<int64_t, int64_t, int64_t>(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent);
}

template <typename C, typename T>
ERROR awkward_indexedarray_getitem_nextcarry(T* tocarry, const C* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  int64_t k = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    C j = fromindex[indexoffset + i];
    if (j < 0  ||  j >= lencontent) {
      return failure("index out of range", i, j);
    }
    else {
      tocarry[k] = j;
      k++;
    }
  }
  return success();
}
ERROR awkward_indexedarray32_getitem_nextcarry_64(int64_t* tocarry, const int32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  return awkward_indexedarray_getitem_nextcarry<int32_t, int64_t>(tocarry, fromindex, indexoffset, lenindex, lencontent);
}
ERROR awkward_indexedarrayU32_getitem_nextcarry_64(int64_t* tocarry, const uint32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  return awkward_indexedarray_getitem_nextcarry<uint32_t, int64_t>(tocarry, fromindex, indexoffset, lenindex, lencontent);
}
ERROR awkward_indexedarray64_getitem_nextcarry_64(int64_t* tocarry, const int64_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  return awkward_indexedarray_getitem_nextcarry<int64_t, int64_t>(tocarry, fromindex, indexoffset, lenindex, lencontent);
}

template <typename C, typename T>
ERROR awkward_indexedarray_getitem_carry(C* toindex, const C* fromindex, const T* fromcarry, int64_t indexoffset, int64_t lenindex, int64_t lencarry) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (fromcarry[i] >= lenindex) {
      return failure("index out of range", i, fromcarry[i]);
    }
    toindex[i] = (C)(fromindex[indexoffset + fromcarry[i]]);
  }
  return success();
}
ERROR awkward_indexedarray32_getitem_carry_64(int32_t* toindex, const int32_t* fromindex, const int64_t* fromcarry, int64_t indexoffset, int64_t lenindex, int64_t lencarry) {
  return awkward_indexedarray_getitem_carry<int32_t, int64_t>(toindex, fromindex, fromcarry, indexoffset, lenindex, lencarry);
}
ERROR awkward_indexedarrayU32_getitem_carry_64(uint32_t* toindex, const uint32_t* fromindex, const int64_t* fromcarry, int64_t indexoffset, int64_t lenindex, int64_t lencarry) {
  return awkward_indexedarray_getitem_carry<uint32_t, int64_t>(toindex, fromindex, fromcarry, indexoffset, lenindex, lencarry);
}
ERROR awkward_indexedarray64_getitem_carry_64(int64_t* toindex, const int64_t* fromindex, const int64_t* fromcarry, int64_t indexoffset, int64_t lenindex, int64_t lencarry) {
  return awkward_indexedarray_getitem_carry<int64_t, int64_t>(toindex, fromindex, fromcarry, indexoffset, lenindex, lencarry);
}

template <typename C, typename I>
ERROR awkward_unionarray_regular_index(I* toindex, const C* fromtags, int64_t tagsoffset, int64_t length) {
  std::vector<I> current;
  for (int64_t i = 0;  i < length;  i++) {
    C tag = fromtags[tagsoffset + i];
    while (current.size() <= (size_t)tag) {
      current.push_back(0);
    }
    toindex[(size_t)i] = current[(size_t)tag];
    current[(size_t)tag]++;
  }
  return success();
}
ERROR awkward_unionarray8_32_regular_index(int32_t* toindex, const int8_t* fromtags, int64_t tagsoffset, int64_t length) {
  return awkward_unionarray_regular_index<int8_t, int32_t>(toindex, fromtags, tagsoffset, length);
}
ERROR awkward_unionarray8_U32_regular_index(uint32_t* toindex, const int8_t* fromtags, int64_t tagsoffset, int64_t length) {
  return awkward_unionarray_regular_index<int8_t, uint32_t>(toindex, fromtags, tagsoffset, length);
}
ERROR awkward_unionarray8_64_regular_index(int64_t* toindex, const int8_t* fromtags, int64_t tagsoffset, int64_t length) {
  return awkward_unionarray_regular_index<int8_t, int64_t>(toindex, fromtags, tagsoffset, length);
}

template <typename T, typename C, typename I>
ERROR awkward_unionarray_project(int64_t* lenout, T* tocarry, const C* fromtags, int64_t tagsoffset, const I* fromindex, int64_t indexoffset, int64_t length, int64_t which) {
  *lenout = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if (fromtags[tagsoffset + i] == which) {
      tocarry[(size_t)(*lenout)] = fromindex[indexoffset + i];
      *lenout = *lenout + 1;
    }
  }
  return success();
}
ERROR awkward_unionarray8_32_project_64(int64_t* lenout, int64_t* tocarry, const int8_t* fromtags, int64_t tagsoffset, const int32_t* fromindex, int64_t indexoffset, int64_t length, int64_t which) {
  return awkward_unionarray_project<int64_t, int8_t, int32_t>(lenout, tocarry, fromtags, tagsoffset, fromindex, indexoffset, length, which);
}
ERROR awkward_unionarray8_U32_project_64(int64_t* lenout, int64_t* tocarry, const int8_t* fromtags, int64_t tagsoffset, const uint32_t* fromindex, int64_t indexoffset, int64_t length, int64_t which) {
  return awkward_unionarray_project<int64_t, int8_t, uint32_t>(lenout, tocarry, fromtags, tagsoffset, fromindex, indexoffset, length, which);
}
ERROR awkward_unionarray8_64_project_64(int64_t* lenout, int64_t* tocarry, const int8_t* fromtags, int64_t tagsoffset, const int64_t* fromindex, int64_t indexoffset, int64_t length, int64_t which) {
  return awkward_unionarray_project<int64_t, int8_t, int64_t>(lenout, tocarry, fromtags, tagsoffset, fromindex, indexoffset, length, which);
}
