// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>

#include "awkward/cpu-kernels/getitem.h"

void awkward_regularize_rangeslice(int64_t& start, int64_t& stop, bool posstep, bool hasstart, bool hasstop, int64_t length) {
  if (posstep) {
    if (!hasstart)          start = 0;
    else if (start < 0)     start += length;
    if (start < 0)          start = 0;
    if (start > length)     start = length;

    if (!hasstop)           stop = length;
    else if (stop < 0)      stop += length;
    if (stop < 0)           stop = 0;
    if (stop > length)      stop = length;
    if (stop < start)       stop = start;
  }

  else {
    if (!hasstart)          start = length - 1;
    else if (start < 0)     start += length;
    if (start < -1)         start = -1;
    if (start > length - 1) start = length - 1;

    if (!hasstop)           stop = -1;
    else if (stop < 0)      stop += length;
    if (stop < -1)          stop = -1;
    if (stop > length - 1)  stop = length - 1;
    if (stop > start)       stop = start;
  }
}

template <typename T>
Error awkward_regularize_arrayslice(T* flatheadptr, int64_t lenflathead, int64_t length) {
  for (int64_t i = 0;  i < lenflathead;  i++) {
    if (flatheadptr[i] < 0) {
      flatheadptr[i] += length;
    }
    if (flatheadptr[i] < 0  ||  flatheadptr[i] >= length) {
      return "index out of range";
    }
  }
  return kNoError;
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

template <typename ID, typename T>
Error awkward_identity_getitem_carry(ID* newidentityptr, const ID* identityptr, const T* carryptr, int64_t lencarry, int64_t offset, int64_t width, int64_t length) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (carryptr[i] >= length) {
      return "index out of range for identity";
    }
    for (int64_t j = 0;  j < width;  j++) {
      newidentityptr[width*i + j] = identityptr[offset + width*carryptr[i] + j];
    }
  }
  return kNoError;
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
void awkward_numpyarray_getitem_next_slice(T* nextcarryptr, const T* carryptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenhead;  j++) {
      nextcarryptr[i*lenhead + j] = skip*carryptr[i] + start + j*step;
    }
  }
}
void awkward_numpyarray_getitem_next_slice_64(int64_t* nextcarryptr, const int64_t* carryptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step) {
  awkward_numpyarray_getitem_next_slice(nextcarryptr, carryptr, lencarry, lenhead, skip, start, step);
}

template <typename T>
void awkward_numpyarray_getitem_next_slice_advanced(T* nextcarryptr, T* nextadvancedptr, const T* carryptr, const T* advancedptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenhead;  j++) {
      nextcarryptr[i*lenhead + j] = skip*carryptr[i] + start + j*step;
      nextadvancedptr[i*lenhead + j] = advancedptr[i];
    }
  }
}
void awkward_numpyarray_getitem_next_slice_advanced_64(int64_t* nextcarryptr, int64_t* nextadvancedptr, const int64_t* carryptr, const int64_t* advancedptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step) {
  awkward_numpyarray_getitem_next_slice_advanced(nextcarryptr, nextadvancedptr, carryptr, advancedptr, lencarry, lenhead, skip, start, step);
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
Error awkward_listarray_getitem_next_array(C* tostarts, C* tostops, T* tocarry, T* toadvanced, const C* fromstarts, const C* fromstops, const T* fromarray, const T* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  if (fromadvanced == nullptr) {
    for (int64_t i = 0;  i < lenstarts;  i++) {
      if (fromstops[stopsoffset + i] < fromstarts[startsoffset + i]) {
        return "stops[i] < starts[i]";
      }
      if (fromstarts[startsoffset + i] != fromstops[stopsoffset + i]  &&  fromstops[stopsoffset + i] > lencontent) {
        return "stops[i] > len(content)";
      }
      int64_t length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
      for (int64_t j = 0;  j < lenarray;  j++) {
        int64_t regular_at = fromarray[j];
        if (regular_at < 0) {
          regular_at += length;
        }
        if (!(0 <= regular_at  &&  regular_at < length)) {
          return "array[i] is out of range for at least one sublist";
        }
        tocarry[i*lenarray + j] = fromstarts[startsoffset + i] + regular_at;
        toadvanced[i*lenarray + j] = j;
      }
      tostarts[i] = (i)*lenarray;
      tostops[i]  = (i + 1)*lenarray;
    }
  }
  else {
    for (int64_t i = 0;  i < lenstarts;  i++) {
      if (fromstops[stopsoffset + i] < fromstarts[startsoffset + i]) {
        return "stops[i] < starts[i]";
      }
      if (fromstarts[startsoffset + i] != fromstops[stopsoffset + i]  &&  fromstops[stopsoffset + i] > lencontent) {
        return "stops[i] > len(content)";
      }
      int64_t length = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
      if (fromadvanced[i] >= lenarray) {
        return "lengths of advanced indexes must match";
      }
      int64_t regular_at = fromarray[fromadvanced[i]];
      if (regular_at < 0) {
        regular_at += length;
      }
      if (!(0 <= regular_at  &&  regular_at < length)) {
        return "array[i] is out of range for at least one sublist";
      }
      tocarry[i] = fromstarts[startsoffset + i] + regular_at;
      toadvanced[i] = i;
      tostarts[i] = i;
      tostops[i]  = i + 1;
    }
  }
  return kNoError;
}
Error awkward_listarray32_getitem_next_array_64(int32_t* tostarts, int32_t* tostops, int64_t* tocarry, int64_t* toadvanced, const int32_t* fromstarts, const int32_t* fromstops, const int64_t* fromarray, const int64_t* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray_getitem_next_array<int32_t, int64_t>(tostarts, tostops, tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}
Error awkward_listarray64_getitem_next_array_64(int64_t* tostarts, int64_t* tostops, int64_t* tocarry, int64_t* toadvanced, const int64_t* fromstarts, const int64_t* fromstops, const int64_t* fromarray, const int64_t* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent) {
  return awkward_listarray_getitem_next_array<int64_t, int64_t>(tostarts, tostops, tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent);
}

template <typename C, typename T>
void awkward_listarray_getitem_carry(C* tostarts, C* tostops, const C* fromstarts, const C* fromstops, const T* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lencarry) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    tostarts[i] = fromstarts[startsoffset + fromcarry[i]];
    tostops[i] = fromstops[stopsoffset + fromcarry[i]];
  }
}
void awkward_listarray32_getitem_carry_64(int32_t* tostarts, int32_t* tostops, const int32_t* fromstarts, const int32_t* fromstops, const int64_t* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lencarry) {
  awkward_listarray_getitem_carry<int32_t, int64_t>(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lencarry);
}
void awkward_listarray64_getitem_carry_64(int64_t* tostarts, int64_t* tostops, const int64_t* fromstarts, const int64_t* fromstops, const int64_t* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lencarry) {
  awkward_listarray_getitem_carry<int64_t, int64_t>(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lencarry);
}
