// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/getitem.h"
#include <cstring>

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
