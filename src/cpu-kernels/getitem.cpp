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
void awkward_numpyarray_contiguous_next(uint8_t* toptr, const uint8_t* fromptr, int64_t len, int64_t stride, int64_t offset, const T* pos) {
  for (int64_t i = 0;  i < len;  i++) {
    memcpy(&toptr[i*stride], &fromptr[offset + (int64_t)pos[i]], (size_t)stride);
  }
}
void awkward_numpyarray_contiguous_next_64(uint8_t* toptr, const uint8_t* fromptr, int64_t len, int64_t stride, int64_t offset, const int64_t* pos) {
  awkward_numpyarray_contiguous_next<int64_t>(toptr, fromptr, len, stride, offset, pos);
}

Error awkward_getitem() {
  return "not implemented";
}
