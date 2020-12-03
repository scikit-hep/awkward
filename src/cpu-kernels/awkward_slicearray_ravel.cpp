// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_slicearray_ravel.cpp", line)

#include "awkward/kernels.h"

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
