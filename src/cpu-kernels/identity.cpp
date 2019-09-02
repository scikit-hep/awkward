// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"

template <typename T>
Error awkward_identity_new(int64_t length, T* to) {
  for (T i = 0;  i < length;  i++) {
    to[i] = i;
  }
  return kNoError;
}
Error awkward_identity_new32(int64_t length, int32_t* to) {
  return awkward_identity_new<int32_t>(length, to);
}
Error awkward_identity_new64(int64_t length, int64_t* to) {
  return awkward_identity_new<int64_t>(length, to);
}

Error awkward_identity_32to64(int64_t length, int32_t* from, int64_t* to) {
  for (int64_t i = 0;  i < length;  i++) {
    to[i]= (int64_t)from[i];
  }
  return kNoError;
}

template <typename T>
Error awkward_identity_from_listfoffsets(int64_t length, int64_t width, T* offsets, T* from, int64_t tolength, T* to) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    for (T subi = 0;  subi < offsets[i + 1] - offsets[i];  subi++) {
      for (int64_t j = 0;  j < width;  j++) {
        to[(width + 1)*k + j] = from[(width)*i + j];
      }
      to[(width + 1)*k + width] = subi;
      k++;
    }
  }
  return kNoError;
}
Error awkward_identity_from_listfoffsets32(int64_t length, int64_t width, int32_t* offsets, int32_t* from, int64_t tolength, int32_t* to) {
  return awkward_identity_from_listfoffsets<int32_t>(length, width, offsets, from, tolength, to);
}
Error awkward_identity_from_listfoffsets64(int64_t length, int64_t width, int64_t* offsets, int64_t* from, int64_t tolength, int64_t* to) {
  return awkward_identity_from_listfoffsets<int64_t>(length, width, offsets, from, tolength, to);
}
