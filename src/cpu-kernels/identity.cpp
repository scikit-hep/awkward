// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"

template <typename T>
void awkward_new_identity(T* toptr, int64_t length) {
  for (T i = 0;  i < length;  i++) {
    toptr[i] = i;
  }
}
void awkward_new_identity32(int32_t* toptr, int64_t length) {
  awkward_new_identity<int32_t>(toptr, length);
}
void awkward_new_identity64(int64_t* toptr, int64_t length) {
  awkward_new_identity<int64_t>(toptr, length);
}

void awkward_identity32_to_identity64(int64_t* toptr, const int32_t* fromptr, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
}

template <typename ID, typename T>
void awkward_identity_from_listoffsets(ID* toptr, const ID* fromptr, int64_t tolength, const T* offsets, int64_t width, int64_t length) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    for (T subi = 0;  subi < offsets[i + 1] - offsets[i];  subi++) {
      for (int64_t j = 0;  j < width;  j++) {
        toptr[(width + 1)*k + j] = fromptr[(width)*i + j];
      }
      toptr[(width + 1)*k + width] = subi;
      k++;
    }
  }
}
void awkward_identity32_from_listoffsets32(int32_t* toptr, const int32_t* fromptr, int64_t tolength, const int32_t* offsets, int64_t width, int64_t length) {
  awkward_identity_from_listoffsets<int32_t, int32_t>(toptr, fromptr, tolength, offsets, width, length);
}
void awkward_identity64_from_listoffsets32(int64_t* toptr, const int64_t* fromptr, int64_t tolength, const int32_t* offsets, int64_t width, int64_t length) {
  awkward_identity_from_listoffsets<int64_t, int32_t>(toptr, fromptr, tolength, offsets, width, length);
}
void awkward_identity64_from_listoffsets64(int64_t* toptr, const int64_t* fromptr, int64_t tolength, const int64_t* offsets, int64_t width, int64_t length) {
  awkward_identity_from_listoffsets<int64_t, int64_t>(toptr, fromptr, tolength, offsets, width, length);
}
