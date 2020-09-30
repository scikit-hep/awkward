// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_carry_arange.cpp", line)

#include "awkward/kernels.h"

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
