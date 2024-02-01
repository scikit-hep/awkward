// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_simplify.cpp", line)

#include "awkward/kernels.h"

template <typename OUTERTAGS,
          typename OUTERINDEX,
          typename INNERTAGS,
          typename INNERINDEX,
          typename TOTAGS,
          typename TOINDEX>
ERROR awkward_UnionArray_simplify(
  TOTAGS* totags,
  TOINDEX* toindex,
  const OUTERTAGS* outertags,
  const OUTERINDEX* outerindex,
  const INNERTAGS* innertags,
  const INNERINDEX* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    if (outertags[i] == outerwhich) {
      OUTERINDEX j = outerindex[i];
      if (innertags[j] == innerwhich) {
        totags[i] = (TOTAGS)towhich;
        toindex[i] = (TOINDEX)(innerindex[j] + base);
      }
    }
  }
  return success();
}
ERROR awkward_UnionArray8_32_simplify8_32_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const int32_t* outerindex,
  const int8_t* innertags,
  const int32_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, int32_t, int8_t, int32_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_32_simplify8_U32_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const int32_t* outerindex,
  const int8_t* innertags,
  const uint32_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, int32_t, int8_t, uint32_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_32_simplify8_64_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const int32_t* outerindex,
  const int8_t* innertags,
  const int64_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, int32_t, int8_t, int64_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_U32_simplify8_32_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const uint32_t* outerindex,
  const int8_t* innertags,
  const int32_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, uint32_t, int8_t, int32_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_U32_simplify8_U32_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const uint32_t* outerindex,
  const int8_t* innertags,
  const uint32_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, uint32_t, int8_t, uint32_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_U32_simplify8_64_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const uint32_t* outerindex,
  const int8_t* innertags,
  const int64_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, uint32_t, int8_t, int64_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_64_simplify8_32_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const int64_t* outerindex,
  const int8_t* innertags,
  const int32_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, int64_t, int8_t, int32_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_64_simplify8_U32_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const int64_t* outerindex,
  const int8_t* innertags,
  const uint32_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, int64_t, int8_t, uint32_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_64_simplify8_64_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const int64_t* outerindex,
  const int8_t* innertags,
  const int64_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, int64_t, int8_t, int64_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
