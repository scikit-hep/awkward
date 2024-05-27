// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long long int64_t;
typedef signed char int_fast8_t;
typedef signed short int_fast16_t;
typedef signed int int_fast32_t;
typedef signed long long int_fast64_t;
typedef signed char int_least8_t;
typedef signed short int_least16_t;
typedef signed int int_least32_t;
typedef signed long long int_least64_t;
typedef signed long long intmax_t;
typedef signed long intptr_t;  //optional
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned char uint_fast8_t;
typedef unsigned short uint_fast16_t;
typedef unsigned int uint_fast32_t;
typedef unsigned long long uint_fast64_t;
typedef unsigned char uint_least8_t;
typedef unsigned short uint_least16_t;
typedef unsigned int uint_least32_t;
typedef unsigned long long uint_least64_t;
typedef unsigned long long uintmax_t;

#define RAISE_ERROR(ERROR_KERNEL_CODE) \
  atomicMin(err_code,                  \
            invocation_index*(1 << ERROR_BITS) + (int)(ERROR_KERNEL_CODE));

void
awkward_regularize_rangeslice(
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
