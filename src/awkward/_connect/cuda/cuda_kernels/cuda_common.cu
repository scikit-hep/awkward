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

// BEGIN PYTHON
// def min_max_type(dtype):
//   supported_types = {
//       'bool': cupy.int32,
//       'int8': cupy.int32,
//       'int16': cupy.int32,
//       'int32': cupy.int32,
//       'int64': cupy.int64,
//       'uint8': cupy.uint32,
//       'uint16': cupy.uint32,
//       'uint32': cupy.uint32,
//       'uint64': cupy.uint64,
//       'float32': cupy.float32,
//       'float64': cupy.float64
//   }
//   if str(dtype) in supported_types:
//       return supported_types[str(dtype)]
//   else:
//       raise ValueError("Unsupported dtype.", dtype)
// END PYTHON

const int64_t  kMaxInt64  = 9223372036854775806;   // 2**63 - 2: see below
const int64_t  kSliceNone = kMaxInt64 + 1;         // for Slice::none()

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
}

__device__ __forceinline__ float atomicMin(float* addr, float value) {
  float old; old = !signbit(value) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) : __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
  return old;
}
__device__ __forceinline__ float atomicMax(float* addr, float value) {
  float old; old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

__device__ int64_t atomicAdd(int64_t* address, int64_t val) {
  uint64_t* address_as_ull = (uint64_t*)address;
  uint64_t old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, assumed + (uint64_t)val);
  } while (assumed != old);
  return (int64_t)old;
}
