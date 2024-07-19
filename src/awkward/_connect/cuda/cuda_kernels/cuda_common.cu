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


// used by awkward_ListArray_getitem_next_range_carrylength
// and awkward_ListArray_getitem_next_range kernels

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


// atomicMin() specializations
template <typename T>
__device__ T atomicMin(T* address, T val);

// atomicMin() specialization for int8_t
template <>
__device__ int8_t atomicMin<int8_t>(int8_t* address, int8_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int sel = selectors[(size_t)address & 3];
  unsigned int old, assumed, min_, new_;
  old = *base_address;
  do {
    assumed = old;
    min_ = min(val, (int8_t)__byte_perm(old, 0, ((size_t)address & 3)));
    new_ = __byte_perm(old, min_, sel);
    old = atomicCAS(base_address, assumed, new_);
  } while (assumed != old);
  return old;
}

// atomicMin() specialization for uint8_t
template <>
__device__ uint8_t atomicMin<uint8_t>(uint8_t* address, uint8_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int sel = selectors[(size_t)address & 3];
  unsigned int old, assumed, min_, new_;
  old = *base_address;
  do {
    assumed = old;
    min_ = min(val, (uint8_t)__byte_perm(old, 0, ((size_t)address & 3)));
    new_ = __byte_perm(old, min_, sel);
    old = atomicCAS(base_address, assumed, new_);
  } while (assumed != old);
  return old;
}

// atomicMin() specialization for int16_t
template <>
__device__ int16_t atomicMin<int16_t>(int16_t* address, int16_t val) {
  uint16_t* address_as_ush = reinterpret_cast<uint16_t*>(address);
  uint16_t old = *address_as_ush, assumed;
  do {
    assumed = old;
    int16_t temp = min(val, reinterpret_cast<int16_t&>(assumed));
    old = atomicCAS(
        address_as_ush, assumed, reinterpret_cast<uint16_t&>(temp)
    );
  } while (assumed != old);
  return reinterpret_cast<int16_t&>(old);
}

// atomicMin() specialization for uint16_t
template <>
__device__ uint16_t atomicMin<uint16_t>(uint16_t* address, uint16_t val) {
  uint16_t old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, min(val, assumed));
  } while (assumed != old);
  return old;
}

// atomicMin() specialization for float
template <>
__device__ float atomicMin<float>(float* addr, float value) {
  float old;
  old = !signbit(value) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
      __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

// atomicMin() specialization for double
template <>
__device__ double atomicMin<double>(double* addr, double value) {
  double old;
  old = !signbit(value) ? __longlong_as_double(atomicMin((long long int*)addr, __double_as_longlong(value))) :
      __ull2double_rz(atomicMax((unsigned long long int*)addr, __double2ull_ru(value)));
  return old;
}


// atomicMax() specializations
template <typename T>
__device__ T atomicMax(T* address, T val);

// atomicMax() specialization for int8_t
template <>
__device__ int8_t atomicMax<int8_t>(int8_t* address, int8_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int sel = selectors[(size_t)address & 3];
  unsigned int old, assumed, max_, new_;
  old = *base_address;
  do {
    assumed = old;
    max_ = max(val, (int8_t)__byte_perm(old, 0, ((size_t)address & 3)));
    new_ = __byte_perm(old, max_, sel);
    old = atomicCAS(base_address, assumed, new_);
  } while (assumed != old);
  return old;
}

// atomicMax() specialization for uint8_t
template <>
__device__ uint8_t atomicMax<uint8_t>(uint8_t* address, uint8_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int sel = selectors[(size_t)address & 3];
  unsigned int old, assumed, max_, new_;
  old = *base_address;
  do {
    assumed = old;
    max_ = max(val, (uint8_t)__byte_perm(old, 0, ((size_t)address & 3)));
    new_ = __byte_perm(old, max_, sel);
    old = atomicCAS(base_address, assumed, new_);
  } while (assumed != old);
  return old;
}

// atomicMax() specialization for int16_t
template <>
__device__ int16_t atomicMax<int16_t>(int16_t* address, int16_t val) {
  uint16_t* address_as_ush = reinterpret_cast<uint16_t*>(address);
  uint16_t old = *address_as_ush, assumed;
  do {
    assumed = old;
    int16_t temp = max(val, reinterpret_cast<int16_t&>(assumed));
    old = atomicCAS(
        address_as_ush, assumed, reinterpret_cast<uint16_t&>(temp)
    );
  } while (assumed != old);
  return reinterpret_cast<int16_t&>(old);
}

// atomicMax() specialization for uint16_t
template <>
__device__ uint16_t atomicMax<uint16_t>(uint16_t* address, uint16_t val) {
  uint16_t old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, max(val, assumed));
  } while (assumed != old);
  return old;
}

// atomicMax() specialization for float
template <>
__device__ float atomicMax<float>(float* addr, float value) {
  float old;
  old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
      __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

// atomicMax() specialization for double
template <>
__device__ double atomicMax<double>(double* addr, double value) {
  double old;
  old = !signbit(value) ? __longlong_as_double(atomicMax((long long int*)addr, __double_as_longlong(value))) :
      __ull2double_rz(atomicMin((unsigned long long int*)addr, __double2ull_ru(value)));
  return old;
}


// atomicAdd() specialization for int64_t
// uses 2's complement
__device__ int64_t atomicAdd(int64_t* address, int64_t val) {
  uint64_t* address_as_ull = (uint64_t*)address;
  uint64_t old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, assumed + (uint64_t)val);
  } while (assumed != old);
  return (int64_t)old;
}


// atomicMul() specializations
template <typename T>
__device__ T atomicMul(T* address, T val);

// atomicMul() specialization for int32_t
template <>
__device__ int32_t atomicMul<int32_t>(int32_t* address, int32_t val) {
  int32_t old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, assumed * val);
  } while (assumed != old);
  return old;
}

// atomicMul() specialization for uint32_t
template <>
__device__ uint32_t atomicMul<uint32_t>(uint32_t* address, uint32_t val) {
  uint32_t old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, assumed * val);
  } while (assumed != old);
  return old;
}

// atomicMul() specialization for int64_t
template <>
__device__ int64_t atomicMul<int64_t>(int64_t* address, int64_t val) {
  uint64_t* address_as_uint64 = reinterpret_cast<uint64_t*>(address);
  uint64_t old = *address_as_uint64, assumed;
  uint64_t val_as_uint64 = *reinterpret_cast<uint64_t*>(&val);
  do {
    assumed = old;
    old = atomicCAS(address_as_uint64, assumed, assumed * val_as_uint64);
  } while (assumed != old);
  return *reinterpret_cast<int64_t*>(&old);
}

// atomicMul() specialization for uint64_t
template <>
__device__ uint64_t atomicMul<uint64_t>(uint64_t* address, uint64_t val) {
  uint64_t old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, assumed * val);
  } while (assumed != old);
  return old;
}

// atomicMul() specialization for float
template <>
__device__ float atomicMul<float>(float* address, float val) {
  float old = *address, assumed;
  do {
    assumed = old;
    old = __int_as_float(atomicCAS((int*)address, __float_as_int(assumed), __float_as_int(assumed * val)));
  } while (assumed != old);
  return old;
}

// atomicMul() specialization for double
template <>
__device__ double atomicMul<double>(double* address, double val) {
  uint64_t* address_as_ull = (uint64_t*)address;
  uint64_t old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(__longlong_as_double(assumed) * val));
  } while (assumed != old);
  return __longlong_as_double(old);
}


// atomicMinComplex() specialization for float
__device__ void atomicMinComplex(float* result_real, float* result_imag, float data_real, float data_imag) {
  unsigned int* real_addr = reinterpret_cast<unsigned int*>(result_real);
  unsigned int* imag_addr = reinterpret_cast<unsigned int*>(result_imag);

  unsigned int val_real_int = __float_as_uint(data_real);
  unsigned int val_imag_int = __float_as_uint(data_imag);
  unsigned int old_real_int = atomicCAS(real_addr, *real_addr, *real_addr);

  while (true) {
      unsigned int assumed_real = old_real_int;
      float old_real = __uint_as_float(old_real_int);

      if (data_real < old_real || (data_real == old_real && data_imag < __uint_as_float(atomicCAS(imag_addr, *imag_addr, *imag_addr)))) {
          old_real_int = atomicCAS(real_addr, assumed_real, val_real_int);
          if (old_real_int == assumed_real) {
              atomicCAS(imag_addr, *imag_addr, val_imag_int);
              break;
          }
      } else {
          break;
      }
  }
}

// atomicMinComplex() specialization for double
__device__ void atomicMinComplex(double* result_real, double* result_imag, double data_real, double data_imag) {
  unsigned long long* real_addr = reinterpret_cast<unsigned long long*>(result_real);
  unsigned long long* imag_addr = reinterpret_cast<unsigned long long*>(result_imag);

  unsigned long long val_real_ll = __double_as_longlong(data_real);
  unsigned long long val_imag_ll = __double_as_longlong(data_imag);
  unsigned long long old_real_ll = atomicCAS(real_addr, *real_addr, *real_addr);

  while (true) {
      unsigned long long assumed_real = old_real_ll;
      double old_real = __longlong_as_double(old_real_ll);

      if (data_real < old_real || (data_real == old_real && data_imag < __longlong_as_double(atomicCAS(imag_addr, *imag_addr, *imag_addr)))) {
          old_real_ll = atomicCAS(real_addr, assumed_real, val_real_ll);
          if (old_real_ll == assumed_real) {
              atomicCAS(imag_addr, *imag_addr, val_imag_ll);
              break;
          }
      } else {
          break;
      }
  }
}

// atomicMinComplex() specialization for float
__device__ void atomicMaxComplex(float* result_real, float* result_imag, float data_real, float data_imag) {
  unsigned int* real_addr = reinterpret_cast<unsigned int*>(result_real);
  unsigned int* imag_addr = reinterpret_cast<unsigned int*>(result_imag);

  unsigned int val_real_int = __float_as_uint(data_real);
  unsigned int val_imag_int = __float_as_uint(data_imag);
  unsigned int old_real_int = atomicCAS(real_addr, *real_addr, *real_addr);

  while (true) {
      unsigned int assumed_real = old_real_int;
      float old_real = __uint_as_float(old_real_int);

      if (data_real > old_real || (data_real == old_real && data_imag > __uint_as_float(atomicCAS(imag_addr, *imag_addr, *imag_addr)))) {
          old_real_int = atomicCAS(real_addr, assumed_real, val_real_int);
          if (old_real_int == assumed_real) {
              atomicCAS(imag_addr, *imag_addr, val_imag_int);
              break;
          }
      } else {
          break;
      }
  }
}

// atomicMinComplex() specialization for double
__device__ void atomicMaxComplex(double* result_real, double* result_imag, double data_real, double data_imag) {
  unsigned long long* real_addr = reinterpret_cast<unsigned long long*>(result_real);
  unsigned long long* imag_addr = reinterpret_cast<unsigned long long*>(result_imag);

  unsigned long long val_real_ll = __double_as_longlong(data_real);
  unsigned long long val_imag_ll = __double_as_longlong(data_imag);
  unsigned long long old_real_ll = atomicCAS(real_addr, *real_addr, *real_addr);

  while (true) {
      unsigned long long assumed_real = old_real_ll;
      double old_real = __longlong_as_double(old_real_ll);

      if (data_real > old_real || (data_real == old_real && data_imag > __longlong_as_double(atomicCAS(imag_addr, *imag_addr, *imag_addr)))) {
          old_real_ll = atomicCAS(real_addr, assumed_real, val_real_ll);
          if (old_real_ll == assumed_real) {
              atomicCAS(imag_addr, *imag_addr, val_imag_ll);
              break;
          }
      } else {
          break;
      }
  }
}


// atomicMulComplex() specialization for float
__device__ void atomicMulComplex(float* addr_real, float* addr_imag, float val_real, float val_imag) {
  unsigned int* addr_real_int = (unsigned int*)addr_real;
  unsigned int* addr_imag_int = (unsigned int*)addr_imag;

  unsigned int old_real, old_imag, new_real, new_imag;
  do {
      old_real = *addr_real_int;
      old_imag = *addr_imag_int;

      new_real = __float_as_int(__int_as_float(old_real) * val_real - __int_as_float(old_imag) * val_imag);
      new_imag = __float_as_int(__int_as_float(old_real) * val_imag + __int_as_float(old_imag) * val_real);
  } while (atomicCAS(addr_real_int, old_real, new_real) != old_real ||
           atomicCAS(addr_imag_int, old_imag, new_imag) != old_imag);
}

// atomicMulComplex() specialization for double
__device__ void atomicMulComplex(double* addr_real, double* addr_imag, double val_real, double val_imag) {
  unsigned long long int old_real, old_imag, new_real, new_imag;
  do {
      old_real = __double_as_longlong(*addr_real);
      old_imag = __double_as_longlong(*addr_imag);
      new_real = __double_as_longlong(__longlong_as_double(old_real) * val_real - __longlong_as_double(old_imag) * val_imag);
      new_imag = __double_as_longlong(__longlong_as_double(old_real) * val_imag + __longlong_as_double(old_imag) * val_real);
  } while (atomicCAS((unsigned long long int*)addr_real, old_real, new_real) != old_real ||
           atomicCAS((unsigned long long int*)addr_imag, old_imag, new_imag) != old_imag);
}
