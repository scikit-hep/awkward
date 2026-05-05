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

__device__ void
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

// Device helper: load flattened complex as (real, imag)
template <typename C>
__device__ inline void load_complex(const C* fromptr, int64_t idx, double& real, double& imag) {
  // idx may be -1 in some checks; caller must avoid calling with idx == -1.
  real = (double)fromptr[2 * idx];
  imag = (double)fromptr[2 * idx + 1];
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
__device__ double atomicMin<double>(double* address, double val) {
    unsigned long long int* addr_as_ull =
        reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        double assumed_val = __longlong_as_double(assumed);
        double new_val = fmin(val, assumed_val);
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(new_val));
    } while (assumed != old);

    return __longlong_as_double(old);
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
__device__ double atomicMax<double>(double* address, double val) {
    unsigned long long int* addr_as_ull =
        reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        double assumed_val = __longlong_as_double(assumed);
        double new_val = fmax(val, assumed_val);
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(new_val));
    } while (assumed != old);

    return __longlong_as_double(old);
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

// simple spin-lock helpers
__device__ inline void acquire_lock(int32_t* lockptr) {
  while (atomicExch(lockptr, 1) != 0) {
    // optional: __nanosleep or __threadfence(); but busy spin is simple
  }
}
__device__ inline void release_lock(int32_t* lockptr) {
  atomicExch(lockptr, 0);
}

#include <cuda_runtime.h>

// ---------------------------
// AtomicCAS helpers
// ---------------------------
__device__ inline float atomicCAS_wrapper(float* addr, float oldval, float newval) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old_int = __float_as_int(oldval);
    int new_int = __float_as_int(newval);
    int prev = atomicCAS(addr_as_int, old_int, new_int);
    return __int_as_float(prev);
}

__device__ inline double atomicCAS_wrapper(double* addr, double oldval, double newval) {
    unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old_ull = __double_as_longlong(oldval);
    unsigned long long new_ull = __double_as_longlong(newval);
    unsigned long long prev = atomicCAS(addr_as_ull, old_ull, new_ull);
    return __longlong_as_double(prev);
}

// ---------------------------
// Atomic min complex (lexicographic)
// ---------------------------
template <typename T>
__device__ void atomicMinComplex(T* result_real, T* result_imag, T val_real, T val_imag) {
    // static_assert(std::is_floating_point<T>::value, "T must be float or double");
    T old_real, old_imag;
    do {
        old_real = *result_real;
        old_imag = *result_imag;
        if (!(val_real < old_real || (val_real == old_real && val_imag < old_imag))) {
            return;
        }
    } while (atomicCAS_wrapper(result_real, old_real, val_real) != old_real);

    *result_imag = val_imag;
}

// ---------------------------
// Atomic max complex (lexicographic)
// ---------------------------
template <typename T>
__device__ void atomicMaxComplex(T* result_real, T* result_imag, T val_real, T val_imag) {
    // static_assert(std::is_floating_point<T>::value, "T must be float or double");
    T old_real, old_imag;
    do {
        old_real = *result_real;
        old_imag = *result_imag;
        if (!(val_real > old_real || (val_real == old_real && val_imag > old_imag))) {
            return;
        }
    } while (atomicCAS_wrapper(result_real, old_real, val_real) != old_real);

    *result_imag = val_imag;
}

// ---------------------------
// Atomic product complex
// ---------------------------
template <typename T>
__device__ void atomicProdComplex(T* result_real, T* result_imag, T val_real, T val_imag) {
    // static_assert(std::is_floating_point<T>::value, "T must be float or double");
    T old_real, old_imag, new_real, new_imag;
    do {
        old_real = *result_real;
        old_imag = *result_imag;
        new_real = old_real * val_real - old_imag * val_imag;
        new_imag = old_real * val_imag + old_imag * val_real;
    } while (atomicCAS_wrapper(result_real, old_real, new_real) != old_real);

    *result_imag = new_imag;
}


// atomicMulComplex() specialization for float
__device__ void atomicMulComplex(float* addr_real, float* addr_imag,
                                 float val_real, float val_imag) {
    // Treat real+imag as one 64-bit value
    unsigned long long* addr = (unsigned long long*)addr_real;

    unsigned long long old_val, new_val;
    float2 old_c, new_c;

    do {
        old_val = *addr;
        old_c = *reinterpret_cast<float2*>(&old_val);

        new_c.x = old_c.x * val_real - old_c.y * val_imag;
        new_c.y = old_c.x * val_imag + old_c.y * val_real;

        new_val = *reinterpret_cast<unsigned long long*>(&new_c);
    } while (atomicCAS(addr, old_val, new_val) != old_val);
}


// atomicMulComplex() specialization for double
__device__ void atomicMulComplex(double* addr_real, double* addr_imag,
                                 double val_real, double val_imag) {
    // Pack two doubles into unsigned long long[2]
    unsigned long long* addr = (unsigned long long*)addr_real;

    unsigned long long old0, old1;
    double2 old_c, new_c;

    do {
        old0 = addr[0];
        old1 = addr[1];

        old_c.x = __longlong_as_double(old0);
        old_c.y = __longlong_as_double(old1);

        new_c.x = old_c.x * val_real - old_c.y * val_imag;
        new_c.y = old_c.x * val_imag + old_c.y * val_real;

    } while (atomicCAS(&addr[0], old0, __double_as_longlong(new_c.x)) != old0 ||
             atomicCAS(&addr[1], old1, __double_as_longlong(new_c.y)) != old1);
}


enum class ARRAY_COMBINATIONS_ERRORS : uint64_t {
    N_NEGATIVE = 1,               // message: "n must be >= 0"
    OVERFLOW_IN_COMBINATORICS = 2 // message: "binomial overflow"
};


template <typename I>
__device__ __forceinline__ I dgcd(I a, I b) {
  // Euclidean algorithm; inputs non-negative
  while (b != 0) { I t = a % b; a = b; b = t; }
  return a;
}


template <typename I>
__device__ __forceinline__ bool mul_will_overflow(I a, I b) {
  // check a * b > INT64_MAX (for signed I=int64_t)
  if (a == 0 || b == 0) return false;
  if (a < 0 || b < 0) return true; // not expected here
  const unsigned long long ua = (unsigned long long)a;
  const unsigned long long ub = (unsigned long long)b;
  const unsigned long long umax = 0x7fffffffffffffffULL;
  return ua > umax / ub;
}


template <typename I>
__device__ __forceinline__ bool binom_safe(
    I n, I k, I& out, uint64_t* err_code) {
  // Compute C(n,k) exactly using gcd reductions to keep intermediates small.
  // Returns false on overflow (err_code set) or invalid inputs.
  if (k < 0 || n < 0 || k > n) { out = 0; return true; }
  if (k == 0 || k == n) { out = 1; return true; }
  if (k + k > n) k = n - k;

  I result = 1;
  I denom = 1;

  for (I j = 1; j <= k; ++j) {
    I a = n - j + 1;   // numerator factor
    I b = j;           // denominator factor

    // Reduce with current numerator/denominator
    I g1 = dgcd(a, b);
    a /= g1; b /= g1;

    // Further reduce denominator against current result
    I g2 = dgcd(result, b);
    result /= g2; b /= g2;

    // b should now be 1; multiply result *= a with overflow check
    if (mul_will_overflow<I>(result, a)) {
        // signal overflow
        err_code[0] = (err_code[0] < static_cast<uint64_t>(
            ARRAY_COMBINATIONS_ERRORS::OVERFLOW_IN_COMBINATORICS))
            ? static_cast<uint64_t>(
                ARRAY_COMBINATIONS_ERRORS::OVERFLOW_IN_COMBINATORICS)
            : err_code[0];
        out = 0;
        return false;
    }
    result *= a;
    // b must be 1 now if inputs were valid
  }

  out = result;
  return true;
}


/**
 * Unrank the k-th combination (lexicographic) of size r from m items.
 * - Without replacement: strictly increasing indices in [0, m-1].
 * - With replacement: nondecreasing indices (via stars-and-bars block sizes).
 *
 * Returns false on overflow; true otherwise. On failure, out[] is unspecified.
 */
__device__ bool unrank_lex_general(
    int64_t m, int64_t r, int64_t k, bool replacement,
    int64_t* out /* length r */, uint64_t* err_code) {

  if (r == 0) return k == 0;  // single empty tuple when k==0

  int64_t prev = -1;  // last chosen index
  for (int64_t pos = 0; pos < r; ++pos) {
    // minimal allowed value for this position
    int64_t v = replacement ? (prev < 0 ? 0 : prev) : (prev + 1);

    // walk v upward until the remaining block covers k
    while (true) {
      if (v >= m) return false;  // out of range

      int64_t rem = r - pos - 1;
      int64_t block_count = 0;

      if (rem == 0) {
        block_count = 1;
      } else if (replacement) {
        // number of (nondecreasing) tails of length rem starting at v
        // = C((m - v) + rem - 1, rem)
        int64_t n_top = (m - v) + rem - 1;
        if (!binom_safe(n_top, rem, block_count, err_code)) return false;
      } else {
        // number of (strictly increasing) tails from v+1
        // = C(m - (v + 1), rem)
        int64_t n_top = m - (v + 1);
        if (!binom_safe(n_top, rem, block_count, err_code)) return false;
      }

      if (k < block_count) {
        out[pos] = v;
        prev = v;
        break;
      } else {
        k -= block_count;
        ++v;
      }
    }
  }
  return true;
}
