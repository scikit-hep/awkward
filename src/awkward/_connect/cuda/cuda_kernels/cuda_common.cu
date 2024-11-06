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

namespace awkward {

namespace detail {

template <typename T_output, typename T_input>
__forceinline__  __device__
T_output type_reinterpret(T_input value)
{
    return *( reinterpret_cast<T_output*>(&value) );
}
    // the implementation of `genericAtomicOperation`
    template <typename T, typename Op, size_t n>
    struct genericAtomicOperationImpl;

    // single byte atomic operation
    template<typename T, typename Op>
    struct genericAtomicOperationImpl<T, Op, 1> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, Op op)
        {
            using T_int = unsigned int;

            T_int * address_uint32 = reinterpret_cast<T_int *>
                (addr - (reinterpret_cast<size_t>(addr) & 3));
            unsigned int shift = ((reinterpret_cast<size_t>(addr) & 3) * 8);

            T_int old = *address_uint32;
            T_int assumed ;

            do {
                assumed = old;
                T target_value = T((old >> shift) & 0xff);
                uint8_t new_value = type_reinterpret<uint8_t, T>
                    ( op(target_value, update_value) );
                old = (old & ~(0x000000ff << shift)) | (T_int(new_value) << shift);
                old = atomicCAS(address_uint32, assumed, old);
            } while (assumed != old);

            return T((old >> shift) & 0xff);
        }
    };

    // 2 bytes atomic operation
    template<typename T, typename Op>
    struct genericAtomicOperationImpl<T, Op, 2> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, Op op)
        {
            using T_int = unsigned int;
            bool is_32_align = (reinterpret_cast<size_t>(addr) & 2) ? false : true;
            T_int * address_uint32 = reinterpret_cast<T_int *>
                (reinterpret_cast<size_t>(addr) - (is_32_align ? 0 : 2));

            T_int old = *address_uint32;
            T_int assumed ;

            do {
                assumed = old;
                T target_value = (is_32_align) ? T(old & 0xffff) : T(old >> 16);
                uint16_t new_value = type_reinterpret<uint16_t, T>
                    ( op(target_value, update_value) );

                old = (is_32_align) ? (old & 0xffff0000) | new_value
                                    : (old & 0xffff) | (T_int(new_value) << 16);
                old = atomicCAS(address_uint32, assumed, old);
            } while (assumed != old);

            return (is_32_align) ? T(old & 0xffff) : T(old >> 16);;
        }
    };

    // 4 bytes atomic operation
    template<typename T, typename Op>
    struct genericAtomicOperationImpl<T, Op, 4> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, Op op)
        {
            using T_int = unsigned int;

            T old_value = *addr;
            T assumed {old_value};

            do {
                assumed  = old_value;
                const T new_value = op(old_value, update_value);

                T_int ret = atomicCAS(
                    reinterpret_cast<T_int*>(addr),
                    type_reinterpret<T_int, T>(assumed),
                    type_reinterpret<T_int, T>(new_value));
                old_value = type_reinterpret<T, T_int>(ret);

            } while (assumed != old_value);

            return old_value;
        }
    };

    // 8 bytes atomic operation
    template<typename T, typename Op>
    struct genericAtomicOperationImpl<T, Op, 8> {
        __forceinline__  __device__
        T operator()(T* addr, T const & update_value, Op op)
        {
            using T_int = unsigned long long int;

            T old_value = *addr;
            T assumed {old_value};

            do {
                assumed  = old_value;
                const T new_value = op(old_value, update_value);

                T_int ret = atomicCAS(
                    reinterpret_cast<T_int*>(addr),
                    type_reinterpret<T_int, T>(assumed),
                    type_reinterpret<T_int, T>(new_value));
                old_value = type_reinterpret<T, T_int>(ret);

            } while (assumed != old_value);

            return old_value;
        }
    };

    // the implementation of `typesAtomicCASImpl`
    template <typename T, size_t n>
    struct typesAtomicCASImpl;

    template<typename T>
    struct typesAtomicCASImpl<T, 4> {
        __forceinline__  __device__
        T operator()(T* addr, T const & compare, T const & update_value)
        {
            using T_int = unsigned int;

            T_int ret = atomicCAS(
                reinterpret_cast<T_int*>(addr),
                type_reinterpret<T_int, T>(compare),
                type_reinterpret<T_int, T>(update_value));

            return type_reinterpret<T, T_int>(ret);
        }
    };

    // 8 bytes atomic operation
    template<typename T>
    struct typesAtomicCASImpl<T, 8> {
        __forceinline__  __device__
        T operator()(T* addr, T const & compare, T const & update_value)
        {
            using T_int = unsigned long long int;

            T_int ret = atomicCAS(
                reinterpret_cast<T_int*>(addr),
                type_reinterpret<T_int, T>(compare),
                type_reinterpret<T_int, T>(update_value));

            return type_reinterpret<T, T_int>(ret);
        }
    };

    // call atomic function with type cast between same underlying type
    template <typename T, typename Functor>
    __forceinline__  __device__
    T typesAtomicOperation32(T* addr, T val, Functor atomicFunc)
    {
        using T_int = int;
        T_int ret = atomicFunc(reinterpret_cast<T_int*>(addr),
            awkward::detail::type_reinterpret<T_int, T>(val));

        return awkward::detail::type_reinterpret<T, T_int>(ret);
    }

    // call atomic function with type cast between same underlying type
    template <typename T, typename Functor>
    __forceinline__  __device__
    T typesAtomicOperation64(T* addr, T val, Functor atomicFunc)
    {
        using T_int = long long int;
        T_int ret = atomicFunc(reinterpret_cast<T_int*>(addr),
            awkward::detail::type_reinterpret<T_int, T>(val));

        return awkward::detail::type_reinterpret<T, T_int>(ret);
    }

    // call atomic function with type cast between same underlying type
    template <typename T, typename Functor>
    __forceinline__  __device__
    T typesAtomicOperationU64(T* addr, T val, Functor atomicFunc)
    {
        using T_int = unsigned long long int;
        T_int ret = atomicFunc(reinterpret_cast<T_int*>(addr),
            awkward::detail::type_reinterpret<T_int, T>(val));

        return awkward::detail::type_reinterpret<T, T_int>(ret);
    }

} // namespace detail


template <typename T, typename BinaryOp>
__forceinline__  __device__
T genericAtomicOperation(T* address, T const & update_value, BinaryOp op)
{
    return awkward::detail::genericAtomicOperationImpl<T, BinaryOp, sizeof(T)>()
        (address, update_value, op);
}

// ------------------------------------------------------------------------
// Binary ops for sum, min, max, prod
struct DeviceSum {
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs + rhs;
    }

    template<typename T>
    static constexpr T identity() { return T{0}; }
};

struct DeviceMin{
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs <= rhs ? lhs : rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::max(); }
};

struct DeviceMax{
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs >= rhs ? lhs : rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::lowest(); }
};

struct DeviceProduct {
    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs)
    {
        return lhs * rhs;
    }

    template<typename T>
    static constexpr T identity() { return T{1}; }
};

} // namespace awkward

/* Overloads for `atomicMin` */
/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory, 
 * computes the minimum of old and val, and stores the result back to memory
 * at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported awkward types for `atomicMin` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * uint8_t, uint16_t, uint32_t, uint64_t.
 * CUDA natively supports `sint32`, `uint32`, `sint64`, `uint64`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
__forceinline__ __device__
int8_t atomicMin(int8_t* address, int8_t val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceMin{});
}

/**
 * @overload uint8_t atomicMin(uint8_t* address, uint8_t val)
 */
__forceinline__ __device__
uint8_t atomicMin(uint8_t* address, uint8_t val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceMin{});
}

/**
 * @overload int16_t atomicMin(int16_t* address, int16_t val)
 */
__forceinline__ __device__
int16_t atomicMin(int16_t* address, int16_t val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceMin{});
}

/**
 * @overload uint16_t atomicMin(uint16_t* address, uint16_t val)
 */
__forceinline__ __device__
uint16_t atomicMin(uint16_t* address, uint16_t val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceMin{});
}

/**
 * @overload int64_t atomicMin(int64_t* address, int64_t val)
 */
__forceinline__ __device__
int64_t atomicMin(int64_t* address, int64_t val)
{
    using T = long long int;
    return awkward::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicMin(a, v);});
}

/**
 * @overload uint64_t atomicMin(uint64_t* address, uint64_t val)
 */
__forceinline__ __device__
uint64_t atomicMin(uint64_t* address, uint64_t val)
{
    using T = unsigned long long int;
    return awkward::detail::typesAtomicOperationU64
        (address, val, [](T* a, T v){return atomicMin(a, v);});
}

/**
 * @overload float atomicMin(float* address, float val)
 */
__forceinline__ __device__
float atomicMin(float* address, float val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceMin{});
}

/**
 * @overload double atomicMin(double* address, double val)
 */
__forceinline__ __device__
double atomicMin(double* address, double val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceMin{});
}

/* Overloads for `atomicMax` */
/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory, 
 * computes the maximum of old and val, and stores the result back to memory
 * at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported awkward types for `atomicMax` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * uint8_t, uint16_t, uint32_t, uint64_t.
 * CUDA natively supports `sint32`, `uint32`, `sint64`, `uint64`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
__forceinline__ __device__
int8_t atomicMax(int8_t* address, int8_t val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceMax{});
}

/**
 * @overload uint8_t atomicMax(uint8_t* address, uint6_t val)
 */
__forceinline__ __device__
uint8_t atomicMax(uint8_t* address, uint8_t val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceMax{});
}

/**
 * @overload int16_t atomicMax(int16_t* address, int16_t val)
 */
__forceinline__ __device__
int16_t atomicMax(int16_t* address, int16_t val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceMax{});
}

/**
 * @overload uint16_t atomicMax(uint16_t* address, uint16_t val)
 */
__forceinline__ __device__
int16_t atomicMax(uint16_t* address, uint16_t val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceMax{});
}

/**
 * @overload int64_t atomicMax(int64_t* address, int64_t val)
 */
__forceinline__ __device__
int64_t atomicMax(int64_t* address, int64_t val)
{
    using T = long long int;
    return awkward::detail::typesAtomicOperation64
        (address, val, [](T* a, T v){return atomicMax(a, v);});
}

/**
 * @overload uint64_t atomicMax(uint64_t* address, uint64_t val)
 */
__forceinline__ __device__
uint64_t atomicMax(uint64_t* address, uint64_t val)
{
    using T = unsigned long long int;
    return awkward::detail::typesAtomicOperationU64
        (address, val, [](T* a, T v){return atomicMax(a, v);});
}

/**
 * @overload float atomicMax(float* address, float val)
 */
__forceinline__ __device__
float atomicMax(float* address, float val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceMax{});
}

/**
 * @overload double atomicMax(double* address, double val)
 */
__forceinline__ __device__
double atomicMax(double* address, double val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceMax{});
}

/* Overloads for `atomicAdd` */
/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory, 
 * computes (old + val), and stores the result back to memory at the same
 * address. These three operations are performed in one atomic transaction.
 *
 * The supported awkward types for `atomicAdd` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * uint8_t, uint16_t, uint32_t, uint64_t.
 * CUDA natively supports `sint32`, `uint32`, `uint64`, `float`, `double`
 * (`double` is supported after Pascal).
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be added
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
__forceinline__ __device__
int8_t atomicAdd(int8_t* address, int8_t val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceSum{});
}

/**
 * @overload uint8_t atomicAdd(uint8_t* address, uint8_t val)
 */
__forceinline__ __device__
uint8_t atomicAdd(uint8_t* address, uint8_t val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceSum{});
}

/**
 * @overload int16_t atomicAdd(int16_t* address, int16_t val)
 */
__forceinline__ __device__
int16_t atomicAdd(int16_t* address, int16_t val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceSum{});
}

/**
 * @overload uint16_t atomicAdd(uint16_t* address, uint16_t val)
 */
__forceinline__ __device__
int16_t atomicAdd(uint16_t* address, uint16_t val)
{
    return awkward::genericAtomicOperation(address, val, awkward::DeviceSum{});
}

/**
 * @overload int64_t atomicAdd(int64_t* address, int64_t val)
 */
__forceinline__ __device__
int64_t atomicAdd(int64_t* address, int64_t val)
{
    // `atomicAdd` supports uint64_t, but not int64_t
    return awkward::genericAtomicOperation(address, val, awkward::DeviceSum{});
}

#if defined(__CUDA_ARCH__) && ( __CUDA_ARCH__ < 600 )
/**
 * @overload double atomicAdd(double* address, double val)
 */
__forceinline__ __device__
double atomicAdd(double* address, double val)
{
    // `atomicAdd` for `double` is supported from Pascal
    return awkward::genericAtomicOperation(address, val, awkward::DeviceSum{});
}
#endif

/* Overloads for `atomicCAS` */
/** --------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory, 
 * computes the maximum of old and val, and stores the result back to memory
 * at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported awkward types for `atomicCAS` are:
 * int32_t, int64_t, float, double, uint32_t, uint64_t.
 * int8_t, int16_t are not supported as overloads
 * CUDA natively supports `sint32`, `uint32`, `uint64`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 *
 * @note int8_t, int16_t are not supported as `atomicCAS` overloads 
 * -------------------------------------------------------------------------**/
__forceinline__ __device__
int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val)
{
    using T = int64_t;
    return awkward::detail::typesAtomicCASImpl<T, sizeof(T)>()(address, compare, val);
}

/**
 * @overload float atomicCAS(float* address, float compare, float val)
 */
__forceinline__ __device__
float atomicCAS(float* address, float compare, float val)
{
    using T = float;
    return awkward::detail::typesAtomicCASImpl<T, sizeof(T)>()(address, compare, val);
}

/**
 * @overload double atomicCAS(double* address, double compare, double val)
 */
__forceinline__ __device__
double atomicCAS(double* address, double compare, double val)
{
    using T = double;
    return awkward::detail::typesAtomicCASImpl<T, sizeof(T)>()(address, compare, val);
}

/* Overloads for `atomicMul` */
/** -------------------------------------------------------------------------*
 * @brief reads the `old` located at the `address` in global or shared memory, 
 * computes (old * val), and stores the result back to memory
 * at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported awkward types for `atomicMul` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * uint8_t, uint16_t, uint32_t, uint64_t.
 * CUDA natively supports `sint32`, `uint32`, `sint64`, `uint64`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
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
