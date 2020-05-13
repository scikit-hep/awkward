#include <awkward/cpu-kernels/util.h>
#include <cstring>
#include "awkward/gpu-kernels/gpu_kernels.h"

template<typename T>
ERROR GPUKernels::awkward_new_identities(
        T *toptr,
        int64_t length) {
    for (T i = 0; i < length; i++) {
        toptr[i] = i;
    }
    return success();
}

ERROR GPUKernels::awkward_new_identities32(
        int32_t *toptr,
        int64_t length) {
    return GPUKernels::awkward_new_identities<int32_t>(
            toptr,
            length);
}

ERROR GPUKernels::awkward_new_identities64(
        int64_t *toptr,
        int64_t length) {
    return GPUKernels::awkward_new_identities<int64_t>(
            toptr,
            length);
}

template<typename T>
ERROR GPUKernels::awkward_carry_arange(
        T *toptr,
        int64_t length) {
    for (int64_t i = 0; i < length; i++) {
        toptr[i] = i;
    }
    return success();
}

ERROR GPUKernels::awkward_carry_arange_32(
        int32_t *toptr,
        int64_t length) {
    return GPUKernels::awkward_carry_arange<int32_t>(
            toptr,
            length);
}

ERROR GPUKernels::awkward_carry_arange_U32(
        uint32_t *toptr,
        int64_t length) {
    return GPUKernels::awkward_carry_arange<uint32_t>(
            toptr,
            length);
}

ERROR GPUKernels::awkward_carry_arange_64(
        int64_t *toptr,
        int64_t length) {
    return GPUKernels::awkward_carry_arange<int64_t>(
            toptr,
            length);
}

template<typename T>
ERROR GPUKernels::awkward_numpyarray_getitem_next_null(
        uint8_t *toptr,
        const uint8_t *fromptr,
        int64_t len,
        int64_t stride,
        int64_t offset,
        const T *pos) {
    for (int64_t i = 0; i < len; i++) {
        std::memcpy(&toptr[i * stride],
                    &fromptr[offset + pos[i] * stride],
                    (size_t) stride);
    }
    return success();
}

ERROR GPUKernels::awkward_numpyarray_getitem_next_null_64(
        uint8_t *toptr,
        const uint8_t *fromptr,
        int64_t len,
        int64_t stride,
        int64_t offset,
        const int64_t *pos) {
    return GPUKernels::awkward_numpyarray_getitem_next_null(
            toptr,
            fromptr,
            len,
            stride,
            offset,
            pos);
}

template<typename T>
ERROR GPUKernels::awkward_regulararray_num(
        T *tonum,
        int64_t size,
        int64_t length) {
    for (int64_t i = 0; i < length; i++) {
        tonum[i] = size;
    }
    return success();
}

ERROR GPUKernels::awkward_regulararray_num_64(
        int64_t *tonum,
        int64_t size,
        int64_t length) {
    return GPUKernels::awkward_regulararray_num<int64_t>(
            tonum,
            size,
            length);
}

template<typename FROM, typename TO>
ERROR GPUKernels::awkward_numpyarray_fill(
        TO *toptr,
        int64_t tooffset,
        const FROM *fromptr,
        int64_t fromoffset,
        int64_t length) {
    for (int64_t i = 0; i < length; i++) {
        toptr[tooffset + i] = (TO) fromptr[fromoffset + i];
    }
    return success();
}

template<typename TO>
ERROR GPUKernels::awkward_numpyarray_fill_frombool(
        TO *toptr,
        int64_t tooffset,
        const bool *fromptr,
        int64_t fromoffset,
        int64_t length) {
    for (int64_t i = 0; i < length; i++) {
        toptr[tooffset + i] = (TO) (fromptr[fromoffset + i] != 0);
    }
    return success();
}

ERROR GPUKernels::awkward_numpyarray_fill_todouble_fromdouble(
        double *toptr,
        int64_t tooffset,
        const double *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<double, double>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_todouble_fromfloat(
        double *toptr,
        int64_t tooffset,
        const float *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<float, double>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_todouble_from64(
        double *toptr,
        int64_t tooffset,
        const int64_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<int64_t, double>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_todouble_fromU64(
        double *toptr,
        int64_t tooffset,
        const uint64_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<uint64_t, double>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_todouble_from32(
        double *toptr,
        int64_t tooffset,
        const int32_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<int32_t, double>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_todouble_fromU32(
        double *toptr,
        int64_t tooffset,
        const uint32_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<uint32_t, double>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_todouble_from16(
        double *toptr,
        int64_t tooffset,
        const int16_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<int16_t, double>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_todouble_fromU16(
        double *toptr,
        int64_t tooffset,
        const uint16_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<uint16_t, double>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_todouble_from8(
        double *toptr,
        int64_t tooffset,
        const int8_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<int8_t, double>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_todouble_fromU8(
        double *toptr,
        int64_t tooffset,
        const uint8_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<uint8_t, double>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_todouble_frombool(
        double *toptr,
        int64_t tooffset,
        const bool *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill_frombool<double>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_toU64_fromU64(
        uint64_t *toptr,
        int64_t tooffset,
        const uint64_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<uint64_t, uint64_t>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_to64_from64(
        int64_t *toptr,
        int64_t tooffset,
        const int64_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<int64_t, int64_t>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_to64_fromU64(
        int64_t *toptr,
        int64_t tooffset,
        const uint64_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    for (int64_t i = 0; i < length; i++) {
        if (fromptr[fromoffset + i] > kMaxInt64) {
            return failure("uint64 value too large for int64 output", i, kSliceNone);
        }
        toptr[tooffset + i] = fromptr[fromoffset + i];
    }
    return success();
}

ERROR GPUKernels::awkward_numpyarray_fill_to64_from32(
        int64_t *toptr,
        int64_t tooffset,
        const int32_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<int32_t, int64_t>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_to64_fromU32(
        int64_t *toptr,
        int64_t tooffset,
        const uint32_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<uint32_t, int64_t>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_to64_from16(
        int64_t *toptr,
        int64_t tooffset,
        const int16_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<int16_t, int64_t>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_to64_fromU16(
        int64_t *toptr,
        int64_t tooffset,
        const uint16_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<uint16_t, int64_t>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_to64_from8(
        int64_t *toptr,
        int64_t tooffset,
        const int8_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<int8_t, int64_t>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_to64_fromU8(
        int64_t *toptr,
        int64_t tooffset,
        const uint8_t *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill<uint8_t, int64_t>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_to64_frombool(
        int64_t *toptr,
        int64_t tooffset,
        const bool *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill_frombool<int64_t>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

ERROR GPUKernels::awkward_numpyarray_fill_tobool_frombool(
        bool *toptr,
        int64_t tooffset,
        const bool *fromptr,
        int64_t fromoffset,
        int64_t length) {
    return awkward_numpyarray_fill_frombool<bool>(
            toptr,
            tooffset,
            fromptr,
            fromoffset,
            length);
}

template <typename T>
ERROR GPUKernels::awkward_numpyarray_getitem_next_array_advanced(
        T* nextcarryptr,
        const T* carryptr,
        const T* advancedptr,
        const T* flatheadptr,
        int64_t lencarry,
        int64_t skip) {
    for (int64_t i = 0;  i < lencarry;  i++) {
        nextcarryptr[i] = skip*carryptr[i] + flatheadptr[advancedptr[i]];
    }
    return success();
}
ERROR GPUKernels::awkward_numpyarray_getitem_next_array_advanced_64(
        int64_t* nextcarryptr,
        const int64_t* carryptr,
        const int64_t* advancedptr,
        const int64_t* flatheadptr,
        int64_t lencarry,
        int64_t skip) {
    return awkward_numpyarray_getitem_next_array_advanced(
            nextcarryptr,
            carryptr,
            advancedptr,
            flatheadptr,
            lencarry,
            skip);
}

ERROR GPUKernels::awkward_numpyarray_getitem_boolean_numtrue(
        int64_t* numtrue,
        const int8_t* fromptr,
        int64_t byteoffset,
        int64_t length,
        int64_t stride) {
    *numtrue = 0;
    for (int64_t i = 0;  i < length;  i += stride) {
        *numtrue = *numtrue + (fromptr[byteoffset + i] != 0);
    }
    return success();
}

