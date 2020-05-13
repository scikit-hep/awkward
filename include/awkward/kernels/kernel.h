#ifndef AWKWARD_KERNEL_H
#define AWKWARD_KERNEL_H

class KernelCore {
public:
    KernelCore() {}

    virtual ERROR awkward_new_identities32(
            int32_t *toptr,
            int64_t length) = 0;

    virtual ERROR awkward_new_identities64(
            int64_t *toptr,
            int64_t length) = 0;

    virtual ERROR awkward_carry_arange_32(
            int32_t *toptr,
            int64_t length) = 0;

    virtual ERROR awkward_carry_arange_U32(
            uint32_t *toptr,
            int64_t length) = 0;

    virtual ERROR awkward_carry_arange_64(
            int64_t *toptr,
            int64_t length) = 0;

    virtual ERROR awkward_numpyarray_getitem_next_null_64(
            uint8_t *toptr,
            const uint8_t *fromptr,
            int64_t len,
            int64_t stride,
            int64_t offset,
            const int64_t *pos) = 0;

    virtual ERROR awkward_regulararray_num_64(
            int64_t *tonum,
            int64_t size,
            int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_todouble_fromdouble(double *toptr, int64_t tooffset, const double *fromptr,
                                                int64_t fromoffset,
                                                int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_todouble_fromfloat(double *toptr, int64_t tooffset, const float *fromptr,
                                               int64_t fromoffset,
                                               int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_todouble_from64(double *toptr, int64_t tooffset, const int64_t *fromptr, int64_t fromoffset,
                                            int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_todouble_fromU64(double *toptr, int64_t tooffset, const uint64_t *fromptr,
                                             int64_t fromoffset,
                                             int64_t length) = 0;


    virtual ERROR
    awkward_numpyarray_fill_todouble_from32(double *toptr, int64_t tooffset, const int32_t *fromptr, int64_t fromoffset,
                                            int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_todouble_fromU32(double *toptr, int64_t tooffset, const uint32_t *fromptr,
                                             int64_t fromoffset,
                                             int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_todouble_from16(double *toptr, int64_t tooffset, const int16_t *fromptr, int64_t fromoffset,
                                            int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_todouble_fromU16(double *toptr, int64_t tooffset, const uint16_t *fromptr,
                                             int64_t fromoffset,
                                             int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_todouble_from8(double *toptr, int64_t tooffset, const int8_t *fromptr, int64_t fromoffset,
                                           int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_todouble_fromU8(double *toptr, int64_t tooffset, const uint8_t *fromptr, int64_t fromoffset,
                                            int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_todouble_frombool(double *toptr, int64_t tooffset, const bool *fromptr, int64_t fromoffset,
                                              int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_toU64_fromU64(uint64_t *toptr, int64_t tooffset, const uint64_t *fromptr,
                                          int64_t fromoffset,
                                          int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_to64_from64(int64_t *toptr, int64_t tooffset, const int64_t *fromptr, int64_t fromoffset,
                                        int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_to64_fromU64(int64_t *toptr, int64_t tooffset, const uint64_t *fromptr, int64_t fromoffset,
                                         int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_to64_from32(int64_t *toptr, int64_t tooffset, const int32_t *fromptr, int64_t fromoffset,
                                        int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_to64_fromU32(int64_t *toptr, int64_t tooffset, const uint32_t *fromptr, int64_t fromoffset,
                                         int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_to64_from16(int64_t *toptr, int64_t tooffset, const int16_t *fromptr, int64_t fromoffset,
                                        int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_to64_fromU16(int64_t *toptr, int64_t tooffset, const uint16_t *fromptr, int64_t fromoffset,
                                         int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_to64_from8(int64_t *toptr, int64_t tooffset, const int8_t *fromptr, int64_t fromoffset,
                                       int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_to64_fromU8(int64_t *toptr, int64_t tooffset, const uint8_t *fromptr, int64_t fromoffset,
                                        int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_to64_frombool(int64_t *toptr, int64_t tooffset, const bool *fromptr, int64_t fromoffset,
                                          int64_t length) = 0;

    virtual ERROR
    awkward_numpyarray_fill_tobool_frombool(bool *toptr, int64_t tooffset, const bool *fromptr, int64_t fromoffset,
                                            int64_t length) = 0;

    virtual ERROR awkward_numpyarray_getitem_next_array_advanced_64(int64_t *nextcarryptr, const int64_t *carryptr,
                                                                    const int64_t *advancedptr,
                                                                    const int64_t *flatheadptr,
                                                                    int64_t lencarry, int64_t skip) = 0;


    virtual ERROR awkward_numpyarray_getitem_boolean_numtrue(int64_t *numtrue, const int8_t *fromptr,
                                                             int64_t byteoffset,
                                                             int64_t length,
                                                             int64_t stride) = 0;

};


#endif //AWKWARD_KERNEL_H
