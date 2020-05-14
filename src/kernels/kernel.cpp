#include "awkward/kernels/kernel.h"
#include "awkward/cpu-kernels/operations.h"
#include "awkward/cpu-kernels/util.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/reducers.h"
#include "awkward/gpu-kernels/identities.h"


ERROR KernelCore::new_identities64(uint8_t memory_loc, int64_t *toptr, int64_t length) {
    if (memory_loc < 0)
        awkward_new_identities64(toptr, length);
    else {
        awkwardgpu_new_identities64(memory_loc, toptr, length);
    }
}

ERROR KernelCore::new_identities32(uint8_t memory_loc, int32_t *toptr, int64_t length) {
    if (memory_loc < 0)
        awkward_new_identities32(toptr, length);
    else {
        awkwardgpu_new_identities32(memory_loc, toptr, length);
    }
}
