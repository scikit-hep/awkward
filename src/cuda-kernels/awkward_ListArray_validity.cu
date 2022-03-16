// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// #define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_validity.cpp", line)

// #include "awkward/kernels.h"

template <typename C>
__global__ void
cuda_ListArray_validity(
  const C* starts,
  const C* stops,
  int64_t length,
  int64_t lencontent,
  int64_t invocation_index,
  int64_t* err_code) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < length && !err_code[0]) {
    C start = starts[thread_id];
    C stop = stops[thread_id];
    if (start != stop) {
      if (start > stop) {
        atomicMin(err_code, invocation_index * 8 + 1); // failure("start[i] > stop[i]", i, kSliceNone, FILENAME(__LINE__));
      }
      if (start < 0) {
        atomicMin(err_code, invocation_index * 8 + 2); // failure("start[i] < 0", i, kSliceNone, FILENAME(__LINE__));
      }
      if (stop > lencontent) {
        atomicMin(err_code, invocation_index * 8 + 3); // failure("stop[i] > len(content)", i, kSliceNone, FILENAME(__LINE__));
      }
    }
  }
}