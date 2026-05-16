// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, offsets, outlength, invocation_index, err_code) = args
//     # Offsets-pipeline: one thread per output bin, mirroring the CPU
//     # awkward_reduce_sum.cpp loop. No host-side parents derivation;
//     # the kernel walks fromptr[offsets[bin] .. offsets[bin+1]) directly.
//     if block[0] > 0:
//         grid_size = math.floor((int(outlength) + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_reduce_sum_kernel', cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, offsets.dtype]))((grid_size,), block, (toptr, fromptr, offsets, outlength, invocation_index, err_code))
// out['awkward_reduce_sum_kernel', {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename V>
__global__ void
awkward_reduce_sum_kernel(
    T* toptr,
    const C* fromptr,
    const V* offsets,
    int64_t outlength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin < outlength) {
      T acc = (T)0;
      int64_t start = (int64_t)offsets[bin];
      int64_t stop  = (int64_t)offsets[bin + 1];
      for (int64_t i = start; i < stop; i++) {
        acc += (T)fromptr[i];
      }
      toptr[bin] = acc;
    }
  }
}
