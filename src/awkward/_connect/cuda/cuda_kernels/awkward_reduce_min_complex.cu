// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, offsets, outlength, identity, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((int(outlength) + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_reduce_min_complex_kernel', cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, offsets.dtype]))((grid_size,), block, (toptr, fromptr, offsets, outlength, identity, invocation_index, err_code))
// out['awkward_reduce_min_complex_kernel', {dtype_specializations}] = None
// END PYTHON

// One thread per bin, mirroring awkward_reduce_min_complex.cpp.
// Lexicographic compare on (real, imag).
template <typename T, typename C, typename V>
__global__ void
awkward_reduce_min_complex_kernel(
    T* toptr,
    const C* fromptr,
    const V* offsets,
    int64_t outlength,
    T identity,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin < outlength) {
      T best_re = identity;
      T best_im = (T)0;
      bool seen = false;
      int64_t start = (int64_t)offsets[bin];
      int64_t stop  = (int64_t)offsets[bin + 1];
      for (int64_t i = start; i < stop; i++) {
        C x = fromptr[i * 2];
        C y = fromptr[i * 2 + 1];
        if (!seen || x < best_re || (x == best_re && y < best_im)) {
          best_re = (T)x;
          best_im = (T)y;
          seen = true;
        }
      }
      toptr[bin * 2]     = best_re;
      toptr[bin * 2 + 1] = best_im;
    }
  }
}
