// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, offsets, outlength, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((int(outlength) + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_reduce_countnonzero_complex_kernel', cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, offsets.dtype]))((grid_size,), block, (toptr, fromptr, offsets, outlength, invocation_index, err_code))
// out['awkward_reduce_countnonzero_complex_kernel', {dtype_specializations}] = None
// END PYTHON

// One thread per bin, mirroring awkward_reduce_countnonzero_complex.cpp.
// A complex value is "nonzero" if either its real or imaginary part is nonzero.
template <typename T, typename C, typename V>
__global__ void
awkward_reduce_countnonzero_complex_kernel(
    T* toptr,
    const C* fromptr,
    const V* offsets,
    int64_t outlength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin < outlength) {
      int64_t c = 0;
      int64_t start = (int64_t)offsets[bin];
      int64_t stop  = (int64_t)offsets[bin + 1];
      for (int64_t i = start; i < stop; i++) {
        if (fromptr[i * 2] != (C)0 || fromptr[i * 2 + 1] != (C)0) c++;
      }
      toptr[bin] = (T)c;
    }
  }
}
