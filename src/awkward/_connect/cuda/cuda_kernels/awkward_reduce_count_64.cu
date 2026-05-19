// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, offsets, outlength, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((int(outlength) + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_reduce_count_64_kernel', cupy.dtype(toptr.dtype).type, offsets.dtype]))((grid_size,), block, (toptr, offsets, outlength, invocation_index, err_code))
// out['awkward_reduce_count_64_kernel', {dtype_specializations}] = None
// END PYTHON

// Per-bin element count = offsets[bin+1] - offsets[bin]. One thread per bin.
template <typename T, typename V>
__global__ void
awkward_reduce_count_64_kernel(
    T* toptr,
    const V* offsets,
    int64_t outlength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin < outlength) {
      toptr[bin] = (T)((int64_t)offsets[bin + 1] - (int64_t)offsets[bin]);
    }
  }
}
