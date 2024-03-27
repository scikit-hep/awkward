// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (fromptr, fromoffsets, offsetslength, target, toptr, invocation_index, err_code) = args
//     diff = cupy.diff(fromoffsets)
//     mask = diff > target
//     scan_in_array = cupy.where(mask, diff, target)
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_NumpyArray_pad_zero_to_length", fromptr.dtype, fromoffsets.dtype, toptr.dtype]))(grid, block, (fromptr, fromoffsets, offsetslength, target, toptr, scan_in_array, invocation_index, err_code))
// out["awkward_NumpyArray_pad_zero_to_length", {dtype_specializations}] = None
// END PYTHON


template <typename T, typename C, typename U>
__global__ void
awkward_NumpyArray_pad_zero_to_length(
    const T* fromptr,
    const C* fromoffsets,
    int64_t offsetslength,
    int64_t target,
    U* toptr,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t l_to_char = 0;

    if (thread_id < offsetslength - 1) {
      if (thread_id > 0) {
        l_to_char = scan_in_array[thread_id - 1];
      }
      // Copy from src to dst
      for (int64_t j_from_char = fromoffsets[thread_id]; j_from_char < fromoffsets[thread_id + 1]; j_from_char++) {
        toptr[l_to_char++] = fromptr[j_from_char];
      }
      // Pad to remaining width
      auto n_to_pad = target - (fromoffsets[thread_id + 1] - fromoffsets[thread_id]);
      for (int64_t j_from_char = 0; j_from_char < n_to_pad; j_from_char++){
        toptr[l_to_char++] = 0;
      }
    }
  }
}
