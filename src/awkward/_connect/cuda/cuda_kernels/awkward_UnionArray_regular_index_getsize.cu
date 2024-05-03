// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (size, fromtags, length, invocation_index, err_code) = args
//     if length > 0:
//         size[0] = 0 if cupy.all(fromtags < 0) else cupy.max(fromtags)
//     else:
//         size[0] = 0
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_UnionArray_regular_index_getsize", size.dtype, fromtags.dtype]))(grid, block, (size, fromtags, length, invocation_index, err_code))
// out["awkward_UnionArray_regular_index_getsize", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_UnionArray_regular_index_getsize(
    T* size,
    const C* fromtags,
    int64_t length,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    *size = *size + 1;
  }
}
