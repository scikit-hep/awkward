// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toindex, current, size, fromtags, length, invocation_index, err_code) = args
//     atomicAdd_toptr = cupy.array(current, dtype=cupy.uint64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_UnionArray_regular_index_a", toindex.dtype, current.dtype, fromtags.dtype]))(grid, block, (toindex, current, size, fromtags, length, atomicAdd_toptr, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_UnionArray_regular_index_b", toindex.dtype, current.dtype, fromtags.dtype]))(grid, block, (toindex, current, size, fromtags, length, atomicAdd_toptr, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_UnionArray_regular_index_c", toindex.dtype, current.dtype, fromtags.dtype]))(grid, block, (toindex, current, size, fromtags, length, atomicAdd_toptr, invocation_index, err_code))
// out["awkward_UnionArray_regular_index_a", {dtype_specializations}] = None
// out["awkward_UnionArray_regular_index_b", {dtype_specializations}] = None
// out["awkward_UnionArray_regular_index_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_UnionArray_regular_index_a(
    T* toindex,
    C* current,
    int64_t size,
    const U* fromtags,
    int64_t length,
    uint64_t* atomicAdd_toptr,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < size) {
      atomicAdd_toptr[thread_id] = 0;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_UnionArray_regular_index_b(
    T* toindex,
    C* current,
    int64_t size,
    const U* fromtags,
    int64_t length,
    uint64_t* atomicAdd_toptr,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      U tag = fromtags[thread_id];
      toindex[(size_t)thread_id] = atomicAdd(atomicAdd_toptr + tag, 1);
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_UnionArray_regular_index_c(
    T* toindex,
    C* current,
    int64_t size,
    const U* fromtags,
    int64_t length,
    uint64_t* atomicAdd_toptr,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      current[thread_id] = (C)atomicAdd_toptr[thread_id];
    }
  }
}
