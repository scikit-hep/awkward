// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nextcarry, nextoffsets, outindex, mask, offsets, outlength, validwhen, invocation_index, err_code) = args
//     # In the offsets-pipeline, total element count = offsets[outlength].
//     length = int(offsets[int(outlength)].item()) if hasattr(offsets[int(outlength)], "item") else int(offsets[int(outlength)])
//     scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     # Phases a and b need `length` threads; phase c needs `outlength + 1`.
//     # Compute per-phase grid sizes; reuse the caller's block size.
//     block_size = block[0] if isinstance(block, tuple) else block
//     if block_size <= 0:
//         block_size = 1
//     grid_ab = ((length + block_size - 1) //block_size,) if length > 0 else (1,)
//     grid_c = ((int(outlength) + 1 + block_size - 1) //block_size,)
//     # Phase a: per-element, mark valid elements with 1 in scan_in_array.
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_reduce_next_64_a', nextcarry.dtype, nextoffsets.dtype, outindex.dtype, mask.dtype, offsets.dtype]))(grid_ab, (block_size,), (nextcarry, nextoffsets, outindex, mask, offsets, outlength, validwhen, scan_in_array, invocation_index, err_code))
//     # Inclusive cumulative sum: scan_in_array[i] now holds the count of valid
//     # elements in [0, i].
//     scan_in_array = cupy.cumsum(scan_in_array)
//     # Phase b: per-element, write nextcarry / outindex using the prefix-sum.
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_reduce_next_64_b', nextcarry.dtype, nextoffsets.dtype, outindex.dtype, mask.dtype, offsets.dtype]))(grid_ab, (block_size,), (nextcarry, nextoffsets, outindex, mask, offsets, outlength, validwhen, scan_in_array, invocation_index, err_code))
//     # Phase c: per-bin-boundary, write nextoffsets[bin] = cumulative valid
//     # count just before offsets[bin]. Needs outlength + 1 threads.
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_reduce_next_64_c', nextcarry.dtype, nextoffsets.dtype, outindex.dtype, mask.dtype, offsets.dtype]))(grid_c, (block_size,), (nextcarry, nextoffsets, outindex, mask, offsets, outlength, validwhen, scan_in_array, invocation_index, err_code))
// out["awkward_ByteMaskedArray_reduce_next_64_a", {dtype_specializations}] = None
// out["awkward_ByteMaskedArray_reduce_next_64_b", {dtype_specializations}] = None
// out["awkward_ByteMaskedArray_reduce_next_64_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ByteMaskedArray_reduce_next_64_a(
    T* nextcarry,
    C* nextoffsets,
    U* outindex,
    const V* mask,
    const W* offsets,
    int64_t outlength,
    bool validwhen,
    int64_t* scan_in_array,
    uint64_t* invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t length = (int64_t)offsets[outlength];

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == validwhen) {
        scan_in_array[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ByteMaskedArray_reduce_next_64_b(
    T* nextcarry,
    C* nextoffsets,
    U* outindex,
    const V* mask,
    const W* offsets,
    int64_t outlength,
    bool validwhen,
    int64_t* scan_in_array,
    uint64_t* invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t length = (int64_t)offsets[outlength];

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == validwhen) {
        nextcarry[scan_in_array[thread_id] - 1] = thread_id;
        outindex[thread_id] = scan_in_array[thread_id] - 1;
      } else {
        outindex[thread_id] = -1;
      }
    }
  }
}

// Phase c: write nextoffsets per-bin boundary. nextoffsets has outlength + 1
// entries; thread `bin_boundary` writes nextoffsets[bin_boundary].
//
//   nextoffsets[0] = 0
//   nextoffsets[b] = #valid elements in [0, offsets[b])  for b > 0
//                  = scan_in_array[offsets[b] - 1]       (when offsets[b] > 0)
//                  = 0                                   (when offsets[b] == 0)
template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ByteMaskedArray_reduce_next_64_c(
    T* nextcarry,
    C* nextoffsets,
    U* outindex,
    const V* mask,
    const W* offsets,
    int64_t outlength,
    bool validwhen,
    int64_t* scan_in_array,
    uint64_t* invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength + 1) {
      int64_t boundary = (int64_t)offsets[thread_id];
      if (boundary == 0) {
        nextoffsets[thread_id] = 0;
      } else {
        nextoffsets[thread_id] = scan_in_array[boundary - 1];
      }
    }
  }
}
