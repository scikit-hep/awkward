// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class LISTARRAY_BROADCAST_TOOFFSETS_ERRORS {
  ERROR_STOP_EXCEED_LENGTH,  // message: "stop[i] > length"
  ERROR_OFFSET_NON_MONOTONIC, // message: "offset[i+1] < offset[i]"
  ERROR_START_STOP_NEQ_COUNT, // message: "(stop[i] - start[i]) != (offset[i+1] - offset[i])"
};

// BEGIN PYTHON
// def f(grid, block, args):
//     (tooffsets, fromstarts, fromstops, length, invocation_index, err_code) = args
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_compact_offsets_a", tooffsets.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tooffsets, fromstarts, fromstops, length, invocation_index, err_code))
//     tooffsets = inclusive_scan(grid, block, (tooffsets, invocation_index, err_code))
// out["awkward_ListArray_compact_offsets_a", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_ListArray_broadcast_tooffsets(T* tocarry,
                                    const T* fromoffsets,
                                    int64_t offsetslength,
                                    const C* fromstarts,
                                    const C* fromstops,
                                    int64_t lencontent,
                                    uint64_t invocation_index,
                                    uint64_t* err_code) {

  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i_sublist = thread_id / (offsetslength - 1);
    int64_t j_carry = thread_id % (offsetslength - 1);

    C start = fromstarts[i_sublist];
    C stop = fromstops[i_sublist];
    if (start != stop  &&  stop > lencontent) {
      RAISE_ERROR(LISTARRAY_BROADCAST_TOOFFSETS_ERRORS::ERROR_STOP_EXCEED_LENGTH);
    }

    T count = fromoffsets[i_sublist + 1] - fromoffsets[i_sublist];
    if (count < 0) {
      RAISE_ERROR(LISTARRAY_BROADCAST_TOOFFSETS_ERRORS::ERROR_OFFSET_NON_MONOTONIC);
    }

    if (stop - start != count) {
      RAISE_ERROR(LISTARRAY_BROADCAST_TOOFFSETS_ERRORS::ERROR_START_STOP_NEQ_COUNT);
    }

    if (j_carry < count) {
        tocarry[thread_id] = (T)(j_carry + start);
    }
}
