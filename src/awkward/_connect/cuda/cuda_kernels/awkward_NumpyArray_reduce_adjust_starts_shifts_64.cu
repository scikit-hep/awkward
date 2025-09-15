// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// Signature preserved: do NOT change arg list
template <typename T, typename C, typename U, typename V>
__global__ void
awkward_NumpyArray_reduce_adjust_starts_shifts_64(
    T* toptr,
    int64_t outlength,
    const C* parents,
    const U* starts,
    const V* shifts,
    uint64_t invocation_index,
    uint64_t* err_code) {

  // Small helper: set first error code atomically (1 = bounds)
  auto set_err_once = [&](uint64_t code) {
    // atomicCAS for unsigned long long on device
    unsigned long long* ec = (unsigned long long*)err_code;
    unsigned long long expected = 0ULL;
    // attempt to set to `code` if it is still zero
    atomicCAS(ec, expected, (unsigned long long)code);
  };

  // If there's already an error, skip work
  if (err_code[0] != 0) {
    return;
  }

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= outlength) {
    return;
  }

  // Read proposed index
  int64_t i = toptr[thread_id];

  // Must be non-negative
  if (i < 0) {
    return;
  }

  // Conservative upper-bound check using outlength
  // NOTE: This is conservative — if parents/shifts are actually longer than outlength,
  // some valid indices i >= outlength will be skipped here (to avoid OOB crashes).
  if (i >= outlength) {
    set_err_once(1);   // record a bounds error for diagnostics
    return;
  }

  // Safe to index parents[i] (within our conservative bound)
  int64_t parent = parents[i];

  // parent must be a valid index into starts; use the same conservative bound.
  if (parent < 0 || parent >= outlength) {
    set_err_once(2);   // parent OOB
    return;
  }

  // Now safe (by our conservative checks) to read starts[parent] and shifts[i]
  int64_t start = starts[parent];
  toptr[thread_id] += (shifts[i] - start);

  // done
}
