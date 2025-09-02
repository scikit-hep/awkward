// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// import math
// import cupy
//
// def f(grid, block, args):
//     (toptr_indices, fromptr, parents, lenparents, outlength,
//      invocation_index, err_code) = args
// 
//     identity = -1  # hardcoded identity
// 
//     # determine grid size
//     if block[0] > 0:
//         grid_size = math.floor((lenparents + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
// 
//     # allocate temporary buffers for values and indices
//     temp_values = cupy.full(lenparents, cupy.array([identity], dtype=fromptr.dtype), dtype=fromptr.dtype)
//     temp_indices = cupy.arange(lenparents, dtype=cupy.int64)
// 
//     # launch kernel pass a (initialize toptr_indices)
//     cuda_kernel_templates.get_function(
//         fetch_specialization([
//             "awkward_argmin_a",
//             cupy.dtype(toptr_indices.dtype).type,
//             cupy.dtype(fromptr.dtype).type,
//             parents.dtype
//         ])
//     )((grid_size,), block, (
//         toptr_indices, fromptr, parents, lenparents, outlength,
//         temp_values, temp_indices, invocation_index, err_code
//     ))
// 
//     # launch kernel pass b (compute argmin per segment)
//     cuda_kernel_templates.get_function(
//         fetch_specialization([
//             "awkward_argmin_b",
//             cupy.dtype(toptr_indices.dtype).type,
//             cupy.dtype(fromptr.dtype).type,
//             parents.dtype
//         ])
//     )((grid_size,), block, (
//         toptr_indices, fromptr, parents, lenparents, outlength,
//         temp_values, temp_indices, err_code
//     ))
// 
// out[f"awkward_argmin_a", {dtype_specializations}] = None
// out[f"awkward_argmin_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename U>
__global__ void
awkward_argmin_a(
    int64_t* toptr_indices,    // output: argmin indices
    const U* fromptr,
    const int32_t* parents,
    int64_t lenparents,
    int64_t outlength,
    T* temp_values,            // temporary minima per thread
    int64_t* temp_indices,     // temporary indices per thread
    uint64_t* err_code) {

  if (err_code[0] != 0) return;

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < outlength) {
    toptr_indices[thread_id] = -1;  // initialize argmin
  }
}

template <typename T, typename U>
__global__ void
awkward_argmin_b(
    int64_t* toptr_indices,    // output: argmin indices
    const U* fromptr,
    const int32_t* parents,
    int64_t lenparents,
    int64_t outlength,
    T* temp_values,
    int64_t* temp_indices,
    uint64_t* err_code) {

  if (err_code[0] != 0) return;

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= lenparents) return;

  // copy input value and index to temporary buffers
  temp_values[thread_id] = fromptr[thread_id];
  temp_indices[thread_id] = thread_id;
  __syncthreads();

  // intra-block reduction for segment-wise argmin
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (thread_id >= stride && parents[thread_id] == parents[thread_id - stride]) {
      T val1 = temp_values[thread_id - stride];
      T val2 = temp_values[thread_id];
      int64_t idx1 = temp_indices[thread_id - stride];
      int64_t idx2 = temp_indices[thread_id];

      if (val1 < val2 || (val1 == val2 && idx1 < idx2)) {
        temp_values[thread_id] = val1;
        temp_indices[thread_id] = idx1;
      }
    }
    __syncthreads();
  }

  // write final per-parent argmin index atomically
  int parent = parents[thread_id];
  if (thread_id == lenparents - 1 || parents[thread_id] != parents[thread_id + 1]) {
    atomicCAS(&toptr_indices[parent], -1, temp_indices[thread_id]);
  }
}

// # BEGIN PYTHON
// import cupy as cp
// import numpy as np
//
// def f(grid, block, args):
//     """
//     Argmin (local indices) implemented with CuPy two-pass approach.
//     args = (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code)
//     - toptr: device array (int32 or int64) to receive local argmin per segment
//     - fromptr: flattened device values (numeric)
//     - parents: device parent labels (sorted, present)
//     - lenparents: int (number of elements in fromptr)
//     - outlength: int (number of segments)
//     """
//     toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code = args
//
//     # Defensive casts to ensure correct types
//     lenparents = int(lenparents)
//     outlength = int(outlength)
//
//     # Quick exit: no elements or no segments
//     if lenparents == 0 or outlength == 0:
//         toptr[:outlength] = -1
//         return
//
//     # parents already on device and sorted/present
//     # compute unique parents, start indices and counts for present labels
//     unique_parents, start_indices, counts = cp.unique(
//         parents, return_index=True, return_counts=True
//     )
//
//     # Build full counts for every segment 0..outlength-1 (zeros for missing)
//     full_counts = cp.zeros(outlength, dtype=cp.int64)
//     full_counts[unique_parents.astype(cp.int64)] = counts
//
//     # Offsets in ListOffsetArray convention (len = outlength + 1)
//     offsets = cp.empty(outlength + 1, dtype=cp.int64)
//     offsets[0] = 0
//     cp.cumsum(full_counts, out=offsets[1:])   # offsets[1:] = cumsum(full_counts)
//
//     # If there are empty segments, ensure toptr has identity (-1)
//     toptr[:outlength] = -1
//
//     # If no non-empty segments, we are done
//     if unique_parents.size == 0:
//         return
//
//     # Build segment_ids for each element (device-only)
//     # Repeat segment ids according to lengths. Using cp.repeat handles this efficiently on the device.
//     lengths = (offsets[1:] - offsets[:-1]).astype(cp.int64)
//     # mask for non-empty segments
//     nonempty_mask = lengths > 0
//     nonempty_indices = cp.nonzero(nonempty_mask)[0]
//     # If there are many segments and few elements, cp.repeat of `arange(n_segments)` with lengths is ok.
//     #segment_ids = cp.repeat(cp.arange(outlength, dtype=cp.int64), lengths)
//
//     segment_ids = cp.concatenate([
//         cp.full((int(l),), i, dtype=cp.int64) for i, l in enumerate(lengths.get())
//     ])
//
//     # First pass: compute minimal value per (non-empty) segment
//     # allocate min_values with +inf for numeric types
//     dtype = fromptr.dtype
//     if np.issubdtype(dtype, np.floating):
//         inf = cp.asarray(np.inf, dtype=dtype)
//     else:
//         # integer types: use max int
//         inf = cp.asarray(np.iinfo(dtype).max, dtype=dtype)
//     min_values = cp.full(outlength, cp.iinfo(cp.int32).max, dtype=cp.int32)
//     segment_ids = segment_ids.astype(cp.int32)
//     fromptr = fromptr.astype(cp.int32)
//     #min_values = cp.full(outlength, inf, dtype=dtype)
//     cp.minimum.at(min_values, segment_ids, fromptr)
//
//     # Second pass: find the first occurrence (flat index) of the segment min
//     # Build a boolean mask of elements equal to their segment's min
//     eq_mask = fromptr == min_values[segment_ids]
//
//     # Candidate flat indices, non-candidates set to a big sentinel
//     flat_idx = cp.arange(lenparents, dtype=cp.int64)
//     sentinel = lenparents  # larger than any valid index
//     flat_idx_masked = cp.where(eq_mask, flat_idx, sentinel)
//
//     # For each segment, we want the first flat index among its elements that is equal to the min.
//     # reduceat with cp.minimum over the flattened masked indices with offsets works:
//     # cp.minimum.reduceat expects offsets into flat_idx_masked; offsets are already correct.
//     # But CuPy's reduceat requires the array length to match; it works with offsets array.
//     first_flat = cp.minimum.reduceat(flat_idx_masked, offsets[:-1])
//
//     # For empty segments, reduceat gives the value at that offset (which equals sentinel).
//     # Compute local indices: first_flat - offsets[:-1], but clamp empty segments to -1.
//     local_indices = first_flat - offsets[:-1]
//     # Where first_flat==sentinel => empty segment -> set -1
//     local_indices = cp.where(first_flat == sentinel, -1, local_indices).astype(toptr.dtype)
//
//     # Write result into toptr
//     toptr[:outlength] = local_indices[:outlength].copy()
// # END PYTHON

// # BEGIN PYTHON
// import cupy as cp
// import numpy as np
// import cuda.cccl.parallel.experimental as parallel
// from awkward._connect.cuda import argmin_reducer
//
// def f(grid, block, args):
//     """
//     Argmin reduction for sorted, present parents on device:
//     (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code)
//     toptr: will hold indices of minima
//     fromptr: input values
//     parents: segment labels
//     """
//     toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code = args
//
//     # Define struct for value + index
//     @parallel.gpu_struct
//     class ValueWithIndex:
//         val: fromptr.dtype
//         idx: np.int32
//
//     # Initialize output indices
//     toptr[:outlength] = -1
//
//     # Create offsets for segmented reduction
//     unique_parents, start_indices, counts = cp.unique(parents, return_index=True, return_counts=True)
//     offsets = cp.empty(outlength + 1, dtype=cp.int64)
//     offsets[0] = 0
//     full_counts = cp.zeros(outlength, dtype=cp.int64)
//     full_counts[unique_parents] = counts
//     cp.cumsum(full_counts, out=offsets[1:])
//
//     # Prepare input structs: store value and local index
//     indices = cp.arange(lenparents, dtype=np.int32)
//     d_struct = cp.empty(lenparents, ValueWithIndex.dtype)
//     d_struct.view(fromptr.dtype)[:lenparents] = fromptr       # val field
//     d_struct.view(np.int32)[lenparents:2*lenparents] = indices  # idx field
//
//     # Temporary array to hold ValueWithIndex reduction results
//     d_out_struct = cp.empty(outlength, ValueWithIndex.dtype)
//
//     # Host identity for the reducer (must be HOST, and hashable)
//     val_dtype = fromptr.dtype
//     if np.issubdtype(val_dtype, np.floating):
//         id_val = np.asarray(np.inf, dtype=val_dtype).item()
//     elif np.issubdtype(val_dtype, np.integer):
//         id_val = np.asarray(np.iinfo(val_dtype).max, dtype=val_dtype).item()
//     else:
//         # If complex or other types show up here, define a policy or raise
//         raise TypeError(f"argmin identity not defined for dtype {val_dtype!r}")
//
//     # Identity element for reduction
//     init_val = ValueWithIndex(id_val, np.int32(-1))
//
//     # Perform segmented reduce
//     parallel.segmented_reduce(
//         d_struct,
//         d_out_struct,
//         offsets[:-1],
//         offsets[1:],
//         argmin_reducer,
//         init_val,
//         outlength
//     )
//
//     print(d_out_struct)
//     # Extract only the indices into toptr
//     toptr[:outlength] = d_out_struct.view(np.int32)[1::2][:outlength].copy()
//
// # END PYTHON
