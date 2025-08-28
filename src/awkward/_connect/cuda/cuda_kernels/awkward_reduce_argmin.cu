// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
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
//     # Initialize output indices on device
//     toptr[:outlength] = -1
// 
//     # Compute segment start indices and counts
//     unique_parents, start_indices, counts = cp.unique(parents, return_index=True, return_counts=True)
// 
//     # Compute offsets
//     offsets = cp.empty(outlength + 1, dtype=cp.int64)
//     offsets[0] = 0
//     full_counts = cp.zeros(outlength, dtype=cp.int64)
//     full_counts[unique_parents] = counts
//     cp.cumsum(full_counts, out=offsets[1:])
// 
//     # Create array of indices to reduce
//     indices = cp.arange(lenparents, dtype=cp.int64)
//     dtype = np.int64
//     identity_host = np.array([0], dtype=dtype)
//     # Segmented reduce
//     parallel.segmented_reduce(
//         indices,
//         toptr,
//         offsets[:-1],
//         offsets[1:],
//         argmin_reducer,
//         identity_host,
//         outlength
//     )
// END PYTHON

// # BEGIN PYTHON
// import cupy as cp
// import numpy as np
// import cuda.cccl.parallel.experimental as parallel
// from awkward._connect.cuda import min_op_real, min_op_complex
// 
// def f(grid, block, args):
//     """
//     Min reduction for sorted, present parents on device:
//     (toptr, fromptr, parents, lenparents, outlength, identity, invocation_index, err_code)
//     """
//     toptr, fromptr, parents, lenparents, outlength, identity, invocation_index, err_code = args
// 
//     # Pick the operation by dtype
//     min_op = min_op_complex if np.issubdtype(fromptr.dtype, np.complexfloating) else min_op_real
// 
//     # Initialize output on device
//     toptr[:outlength] = identity
// 
//     # Keep host copy of identity for segmented_reduce
//     identity_host = np.asarray(identity, dtype=fromptr.dtype)
// 
//     # Parents are already sorted and on device
//     unique_parents, start_indices, counts = cp.unique(parents, return_index=True, return_counts=True)
// 
//     # Build full counts array for all segments
//     full_counts = cp.zeros(outlength, dtype=cp.int64)
//     full_counts[unique_parents] = counts
// 
//     # Compute offsets (length = outlength + 1)
//     offsets = cp.empty(outlength + 1, dtype=cp.int64)
//     offsets[0] = 0
//     cp.cumsum(full_counts, out=offsets[1:])
// 
//     # Segmented reduction (identity must be host scalar/array)
//     parallel.segmented_reduce(
//         fromptr, toptr, offsets[:-1], offsets[1:], min_op, identity_host, outlength
//     )
// # END PYTHON
