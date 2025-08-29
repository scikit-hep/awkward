// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// import cupy as cp
// import numpy as np
// import cuda.cccl.parallel.experimental as parallel
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
//         val: np.float32
//         idx: np.int32
// 
//     # Reducer: return element with smaller value
//     def argmin_reducer(x, y):
//         return x if x.val < y.val else y
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
//     d_struct.view(np.float32)[:lenparents] = fromptr       # val field
//     d_struct.view(np.int32)[lenparents:2*lenparents] = indices  # idx field
// 
//     # Temporary array to hold ValueWithIndex reduction results
//     d_out_struct = cp.empty(outlength, ValueWithIndex.dtype)
// 
//     # Identity element for reduction
//     init_val = ValueWithIndex(np.float32(np.inf), np.int32(-1))
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
//     # Extract only the indices into toptr
//     toptr[:outlength] = d_out_struct.view(np.int32)[1::2][:outlength]
// 
// END PYTHON
