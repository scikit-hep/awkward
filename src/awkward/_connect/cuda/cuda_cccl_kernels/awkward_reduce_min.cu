// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// import cuda.cccl.parallel.experimental as parallel
// import awkward._connect.cuda
// 
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, identity, invocation_index, err_code) = args
//     """
//     Min reduction using the API:
//     (toptr, fromptr, parents, lenparents, outlength, identity, invocation_index, err_code)
//     """
//     (toptr, fromptr, parents, lenparents, outlength, identity, invocation_index, err_code) = args
// 
//     # fromptr: input CuPy array
//     # toptr: output CuPy array
//     # lenparents: length of the input array (or segments)
//     # outlength: length of the output array
//     # identity: identity value for min (max integer)
//       
//     # If segmented reduction
//     if parents is not None and len(parents) > 0:
//         # parents defines segment offsets
//         start_o = cp.array(parents[:-1], dtype=np.int64)
//         end_o = cp.array(parents[1:], dtype=np.int64)
//         n_segments = outlength
//         parallel.segmented_reduce(
//             fromptr, toptr, start_o, end_o, awkward._connect.cuda.min_op, identity, n_segments
//         )
//     else:
//         # Simple reduction over whole array
//         parallel.reduce_into(fromptr, toptr, awkward._connect.cuda.min_op, lenparents, identity)
// END PYTHON
          
// //     if block[0] > 0:
// //         grid_size = math.floor((lenparents + block[0] - 1) / block[0])
// //     else:
// //         grid_size = 1
// //     temp = cupy.full(lenparents, cupy.array([identity]), dtype=toptr.dtype)
// //     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_min_a", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, toptr.dtype.type(identity), temp, invocation_index, err_code))
// //     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_min_b", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, toptr.dtype.type(identity), temp, invocation_index, err_code))
// // out["awkward_reduce_min_a", {dtype_specializations}] = None
// // out["awkward_reduce_min_b", {dtype_specializations}] = None
// // END PYTHON

// template <typename T, typename C, typename U>
// __global__ void
// awkward_reduce_min_a(
//     T* toptr,
//     const C* fromptr,
//     const U* parents,
//     int64_t lenparents,
//     int64_t outlength,
//     T identity,
//     T* temp,
//     uint64_t invocation_index,
//     uint64_t* err_code) {
//   if (err_code[0] == NO_ERROR) {
//     int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

//     if (thread_id < outlength) {
//       toptr[thread_id] = identity;
//     }
//   }
// }

// template <typename T, typename C, typename U>
// __global__ void
// awkward_reduce_min_b(
//     T* toptr,
//     const C* fromptr,
//     const U* parents,
//     int64_t lenparents,
//     int64_t outlength,
//     T identity,
//     T* temp,
//     uint64_t invocation_index,
//     uint64_t* err_code) {
//   if (err_code[0] == NO_ERROR) {
//     int64_t idx = threadIdx.x;
//     int64_t thread_id = blockIdx.x * blockDim.x + idx;

//     if (thread_id < lenparents) {
//       temp[thread_id] = fromptr[thread_id];
//     }
//     __syncthreads();

//         for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
//         T val = identity;
//         if (thread_id < lenparents && idx >= stride && parents[thread_id] == parents[thread_id - stride]) {
//             val = temp[thread_id - stride];
//         }
//         __syncthreads();
//         if (thread_id < lenparents) {
//             temp[thread_id] = val < temp[thread_id] ? val : temp[thread_id];
//         }
//         __syncthreads();
//     }

//     if (thread_id < lenparents) {
//         bool is_last_in_group =
//             (thread_id == lenparents - 1) ||
//             (parents[thread_id] != parents[thread_id + 1]);

//         if (is_last_in_group) {
//             int64_t parent = parents[thread_id];
//             atomicMin(&toptr[parent], temp[thread_id]);
//         }
//     }
//   }
// }
