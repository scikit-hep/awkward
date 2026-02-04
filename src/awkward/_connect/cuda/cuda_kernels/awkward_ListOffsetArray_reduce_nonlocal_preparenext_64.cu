// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nextcarry, nextparents, nextlen, maxnextparents, distincts, distinctslen, offsetscopy, offsets, lenstarts, parents, maxcount, invocation_index, err_code) = args
//     distincts[:] = -1
//     maxnextparents[0] = 0
//     scan_in_array = cupy.zeros(lenstarts, dtype=cupy.int64)
//     k = 0
//     iteration = 0
//     max_iterations = nextlen  # Safety limit
//     while k < nextlen and iteration < max_iterations:
//         scan_in_array[:] = 0
//         cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_preparenext_64_a", 
//             nextcarry.dtype, nextparents.dtype, maxnextparents.dtype, distincts.dtype, offsets.dtype, offsetscopy.dtype, parents.dtype]))(grid, block,
//             (nextcarry, nextparents, nextlen, maxnextparents, distincts, distinctslen, offsetscopy, offsets, lenstarts, parents, maxcount, scan_in_array, invocation_index, err_code))
//         scan_out_array = cupy.cumsum(scan_in_array)
//         total_count = int(scan_out_array[-1]) if lenstarts > 0 else 0
//         if total_count == 0:
//             break
//         j_counter = cupy.zeros(1, dtype=cupy.int64)
//         cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_preparenext_64_b",
//             nextcarry.dtype, nextparents.dtype, maxnextparents.dtype, distincts.dtype, offsets.dtype, offsetscopy.dtype, parents.dtype]))(grid, block, 
//             (nextcarry, nextparents, nextlen, maxnextparents, distincts, distinctslen, offsetscopy, offsets, lenstarts, parents, maxcount, scan_in_array, scan_out_array, j_counter, k, invocation_index, err_code))
//         k += total_count
//         iteration += 1
// out["awkward_ListOffsetArray_reduce_nonlocal_preparenext_64_a", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_reduce_nonlocal_preparenext_64_b", {dtype_specializations}] = None
// END PYTHON

template <typename T_nextcarry, 
	  typename T_nextparents, 
	  typename T_maxnextparents, 
	  typename T_distincts, 
	  typename T_offsetscopy, 
	  typename T_offsets, 
	  typename T_parents>
__global__ void awkward_ListOffsetArray_reduce_nonlocal_preparenext_64_a(
  T_nextcarry* nextcarry, 
  T_nextparents* nextparents, 
  int64_t nextlen, 
  T_maxnextparents* maxnextparents, 
  T_distincts* distincts, 
  int64_t distinctslen, 
  T_offsetscopy* offsetscopy, 
  const T_offsets* offsets, 
  int64_t lenstarts, 
  const T_parents* parents, 
  int64_t maxcount,
  int64_t* scan_in_array, 
  uint64_t invocation_index,
  uint64_t* err_code) {
    
  if (err_code[0] != NO_ERROR) {
    return;
  }
  
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
  if (thread_id < lenstarts) {
    if (offsetscopy[thread_id] < offsets[thread_id + 1]) {
      scan_in_array[thread_id] = 1;
    } else {
      scan_in_array[thread_id] = 0;
    }
  }
}

template <typename T_nextcarry, 
	  typename T_nextparents, 
	  typename T_maxnextparents, 
	  typename T_distincts, 
	  typename T_offsetscopy, 
	  typename T_offsets, 
	  typename T_parents>
__global__ void awkward_ListOffsetArray_reduce_nonlocal_preparenext_64_b(
  T_nextcarry* nextcarry,
  T_nextparents* nextparents,
  int64_t nextlen,                    
  T_maxnextparents* maxnextparents,
  T_distincts* distincts,
  int64_t distinctslen,               
  T_offsetscopy* offsetscopy,
  const T_offsets* offsets,
  int64_t lenstarts,                  
  const T_parents* parents,
  int64_t maxcount,                  
  int64_t* scan_in_array,             
  int64_t* scan_out_array,            
  int64_t* j_counter,                
  int64_t k,                        
  uint64_t invocation_index,
  uint64_t* err_code) {

  if (err_code[0] != NO_ERROR) {
    return;
  }

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
  if (thread_id < lenstarts) {
    if (offsetscopy[thread_id] < offsets[thread_id + 1]) {
      int64_t output_idx = (thread_id > 0) ? scan_out_array[thread_id - 1] : 0;
      output_idx += k;
            
      if (output_idx < nextlen) {
        T_offsets diff = offsetscopy[thread_id] - offsets[thread_id];
        T_parents parent = parents[thread_id];
                
        nextcarry[output_idx] = (T_nextcarry)offsetscopy[thread_id];
        T_nextparents nextparent_value = (T_nextparents)(parent * maxcount + diff);
        nextparents[output_idx] = nextparent_value;
                
        atomicMax((unsigned long long*)maxnextparents, (unsigned long long)nextparent_value);
                
        if (nextparent_value >= 0 && nextparent_value < distinctslen) {
          T_distincts current = distincts[nextparent_value];
                    
          if (current == -1) {
            unsigned long long* addr = (unsigned long long*)&distincts[nextparent_value];
            unsigned long long expected = (unsigned long long)(-1LL);
            unsigned long long desired = (unsigned long long)(-2LL);
            unsigned long long old = atomicCAS(addr, expected, desired);
                        
            if (old == expected) {
              int64_t j_value = atomicAdd(j_counter, 1);
              distincts[nextparent_value] = (T_distincts)j_value;
            }
          }
        }
                
        if (sizeof(T_offsetscopy) == sizeof(unsigned long long)) {
          atomicAdd((unsigned long long*)&offsetscopy[thread_id], 1ULL);
        } else if (sizeof(T_offsetscopy) == sizeof(unsigned int)) {
          atomicAdd((unsigned int*)&offsetscopy[thread_id], 1U);
        } else {
          offsetscopy[thread_id]++;
        }
      }
    }
  }
}

