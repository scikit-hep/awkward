// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long long int64_t;
typedef signed char int_fast8_t;
typedef signed short int_fast16_t;
typedef signed int int_fast32_t;
typedef signed long long int_fast64_t;
typedef signed char int_least8_t;
typedef signed short int_least16_t;
typedef signed int int_least32_t;
typedef signed long long int_least64_t;
typedef signed long long intmax_t;
typedef signed long intptr_t;  //optional
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned char uint_fast8_t;
typedef unsigned short uint_fast16_t;
typedef unsigned int uint_fast32_t;
typedef unsigned long long uint_fast64_t;
typedef unsigned char uint_least8_t;
typedef unsigned short uint_least16_t;
typedef unsigned int uint_least32_t;
typedef unsigned long long uint_least64_t;
typedef unsigned long long uintmax_t;

#define RAISE_ERROR(ERROR_KERNEL_CODE) \
  atomicMin(err_code,                  \
            invocation_index*(1 << ERROR_BITS) + (int)(ERROR_KERNEL_CODE));

// BEGIN PYTHON
// def inclusive_scan(grid, block, args):
//     (d_in, invocation_index, err_code) = args
//     import math
//     d_out = cupy.empty(len(d_in), dtype=cupy.int64)
//     d_final = cupy.empty(len(d_in), dtype=cupy.int64)
//     stride = 1
//     total_steps = math.ceil(math.log2(len(d_in)))
//     for curr_step in range(1, total_steps + 1):
//         in_out_flag = (curr_step % 2) != 0
//         cuda_kernel_templates.get_function(fetch_specialization(['inclusive_scan_kernel', cupy.int64]))(grid, block, (d_in, d_out, d_final, curr_step, total_steps, stride, in_out_flag, len(d_in), invocation_index, err_code))
//         stride = stride * 2
//     return d_final
// out['inclusive_scan_kernel', cupy.int64] = inclusive_scan
// END PYTHON

template <typename T>
__global__ void
inclusive_scan_kernel(T* d_in,
                      T* d_out,
                      T* d_final,
                      int64_t curr_step,
                      int64_t total_steps,
                      int64_t stride,
                      bool in_out_flag,
                      int64_t length,
                      uint64_t* invocation_index,
                      uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t block_id = blockIdx.x + blockIdx.y * gridDim.x +
                       gridDim.x * gridDim.y * blockIdx.z;

    int64_t thread_id = block_id * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if (!in_out_flag) {
        if (thread_id < stride) {
          d_in[thread_id] = d_out[thread_id];
        } else {
          d_in[thread_id] = d_out[thread_id] + d_out[thread_id - stride];
        }
      } else {
        if (thread_id < stride) {
          d_out[thread_id] = d_in[thread_id];
        } else {
          d_out[thread_id] = d_in[thread_id] + d_in[thread_id - stride];
        }
      }

      if (curr_step == total_steps) {
        d_final[thread_id] = in_out_flag ? d_out[thread_id] : d_in[thread_id];
      }
    }
  }
}
