import cupy as cp

cuda_kernel = """
extern "C" {
    __global__ void reduce_sum_a(int* toptr, int* fromptr, int* parents, int lenparents, int outlength) {
       int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

       if (thread_id < outlength) {
          toptr[thread_id] = 0;
       }
    }
}
extern "C" {
    __global__ void reduce_sum_b(int* toptr, int* fromptr, int* parents, int lenparents, int outlength) {
       int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
       int stride = blockDim.x * gridDim.x;

       for (int i = thread_id; i < lenparents; i += stride) {
           atomicAdd(&toptr[parents[i]], fromptr[i]);
       }
    }
}
"""

parents = cp.array([0, 1, 1, 2, 2, 3, 3, 3, 5], dtype=cp.int32)
fromptr = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=cp.int32)
lenparents = len(parents)
outlength = int(cp.max(parents)) + 1
toptr = cp.zeros(outlength, dtype=cp.int32)

block_size = 256
grid_size = (lenparents + block_size - 1) // block_size

raw_module = cp.RawModule(code=cuda_kernel)

reduce_sum_a = raw_module.get_function('reduce_sum_a')
reduce_sum_b = raw_module.get_function('reduce_sum_b')

reduce_sum_a((grid_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength))
reduce_sum_b((grid_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength))

toptr_host = toptr.get()
print("atomic toptr:", toptr_host)