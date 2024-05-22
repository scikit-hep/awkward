import cupy as cp

cuda_kernel = """
extern "C" {
    __global__ void awkward_reduce_sum_a(int* toptr, int* fromptr, int* parents, int lenparents, int outlength, int* partial) {
       int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

       if (thread_id < outlength) {
          toptr[thread_id] = 0;
       }
    }
}
    
extern "C" {
    __global__ void awkward_reduce_sum_b(int* toptr, int* fromptr, int* parents, int lenparents, int outlength, int* partial) {
        extern __shared__ int shared[];

        int idx = threadIdx.x;
        int thread_id = blockIdx.x * blockDim.x + idx;

        if (thread_id < lenparents) {
            shared[idx] = fromptr[thread_id];
        }
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int val = 0;
            if (idx >= stride && thread_id < lenparents && parents[thread_id] == parents[thread_id - stride]) {
                val = shared[idx - stride];
            }
            __syncthreads();
            shared[idx] += val;
            __syncthreads();
        }

        if (thread_id < lenparents) {
            int parent = parents[thread_id];
            if (idx == blockDim.x - 1 || thread_id == lenparents - 1 || parents[thread_id] != parents[thread_id + 1]) {
                partial[blockIdx.x * outlength + parent] = shared[idx];
            }
        }
    }
}

extern "C" {
    __global__ void awkward_reduce_sum_c(int* toptr, int* fromptr, int* parents, int lenparents, int outlength, int* partial) {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (thread_id < outlength) {
            int sum = 0;
            int blocks = (lenparents + blockDim.x - 1) / blockDim.x;
            for (int i = 0; i < blocks; ++i) {
                sum += partial[i * outlength + thread_id];
            }
            toptr[thread_id] = sum;
        }
    }
}
"""

parents = cp.array([0, 1, 1, 2, 2, 2, 2, 2, 2, 5], dtype=cp.int32)
fromptr = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=cp.int32)
lenparents = len(parents)
outlength = int(cp.max(parents)) + 1
toptr = cp.zeros(outlength, dtype=cp.int32)

block_size = 2
partial = cp.zeros((outlength * ((lenparents + block_size - 1) // block_size)), dtype=cp.int32)
grid_size = (lenparents + block_size - 1) // block_size
shared_mem_size = block_size * cp.int32().nbytes

raw_module = cp.RawModule(code=cuda_kernel)

awkward_reduce_sum_a = raw_module.get_function('awkward_reduce_sum_a')
awkward_reduce_sum_b = raw_module.get_function('awkward_reduce_sum_b')
awkward_reduce_sum_c = raw_module.get_function('awkward_reduce_sum_c')

awkward_reduce_sum_a((grid_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength, partial))
awkward_reduce_sum_b((grid_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength, partial), shared_mem=shared_mem_size)
awkward_reduce_sum_c(((outlength + block_size - 1) // block_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength, partial))

assert cp.array_equal(toptr, cp.array([1, 5, 39, 0, 0, 10]))