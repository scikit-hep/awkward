import cupy as cp

cuda_kernel = """
extern "C" {
    __global__ void awkward_reduce_argmax_a(int* toptr, int* fromptr, int* parents, int lenparents, int outlength, int* partial) {
       int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

       if (thread_id < outlength) {
          toptr[thread_id] = -1;
       }
    }
}
    
extern "C" {
    __global__ void awkward_reduce_argmax_b(int* toptr, int* fromptr, int* parents, int lenparents, int outlength, int* partial) {
        extern __shared__ int shared[];

        int idx = threadIdx.x;
        int thread_id = blockIdx.x * blockDim.x + idx;

        if (thread_id < lenparents) {
            shared[idx] = thread_id;
        }
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int index = -1;
            if (idx >= stride && thread_id < lenparents && parents[thread_id] == parents[thread_id - stride]) {
                index = shared[idx - stride];
            }
            __syncthreads();
            if (index != -1 && (shared[idx] == -1 || fromptr[index] > fromptr[shared[idx]])) {
                shared[idx] = index;
            }
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
    __global__ void awkward_reduce_argmax_c(int* toptr, int* fromptr, int* parents, int lenparents, int outlength, int* partial) {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (thread_id < outlength) {
            int max_index = -1;
            int blocks = (lenparents + blockDim.x - 1) / blockDim.x;
            for (int i = 0; i < blocks; ++i) {
                int index = partial[i * outlength + thread_id];
                if (index != -1 && (max_index == -1 || fromptr[index] > fromptr[max_index])) {
                    max_index = index;
                }
            }
            toptr[thread_id] = max_index;
        }
    }
}
"""

parents = cp.array([0, 1, 1, 2, 2, 2, 2, 2, 2, 5], dtype=cp.int32)
fromptr = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=cp.int32)
lenparents = len(parents)
outlength = int(cp.max(parents)) + 1
toptr = cp.full(outlength, -1, dtype=cp.int32)


block_size = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
for i in range (len(block_size)):
    partial = cp.full((outlength * ((lenparents + block_size[i] - 1) // block_size[i])), -1, dtype=cp.int32)
    grid_size = (lenparents + block_size[i] - 1) // block_size[i]
    shared_mem_size = block_size[i] * cp.int32().nbytes

    raw_module = cp.RawModule(code=cuda_kernel)

    awkward_reduce_argmax_a = raw_module.get_function('awkward_reduce_argmax_a')
    awkward_reduce_argmax_b = raw_module.get_function('awkward_reduce_argmax_b')
    awkward_reduce_argmax_c = raw_module.get_function('awkward_reduce_argmax_c')

    awkward_reduce_argmax_a((grid_size,), (block_size[i],), (toptr, fromptr, parents, lenparents, outlength, partial))
    awkward_reduce_argmax_b((grid_size,), (block_size[i],), (toptr, fromptr, parents, lenparents, outlength, partial), shared_mem=shared_mem_size)
    awkward_reduce_argmax_c(((outlength + block_size[i] - 1) // block_size[i],), (block_size[i],), (toptr, fromptr, parents, lenparents, outlength, partial))

    assert cp.array_equal(toptr, cp.array([0, 2, 8, -1, -1, 9]))