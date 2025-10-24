import cupy as cp

cuda_kernel = """
extern "C" {
    __global__ void awkward_reduce_max_a(int* toptr, int* fromptr, int* parents, int lenparents, int outlength, int identity, int* partial) {
       int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

       if (thread_id < outlength) {
          toptr[thread_id] = identity;
       }
    }
}
    
extern "C" {
    __global__ void awkward_reduce_max_b(int* toptr, int* fromptr, int* parents, int lenparents, int outlength, int identity, int* partial) {
        extern __shared__ int shared[];

        int idx = threadIdx.x;
        int thread_id = blockIdx.x * blockDim.x + idx;

        if (thread_id < lenparents) {
            shared[idx] = fromptr[thread_id];
        } else {
            shared[idx] = identity;
        }
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int val = identity;
            if (idx >= stride && thread_id < lenparents && parents[thread_id] == parents[thread_id - stride]) {
                val = shared[idx - stride];
            }
            shared[idx] = max(shared[idx], val);
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
    __global__ void awkward_reduce_max_c(int* toptr, int* fromptr, int* parents, int lenparents, int outlength, int identity, int* partial) {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (thread_id < outlength) {
            int maximum = identity;
            int blocks = (lenparents + blockDim.x - 1) / blockDim.x;
            for (int i = 0; i < blocks; ++i) {
                maximum = max(maximum, partial[i * outlength + thread_id]);
            }
            toptr[thread_id] = maximum;
        }
    }
}
"""

parents = cp.array([0, 1, 1, 2, 2, 2, 2, 2, 2, 5], dtype=cp.int32)
fromptr = cp.array([1, -2, -3, 4, 5, 6, 7, 8, 9, 10], dtype=cp.int32)
lenparents = len(parents)
outlength = int(cp.max(parents)) + 1
identity = cp.iinfo(cp.int32).min
toptr = cp.full(outlength, identity, dtype=cp.int32)


block_size = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
for i in range (len(block_size)):
    partial = cp.full((outlength * ((lenparents + block_size[i] - 1) // block_size[i])), identity, dtype=cp.int32)
    grid_size = (lenparents + block_size[i] - 1) // block_size[i]
    shared_mem_size = block_size[i] * cp.int32().nbytes

    raw_module = cp.RawModule(code=cuda_kernel)

    awkward_reduce_max_a = raw_module.get_function('awkward_reduce_max_a')
    awkward_reduce_max_b = raw_module.get_function('awkward_reduce_max_b')
    awkward_reduce_max_c = raw_module.get_function('awkward_reduce_max_c')

    awkward_reduce_max_a((grid_size,), (block_size[i],), (toptr, fromptr, parents, lenparents, outlength, identity, partial))
    awkward_reduce_max_b((grid_size,), (block_size[i],), (toptr, fromptr, parents, lenparents, outlength, identity, partial), shared_mem=shared_mem_size)
    awkward_reduce_max_c(((outlength + block_size[i] - 1) // block_size[i],), (block_size[i],), (toptr, fromptr, parents, lenparents, outlength, identity, partial))

    assert cp.array_equal(toptr, cp.array([1, -2, 9, -2147483648, -2147483648, 10]))