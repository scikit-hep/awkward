import cupy as cp

cuda_kernel = """
extern "C" {
    __global__ void awkward_reduce_countnonzero_a(long long* toptr, bool* fromptr, long long* parents, long long lenparents, long long outlength, long long* partial) {
       long long thread_id = blockIdx.x * blockDim.x + threadIdx.x;

       if (thread_id < outlength) {
          toptr[thread_id] = 0;
       }
    }
}
    
extern "C" {
    __global__ void awkward_reduce_countnonzero_b(long long* toptr, bool* fromptr, long long* parents, long long lenparents, long long outlength, long long* partial) {
        extern __shared__ long long shared[];

        long long idx = threadIdx.x;
        long long thread_id = blockIdx.x * blockDim.x + idx;

        if (thread_id < lenparents) {
            shared[idx] = (fromptr[thread_id] != 0) ? 1 : 0;
        }
        __syncthreads();

        for (long long stride = 1; stride < blockDim.x; stride *= 2) {
            long long val = 0;
            if (idx >= stride && thread_id < lenparents && parents[thread_id] == parents[thread_id - stride]) {
                val = shared[idx - stride];
            }
            __syncthreads();
            shared[idx] += val;
            __syncthreads();
        }

        if (thread_id < lenparents) {
            long long parent = parents[thread_id];
            if (idx == blockDim.x - 1 || thread_id == lenparents - 1 || parents[thread_id] != parents[thread_id + 1]) {
                partial[blockIdx.x * outlength + parent] = shared[idx];
            }
        }
    }
}

extern "C" {
    __global__ void awkward_reduce_countnonzero_c(long long* toptr, bool* fromptr, long long* parents, long long lenparents, long long outlength, long long* partial) {
        long long thread_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (thread_id < outlength) {
            long long countnonzero = 0;
            long long blocks = (lenparents + blockDim.x - 1) / blockDim.x;
            for (long long i = 0; i < blocks; ++i) {
                countnonzero += partial[i * outlength + thread_id];
            }
            toptr[thread_id] = countnonzero;
        }
    }
}
"""

parents = cp.array([0, 1, 1, 2, 2, 2, 2, 2, 2, 5], dtype=cp.int64)
fromptr = cp.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=cp.bool_)
lenparents = len(parents)
outlength = int(cp.max(parents)) + 1
toptr = cp.zeros(outlength, dtype=cp.int64)

block_size = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
for i in range (len(block_size)):
    partial = cp.zeros((outlength * ((lenparents + block_size[i] - 1) // block_size[i])), dtype=cp.int64)
    grid_size = (lenparents + block_size[i] - 1) // block_size[i]
    shared_mem_size = block_size[i] * cp.int64().nbytes

    raw_module = cp.RawModule(code=cuda_kernel)

    awkward_reduce_countnonzero_a = raw_module.get_function('awkward_reduce_countnonzero_a')
    awkward_reduce_countnonzero_b = raw_module.get_function('awkward_reduce_countnonzero_b')
    awkward_reduce_countnonzero_c = raw_module.get_function('awkward_reduce_countnonzero_c')

    awkward_reduce_countnonzero_a((grid_size,), (block_size[i],), (toptr, fromptr, parents, lenparents, outlength, partial))
    awkward_reduce_countnonzero_b((grid_size,), (block_size[i],), (toptr, fromptr, parents, lenparents, outlength, partial), shared_mem=shared_mem_size)
    awkward_reduce_countnonzero_c(((outlength + block_size[i] - 1) // block_size[i],), (block_size[i],), (toptr, fromptr, parents, lenparents, outlength, partial))

    assert cp.array_equal(toptr, cp.array([1, 2, 4, 0, 0, 0]))