import cupy as cp

cuda_kernel = """
extern "C" {
    __global__ void awkward_reduce_max_complex_a(float* toptr, float* fromptr, int* parents, int lenparents, int outlength, int identity, int* partial) {
       int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

       if (thread_id < outlength) {
          toptr[thread_id * 2] = identity;
          toptr[thread_id * 2 + 1] = 0;
       }
    }
}
    
extern "C" {
    __global__ void awkward_reduce_max_complex_b(float* toptr, float* fromptr, int* parents, int lenparents, int outlength, int identity, int* partial) {
        extern __shared__ float shared[];

        int idx = threadIdx.x;
        int thread_id = blockIdx.x * blockDim.x + idx;

        if (thread_id < lenparents) {
            shared[idx * 2] = fromptr[thread_id * 2];
            shared[idx * 2 + 1] = fromptr[thread_id * 2 + 1];
        }
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            float real = identity;
            float imag = 0;
            if (idx >= stride && thread_id < lenparents && parents[thread_id] == parents[thread_id - stride]) {
                real = shared[(idx - stride) * 2];
                imag = shared[(idx - stride) * 2 + 1];
            }
            __syncthreads();
            if (shared[idx * 2] < real || shared[idx * 2] == real && shared[idx * 2 + 1] < imag) {
                shared[idx * 2] = real;
                shared[idx * 2 + 1] = imag;
            }
            __syncthreads();
        }

        if (thread_id < lenparents) {
            int parent = parents[thread_id];
            if (idx == blockDim.x - 1 || thread_id == lenparents - 1 || parents[thread_id] != parents[thread_id + 1]) {
                partial[(blockIdx.x * outlength + parent) * 2 ] = shared[idx * 2];
                partial[(blockIdx.x * outlength + parent) * 2 + 1] = shared[idx * 2 + 1];
            }
        }
    }
}

extern "C" {
    __global__ void awkward_reduce_max_complex_c(float* toptr, float* fromptr, int* parents, int lenparents, int outlength, int identity, int* partial) {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (thread_id < outlength) {
            float real = identity;
            float imag = 0;
            int blocks = (lenparents + blockDim.x - 1) / blockDim.x;
            for (int i = 0; i < blocks; ++i) {
            if (real < partial[2 * (i * outlength + thread_id)] ||
                (partial[2 * (i * outlength + thread_id)] == real &&
                imag < partial[2 * (i * outlength + thread_id) + 1])) {
                    real = partial[(i * outlength + thread_id) * 2];
                    imag = partial[(i * outlength + thread_id) * 2 + 1];
                }
            }
            toptr[thread_id * 2] = real;
            toptr[thread_id * 2 + 1] = imag;
        }
    }
}
"""

parents = cp.array([0, 1, 1, 2, 2, 2, 2, 2, 2, 5], dtype=cp.int32)
fromptr = cp.array([1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 0], dtype=cp.float32)
lenparents = len(parents)
outlength = int(cp.max(parents)) + 1
identity = cp.finfo(cp.float32).min
toptr = cp.full(outlength * 2, identity, dtype=cp.float32)
block_size = 2
partial = cp.full((outlength * ((lenparents + block_size - 1) // block_size)), identity, dtype=cp.float32)
grid_size = (lenparents + block_size - 1) // block_size
shared_mem_size = block_size * cp.float32().nbytes

raw_module = cp.RawModule(code=cuda_kernel)

awkward_reduce_max_complex_a = raw_module.get_function('awkward_reduce_max_complex_a')
awkward_reduce_max_complex_b = raw_module.get_function('awkward_reduce_max_complex_b')
awkward_reduce_max_complex_c = raw_module.get_function('awkward_reduce_max_complex_c')

awkward_reduce_max_complex_a((grid_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength, identity, partial))
awkward_reduce_max_complex_b((grid_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength, identity, partial), shared_mem=shared_mem_size)
awkward_reduce_max_complex_c(((outlength + block_size - 1) // block_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength, identity, partial))

toptr_host = toptr[0::2] + 1j * toptr[1::2]
print("tree reduction toptr:", toptr_host)