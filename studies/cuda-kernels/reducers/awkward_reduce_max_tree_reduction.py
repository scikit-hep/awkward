import cupy as cp

cuda_kernel = """
extern "C" {
    __global__ void awkward_reduce_max_a(int* toptr, int* fromptr, int* parents, int lenparents, int outlength, int identity) {
       int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

       if (thread_id < outlength) {
          toptr[thread_id] = identity;
       }
    }
}
    extern "C" {
    __global__ void awkward_reduce_max_b(int *toptr, int *fromptr, int *parents, int lenparents, int outlength, int identity) {
        extern __shared__ int shared[];

        int idx = threadIdx.x;
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (thread_id < lenparents) {
            shared[idx] = fromptr[thread_id];
            __syncthreads();

            for (int stride = 1; stride < blockDim.x; stride *= 2) {
                int index = idx - stride;
                if (index >= 0 && parents[index] == parents[idx]) {
                    shared[idx] = max(shared[idx], shared[index]);
                }
                __syncthreads();
            }

            fromptr[thread_id] = shared[idx];

            if (idx == blockDim.x - 1 || thread_id == lenparents - 1 || parents[thread_id] != parents[thread_id + 1]) {
                int parent = parents[thread_id];
                if (parent < lenparents) {
                    toptr[parent] = shared[idx];
                }
            }
        }
    }
}
"""

parents = cp.array([0, 1, 1, 2, 2, 3, 3, 3, 5], dtype=cp.int32)
fromptr = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=cp.int32)
lenparents = len(parents)
outlength = int(cp.max(parents)) + 1
identity = cp.int32(cp.iinfo(cp.int32).min)
toptr = cp.zeros(outlength, dtype=cp.int32)

block_size = 256
grid_size = (lenparents + block_size - 1) // block_size

raw_module = cp.RawModule(code=cuda_kernel)

awkward_reduce_max_a = raw_module.get_function('awkward_reduce_max_a')
awkward_reduce_max_b = raw_module.get_function('awkward_reduce_max_b')

awkward_reduce_max_a((grid_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength, identity))
awkward_reduce_max_b((grid_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength, identity))

toptr_host = toptr.get()
print("tree reduction toptr:", toptr_host)