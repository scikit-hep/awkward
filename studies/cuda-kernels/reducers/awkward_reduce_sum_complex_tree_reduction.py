import cupy as cp

cuda_kernel = """
#include <cuda/std/complex>

extern "C" {
    __global__ void awkward_reduce_sum_complex_a(float* toptr, float* fromptr, int* parents, int lenparents, int outlength, float* partial) {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (thread_id < outlength) {
            toptr[thread_id * 2] = 0.0f;
            toptr[thread_id * 2 + 1] = 0.0f;
        }
    }

    __global__ void awkward_reduce_sum_complex_b(float* toptr, float* fromptr, int* parents, int lenparents, int outlength, float* partial) {
        extern __shared__ cuda::std::complex<float> shared[];

        int idx = threadIdx.x;
        int thread_id = blockIdx.x * blockDim.x + idx;

        if (thread_id < lenparents) {
            shared[idx] = cuda::std::complex<float>(fromptr[thread_id * 2], fromptr[thread_id * 2 + 1]);
        }
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            cuda::std::complex<float> val(0.0f, 0.0f);
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
                partial[(blockIdx.x * outlength + parent) * 2] = shared[idx].real();
                partial[(blockIdx.x * outlength + parent) * 2 + 1] = shared[idx].imag();
            }
        }
    }

    __global__ void awkward_reduce_sum_complex_c(float* toptr, float* fromptr, int* parents, int lenparents, int outlength, float* partial) {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (thread_id < outlength) {
            cuda::std::complex<float> sum(0.0f, 0.0f);
            int blocks = (lenparents + blockDim.x - 1) / blockDim.x;
            for (int i = 0; i < blocks; ++i) {
                cuda::std::complex<float> val(partial[(i * outlength + thread_id) * 2], partial[(i * outlength + thread_id) * 2 + 1]);
                sum += val;
            }
            toptr[thread_id * 2] = sum.real();
            toptr[thread_id * 2 + 1] = sum.imag();
        }
    }
}
"""

raw_module = cp.RawModule(code=cuda_kernel, options=('-I', '/usr/local/cuda-12.3/include/'),)

parents = cp.array([0, 1, 1, 2, 2, 2, 2, 2, 2, 5], dtype=cp.int32)
fromptr = cp.array([1, 0, 2.5677, 1.2345, 3.2367, 2.256576, 4.3456, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 0], dtype=cp.float32)
lenparents = len(parents)
outlength = int(cp.max(parents)) + 1
toptr = cp.zeros(2 * outlength, dtype=cp.float32)

block_size = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
for i in range(len(block_size)):
    partial = cp.zeros(2 * outlength * ((lenparents + block_size[i] - 1) // block_size[i]), dtype=cp.float32)
    grid_size = (lenparents + block_size[i] - 1) // block_size[i]
    shared_mem_size = block_size[i] * 2 * cp.float32().nbytes


    awkward_reduce_sum_complex_a = raw_module.get_function('awkward_reduce_sum_complex_a')
    awkward_reduce_sum_complex_b = raw_module.get_function('awkward_reduce_sum_complex_b')
    awkward_reduce_sum_complex_c = raw_module.get_function('awkward_reduce_sum_complex_c')

    awkward_reduce_sum_complex_a((grid_size,), (block_size[i],), (toptr, fromptr, parents, lenparents, outlength, partial))
    awkward_reduce_sum_complex_b((grid_size,), (block_size[i],), (toptr, fromptr, parents, lenparents, outlength, partial), shared_mem=shared_mem_size)
    awkward_reduce_sum_complex_c(((outlength + block_size[i] - 1) // block_size[i],), (block_size[i],), (toptr, fromptr, parents, lenparents, outlength, partial))

    print(block_size[i], toptr.get())