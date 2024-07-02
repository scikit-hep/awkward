#include <iostream>
#include <cuda_runtime.h>
#include <vector>

__device__ int64_t choose(int n, int k) {
    if (n < k) return 0;
    if (n == k) return 1;

    int delta, imax;
    if (k < n - k) {
        delta = n - k;
        imax = k;
    } else {
        delta = k;
        imax = n - k;
    }

    int64_t ans = delta + 1;
    for (int i = 2; i <= imax; ++i) {
        ans = (ans * (delta + i)) / i;
    }

    return ans;
}

__device__ int64_t largestV(int a, int b, int64_t x) {
    int64_t v = a - 1;
    while (choose(v, b) > x) {
        --v;
    }
    return v;
}

__global__ void cuda_calculateMth(int n, int k, int** d_result, int totalcount, int start, int end) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= totalcount) return;

    int a = end - start;
    int b = k;
    int64_t x = choose(a, b) - 1 - pos;

    for (int i = 0; i < k; ++i) {
        int v = largestV(a, b, x);
        d_result[i][pos] = (end - 1) - v;
        x -= choose(v, b);
        a = v;
        b--;
    }
}

int main() {
    int n = 10;
    int k = 2;
    int start = 2;
    int end = 7;

    int totalcount = 1;
    int factorial = 1;
    for (int i = 0; i < k; ++i) {
        totalcount *= (end - start - i);
        factorial *= (i + 1);
    }
    totalcount /= factorial;

    std::vector<int*> result_ptrs(k);
    std::vector<int*> d_result_ptrs(k);

    for (int i = 0; i < k; ++i) {
        result_ptrs[i] = new int[totalcount];
        cudaMalloc(&d_result_ptrs[i], totalcount * sizeof(int));
    }

    int** d_result;
    cudaMalloc(&d_result, k * sizeof(int*));
    cudaMemcpy(d_result, d_result_ptrs.data(), k * sizeof(int*), cudaMemcpyHostToDevice);

    int threadsperblock = 512;
    int blockspergrid = (totalcount + threadsperblock - 1) / threadsperblock;

    cuda_calculateMth<<<blockspergrid, threadsperblock>>>(n, k, d_result, totalcount, start, end);

    for (int i = 0; i < k; ++i) {
        cudaMemcpy(result_ptrs[i], d_result_ptrs[i], totalcount * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_result);
    for (int i = 0; i < k; ++i) {
        cudaFree(d_result_ptrs[i]);
    }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < totalcount; ++j) {
            std::cout << result_ptrs[i][j] << " ";
        }
        std::cout << std::endl;
        delete[] result_ptrs[i];
    }

    return 0;
}
