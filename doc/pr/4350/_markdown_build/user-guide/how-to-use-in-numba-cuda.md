# How to use Awkward Arrays in Numba’s CUDA target

Awkward Array defines extentions to the Numba compiler so that Numba can understand Awkward Array types, and use them to generate efficient compiled code for execution on GPUs or multicore CPUs. The programming effort required is as simple as adding a function decorator to instruct Numba to compile for the GPU.

```ipython3
import awkward as ak
import numpy as np
import cupy as cp

import numba.cuda
```

Note, CUDA has an execution model unlike the traditional sequential model used for programming CPUs. In CUDA, the code you write will be executed by multiple threads at once (often hundreds or thousands). Your solution will be modeled by defining a thread hierarchy of grid, blocks, and threads.

## Writing CUDA kernels that recognize Awkward Arrays

For the most part, writing a CUDA kernel in Numba that reads Awkward Arrays is like writing a CUDA kernel in Numba generally. See the [Numba documentation](https://numba.pydata.org/numba-doc/latest/cuda/index.html) for the general case.

At the time of writing, Numba’s CUDA backend does not recognize Awkward Arrays until they are explicitly registered. (This may improve in the future.)

```ipython3
ak.numba.register_and_check()
```

The `@numba.cuda.jit` decorator is used to create a CUDA kernel. A kernel function is a GPU function that is meant to be called from CPU code. To understand Awkward Array types the decorator extensions must include an `ak.numba.cuda` object that prepares the `ak.Array` arguments to be passed into Numba’s default argument marshalling logic.

```ipython3
@numba.cuda.jit(extensions=[ak.numba.cuda])
def path_length(out, array):
    tid = numba.cuda.grid(1)
    if tid < len(array):
        out[tid] = 0
        for i, x in enumerate(array[tid]):
            out[tid] += x
```

The kernels cannot explicitly return a value. The result data must be written to an `out` array passed to the function (if computing a scalar, you will probably pass a one-element array).

The kernels explicitly declare their thread hierarchy when called: i.e. the number of thread blocks and the number of threads per block (note that while a kernel is compiled once, it can be called multiple times with different block sizes or grid sizes). The `tid` is the absolute position of the current thread in the entire grid of blocks.

## Awkward Arrays on the GPU

It is a user responsibility to allocate and manage memory, for example, transferring device memory back to the host when a kernel finishes. The `ak.numba.cuda` extention only accepts `ak.Array` with a cuda backend. That way the array data are already on the device and do not need to be copied.

```ipython3
N = 2**20

counts = ak.Array(cp.random.poisson(1.5, N).astype(np.int32))
content = ak.Array(cp.random.normal(0, 45.0, int(ak.sum(counts))).astype(np.float32))
array = ak.unflatten(content, counts)
array
```

For all but the simplest algorithms, it is important that you carefully consider how to use and access memory in order to minimize bandwidth requirements and contention.

Awkward Array can operate on CUDA-device arrays through the `cupy` library.

## Kernel invocation

Numba can use the CUDA array protocol (`__cuda_array_interface__`) to obtain a zero-copy reference to the CuPy array. We can launch a Numba kernel that operates upon our source `array` and target `result` as follows:

```ipython3
blocksize = 256
numblocks = (N + blocksize - 1) // blocksize
```

```ipython3
result = cp.empty(len(array), dtype=np.float32)

path_length[numblocks, blocksize](result, array)

result
```

The calculation on the GPU is much faster than its CPU equivalent:

```ipython3
%%timeit

path_length[numblocks, blocksize](result, array)
```

```myst-ansi
180 µs ± 1.41 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```

```ipython3
cpu_array = ak.to_backend(array, "cpu")
```

```ipython3
%%timeit

ak.sum(cpu_array, axis=-1)
```

```myst-ansi
7.5 ms ± 43.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

But the result is the same.

```ipython3
check_result = ak.sum(cpu_array, axis=-1)
```

```ipython3
ak.all(ak.isclose(check_result, ak.Array(result, backend="cpu"), atol=1e-05))
```
