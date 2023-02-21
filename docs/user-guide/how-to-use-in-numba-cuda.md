---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

How to use Awkward Arrays in Numba's CUDA target
================================================

Awkward Array defines extentions to the Numba compiler so that Numba can understand Awkward Array types, and use them to generate efficient compiled code for execution on GPUs or multicore CPUs. The programming effort required is as simple as adding a function decorator to instruct Numba to compile for the GPU.

```{code-cell} ipython3
import awkward as ak
import numba
from numba import cuda
```

The Numba entry point registration happens too late for the Awkward CUDA extention, that is why we need to register it manually:   
```{code-cell} ipython3
ak.numba.register_and_check()
```

Note, CUDA has an execution model unlike the traditional sequential model used for programming CPUs. In CUDA, the code you write will be executed by multiple threads at once (often hundreds or thousands). Your solution will be modeled by defining a thread hierarchy of grid, blocks, and threads. Here we'll disable low-occupancy and implicit-copy warnings for this guide examples:

```{code-cell} ipython3
from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = False
config.CUDA_WARN_ON_IMPLICIT_COPY = False
```

Writing CUDA kernels that understand Awkward Array types 
--------------------------------------------------------

The `@cuda.jit` decorator is used to create a CUDA kernel. A kernel function is a GPU function that is meant to be called from CPU code. To understand Awkward Array types the decorator extensions must include an `ak.numba.cuda` object that prepares the `ak.Array` arguments to be passed into Numba’s default argument marshalling logic.

```{code-cell} ipython3
@cuda.jit(extensions=[ak.numba.cuda])
def multiply(array, n, out):
    tid = cuda.grid(1)
    out[tid] = array[tid] * n
```
The kernels cannot explicitly return a value. The result data must be written to an `out` array passed to the function (if computing a scalar, you will probably pass a one-element array).

The kernels explicitly declare their thread hierarchy when called: i.e. the number of thread blocks and the number of threads per block (note that while a kernel is compiled once, it can be called multiple times with different block sizes or grid sizes). The `tid` is the absolute position of the current thread in the entire grid of blocks.

Memory management
-----------------

It is a user responsibility to allocate and manage memory, for example, transferring device memory back to the host when a kernel finishes. The `ak.numba.cuda` extention only accepts `ak.Array` with a cuda backend. That way the array data are already on the device and do not need to be copied.

```{code-cell} ipython3
array = ak.Array([0, 1, 2, 3], backend="cuda")
```

For all but the simplest algorithms, it is important that you carefully consider how to use and access memory in order to minimize bandwidth requirements and contention.

Allocate the result:
```{code-cell} ipython3
result = nb_cuda.to_device(np.empty(4, dtype=np.int32))
```

Kernel invocation
-----------------

Launch a kernel as follows:

```{code-cell} ipython3
multiply[1, 4](array, 3, result)
```