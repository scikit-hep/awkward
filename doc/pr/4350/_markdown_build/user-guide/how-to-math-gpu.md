# Awkward Arrays on GPUs

At the time when this article is being written, Awkward Arrays can be used on Nvidia GPUs on Linux, assuming that you have the [CuPy](https://cupy.dev/) package installed. In the future, we may support more GPU vendors on more platforms, so check [Awkward Array on GitHub](https://github.com/scikit-hep/awkward) for more up-to-date information.

```python
import awkward as ak
import numpy as np
import cupy as cp
```

## Copying data from RAM to a GPU

An Awkward Array might either reside in main memory (RAM), to be processed by the CPU, or in a GPU’s global memory, to be processed by a GPU. Arrays can be copied between devices using the [`ak.to_backend()`](sphinx-llm:fb9bd58d1c894eed98bb17af16147d71#ak.to_backend) function, and their device can be checked with [`ak.backend()`](sphinx-llm:a8778e70cfb34139919a76f6ff6747f7#ak.backend).

```python
array_cpu = ak.Array(
    [[0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]] * 10000
)
ak.backend(array_cpu)
```

```default
'cpu'
```

```python
array_gpu = ak.to_backend(array_cpu, "cuda")
ak.backend(array_gpu)
```

```default
'cuda'
```

The backend names, `"cpu"` and `"cuda"`, refer to the software that performs the calculations, which is written in conventional C code for the CPU and CUDA for (Nvidia) GPUs. By passing `"cpu"` to [`ak.to_backend()`](sphinx-llm:fb9bd58d1c894eed98bb17af16147d71#ak.to_backend), you can copy an array from the GPU back to main memory.

Arrays are never copied without an explicit [`ak.to_backend()`](sphinx-llm:fb9bd58d1c894eed98bb17af16147d71#ak.to_backend) call, so if you pass arrays from different devices to the same function, it will raise an error.

## Array calculations on a GPU

All of the `ak.*` functions (excluding `ak.str.*`), slices, and NumPy functions that work on Awkward Arrays on CPUs will work on Awkward Arrays on GPUs. At the time of writing, the implementation is nearing completion, so check [Awkward Array on GitHub](https://github.com/scikit-hep/awkward) if a function doesn’t work.

Here’s an example, using [`ak.num()`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num):

```python
ak.num(array_gpu)
```

```default
[3,
 0,
 2,
 1,
 4,
 3,
 0,
 2,
 1,
 4,
 ...,
 0,
 2,
 1,
 4,
 3,
 0,
 2,
 1,
 4]
-------------------
type: 50000 * int64
```

and here is a slice:

```python
array_gpu[100:]
```

```default
[[0.0, 1.1, 2.2],
 [],
 [3.3, 4.4],
 [5.5],
 [6.6, 7.7, 8.8, 9.9],
 [0.0, 1.1, 2.2],
 [],
 [3.3, 4.4],
 [5.5],
 [6.6, 7.7, 8.8, 9.9],
 ...,
 [],
 [3.3, 4.4],
 [5.5],
 [6.6, 7.7, 8.8, 9.9],
 [0.0, 1.1, 2.2],
 [],
 [3.3, 4.4],
 [5.5],
 [6.6, 7.7, 8.8, 9.9]]
---------------------------
type: 49900 * var * float64
```

All [NumPy universal functions (ufuncs)](https://numpy.org/doc/stable/reference/ufuncs.html) *for which there is a CuPy equivalent* also work:

```python
np.sqrt(array_gpu)
```

```default
[[0.0, 1.0488088481701516, 1.4832396974191326],
 [],
 [1.816590212458495, 2.0976176963403033],
 [2.345207879911715],
 [2.569046515733026, 2.7748873851023217, ..., 3.146426544510455],
 [0.0, 1.0488088481701516, 1.4832396974191326],
 [],
 [1.816590212458495, 2.0976176963403033],
 [2.345207879911715],
 [2.569046515733026, 2.7748873851023217, ..., 3.146426544510455],
 ...,
 [],
 [1.816590212458495, 2.0976176963403033],
 [2.345207879911715],
 [2.569046515733026, 2.7748873851023217, ..., 3.146426544510455],
 [0.0, 1.0488088481701516, 1.4832396974191326],
 [],
 [1.816590212458495, 2.0976176963403033],
 [2.345207879911715],
 [2.569046515733026, 2.7748873851023217, ..., 3.146426544510455]]
-----------------------------------------------------------------
type: 50000 * var * float64
```

## JIT-compilation in Numba

Just as Awkward Arrays in main memory can be iterated over in functions that have been JIT-compiled by [Numba](https://numba.pydata.org/), Awkward Arrays on GPUs can be iterated over in functions JIT-compiled by `@numba.cuda.jit`. The same restrictions apply (iteration only; no `ak.*` functions); see how-to-use-in-numba-cuda.md.
