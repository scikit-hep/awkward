# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

numba = pytest.importorskip("numba")

from numba import config, cuda, types  # noqa: F401, E402
from numba.core.typing.typeof import typeof, typeof_impl  # noqa: F401, E402

config.CUDA_LOW_OCCUPANCY_WARNINGS = False
config.CUDA_WARN_ON_IMPLICIT_COPY = False
# Numba extension for cuRAND functions. Presently only implements:
#
# - curand_init()
# - curand()
#
# This is enough for a proof-of-concept embedded calls to cuRAND functions in
# Numba kernels.

from pathlib import Path

import numpy as np
from numba import cuda, types
from numba.core.extending import models, register_model, typeof_impl

path_to_shim = Path(__file__).parent / "shim.cu"

# cuRAND state type as a NumPy dtype - this mirrors the state defined in
# curand_kernel.h. Can be used to inspect the state through the device array
# held by CurandStates.

state_fields = [
    ("d", np.int32),
    ("v", np.int32, 5),
    ("boxmuller_flag", np.int32),
    ("boxmuller_flag_double", np.int32),
    ("boxmuller_extra", np.float32),
    ("boxmuller_extra_double", np.float64),
]

curandState = np.dtype(state_fields, align=True)


# Hold an array of cuRAND states - somewhat analagous to a curandState* in
# C/C++.


class CurandStates:
    def __init__(self, n):
        self._array = cuda.device_array(n, dtype=curandState)

    @property
    def data(self):
        return self._array.__cuda_array_interface__["data"][0]


# Numba typing for cuRAND state.


class CurandState(types.Type):
    def __init__(self):
        super().__init__(name="CurandState")


curand_state = CurandState()


class CurandStatePointer(types.Type):
    def __init__(self):
        self.dtype = curand_state
        super().__init__(name="CurandState*")


curand_state_pointer = CurandStatePointer()


@typeof_impl.register(CurandStates)
def typeof_curand_states(val, c):
    return curand_state_pointer


# The CurandState model mirrors the C/C++ structure, and the state pointer
# represented similarly to other pointers.


@register_model(CurandState)
class curand_state_model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("d", types.int32),
            ("v", types.UniTuple(types.int32, 5)),
            ("boxmuller_flag", types.int32),
            ("boxmuller_flag_double", types.int32),
            ("boxmuller_extra", types.float32),
            ("boxmuller_extra_double", types.float64),
        ]
        super().__init__(dmm, fe_type, members)


register_model(CurandStatePointer)(models.PointerModel)


# Numba forward declarations of cuRAND functions. These call shim functions
# prepended with _numba, that simply forward arguments to the named cuRAND
# function.

curand_init_sig = types.void(
    types.uint64, types.uint64, types.uint64, curand_state_pointer, types.uint64
)

curand_init = cuda.declare_device("_numba_curand_init", curand_init_sig)
curand = cuda.declare_device(
    "_numba_curand", types.uint32(curand_state_pointer, types.uint64)
)


# Argument handling. When a CurandStatePointer is passed into a kernel, we
# really only need to pass the pointer to the data, not the whole underlying
# array structure. Our handler here transforms these arguments into a uint64
# holding the pointer.


class CurandStateArgHandler:
    def prepare_args(self, ty, val, **kwargs):
        if isinstance(val, CurandStates):
            assert ty == curand_state_pointer
            return types.uint64, val.data
        else:
            return ty, val


curand_state_arg_handler = CurandStateArgHandler()
# Demonstration of calling cuRAND functions from Numba kernels. Shim functions
# in a .cu file are used to access the cuRAND functions from Numba. This is
# based on the cuRAND device API example in:
# https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
#
# The result produced by this example agrees with the documentation example.
# E.g. on a particular configuration:
#
# Output from this example:                  0.4999931156635
# Output from cuRAND documentation example:  0.4999931156635
#
# Note that this example requires an installation of the CUDA toolkit 11.0 or
# later, because NVRTC will use the include files from the installed CUDA
# toolkit.

import sys

try:
    from cuda import cuda as cuda_driver  # noqa: F401
    from numba import config

    config.CUDA_USE_NVIDIA_BINDING = True
except ImportError:
    print(
        "This example requires the NVIDIA CUDA Python Bindings. "
        "Please see https://nvidia.github.io/cuda-python/install.html for "
        "installation instructions."
    )
    sys.exit(1)

# from numba_curand import (curand_init, curand, curand_state_arg_handler,
#                          CurandStates)
import numpy as np
from numba import cuda

# Various parameters

threads = 64
blocks = 64
nthreads = blocks * threads

sample_count = 10000
repetitions = 50


# State initialization kernel


@cuda.jit(link=[path_to_shim], extensions=[curand_state_arg_handler], target="cuda")
def setup(states):
    i = cuda.grid(1)
    curand_init(1234, i, 0, states, i)


# Random sampling kernel - computes the fraction of numbers with low bits set
# from a random distribution.


@cuda.jit(link=[path_to_shim], extensions=[curand_state_arg_handler], target="cuda")
def count_low_bits_native(states, sample_count, results):
    i = cuda.grid(1)
    count = 0

    # Copy state to local memory
    # XXX: TBC

    # Generate pseudo-random numbers
    for _sample in range(sample_count):
        x = curand(states, i)

        # Check if low bit set
        if x & 1:
            count += 1

    # Copy state back to global memory
    # XXX: TBC

    # Store results
    results[i] += count


# "NotImplementedError: Linking CUDA source files is not supported with the ctypes binding."
def test_curand_example():

    # Create state on the device. The CUDA Array Interface provides a convenient
    # way to get the pointer needed for the shim functions.

    # Initialise cuRAND state

    states = CurandStates(nthreads)
    setup[blocks, threads](states)

    # Run random sampling kernel

    results = cuda.to_device(np.zeros(nthreads, dtype=np.int32))

    for _i in range(repetitions):
        count_low_bits_native[blocks, threads](states, sample_count, results)

    # Collect the results and summarize them. This could have been done on
    # device, but the corresponding CUDA C++ sample does it on the host, and
    # we're following that example.

    host_results = results.copy_to_host()

    total = 0
    for i in range(nthreads):
        total += host_results[i]

    # Use float32 to show an exact match between this and the cuRAND
    # documentation example
    fraction = np.float32(total) / np.float32(nthreads * sample_count * repetitions)

    print(f"Fraction with low bit set was {fraction:17.13f}")
