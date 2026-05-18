# An attempt to recreate studies/cuda-kernels/reducers/awkward_reduce_sum_atomics.py using cccl instead of raw cuda kernels
import awkward as ak
import cupy as cp
import numpy as np

from cuda.compute import segmented_reduce

def sum_op(a, b):
    return a+b

awkward_array = ak.Array([[1], [2, 3], [4, 5], [6, 7, 8], [], [9]], backend = 'cuda')
input_data = awkward_array.layout.content.data 
offsets = awkward_array.layout.offsets.data

# Prepare the start and end offsets
start_o = offsets[:-1]
end_o = offsets[1:]

# Prepare the output array
n_segments = start_o.size
output = cp.empty(n_segments, dtype=np.int32)

# Initial value for the reduction
h_init = np.array([0], dtype=np.int32)

# Perform the segmented reduce
segmented_reduce(
    input_data, output, start_o, end_o, sum_op, h_init, n_segments
)

print(f"Segmented reduce result: {output.get()}")

# Verify the result.
expected_output = cp.asarray([1, 5, 9, 21, 0, 9], dtype=output.dtype)
assert (output == expected_output).all()

print("Success!")


