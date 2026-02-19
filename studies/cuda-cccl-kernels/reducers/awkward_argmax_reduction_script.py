import awkward as ak
import cupy as cp
import numpy as np
import time
import nvtx

from cuda.compute import segmented_reduce, ZipIterator, gpu_struct, reduce_into

# An attempt to recreate studies/cuda-kernels/reducers/awkward_reduce_argmax_tree_reduction.py using cccl instead of raw cuda kernels
def cccl_argmax(awkward_array):
    @gpu_struct
    class ak_array:
        data: cp.float64
        local_index: cp.int64
    
    # compare the values of the arrays
    def max_op(a: ak_array, b: ak_array):
        return a if a.data > b.data else b
    
    input_data = awkward_array.layout.content.data
    # use an internal awkward function to get the local indicies
    local_indicies = ak.local_index(awkward_array, axis=1)
    local_indicies = local_indicies.layout.content.data
    
    #Combine data and their indicies into a single structure
    #input_struct = cp.stack((input_data, parents), axis=1).view(ak_array.dtype)
    input_struct = ZipIterator(input_data, local_indicies)
    
    # Prepare the start and end offsets
    offsets = awkward_array.layout.offsets.data
    start_o = offsets[:-1]
    end_o = offsets[1:]
    
    # Prepare the output array
    n_segments = start_o.size
    output = cp.zeros([n_segments], dtype= ak_array.dtype)
    
    # Initial value for the reduction
    h_init = ak_array(-1, -1)
    
    # Perform the segmented reduce
    segmented_reduce(
        input_struct, output, start_o, end_o, max_op, h_init, n_segments
    )
    
    return output

print("Loading the array...")
awkward_array = ak.to_backend(ak.from_parquet("random_listoffset_small.parquet"), 'cuda')
print("The array to reduce from: ", awkward_array)

print("Running the reduction function...")
res = cccl_argmax(awkward_array)
print("Result: ", res)
