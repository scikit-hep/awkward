import awkward as ak

array = ak._v2.from_parquet("/home/swish/Downloads/zlib9-jagged3.parquet", row_groups = range(35))

print(array)

import cupy

a = ak._v2.Array([[1,2,3], [], [4,5]])

a = a.layout

starts = a.starts

stops = a.stops

err_array = ak._v2.contents.ListArray(stops, starts, a.content)

cuda_array_err = err_array.to_backend("cuda")

cuda_array = ak._v2.to_backend(array, "cuda")

print(f"Memory used GPU: {cupy.get_default_memory_pool().used_bytes()}")

cuda_stream_1 =cupy.cuda.Stream(non_blocking=True) 
cuda_stream_2 =cupy.cuda.Stream(non_blocking=True) 
cuda_stream_3 =cupy.cuda.Stream(non_blocking=True) 

with cuda_stream_1:
    a = ak._v2.num(cuda_array, 2)
    a = ak._v2.num(cuda_array, 1)
#print(f"Memory used GPU: {cupy.get_default_memory_pool().used_bytes()}")
with cuda_stream_2:
    cuda_array_err._compact_offsets64(True)
    b = ak._v2.num(cuda_array, 3)
#print(f"Memory used GPU: {cupy.get_default_memory_pool().used_bytes()}")
with cuda_stream_3:
    c = ak._v2.num(cuda_array, 1)
#print(f"Memory used GPU: {cupy.get_default_memory_pool().used_bytes()}")


import awkward._v2._connect.cuda
awkward._v2._connect.cuda.synchronize_cuda(cuda_stream_1)
print(a)
try:
    awkward._v2._connect.cuda.synchronize_cuda(cuda_stream_2)
except ValueError:
    pass
print(b)
awkward._v2._connect.cuda.synchronize_cuda(cuda_stream_3)
print(c)
