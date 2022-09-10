import awkward as ak

array = ak.from_parquet("/home/swish/Downloads/zlib9-jagged3.parquet", row_groups = range(25))

print(array)

import cupy

cuda_array = ak.to_backend(array, "cuda")

print(f"Memory used GPU: {cupy.get_default_memory_pool().used_bytes()}")

cuda_stream_1 =cupy.cuda.Stream(non_blocking=True) 
cuda_stream_2 =cupy.cuda.Stream(non_blocking=True) 
cuda_stream_3 =cupy.cuda.Stream(non_blocking=True) 

with cuda_stream_1:
    for i in range(10):
        a = ak.num(cuda_array, 2)
        a = ak.num(cuda_array, 1)

with cuda_stream_2:
    for i in range(10):
        b = ak.num(cuda_array, 3)

with cuda_stream_3:
    for i in range(10):
        c = ak.num(cuda_array, 1)


import awkward._connect.cuda

awkward._connect.cuda.synchronize_cuda(cuda_stream_1)
print(a)

awkward._connect.cuda.synchronize_cuda(cuda_stream_2)
print(b)

awkward._connect.cuda.synchronize_cuda(cuda_stream_3)
print(c)
