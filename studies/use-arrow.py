import pyarrow as pa
import pandas as pd
import numpy as np
from ctypes import *

class ArrowSchema(Structure):
    pass

release_func_ptr_schema = CFUNCTYPE(c_void_p, POINTER(ArrowSchema))

ArrowSchema._fields_ = [("format", c_char_p),
                        ("name", c_char_p),
                        ("metadata", c_char_p),
                        ("flags", c_longlong),
                        ("n_children", c_longlong),
                        ("children", POINTER(POINTER(ArrowSchema))),
                        ("dictionary", POINTER(ArrowSchema)),
                        ("release", release_func_ptr_schema),
                        ("private_data", c_void_p)]


class ArrowArray(Structure):
    pass

release_func_ptr_array = CFUNCTYPE(c_void_p, POINTER(ArrowArray))

ArrowArray._fields_ =[("length", c_longlong),
                      ("null_count", c_longlong),
                      ("offset", c_longlong),
                      ("n_buffers", c_longlong),
                      ("n_children", c_longlong),
                      ("buffers", POINTER(c_void_p)),
                      ("children", POINTER(POINTER(ArrowArray))),
                      ("dictionary", POINTER(ArrowArray)),
                      ("release", release_func_ptr_array),
                      ("private_data", c_void_p)]

lib = cdll.LoadLibrary('./libfoo.so')
arr = pa.array([1, 2, 3, 4])
# print(np.frombuffer(arr , dtype=np.uint8))
int32_arr = c_int * 4
arr_ptr = int32_arr(1, 2, 3, 4)
# print(arr_ptr)
scheme = ArrowSchema()
lib.export_int32_type(pointer(scheme))
# print(scheme)
arrow_arr = ArrowArray()
lib.export_int32_array(arr_ptr, 10, pointer(arrow_arr))
print(np.frombuffer(arrow_arr, dtype=np.uint8))