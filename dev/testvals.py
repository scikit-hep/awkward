import ctypes

length = [5, 10]

# FIXME: Need to get path dynamically
awkward = ctypes.CDLL("/home/reik/awkward-1.0/localbuild/libawkward-cpu-kernels.so")

class Error(ctypes.Structure):
    _fields_ = [("str", ctypes.POINTER(ctypes.c_char)),
                ("identity", ctypes.c_int64),
                ("attempt", ctypes.c_int64),
                ("extra", ctypes.c_int64)]

awkward.awkward_new_Identities32.restype = Error
awkward.awkward_new_Identities32.argtypes = ctypes.POINTER(ctypes.c_int32), ctypes.c_int64

for i in length:
    outarray = [0]*i
    outarray = (ctypes.c_int32 * i)(*outarray)
    awkward.awkward_new_Identities32(outarray, i)
    print(list(outarray))
