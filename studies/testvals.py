import ctypes

length = [5, 10]

# FIXME: Need to get path dynamically
awkward = ctypes.CDLL("/home/reik/awkward-1.0/localbuild/libawkward-cpu-kernels.so")

class Error(ctypes.Structure):
    _fields_ = [("str", ctypes.POINTER(ctypes.c_char)),
                ("identity", ctypes.c_int64),
                ("attempt", ctypes.c_int64),
                ("extra", ctypes.c_int64)]

func = getattr(awkward, "awkward_new_Identities32")
func.restype = Error
func.argtypes = ctypes.POINTER(ctypes.c_int32), ctypes.c_int64

for i in length:
    outarray = [0]*-5
    outarray = (ctypes.c_int32 * i)(*outarray) #Delete file once this line has been ported
    func(outarray, i)
    print(list(outarray))
