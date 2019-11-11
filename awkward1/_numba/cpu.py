# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import os
import ctypes
import platform
import glob
import xml.etree.ElementTree

import numba.typing.ctypes_utils

if platform.system() == "Windows":
    libname = "awkward-cpu-kernels.dll"
elif platform.system() == "Darwin":
    libname = "libawkward-cpu-kernels.dylib"
else:
    libname = "libawkward-cpu-kernels.so"

libpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), libname)

kernels = ctypes.cdll.LoadLibrary(libpath)

class ErrorType(numba.types.Type):
    def __init__(self):
        super(ErrorType, self).__init__(name="awkward._numba.cpu.ErrorType")

class Error(ctypes.Structure):
    _fields_ = [("str", ctypes.c_char_p),
                ("location", ctypes.c_int64),
                ("attempt", ctypes.c_int64),
                ("extra", ctypes.c_int64)]
    numbatpe = ErrorType()

@numba.extending.register_model(ErrorType)
class ErrorModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("str", numba.intp),
                   ("location", numba.int64),
                   ("attempt", numba.int64),
                   ("extra", numba.int64)]
        super(ErrorModel, self).__init__(dmm, fe_type, members)

h2ctypes = {
    "bool": ctypes.c_uint8,
    "uint8_t *": ctypes.POINTER(ctypes.c_uint8),
    "const uint8_t *": ctypes.POINTER(ctypes.c_uint8),
    "int32_t": ctypes.c_int32,
    "int32_t *": ctypes.POINTER(ctypes.c_int32),
    "const int32_t *": ctypes.POINTER(ctypes.c_int32),
    "uint32_t": ctypes.c_uint32,
    "uint32_t *": ctypes.POINTER(ctypes.c_uint32),
    "const uint32_t *": ctypes.POINTER(ctypes.c_uint32),
    "int64_t": ctypes.c_int64,
    "int64_t *": ctypes.POINTER(ctypes.c_int64),
    "const int64_t *": ctypes.POINTER(ctypes.c_int64),
    "Error": Error,
    "ERROR": Error,
    "void": None,
    }

for hfile in glob.glob(os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), "signatures"), "*_8cpp.xml")):
    xfile = xml.etree.ElementTree.parse(hfile)
    if xfile.find("./compounddef/location").attrib["file"].startswith("src/cpu-kernels"):
        for xfcn in xfile.findall(".//memberdef[@kind='function']"):
            name = xfcn.find("./name").text
            if hasattr(kernels, name) and name not in ("success", "failure"):
                rettype = xfcn.find("./type").text
                params = [(x.find("./declname").text, x.find("./type").text) for x in xfcn.findall("./param")]
                getattr(kernels, name).name = name
                getattr(kernels, name).argtypes = [h2ctypes[t] for n, t in params]
                if rettype == "Error" or rettype == "ERROR":
                    getattr(kernels, name).restype = None
                    tmp = numba.typing.ctypes_utils.make_function_type(getattr(kernels, name))
                    getattr(kernels, name).numbatpe = numba.types.functions.ExternalFunctionPointer(Error.numbatpe(*tmp.sig.args), tmp.get_pointer, cconv=tmp.cconv)
                    getattr(kernels, name).restype = h2ctypes[rettype]
                else:
                    getattr(kernels, name).restype = h2ctypes[rettype]
                    getattr(kernels, name).numbatpe = numba.typing.ctypes_utils.make_function_type(getattr(kernels, name))
