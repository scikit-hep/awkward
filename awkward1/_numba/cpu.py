# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import os
import ctypes
import platform
import glob
import xml.etree.ElementTree

if platform.system() == "Windows":
    libname = "awkward-cpu-kernels.dll"
elif platform.system() == "Darwin":
    libname = "libawkward-cpu-kernels.dylib"
else:
    libname = "libawkward-cpu-kernels.so"

libpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), libname)

kernels = ctypes.cdll.LoadLibrary(libpath)

h2cskip = [
    "T *",
    "const T *",
    "ID *",
    "const ID *",
    "C *",
    "const C *",
    ]

h2ctypes = {
    "bool": ctypes.c_uint8,
    "uint8_t *": ctypes.POINTER(ctypes.c_uint8),
    "const uint8_t *": ctypes.POINTER(ctypes.c_uint8),
    "int32_t": ctypes.c_int32,
    "int32_t *": ctypes.POINTER(ctypes.c_int32),
    "const int32_t *": ctypes.POINTER(ctypes.c_int32),
    "int64_t": ctypes.c_int64,
    "int64_t *": ctypes.POINTER(ctypes.c_int64),
    "const int64_t *": ctypes.POINTER(ctypes.c_int64),
    "Error": ctypes.c_char_p,
    "void": None,
    }

for hfile in glob.glob(os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), "signatures"), "*_8cpp.xml")):
    xfile = xml.etree.ElementTree.parse(hfile)
    if xfile.find("./compounddef/location").attrib["file"].startswith("src/cpu-kernels"):
        for xfcn in xfile.findall(".//memberdef[@kind='function']"):
            name = xfcn.find("./name").text
            if hasattr(kernels, name):
                rettype = xfcn.find("./type").text
                params = [(x.find("./declname").text, x.find("./type").text) for x in xfcn.findall("./param")]
                getattr(kernels, name).argtypes = [h2ctypes[t] for n, t in params]
                getattr(kernels, name).restype = h2ctypes[rettype]

# builder.call(fcnptr, args)
############################
# builder <class 'llvmlite.llvmpy.core.Builder'> <llvmlite.llvmpy.core.Builder object at 0x7a92c91cc748>
#
# funcptr <class 'llvmlite.ir.instructions.LoadInstr'> <ir.LoadInstr '.14' of type 'i32 (i32)*', opname 'load', operands [<ir.AllocaInstr '$0.1' of type 'i32 (i32)**', opname 'alloca', operands ()>]>
#
# args 1 [<class 'llvmlite.ir.instructions.LoadInstr'>] [<ir.LoadInstr '.13' of type 'i32', opname 'load', operands [<ir.AllocaInstr 'x' of type 'i32*', opname 'alloca', operands ()>]>]
#
# cconv <class 'NoneType'> None

# constant_function_pointer(context, builder, ty, pyval)
########################################################
# @lower_constant(types.ExternalFunctionPointer)
# def constant_function_pointer(context, builder, ty, pyval):
#     ptrty = context.get_function_pointer_type(ty)
#     ptrval = context.add_dynamic_addr(builder, ty.get_pointer(pyval),
#                                       info=str(pyval))
#     return builder.bitcast(ptrval, ptrty)
#
# ty <class 'numba.types.functions.ExternalFunctionPointer'> ExternalFunctionPointer((int32,) -> int32)
#
# pyval <class 'ctypes.CDLL.__init__.<locals>._FuncPtr'> <_FuncPtr object at 0x7f12b501f750>
#
# ptrty <class 'llvmlite.ir.types.PointerType'> <<class 'llvmlite.ir.types.PointerType'> i32 (i32)*>
#
# ty.get_pointer(pyval) <class 'int'> 139718322045248
# str(pyval) '<_FuncPtr object at 0x7f12b501f750>'
#
# ptrval <class 'llvmlite.ir.instructions.LoadInstr'> <ir.LoadInstr '.8' of type 'i8*', opname 'load', operands [<ir.GlobalVariable 'numba.dynamic.globals.7f12b4f44140' of type 'i8**'>]>
#
# builder.bitcast(ptrval, ptrty) <class 'llvmlite.ir.instructions.CastInstr'> <ir.CastInstr '.9' of type 'i32 (i32)*', opname 'bitcast', operands [<ir.LoadInstr '.8' of type 'i8*', opname 'load', operands [<ir.GlobalVariable 'numba.dynamic.globals.7f12b4f44140' of type 'i8**'>]>]>
