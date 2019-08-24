# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import os
import ctypes
import platform

if platform.system() == "Windows":
    libname = "awkward-cpu-kernels.dll"
elif platform.system() == "Darwin":
    libname = "libawkward-cpu-kernels.dylib"
else:
    libname = "libawkward-cpu-kernels.so"

libpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), libname)

kernels = ctypes.cdll.LoadLibrary(libpath)

# builder.call(fcnptr, args)

# builder <class 'llvmlite.llvmpy.core.Builder'> <llvmlite.llvmpy.core.Builder object at 0x7a92c91cc748>

# funcptr <class 'llvmlite.ir.instructions.LoadInstr'> <ir.LoadInstr '.14' of type 'i32 (i32)*', opname 'load', operands [<ir.AllocaInstr '$0.1' of type 'i32 (i32)**', opname 'alloca', operands ()>]>

# args 1 [<class 'llvmlite.ir.instructions.LoadInstr'>] [<ir.LoadInstr '.13' of type 'i32', opname 'load', operands [<ir.AllocaInstr 'x' of type 'i32*', opname 'alloca', operands ()>]>]

# cconv <class 'NoneType'> None
