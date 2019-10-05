# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numpy
import numba
import llvmlite.ir.types

from .._numba import cpu

RefType = numba.int64

index64tpe = numba.types.Array(numba.int64, 1, "C")

def cast(context, builder, fromtpe, totpe, val):
    if fromtpe.bitwidth < totpe.bitwidth:
        return builder.zext(val, context.get_value_type(totpe))
    elif fromtpe.bitwidth > totpe.bitwidth:
        return builder.trunc(val, context.get_value_type(totpe))
    else:
        return val

def arrayptr(context, builder, tpe, val):
    return numba.targets.arrayobj.make_array(tpe)(context, builder, val).data

def arraylen(context, builder, tpe, val, totpe=None):
    if isinstance(tpe, numba.types.Array):
        out = numba.targets.arrayobj.array_len(context, builder, numba.intp(tpe), (val,))
    else:
        out = tpe.lower_len(context, builder, numba.intp(tpe), (val,))
    if totpe is None:
        return out
    else:
        return cast(context, builder, numba.intp, totpe, out)

def call(context, builder, fcn, args, errormessage=None):
    fcntpe = context.get_function_pointer_type(fcn.numbatpe)
    fcnval = context.add_dynamic_addr(builder, fcn.numbatpe.get_pointer(fcn), info=fcn.name)
    fcnptr = builder.bitcast(fcnval, fcntpe)

    err = context.call_function_pointer(builder, fcnptr, args)

    if fcn.restype is cpu.Error:
        assert errormessage is not None, "this function can return an error"
        proxyerr = numba.cgutils.create_struct_proxy(cpu.Error.numbatpe)(context, builder, value=err)
        with builder.if_then(builder.icmp_signed("!=", proxyerr.str, context.get_constant(numba.intp, 0)), likely=False):
            context.call_conv.return_user_exc(builder, ValueError, (errormessage,))

            # pyapi = context.get_python_api(builder)
            # exc = pyapi.serialize_object(ValueError(errormessage))
            # excptr = context.call_conv._get_excinfo_argument(builder.function)
            # if excptr.name == "excinfo" and excptr.type == llvmlite.llvmpy.core.Type.pointer(llvmlite.llvmpy.core.Type.pointer(llvmlite.llvmpy.core.Type.struct([llvmlite.llvmpy.core.Type.pointer(llvmlite.llvmpy.core.Type.int(8)), llvmlite.llvmpy.core.Type.int(32)]))):
            #     builder.store(exc, excptr)
            #     builder.ret(numba.targets.callconv.RETCODE_USEREXC)
            # elif excptr.name == "py_args" and excptr.type == llvmlite.llvmpy.core.Type.pointer(llvmlite.llvmpy.core.Type.int(8)):
            #     pyapi.raise_object(exc)
            #     builder.ret(llvmlite.llvmpy.core.Constant.null(context.get_value_type(numba.types.pyobject)))
            # else:
            #     raise AssertionError("unrecognized exception calling convention: {}".format(excptr))

def newindex64(context, builder, lentpe, lenval):
    return numba.targets.arrayobj.numpy_empty_nd(context, builder, index64tpe(lentpe), (lenval,))

@numba.jit(nopython=True)
def shapeat(shapeat, array, at, ndim):
    redat = at - (ndim - array.ndim)
    if redat < 0:
        return 1
    elif shapeat == 1:
        return array.shape[redat]
    elif shapeat == array.shape[redat] or array.shape[redat] == 1:
        return shapeat
    else:
        raise ValueError("cannot broadcast arrays to the same shape")

@numba.generated_jit(nopython=True)
def broadcast_to(array, shape):
    if isinstance(array, numba.types.Array):
        def impl(array, shape):
            out = numpy.empty(shape, array.dtype)
            out[:] = array
            return out
        return impl
    elif isinstance(array, numba.types.Integer):
        def impl(array, shape):
            return numpy.full(shape, array, numpy.int64)
        return impl
    else:
        return lambda array, shape: array

@numba.generated_jit(nopython=True)
def broadcast_arrays(arrays):
    if not isinstance(arrays, numba.types.BaseTuple) or not any(isinstance(x, numba.types.Array) for x in arrays.types):
        return lambda arrays: arrays

    else:
        ndim = max(t.ndim if isinstance(t, numba.types.Array) else 1 for t in arrays.types)
        def getshape(i, at):
            if i == -1:
                return "1"
            elif isinstance(arrays.types[i], numba.types.Array):
                return "shapeat({}, arrays[{}], {}, {})".format(getshape(i - 1, at), i, at, ndim)
            else:
                return getshape(i - 1, at)
        g = {"shapeat": shapeat, "broadcast_to": broadcast_to}
        exec("""
def impl(arrays):
    shape = ({})
    return ({})
""".format(" ".join(getshape(len(arrays.types) - 1, at) + "," for at in range(ndim)),
           " ".join("broadcast_to(arrays[{}], shape),".format(at) if isinstance(arrays.types[at], (numba.types.Array, numba.types.Integer)) else "arrays[{}],".format(at) for at in range(len(arrays.types)))), g)
        return g["impl"]

def typing_broadcast_arrays(arrays):
    if not isinstance(arrays, numba.types.BaseTuple) or not any(isinstance(x, numba.types.Array) for x in arrays.types):
        return arrays
    else:
        return numba.types.Tuple([numba.types.Array(numba.int64, 1, "C") if isinstance(t, numba.types.Integer) else t for t in arrays.types])

@numba.generated_jit(nopython=True)
def regularize_slice(arrays):
    if not isinstance(arrays, numba.types.BaseTuple) and isinstance(arrays, (numba.types.ArrayCompatible, numba.types.Sequence)) and isinstance(arrays.dtype, numba.types.Boolean):
        return lambda arrays: numpy.nonzero(arrays)

    elif not isinstance(arrays, numba.types.BaseTuple) or not any(isinstance(t, (numba.types.ArrayCompatible, numba.types.Sequence)) for t in arrays.types):
        return lambda arrays: arrays

    else:
        code = "def impl(arrays):\n"
        indexes = []
        for i, t in enumerate(arrays.types):
            if isinstance(t, (numba.types.ArrayCompatible, numba.types.Sequence)) and isinstance(t.dtype, numba.types.Boolean):
                code += "    x{} = numpy.nonzero(arrays[{}])\n".format(i, i)
                indexes.extend(["x{}[{}],".format(i, j) for j in range(arrays.types[i].ndim)])
            elif isinstance(t, (numba.types.ArrayCompatible, numba.types.Sequence)) and isinstance(t.dtype, numba.types.Integer):
                indexes.append("numpy.asarray(arrays[{}], numpy.int64),".format(i))
            elif isinstance(t, (numba.types.ArrayCompatible, numba.types.Sequence)):
                raise TypeError("arrays must have boolean or integer type")
            else:
                indexes.append("arrays[{}]".format(i))
        code += "    return ({})".format(" ".join(indexes))
        g = {"numpy": numpy}
        exec(code, g)
        return g["impl"]

def typing_regularize_slice(arrays):
    out = ()
    if not isinstance(arrays, numba.types.BaseTuple) and isinstance(arrays, (numba.types.ArrayCompatible, numba.types.Sequence)) and isinstance(arrays.dtype, numba.types.Boolean):
        return numba.types.Tuple(arrays.ndims*(numba.types.Array(numba.int64, 1, "C"),))

    elif not isinstance(arrays, numba.types.BaseTuple) or not any(isinstance(t, (numba.types.ArrayCompatible, numba.types.Sequence)) for t in arrays.types):
        return arrays

    else:
        for t in arrays.types:
            if isinstance(t, (numba.types.ArrayCompatible, numba.types.Sequence)) and isinstance(t.dtype, numba.types.Boolean):
                out = out + t.ndims*(numba.types.Array(numba.int64, 1, "C"),)
            elif isinstance(t, (numba.types.ArrayCompatible, numba.types.Sequence)) and isinstance(t.dtype, numba.types.Integer):
                out = out + (numba.types.Array(numba.int64, 1, "C"),)
            elif isinstance(t, (numba.types.ArrayCompatible, numba.types.Sequence)):
                raise TypeError("arrays must have boolean or integer type")
            else:
                out = out + (t,)
        return numba.types.Tuple(out)

def preprocess_slicetuple(context, builder, wheretpe1, whereval1):
    wheretpe2 = typing_regularize_slice(wheretpe1)
    regularize_slice.compile(wheretpe2(wheretpe1))
    cres = regularize_slice.overloads[(wheretpe1,)]
    whereval2 = context.call_internal(builder, cres.fndesc, wheretpe2(wheretpe1), (whereval1,))

    wheretpe3 = typing_broadcast_arrays(wheretpe2)
    broadcast_arrays.compile(wheretpe3(wheretpe2))
    cres2 = broadcast_arrays.overloads[(wheretpe2,)]
    whereval3 = context.call_internal(builder, cres2.fndesc, wheretpe3(wheretpe2), (whereval2,))

    return wheretpe3, whereval3
