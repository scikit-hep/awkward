# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numpy
import numba

RefType = numba.int64

@numba.jit(nopython=True)
def _shapeat(shapeat, array, at, ndim):
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
            return numpy.full(shape, array)
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
                return "_shapeat({}, arrays[{}], {}, {})".format(getshape(i - 1, at), i, at, ndim)
            else:
                return getshape(i - 1, at)
        g = {"_shapeat": _shapeat, "broadcast_to": broadcast_to}
        exec("""
def impl(arrays):
    shape = ({})
    return ({})
""".format(" ".join(getshape(len(arrays.types) - 1, at) + "," for at in range(ndim)),
           " ".join("broadcast_to(arrays[{}], shape),".format(at) if isinstance(arrays.types[at], (numba.types.Array, numba.types.Integer)) else "arrays[{}],".format(at) for at in range(len(arrays.types)))), g)
        return g["impl"]

@numba.generated_jit(nopython=True)
def maskarrays_to_indexarrays(arrays):
    if not isinstance(arrays, numba.types.BaseTuple) and isinstance(arrays, numba.types.Array) and isinstance(arrays.dtype, numba.types.scalars.Boolean):
        return lambda arrays: numpy.nonzero(arrays)

    elif not isinstance(arrays, numba.types.BaseTuple) or not any(isinstance(t, numba.types.Array) and isinstance(t.dtype, numba.types.scalars.Boolean) for t in arrays.types):
        return lambda arrays: arrays

    else:
        code = "def impl(arrays):\n"
        indexes = []
        for i, t in enumerate(arrays.types):
            if isinstance(t, numba.types.Array) and isinstance(t.dtype, numba.types.scalars.Boolean):
                code += "    x{} = numpy.nonzero(arrays[{}])\n".format(i, i)
                indexes.extend(["x{}[{}],".format(i, j) for j in range(arrays.types[i].ndim)])
            else:
                indexes.append("arrays[{}],".format(i))
        code += "    return ({})".format(" ".join(indexes))
        print(code)
        g = {"numpy": numpy}
        exec(code, g)
        return g["impl"]
