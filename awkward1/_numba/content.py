# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba

from .._numba import cpu, identity

class ContentType(numba.types.Type):
    pass

@numba.typing.templates.infer_global(len)
class type_len(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            arraytpe, = args
            if isinstance(arraytpe, ContentType):
                return numba.typing.templates.signature(numba.types.intp, arraytpe)

@numba.typing.templates.infer_global(operator.getitem)
class type_getitem(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            arraytpe, wheretpe = args
            if isinstance(arraytpe, ContentType):
                original_wheretpe = wheretpe
                if not isinstance(wheretpe, numba.types.BaseTuple):
                    wheretpe = numba.types.Tuple((wheretpe,))
                if len(wheretpe.types) == 0:
                    return arraytpe
                if any(isinstance(x, numba.types.Array) and x.ndim == 1 for x in wheretpe.types):
                    wheretpe = numba.types.Tuple(tuple(numba.types.Array(x, 1, "C") if isinstance(x, numba.types.Integer) else x for x in wheretpe))
                return numba.typing.templates.signature(arraytpe.getitem(wheretpe), arraytpe, original_wheretpe)

@numba.generated_jit(nopython=True)
def _shapeat(shapeat, array, at, ndim):
    if isinstance(array, numba.types.Array):
        def impl(shapeat, array, at, ndim):
            redat = at - (ndim - array.ndim)
            if at < ndim - array.ndim:
                return 1
            elif shapeat == 1:
                return array.shape[redat]
            elif shapeat == array.shape[redat] or array.shape[redat] == 1:
                return shapeat
            else:
                raise ValueError("cannot broadcast arrays to the same shape")
        return impl
    else:
        return lambda shapeat, array, at, ndim: shapeat

@numba.generated_jit(nopython=True)
def broadcast_to(array, shape):
    if isinstance(array, numba.types.Array):
        def impl(array, shape):
            out = numpy.empty(shape, array.dtype)
            out[:] = array
            return out
        return impl
    elif isinstance(array, numba.types.Number):
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
            if i == 0:
                return "_shapeat(1, arrays[{}], {}, {})".format(i, at, ndim)
            else:
                return "_shapeat({}, arrays[{}], {}, {})".format(getshape(i - 1, at), i, at, ndim)
        g = {"_shapeat": _shapeat, "broadcast_to": broadcast_to}
        exec("""
def impl(arrays):
    shape = ({})
    return ({})
""".format(" ".join(getshape(len(arrays.types) - 1, at) + "," for at in range(ndim)),
           " ".join("broadcast_to(arrays[{}], shape),".format(at) for at in range(len(arrays.types)))), g)
        return g["impl"]
