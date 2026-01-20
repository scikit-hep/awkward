# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._nplikes.cupy import Cupy

cupy_nplike = Cupy.instance()
cp = cupy_nplike._module

CUPY_UFUNC_AT_PROMOTION = {
    "bool": {"promoted": "uint32", "reinterpret": False},
    "int8": {"promoted": "int32", "reinterpret": False},
    "uint8": {"promoted": "uint32", "reinterpret": False},
    "int16": {"promoted": "int32", "reinterpret": False},
    "uint16": {"promoted": "uint32", "reinterpret": False},
    "int32": {"promoted": "int32", "reinterpret": False},
    "uint32": {"promoted": "uint32", "reinterpret": False},
    "int64": {"promoted": "uint64", "reinterpret": True},
    "uint64": {"promoted": "uint64", "reinterpret": False},
    "float16": {"promoted": "float32", "reinterpret": False},
    "float32": {"promoted": "float32", "reinterpret": False},
    "float64": {"promoted": "float64", "reinterpret": False},
}

_multiply_at_kernel = cp.ElementwiseKernel(
    "S val, raw I parents, raw S fromptr",  # S is input type, T is output type
    "raw T toptr",
    """
    // We must cast the address to 64-bit for atomicCAS
    unsigned long long* address = (unsigned long long*)&toptr[parents[i]];
    unsigned long long old_val = *address;
    unsigned long long assumed;

    do {
        assumed = old_val;
        // Perform math in the promoted type T
        T result = (T)assumed * (T)fromptr[i];
        old_val = atomicCAS(address, assumed, (unsigned long long)result);
    } while (assumed != old_val);
    """,
    "awkward_multiply_at",
)


# FIXME: CuPy: cupy.multiply.at is NotImplementedError
def _multiply_at_cuda(toptr, parents, fromptr):
    # This invokes the element-wise kernel which handles the loop per element
    _multiply_at_kernel(fromptr, parents, fromptr, toptr)


def reduce_with_cupy_at(op, toptr, fromptr, parents, identity):
    import cupy

    orig_dtype = toptr.dtype
    info = CUPY_UFUNC_AT_PROMOTION[orig_dtype.name]
    promoted_dtype = getattr(cupy, info["promoted"])
    reinterpret = info["reinterpret"]

    if reinterpret:
        identity_prom = orig_dtype.type(identity).view(promoted_dtype)
    else:
        identity_prom = promoted_dtype(identity)

    if orig_dtype == promoted_dtype:
        toptr_prom = toptr
        fromptr_prom = fromptr
    else:
        toptr_prom = toptr.astype(promoted_dtype, copy=True)
        fromptr_prom = fromptr.astype(promoted_dtype, copy=False)

    toptr_prom.fill(identity_prom)
    if op == cupy.multiply:
        _multiply_at_cuda(toptr_prom, parents, fromptr_prom)
    else:
        op.at(toptr_prom, parents, fromptr_prom)

    if orig_dtype != promoted_dtype:
        if reinterpret:
            toptr[:] = toptr_prom.view(orig_dtype)
        else:
            toptr[:] = toptr_prom.astype(orig_dtype, copy=False)
