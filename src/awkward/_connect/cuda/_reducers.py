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
    op.at(toptr_prom, parents, fromptr_prom)

    if orig_dtype != promoted_dtype:
        if reinterpret:
            toptr[:] = toptr_prom.view(orig_dtype)
        else:
            toptr[:] = toptr_prom.astype(orig_dtype, copy=False)
