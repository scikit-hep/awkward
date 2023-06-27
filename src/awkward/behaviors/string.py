# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
import awkward as ak
from awkward._errors import deprecate
from awkward._nplikes.numpylike import NumpyMetadata
from awkward.highlevel import Array

np = NumpyMetadata.instance()


class ByteBehavior(Array):
    __name__ = "Array"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        deprecate(
            f"{type(self).__name__} is deprecated: string-types are now considered a built-in feature "
            f"provided by Awkward Array, rather than an extension.",
            version="2.4.0",
        )


class CharBehavior(Array):
    __name__ = "Array"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        deprecate(
            f"{type(self).__name__} is deprecated: string-types are now considered a built-in feature "
            f"provided by Awkward Array, rather than an extension.",
            version="2.4.0",
        )


class ByteStringBehavior(Array):
    __name__ = "Array"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        deprecate(
            f"{type(self).__name__} is deprecated: string-types are now considered a built-in feature "
            f"provided by Awkward Array, rather than an extension.",
            version="2.4.0",
        )


class StringBehavior(Array):
    __name__ = "Array"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        deprecate(
            f"{type(self).__name__} is deprecated: string-types are now considered a built-in feature "
            f"provided by Awkward Array, rather than an extension.",
            version="2.4.0",
        )


def _string_numba_typer(viewtype):
    import numba

    if viewtype.type.parameters["__array__"] == "string":
        return numba.types.string
    else:
        return numba.cpython.charseq.bytes_type


def _string_numba_lower(
    context, builder, rettype, viewtype, viewval, viewproxy, attype, atval
):
    import llvmlite.ir
    import numba

    whichpos = ak._connect.numba.layout.posat(
        context, builder, viewproxy.pos, viewtype.type.CONTENT
    )
    nextpos = ak._connect.numba.layout.getat(
        context, builder, viewproxy.arrayptrs, whichpos
    )

    whichnextpos = ak._connect.numba.layout.posat(
        context, builder, nextpos, viewtype.type.contenttype.ARRAY
    )

    startspos = ak._connect.numba.layout.posat(
        context, builder, viewproxy.pos, viewtype.type.STARTS
    )
    startsptr = ak._connect.numba.layout.getat(
        context, builder, viewproxy.arrayptrs, startspos
    )
    startsarraypos = builder.add(viewproxy.start, atval)
    start = ak._connect.numba.layout.getat(
        context, builder, startsptr, startsarraypos, viewtype.type.indextype.dtype
    )

    stopspos = ak._connect.numba.layout.posat(
        context, builder, viewproxy.pos, viewtype.type.STOPS
    )
    stopsptr = ak._connect.numba.layout.getat(
        context, builder, viewproxy.arrayptrs, stopspos
    )
    stopsarraypos = builder.add(viewproxy.start, atval)
    stop = ak._connect.numba.layout.getat(
        context, builder, stopsptr, stopsarraypos, viewtype.type.indextype.dtype
    )

    baseptr = ak._connect.numba.layout.getat(
        context, builder, viewproxy.arrayptrs, whichnextpos
    )
    rawptr = builder.add(
        baseptr,
        ak._connect.numba.layout.castint(
            context, builder, viewtype.type.indextype.dtype, numba.intp, start
        ),
    )
    rawptr_cast = builder.inttoptr(
        rawptr,
        llvmlite.ir.PointerType(llvmlite.ir.IntType(numba.intp.bitwidth // 8)),
    )
    strsize = builder.sub(stop, start)
    strsize_cast = ak._connect.numba.layout.castint(
        context, builder, viewtype.type.indextype.dtype, numba.intp, strsize
    )

    pyapi = context.get_python_api(builder)
    gil = pyapi.gil_ensure()

    strptr = builder.bitcast(rawptr_cast, pyapi.cstring)

    if viewtype.type.parameters["__array__"] == "string":
        kind = context.get_constant(numba.types.int32, pyapi.py_unicode_1byte_kind)
        pystr = pyapi.string_from_kind_and_data(kind, strptr, strsize_cast)
    else:
        pystr = pyapi.bytes_from_string_and_size(strptr, strsize_cast)

    out = pyapi.to_native_value(rettype, pystr).value

    pyapi.decref(pystr)

    pyapi.gil_release(gil)

    return out


def _cast_bytes_or_str_to_string(string):
    return ak.to_layout([string])


def register(behavior):
    behavior["__numba_typer__", "bytestring"] = _string_numba_typer
    behavior["__numba_lower__", "bytestring"] = _string_numba_lower
    behavior["__numba_typer__", "string"] = _string_numba_typer
    behavior["__numba_lower__", "string"] = _string_numba_lower

    behavior["__cast__", str] = _cast_bytes_or_str_to_string
    behavior["__cast__", bytes] = _cast_bytes_or_str_to_string
