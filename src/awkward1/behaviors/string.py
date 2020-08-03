# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import awkward1.highlevel
import awkward1.operations.convert


class ByteBehavior(awkward1.highlevel.Array):
    __name__ = "Array"

    def __bytes__(self):
        tmp = numpy.asarray(self.layout)
        if hasattr(tmp, "tobytes"):
            return tmp.tobytes()
        else:
            return tmp.tostring()

    def __str__(self):
        return str(self.__bytes__())

    def __repr__(self):
        return repr(self.__bytes__())

    def __iter__(self):
        for x in self.__bytes__():
            yield x


class CharBehavior(awkward1.highlevel.Array):
    __name__ = "Array"

    def __bytes__(self):
        tmp = numpy.asarray(self.layout)
        if hasattr(tmp, "tobytes"):
            return tmp.tobytes()
        else:
            return tmp.tostring()

    def __str__(self):
        return self.__bytes__().decode("utf-8", "surrogateescape")

    def __repr__(self):
        return repr(self.__bytes__().decode("utf-8", "surrogateescape"))

    def __iter__(self):
        for x in self.__str__():
            yield x


awkward1.behavior["byte"] = ByteBehavior
awkward1.behavior["__typestr__", "byte"] = "byte"
awkward1.behavior["char"] = CharBehavior
awkward1.behavior["__typestr__", "char"] = "char"


class ByteStringBehavior(awkward1.highlevel.Array):
    __name__ = "Array"

    def __iter__(self):
        for x in super(ByteStringBehavior, self).__iter__():
            yield x.__bytes__()


class StringBehavior(awkward1.highlevel.Array):
    __name__ = "Array"

    def __iter__(self):
        for x in super(StringBehavior, self).__iter__():
            yield x.__str__()


awkward1.behavior["bytestring"] = ByteStringBehavior
awkward1.behavior["__typestr__", "bytestring"] = "bytes"
awkward1.behavior["string"] = StringBehavior
awkward1.behavior["__typestr__", "string"] = "string"


def _string_equal(one, two):
    one, two = one.layout, two.layout

    # first condition: string lengths must be the same
    counts1 = numpy.asarray(one.count(axis=-1))
    counts2 = numpy.asarray(two.count(axis=-1))

    out = counts1 == counts2

    # only compare characters in strings that are possibly equal (same length)
    possible = numpy.logical_and(out, counts1)
    possible_counts = counts1[possible]

    if len(possible_counts) > 0:
        onepossible = awkward1.without_parameters(one[possible])
        twopossible = awkward1.without_parameters(two[possible])

        reduced = awkward1.all(onepossible == twopossible, axis=-1).layout

        # update same-length strings with a verdict about their characters
        out[possible] = reduced

    return awkward1.highlevel.Array(awkward1.layout.NumpyArray(out))


awkward1.behavior[numpy.equal, "bytestring", "bytestring"] = _string_equal
awkward1.behavior[numpy.equal, "string", "string"] = _string_equal


def _string_broadcast(layout, offsets):
    offsets = numpy.asarray(offsets)
    counts = offsets[1:] - offsets[:-1]
    if awkward1._util.win:
        counts = counts.astype(numpy.int32)
    parents = numpy.repeat(numpy.arange(len(counts), dtype=counts.dtype), counts)
    return awkward1.layout.IndexedArray64(
        awkward1.layout.Index64(parents), layout
    ).project()


awkward1.behavior["__broadcast__", "bytestring"] = _string_broadcast
awkward1.behavior["__broadcast__", "string"] = _string_broadcast


def _string_numba_typer(viewtype):
    import numba

    return numba.types.string


def _string_numba_lower(
    context, builder, rettype, viewtype, viewval, viewproxy, attype, atval
):
    import numba
    import llvmlite.llvmpy.core
    import awkward1._connect._numba.layout

    whichpos = awkward1._connect._numba.layout.posat(
        context, builder, viewproxy.pos, viewtype.type.CONTENT
    )
    nextpos = awkward1._connect._numba.layout.getat(
        context, builder, viewproxy.arrayptrs, whichpos
    )

    whichnextpos = awkward1._connect._numba.layout.posat(
        context, builder, nextpos, viewtype.type.contenttype.ARRAY
    )

    startspos = awkward1._connect._numba.layout.posat(
        context, builder, viewproxy.pos, viewtype.type.STARTS
    )
    startsptr = awkward1._connect._numba.layout.getat(
        context, builder, viewproxy.arrayptrs, startspos
    )
    startsarraypos = builder.add(viewproxy.start, atval)
    start = awkward1._connect._numba.layout.getat(
        context, builder, startsptr, startsarraypos, viewtype.type.indextype.dtype
    )

    stopspos = awkward1._connect._numba.layout.posat(
        context, builder, viewproxy.pos, viewtype.type.STOPS
    )
    stopsptr = awkward1._connect._numba.layout.getat(
        context, builder, viewproxy.arrayptrs, stopspos
    )
    stopsarraypos = builder.add(viewproxy.start, atval)
    stop = awkward1._connect._numba.layout.getat(
        context, builder, stopsptr, stopsarraypos, viewtype.type.indextype.dtype
    )

    baseptr = awkward1._connect._numba.layout.getat(
        context, builder, viewproxy.arrayptrs, whichnextpos
    )
    rawptr = builder.add(
        baseptr,
        awkward1._connect._numba.castint(
            context, builder, viewtype.type.indextype.dtype, numba.intp, start
        ),
    )
    rawptr_cast = builder.inttoptr(
        rawptr,
        llvmlite.llvmpy.core.Type.pointer(
            llvmlite.llvmpy.core.Type.int(numba.intp.bitwidth // 8)
        ),
    )
    strsize = builder.sub(stop, start)
    strsize_cast = awkward1._connect._numba.castint(
        context, builder, viewtype.type.indextype.dtype, numba.intp, strsize
    )

    pyapi = context.get_python_api(builder)
    gil = pyapi.gil_ensure()

    strptr = builder.bitcast(rawptr_cast, pyapi.cstring)
    kind = context.get_constant(numba.types.int32, pyapi.py_unicode_1byte_kind)
    pystr = pyapi.string_from_kind_and_data(kind, strptr, strsize_cast)

    out = pyapi.to_native_value(rettype, pystr).value

    pyapi.gil_release(gil)

    return out


# awkward1.behavior["__numba_typer__", "bytestring"] = _string_numba_typer
# awkward1.behavior["__numba_lower__", "bytestring"] = _string_numba_lower
awkward1.behavior["__numba_typer__", "string"] = _string_numba_typer
awkward1.behavior["__numba_lower__", "string"] = _string_numba_lower
